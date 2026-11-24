import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from omegaconf import OmegaConf
import sentencepiece
import torch
from torch.nn import functional as F
from transformers import T5Config, T5EncoderModel, T5Tokenizer

from fairseq.criterions.model_criterion import ModelCriterion
from shubert_pose import SHubertPoseModel, semantic
from shubert_pose.packed_dataset import PackedPoseDataset
from test_pose import tiny_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_pose_backbone import export  # noqa: E402
from prepare_pose_captions import prepare  # noqa: E402

SENTENCES = ["hello my name is sam", "what is your name", "i like to sign with my friends",
             "the weather is nice today", "thank you very much", "see you later my friend"]


def tiny_text_model(root):
    """A random T5 encoder and a matching tokenizer, written without any download."""
    root = Path(root)
    (root / "corpus.txt").write_text("\n".join(SENTENCES * 4) + "\n")
    sentencepiece.SentencePieceTrainer.train(
        input=str(root / "corpus.txt"), model_prefix=str(root / "spiece"), vocab_size=40, model_type="unigram",
        hard_vocab_limit=False, pad_id=0, eos_id=1, unk_id=2, bos_id=-1, minloglevel=2)
    tokenizer = T5Tokenizer(vocab_file=str(root / "spiece.model"), extra_ids=0, legacy=True)
    tokenizer.save_pretrained(root)
    torch.manual_seed(11)
    T5EncoderModel(T5Config(vocab_size=len(tokenizer), d_model=12, d_kv=4, d_ff=24, num_layers=1, num_heads=2,
                            dropout_rate=0.0)).save_pretrained(root)
    return tokenizer


def semantic_model(text_model, **kwargs):
    torch.manual_seed(3)
    settings = dict(semantic=True, text_model=str(text_model), text_dim=12, align_dim=8, token_top_ratio=1.0)
    settings.update(kwargs)
    return SHubertPoseModel(tiny_config(**settings))


def batch(clips=4, frames=6, uncaptioned=(2,)):
    generator = torch.Generator().manual_seed(5)
    source = {s: torch.randn(clips, frames, d, generator=generator)
              for s, d in zip(["body", "left_hand", "right_hand"], [4, 6, 6])}
    padding = torch.zeros(clips, frames, dtype=torch.bool)
    padding[1, 4:] = True
    labels = {s: torch.randint(5, (clips, frames), generator=generator) for s in source}
    mask = torch.zeros(clips, frames, 3, dtype=torch.bool)
    mask[:, 1] = True
    text = torch.randint(3, 30, (clips, 5), generator=generator)
    text_mask = torch.ones(clips, 5, dtype=torch.bool)
    text_mask[0, 3:] = False
    for index in uncaptioned:
        text_mask[index] = False
    return dict(source=source, padding_mask=padding, labels=labels, mask_indices=mask, text_ids=text,
                text_mask=text_mask)


def rows(values, chosen):
    return {name: ({s: v[chosen] for s, v in value.items()} if isinstance(value, dict) else value[chosen])
            for name, value in values.items()}


def distributed_worker(rank, world, root, text_model):
    """One rank of a CPU process group: align its shard against every rank's candidates."""
    root = Path(root)
    torch.distributed.init_process_group("gloo", init_method=f"file://{root / 'group'}", rank=rank, world_size=world)
    model = semantic_model(text_model).train()
    result = model(**torch.load(root / f"shard{rank}.pt"))
    losses = {name: value[0] for name, value in result["losses"].items() if name != "kmeans_loss"}
    sum(losses.values()).backward()
    (root / f"result{rank}.json").write_text(json.dumps({
        "losses": {name: float(value) for name, value in losses.items()}, "logs": result["logs"],
        "gradient": float(sum(p.grad.abs().sum() for p in model.semantic.parameters() if p.grad is not None))}))
    torch.distributed.destroy_process_group()


class SemanticAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.text_model = Path(cls.temp.name) / "text"
        cls.text_model.mkdir()
        cls.tokenizer = tiny_text_model(cls.text_model)

    def features(self, model, inputs):
        return model(inputs["source"], inputs["padding_mask"], features_only=True,
                     mask_indices=inputs["mask_indices"])["x"]

    def test_losses_reach_pose_and_alignment_heads_but_not_the_text_encoder(self):
        model, inputs = semantic_model(self.text_model), batch()
        result = model(**inputs)
        count = result["losses"]["kmeans_loss"].numel()
        self.assertEqual(set(result["losses"]), {"kmeans_loss", "sent_nce_loss", "token_nce_loss"})
        for name in ("sent_nce_loss", "token_nce_loss"):
            self.assertEqual(result["losses"][name].shape, (count,))
            self.assertTrue(torch.isfinite(result["losses"][name]).all())
            self.assertGreater(float(result["losses"][name][0]), 0)
        self.assertEqual(result["logs"]["caption_pairs"], 3.0)
        (result["losses"]["sent_nce_loss"][0] + result["losses"]["token_nce_loss"][0]).backward()
        heads = model.semantic
        for layer in (heads.pose_sentence[0], heads.text_sentence[0], heads.pose_token[0], heads.text_token[0],
                      model.projections["left_hand"], model.encoder.layers[0].fc1):
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
        encoder = heads.text_encoder()
        self.assertFalse(any(p.requires_grad or p.grad is not None for p in encoder.parameters()))
        self.assertFalse(encoder.training)
        self.assertFalse(any("text_encoder" in name or "shared" in name for name in model.state_dict()))
        owned = {id(p) for p in model.parameters()}
        self.assertFalse(any(id(p) in owned for p in encoder.parameters()))
        self.assertEqual(next(model.half().semantic.text_encoder().parameters()).dtype, torch.float32)

    def test_losses_match_a_direct_computation(self):
        model, inputs = semantic_model(self.text_model), batch()
        result = model(**inputs)
        kept = torch.tensor([0, 1, 3])
        x, valid = self.features(model, inputs)[kept], ~inputs["padding_mask"][kept]
        text_mask = inputs["text_mask"][kept]
        with torch.no_grad():
            tokens = model.semantic.text_encoder()(input_ids=inputs["text_ids"][kept],
                                                   attention_mask=text_mask.long()).last_hidden_state
        pose = torch.stack([x[i, valid[i]].mean(0) for i in range(3)])
        sentence = torch.stack([tokens[i, text_mask[i]].mean(0) for i in range(3)])
        pose = F.normalize(model.semantic.pose_sentence(pose), dim=-1)
        sentence = F.normalize(model.semantic.text_sentence(sentence), dim=-1)
        logits, target = pose @ sentence.t() / 0.07, torch.arange(3)
        expected = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))
        torch.testing.assert_close(result["losses"]["sent_nce_loss"][0], expected, rtol=1e-4, atol=1e-5)

        steps = F.normalize(model.semantic.pose_token(x), dim=-1)
        words = F.normalize(model.semantic.text_token(tokens), dim=-1)
        text_to_pose, pose_to_text = torch.zeros(3, 3), torch.zeros(3, 3)
        for i in range(3):
            for j in range(3):
                similarity = words[i, text_mask[i]] @ steps[j, valid[j]].t()
                text_to_pose[i, j] = similarity.max(1).values.mean()
                similarity = steps[i, valid[i]] @ words[j, text_mask[j]].t()
                pose_to_text[i, j] = similarity.max(1).values.mean()
        expected = 0.5 * (F.cross_entropy(text_to_pose / 0.07, target) + F.cross_entropy(pose_to_text / 0.07, target))
        torch.testing.assert_close(result["losses"]["token_nce_loss"][0], expected, rtol=1e-4, atol=1e-5)

    def test_padding_and_uncaptioned_clips_do_not_change_alignment(self):
        model, inputs = semantic_model(self.text_model, token_top_ratio=0.5), batch()
        first = model(**inputs)["losses"]
        changed = batch()
        for stream in changed["source"]:
            changed["source"][stream][1, 4:] = 9.0   # padded frames
            changed["source"][stream][2] += 3.0      # the clip without a caption
        changed["text_ids"][0, 3:] = 29              # padded caption tokens
        changed["text_ids"][2] = 7                   # masked-out caption
        second = model(**changed)["losses"]
        for name in ("sent_nce_loss", "token_nce_loss"):
            torch.testing.assert_close(first[name], second[name], rtol=1e-4, atol=1e-5)

    def test_batches_without_captions_give_zero_alignment_loss(self):
        model, inputs = semantic_model(self.text_model), batch(uncaptioned=(0, 1, 2, 3))
        result = model(**inputs)
        self.assertEqual(result["logs"], {"sent_acc": 0.0, "caption_pairs": 0.0, "token_acc": 0.0})
        self.assertEqual(float(result["losses"]["sent_nce_loss"].sum() + result["losses"]["token_nce_loss"].sum()), 0)
        sum(value.sum() for value in result["losses"].values()).backward()
        with self.assertRaisesRegex(ValueError, "task.captions"):
            model(inputs["source"], inputs["padding_mask"], inputs["labels"], mask_indices=inputs["mask_indices"])

    def test_sentence_only_alignment_has_no_token_modules(self):
        model = semantic_model(self.text_model, token_align=False)
        result = model(**batch())
        self.assertEqual(set(result["losses"]), {"kmeans_loss", "sent_nce_loss"})
        self.assertEqual(set(result["logs"]), {"sent_acc", "caption_pairs"})
        self.assertFalse(any("token" in name for name in model.state_dict()))

    def test_chunked_interaction_and_top_token_selection(self):
        generator = torch.Generator().manual_seed(2)
        words = F.normalize(torch.randn(5, 4, 8, generator=generator), dim=-1)
        steps = F.normalize(torch.randn(5, 7, 8, generator=generator), dim=-1)
        word_mask, step_mask = torch.ones(5, 4, dtype=torch.bool), torch.ones(5, 7, dtype=torch.bool)
        word_mask[0, 2:], step_mask[3, 3:] = False, False
        whole = semantic.interaction(words, word_mask, steps, step_mask)
        scores = semantic.best_local_matches(words, word_mask, steps, step_mask)
        with mock.patch.object(semantic, "BLOCK_ELEMENTS", 1):
            torch.testing.assert_close(semantic.interaction(words, word_mask, steps, step_mask), whole)
            for chunked, direct in zip(semantic.best_local_matches(words, word_mask, steps, step_mask), scores):
                torch.testing.assert_close(chunked, direct)
        expected = torch.stack([(words[0, :2] @ steps[j, step_mask[j]].t()).max(1).values.mean() for j in range(5)])
        torch.testing.assert_close(whole[0], expected)
        similarity = torch.einsum("nld,cmd->nclm", words, steps)
        torch.testing.assert_close(scores[1][2], similarity[:, 2, :, :][word_mask].max(0).values)
        kept, kept_mask = semantic.keep_top_tokens(steps, step_mask, scores[1], 0.5)
        self.assertEqual(kept.shape, (5, 4, 8))
        self.assertEqual(kept_mask.sum(1).tolist(), [4, 4, 4, 3, 4])   # padding is only kept after valid frames
        best = scores[1][0].argmax()
        self.assertTrue((kept[0] == steps[0, best]).all(-1).any())

    def test_criterion_weights_alignment_like_the_cluster_loss(self):
        model, inputs = semantic_model(self.text_model), batch()
        weights = {"kmeans_loss": 1.0, "sent_nce_loss": 0.5, "token_nce_loss": 0.25}
        torch.manual_seed(0)
        loss, sample_size, logs = ModelCriterion(None, weights, [])(model, {"net_input": inputs, "id": torch.arange(4)})
        direct = model(**inputs)["losses"]
        expected = direct["kmeans_loss"].mean() + 0.5 * direct["sent_nce_loss"][0] + 0.25 * direct["token_nce_loss"][0]
        torch.testing.assert_close(loss / sample_size, expected, rtol=1e-4, atol=1e-5)
        self.assertAlmostEqual(logs["loss_sent_nce_loss"] / sample_size, 0.5 * float(direct["sent_nce_loss"][0]),
                               places=4)
        self.assertEqual(logs["caption_pairs"], 3.0)

    def distributed(self, shards):
        root = Path(tempfile.mkdtemp(dir=self.temp.name))
        for rank, shard in enumerate(shards):
            torch.save(shard, root / f"shard{rank}.pt")
        environment = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        code = "import sys, test_pose_semantic as t; t.distributed_worker(int(sys.argv[1]), {}, {!r}, {!r})".format(
            len(shards), str(root), str(self.text_model))
        workers = [subprocess.Popen([sys.executable, "-c", code, str(rank)], env=environment,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                   for rank in range(len(shards))]
        for worker in workers:
            output, _ = worker.communicate(timeout=300)
            self.assertEqual(worker.returncode, 0, output)
        return [json.loads((root / f"result{rank}.json").read_text()) for rank in range(len(shards))]

    def test_gathered_candidates_reproduce_the_single_process_loss(self):
        inputs = batch(clips=5, uncaptioned=(2,))
        whole = semantic_model(self.text_model).train()(**inputs)
        results = self.distributed([rows(inputs, slice(0, 3)), rows(inputs, slice(3, 5))])
        self.assertEqual([r["logs"]["caption_pairs"] for r in results], [2.0, 2.0])
        for name in ("sent_nce_loss", "token_nce_loss"):
            # Each rank averages over its own anchors, against every rank's candidates.
            combined = sum(r["losses"][name] * r["logs"]["caption_pairs"] for r in results) / 4
            self.assertAlmostEqual(combined, float(whole["losses"][name][0]), places=4)
        self.assertTrue(all(r["gradient"] > 0 for r in results))

    def test_a_rank_without_captions_still_supplies_no_candidates(self):
        inputs = batch(clips=4, uncaptioned=(2, 3))
        alone = semantic_model(self.text_model).train()(**rows(inputs, slice(0, 2)))
        results = self.distributed([rows(inputs, slice(0, 2)), rows(inputs, slice(2, 4))])
        self.assertEqual(results[1]["losses"], {"sent_nce_loss": 0.0, "token_nce_loss": 0.0})
        self.assertEqual(results[1]["gradient"], 0)
        for name in ("sent_nce_loss", "token_nce_loss"):
            self.assertAlmostEqual(results[0]["losses"][name], float(alone["losses"][name][0]), places=4)

    def test_evaluation_does_not_gather(self):
        model, inputs = semantic_model(self.text_model).eval(), batch()
        with mock.patch.object(semantic, "process_group", return_value=(0, 2)), \
                mock.patch.object(semantic, "gather_counts", side_effect=AssertionError("gathered in evaluation")):
            model(**inputs)
            with self.assertRaisesRegex(AssertionError, "gathered"):
                model.train()(**inputs)

    def checkpoint(self, model, path):
        cfg = {"model": dict(OmegaConf.to_container(OmegaConf.structured(model.cfg), enum_to_str=True),
                             _name="shubert_pose"),
               "task": {"captions": "/somewhere/captions.json"}}
        torch.save({"cfg": cfg, "model": model.state_dict()}, path)

    def test_initialization_requires_exactly_the_backbone(self):
        root = Path(tempfile.mkdtemp(dir=self.temp.name))
        torch.manual_seed(8)
        pretrained = SHubertPoseModel(tiny_config())
        self.checkpoint(pretrained, root / "masked.pt")
        model = semantic_model(self.text_model)
        model.initialize_from(root / "masked.pt")
        for name, value in pretrained.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value)
        self.checkpoint(SHubertPoseModel(tiny_config(stream_names=["body"], stream_dims=[4])), root / "other.pt")
        with self.assertRaisesRegex(ValueError, "backbone keys differ"):
            model.initialize_from(root / "other.pt")

    def test_exported_backbone_loads_strictly_without_alignment(self):
        root = Path(tempfile.mkdtemp(dir=self.temp.name))
        model = semantic_model(self.text_model)
        self.checkpoint(model, root / "aligned.pt")
        export(root / "aligned.pt", root / "backbone.pt")
        saved = torch.load(root / "backbone.pt", weights_only=False)
        self.assertFalse(saved["cfg"]["model"]["semantic"] or saved["cfg"]["model"]["text_model"])
        self.assertEqual(saved["cfg"]["task"]["captions"], "")
        self.assertTrue(saved["export"]["removed"])
        self.assertFalse(any(name.startswith("semantic.") for name in saved["model"]))
        plain = SHubertPoseModel(tiny_config())
        plain.load_state_dict(saved["model"], strict=True)
        inputs = batch()
        torch.testing.assert_close(self.features(plain.eval(), inputs), self.features(model.eval(), inputs))
        with self.assertRaises(FileExistsError):
            export(root / "aligned.pt", root / "backbone.pt")
        with self.assertRaisesRegex(ValueError, "model.semantic=true"):
            export(root / "backbone.pt", root / "again.pt")


class CaptionPackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "text").mkdir()
        self.tokenizer = tiny_text_model(self.root / "text")
        np.save(self.root / "offsets.npy", np.array([0, 4, 11, 14]))
        np.save(self.root / "indices.npy", np.array([0, 1, 2]))
        np.save(self.root / "body.npy", np.repeat(np.arange(14, dtype=np.float32)[:, None], 4, axis=1))
        np.save(self.root / "labels.npy", np.arange(14, dtype=np.uint16))
        meta = {"format": "shubert-pose-pack-v1", "dimensions": {"body": 4}, "codebook_size": 16,
                "features": {"body": str(self.root / "body.npy")}, "labels": {"body": str(self.root / "labels.npy")}}
        for key in ("offsets", "indices"):
            meta[key] = str(self.root / f"{key}.npy")
            meta[f"{key}_sha256"] = hashlib.sha256((self.root / f"{key}.npy").read_bytes()).hexdigest()
        self.manifest = self.root / "train.pack.json"
        self.manifest.write_text(json.dumps(meta))
        self.records = self.root / "records.jsonl"
        self.records.write_text("".join(json.dumps(row) + "\n" for row in (
            {"id": "c", "pack_index": 2, "text": " ".join(SENTENCES)},
            {"id": "a", "pack_index": 0, "text": SENTENCES[0]})))

    def prepare(self, **kwargs):
        settings = dict(records=[str(self.records)], pack=str(self.manifest), text_model=str(self.root / "text"),
                        max_tokens=32, output=str(self.root / "captions"))
        settings.update(kwargs)
        prepare(type("Arguments", (), settings)())
        return Path(settings["output"]) / "captions.json"

    def data(self, captions):
        return PackedPoseDataset(self.manifest, ["body"], [4], 16, max_frames=5, shuffle=False, captions=captions)

    def test_captions_follow_their_clips_through_batching_and_pickling(self):
        captions = self.prepare()
        summary = json.loads(captions.read_text())
        self.assertEqual((summary["clips"], summary["captioned_clips"], summary["captions_at_token_limit"]), (3, 2, 1))
        data = self.data(captions)
        expected = self.tokenizer(SENTENCES[0])["input_ids"]
        self.assertEqual(data[0]["text"].tolist(), expected)
        self.assertEqual(len(data[1]["text"]), 0)
        self.assertEqual(data[2]["text"].tolist(), self.tokenizer(" ".join(SENTENCES), truncation=True,
                                                                  max_length=32)["input_ids"])
        net_input = data.collater([data[2], data[1], data[0]])["net_input"]
        self.assertEqual(net_input["text_ids"].shape, (3, 32))
        self.assertLess(len(expected), 32)
        self.assertEqual(net_input["text_mask"].sum(1).tolist(), [32, 0, len(expected)])
        self.assertEqual(net_input["text_ids"][2, :len(expected)].tolist(), expected)
        restored = pickle.loads(pickle.dumps(data))
        self.assertIsInstance(restored.text, np.memmap)
        self.assertEqual(restored[0]["text"].tolist(), expected)
        self.assertNotIn("text", PackedPoseDataset(self.manifest, ["body"], [4], 16, max_frames=5)[0])
        self.assertNotIn("text_ids", self.data(None).collater([self.data(None)[0]])["net_input"])

    def test_preparation_is_reused_and_rejects_changed_inputs(self):
        captions = self.prepare()
        before = captions.read_text()
        self.prepare()
        self.assertEqual(captions.read_text(), before)
        with self.assertRaisesRegex(ValueError, "settings changed"):
            self.prepare(max_tokens=6)
        self.records.write_text(json.dumps({"pack_index": 3, "text": "x"}) + "\n")
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            self.prepare(output=str(self.root / "other"))

    def test_captions_for_other_clips_or_changed_tokens_are_rejected(self):
        captions = self.prepare()
        tokens = np.load(self.root / "captions" / "captions.tokens.npy")
        np.save(self.root / "captions" / "captions.tokens.npy", tokens[::-1].copy())
        with self.assertRaisesRegex(ValueError, "caption tokens changed"):
            self.data(captions)
        np.save(self.root / "captions" / "captions.tokens.npy", tokens)
        self.data(captions)
        summary = json.loads(captions.read_text())
        summary["pack_offsets_sha256"] = "0" * 64
        captions.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "different packed clips"):
            self.data(captions)


if __name__ == "__main__":
    unittest.main()
