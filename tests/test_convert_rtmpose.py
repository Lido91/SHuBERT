import pickle
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from convert_rtmpose import HANDS, convert, fill_missing, load_pose  # noqa: E402


def synthetic(frames=5, seed=0):
    rng = np.random.default_rng(seed)
    keypoints = rng.uniform(100, 500, size=(frames, 133, 2)).astype(np.float32)
    keypoints[:, 5], keypoints[:, 6] = [360, 300], [240, 300]  # shoulders 120 px apart: unit = 60
    keypoints[:, 0], keypoints[:, 1] = [300, 200], [310, 190]  # nose, left eye
    return keypoints, np.ones((frames, 133), dtype=np.float32)


class ConvertTests(unittest.TestCase):
    def test_shapes_and_signing_space(self):
        keypoints, scores = synthetic()
        streams = convert(keypoints, scores)
        self.assertEqual({k: v.shape for k, v in streams.items()},
                         {"body": (5, 14), "left_hand": (5, 42), "right_hand": (5, 42)})
        self.assertTrue(all(v.dtype == np.float32 and np.isfinite(v).all() for v in streams.values()))
        body = streams["body"].reshape(5, 7, 2)
        np.testing.assert_allclose(body[:, 0, 0], 0, atol=1e-6)  # nose is the horizontal centre
        np.testing.assert_allclose(body[:, 1, 0] - body[:, 2, 0], 120 / 360, atol=1e-6)  # 2 of 6 units
        np.testing.assert_allclose(body[:, 0, 1], (200 - 160) / 420 - 0.5, atol=1e-6)

    def test_hands_are_wrist_relative_and_position_free(self):
        keypoints, scores = synthetic()
        moved = keypoints.copy()
        moved[:, HANDS["left_hand"]] += [70, -40]  # move the whole hand, keep its shape
        before, after = convert(keypoints, scores), convert(moved, scores)
        np.testing.assert_allclose(before["left_hand"], after["left_hand"], atol=1e-5)
        np.testing.assert_allclose(before["left_hand"][:, :2], 0, atol=1e-6)  # wrist is the origin
        expected = (keypoints[:, 92] - keypoints[:, 91]) / 60
        np.testing.assert_allclose(before["left_hand"][:, 2:4], expected, atol=1e-5)

    def test_invariant_to_image_translation_and_scale(self):
        keypoints, scores = synthetic()
        before, after = convert(keypoints, scores), convert(keypoints * 2.5 + [40, -15], scores)
        for stream in before:
            np.testing.assert_allclose(before[stream], after[stream], atol=1e-4)

    def test_missing_detections(self):
        keypoints, scores = synthetic()
        scores[2, HANDS["right_hand"]] = 0.1  # hand lost for one frame
        scores[1, 9] = 0.0  # left wrist lost for one frame
        keypoints[3, 7] = np.nan  # non-finite elbow
        streams = convert(keypoints, scores)
        self.assertTrue(all(np.isfinite(v).all() for v in streams.values()))
        self.assertTrue((streams["right_hand"][2] == 0).all())
        self.assertTrue((streams["right_hand"][1] != 0).any())
        body = streams["body"].reshape(5, 7, 2)
        np.testing.assert_allclose(body[1, 5], body[0, 5], atol=1e-6)  # held from the previous frame
        np.testing.assert_allclose(body[3, 3], body[2, 3], atol=1e-6)
        empty = convert(keypoints, np.zeros_like(scores))
        self.assertTrue(all((v == 0).all() for v in empty.values()))

    def test_fill_missing(self):
        values = np.arange(5, dtype=np.float32)[:, None]
        valid = np.array([False, True, False, False, True])
        np.testing.assert_array_equal(fill_missing(values, valid)[:, 0], [1, 1, 1, 1, 4])

    def test_load_pose_accepts_person_axis_and_bounds(self):
        keypoints, scores = synthetic(frames=8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.pkl"
            with path.open("wb") as handle:
                pickle.dump({"keypoints": keypoints[:, None], "scores": scores[:, None], "start": 2, "end": 6}, handle)
            loaded, confidence = load_pose(path)
        self.assertEqual((loaded.shape, confidence.shape), ((4, 133, 2), (4, 133)))
        np.testing.assert_array_equal(loaded, keypoints[2:6])


if __name__ == "__main__":
    unittest.main()
