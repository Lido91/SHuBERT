import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class SupervisorTests(unittest.TestCase):
    def load_supervisor(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/supervise_pose_sft.py'
        spec = importlib.util.spec_from_file_location('pose_supervisor_test_module', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_exhausted_retries_mark_existing_training_progress_failed(self):
        module = self.load_supervisor()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / module.STAGES['how2sign_grpo'][0]
            (output / 'train').mkdir(parents=True)
            progress = output / 'train/progress.json'
            progress.write_text(json.dumps({'state': 'running', 'step': 24, 'updated_unix': 123.}))
            process = SimpleNamespace(pid=1, poll=lambda: 1, returncode=1)
            with patch.object(module, 'OUTPUTS', root), \
                    patch.object(module.subprocess, 'run', return_value=SimpleNamespace(stdout='RUNNING\n')), \
                    patch.object(module.subprocess, 'Popen', return_value=process):
                with self.assertRaises(SystemExit):
                    module.main(SimpleNamespace(stage='how2sign_grpo', gpu_job='123', retries=0))
            training = json.loads(progress.read_text())
            self.assertEqual(training['state'], 'failed')
            self.assertEqual(training['step'], 24)
            self.assertEqual(training['last_train_progress_unix'], 123.)
            self.assertEqual(training['exit_code'], 1)

    def test_completed_long_run_is_allowed_to_exit_without_stall_termination(self):
        from unittest.mock import Mock
        module = self.load_supervisor()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / module.STAGES['how2sign_grpo'][0]
            (output / 'train').mkdir(parents=True)
            (output / 'train/progress.json').write_text('{"state":"complete","step":3884}')
            (output / 'train/completed.json').write_text('{"state":"complete","step":3884}')
            clock = [0.]
            polls = iter([None, 0])

            def poll():
                clock[0] = 4 * 3600
                return next(polls)

            process = SimpleNamespace(pid=1, poll=poll, returncode=0, terminate=Mock(), wait=Mock())
            with patch.object(module, 'OUTPUTS', root), \
                    patch.object(module.subprocess, 'run', return_value=SimpleNamespace(stdout='RUNNING\n')), \
                    patch.object(module.subprocess, 'Popen', return_value=process), \
                    patch.object(module.time, 'time', side_effect=lambda: clock[0]), \
                    patch.object(module.time, 'sleep'):
                module.main(SimpleNamespace(stage='how2sign_grpo', gpu_job='123', retries=0))
            process.terminate.assert_not_called()
            status = json.loads((output / 'supervisor.json').read_text())
            self.assertEqual(status['state'], 'complete')
            self.assertEqual(status['exit_code'], 0)

    def test_continuation_does_not_mistake_original_completion_for_its_own(self):
        module = self.load_supervisor()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / module.STAGES['how2sign'][0] / 'train'
            original.mkdir(parents=True)
            (original / 'completed.json').write_text('{"state":"complete"}')
            output = root / module.STAGES['how2sign_continue'][0]
            process = SimpleNamespace(pid=1, poll=lambda: 1, returncode=1)
            with patch.object(module, 'OUTPUTS', root), \
                    patch.object(module.subprocess, 'run', return_value=SimpleNamespace(stdout='RUNNING\n')), \
                    patch.object(module.subprocess, 'Popen', return_value=process) as launch:
                with self.assertRaises(SystemExit):
                    module.main(SimpleNamespace(stage='how2sign_continue', gpu_job='123', retries=0))
            self.assertEqual(json.loads((output / 'supervisor.json').read_text())['state'], 'failed')
            self.assertIn('run_how2sign_pose_mt5_continue.sh', launch.call_args.args[0][-1])

    def test_failed_step_retries_and_completes_without_ending_reservation(self):
        module = self.load_supervisor()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / module.STAGES['how2sign'][0]
            calls = []

            def start(command, **kwargs):
                calls.append(command)
                code = 1 if len(calls) == 1 else 0
                if code == 0:
                    (output / 'train').mkdir()
                    (output / 'train/completed.json').write_text('{"state":"complete"}')
                return SimpleNamespace(pid=100 + len(calls), poll=lambda: code, returncode=code)

            with patch.object(module, 'OUTPUTS', root), \
                    patch.object(module.subprocess, 'run', return_value=SimpleNamespace(stdout='RUNNING\n')) as query, \
                    patch.object(module.subprocess, 'Popen', side_effect=start), patch.object(module.time, 'sleep'):
                module.main(SimpleNamespace(stage='how2sign', gpu_job='123', retries=3))
            status = json.loads((output / 'supervisor.json').read_text())
            self.assertEqual(status['state'], 'complete')
            self.assertEqual(status['attempt'], 2)
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(command[0] == 'srun' and '--jobid=123' in command for command in calls))
            self.assertTrue(all(call.args[0][0] == 'squeue' for call in query.call_args_list))

    def test_retries_are_bounded_and_explicit_pause_is_respected(self):
        for paused in (False, True):
            with self.subTest(paused=paused), tempfile.TemporaryDirectory() as directory:
                module = self.load_supervisor()
                root = Path(directory)
                output = root / module.STAGES['how2sign'][0]
                if paused:
                    (output / 'train').mkdir(parents=True)
                    (output / 'train/progress.json').write_text('{"state":"paused"}')
                process = SimpleNamespace(pid=1, poll=lambda: 1, returncode=1)
                with patch.object(module, 'OUTPUTS', root), \
                        patch.object(module.subprocess, 'run', return_value=SimpleNamespace(stdout='RUNNING\n')), \
                        patch.object(module.subprocess, 'Popen', return_value=process) as launch, \
                        patch.object(module.time, 'sleep'):
                    args = SimpleNamespace(stage='how2sign', gpu_job='123', retries=1)
                    if paused:
                        module.main(args)
                    else:
                        with self.assertRaises(SystemExit):
                            module.main(args)
                status = json.loads((output / 'supervisor.json').read_text())
                self.assertEqual(status['state'], 'paused' if paused else 'failed')
                self.assertEqual(launch.call_count, 1 if paused else 2)


if __name__ == '__main__':
    unittest.main()
