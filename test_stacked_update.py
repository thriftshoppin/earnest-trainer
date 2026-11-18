import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

import baseline
import stacked


class StackedUpdateTest(unittest.TestCase):
    def test_fork_cache_resume_and_reject_without_freezing(self):
        torch.set_num_threads(1)

        def cli(*args):
            with patch.object(sys, 'argv', ['stacked.py', *map(str, args)]):
                stacked.main()

        def read_split(split, budget, revision, exclude=()):
            story = (b'Willow reads the evidence. ' if split == 'train'
                     else b'Willow checks the conclusion. ') * 16
            return [story], {'received_sha256': split, 'story_hashes': [split]}

        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            source = Path(temp) / 'baseline'
            target = Path(temp) / 'phase-two'
            extra = Path(temp) / 'examples.txt'
            extra.write_bytes(b'User: Here is new evidence. Willow: I changed my conclusion.\n' * 60)
            with patch.object(baseline, 'read_split', side_effect=read_split), patch.dict(stacked._corpus, clear=True):
                cli('new', '--stack', source, '--width', 8, '--context', 16, '--train-bytes', 65536)
                cli('train', '--stack', source, '--steps', 1)
                cli('freeze', '--stack', source)
                original = (source / 'stack.json').read_bytes()
                cli('fork', '--source', source, '--stack', target, '--extra', extra,
                    '--validation-batches', 3)
                self.assertFalse((target / 'cache').exists())
                self.assertEqual((source / 'block-000.pt').read_bytes(), (target / 'block-000.pt').read_bytes())
                with self.assertRaises(SystemExit):
                    cli('fork', '--source', source, '--stack', target)
                cli('grow', '--stack', target, '--blocks', 2, '--round-steps', 1,
                    '--stage-steps', 1, '--accept-gain', 100)
                self.assertEqual(len(stacked.read_meta(target)['blocks']), 1)
                stage = target / 'stage-001'
                self.assertTrue((stage / 'best.pt').is_file())
                cache = json.loads((target / 'cache' / 'cache.json').read_text())
                self.assertEqual(cache['windows']['valid'], 6)
                final = json.loads((stage / 'evaluations.jsonl').read_text().splitlines()[-1])
                self.assertEqual({sample['seed'] for sample in final['samples']}, {2026, 2027, 2028})
                self.assertIsNotNone(final['own_data_validation_bpb'])
                cli('grow', '--stack', target, '--blocks', 2, '--round-steps', 1,
                    '--stage-steps', 2, '--accept-gain', 0)
                self.assertEqual(len(stacked.read_meta(target)['blocks']), 2)
                self.assertEqual((source / 'stack.json').read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
