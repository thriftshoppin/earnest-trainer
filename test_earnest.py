import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import earnest


class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_causality(self):
        torch.manual_seed(5)
        model = earnest.Earnest(earnest.Config()).eval()
        x = torch.randint(0, 256, (1, 32))
        other = x.clone()
        other[:, 16:] = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            torch.testing.assert_close(model(x)[:, :16], model(other)[:, :16])

    def test_checkpoint_preserves_next_update(self):
        torch.manual_seed(7)
        cfg = earnest.Config()
        model = earnest.Earnest(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        documents, _ = earnest.split_documents([bytes(range(256)) * 10], cfg.context)
        rng = random.Random(7)
        def update(m, opt, generator):
            x, y = earnest.batch(documents, cfg.context, 2, generator)
            opt.zero_grad(set_to_none=True)
            loss = earnest.loss_for(m, x, y)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            opt.step()
        update(model, optimizer, rng)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.pt'
            earnest.save_checkpoint(path, model, optimizer, 1, rng, [])
            update(model, optimizer, rng)
            state = torch.load(path, weights_only=True)
            restored = earnest.Earnest(earnest.Config(**state['config']))
            restored.load_state_dict(state['model'])
            restored_opt = torch.optim.AdamW(restored.parameters(), lr=0.001)
            restored_opt.load_state_dict(state['optimizer'])
            restored_rng = random.Random()
            restored_rng.setstate(state['python_rng'])
            torch.set_rng_state(state['torch_rng'])
            update(restored, restored_opt, restored_rng)
            for first, second in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_split_has_no_shared_positions(self):
        data = bytes(range(256)) * 10
        train, valid = earnest.split_documents([data], 64)
        boundary = int(len(data) * .85)
        self.assertEqual(train[0].tolist(), list(data[:boundary]))
        self.assertEqual(valid[0].tolist(), list(data[boundary:]))

    def test_reader_strips_scripts_and_keeps_provenance(self):
        html = b'<p>Train on this.</p><script>ignore this</script><p>More text.</p>'
        with patch('earnest.fetch', return_value=(html, 'text/html', 'https://example.org/text')):
            docs, records = earnest.read_sources(['https://example.org/text'])
        self.assertNotIn(b'ignore', docs[0])
        self.assertEqual(records[0]['bytes_received'], len(html))
        self.assertEqual(len(records[0]['sha256']), 64)

    def test_reject_insecure_source(self):
        with self.assertRaises(ValueError):
            earnest.fetch('http://example.org/text', 100)


if __name__ == '__main__':
    unittest.main()
