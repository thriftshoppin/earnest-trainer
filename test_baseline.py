import unittest
import torch
import baseline
import earnest


class BaselineTests(unittest.TestCase):
    def test_complete_stories_only_with_budget(self):
        raw = b'aaa<|endoftext|>bbb<|endoftext|>unfinished'
        self.assertEqual(baseline.select_stories(raw, 4, min_bytes=1), [b'aaa\n'])
        self.assertEqual(baseline.select_stories(raw, 100, min_bytes=1), [b'aaa\n', b'bbb\n'])

    def test_validation_excludes_normalized_duplicates(self):
        raw = b'a  b<|endoftext|>c d<|endoftext|>c  d<|endoftext|>'
        result = baseline.select_stories(raw, 100, [baseline.digest(b'a b')], min_bytes=1)
        self.assertEqual(result, [b'c d\n'])

    def test_sampling_preserves_training_rng(self):
        torch.set_num_threads(1)
        model = earnest.Earnest(earnest.Config())
        state = torch.get_rng_state().clone()
        result = baseline.samples(model)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(len(result), 3)
        self.assertTrue(model.training)


if __name__ == '__main__':
    unittest.main()
