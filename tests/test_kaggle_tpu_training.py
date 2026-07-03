import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_kaggle_tpu_hop1 import TokenBlockDataset, learning_rate_for_step  # noqa: E402


class KaggleTpuTrainingTests(unittest.TestCase):
    def test_token_block_dataset_returns_shifted_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            token_path = Path(temporary_directory) / "tokens.bin"
            np.arange(11, dtype=np.uint16).tofile(token_path)

            dataset = TokenBlockDataset(token_path, seq_len=5)

            self.assertEqual(len(dataset), 2)
            input_ids, target_ids = dataset[1]
            self.assertTrue(torch.equal(input_ids, torch.tensor([5, 6, 7, 8, 9])))
            self.assertTrue(torch.equal(target_ids, torch.tensor([6, 7, 8, 9, 10])))
            dataset.close()

    def test_learning_rate_warms_up_and_decays(self) -> None:
        args = argparse.Namespace(
            warmup_steps=10,
            decay_steps=100,
            learning_rate=2.0e-4,
            min_learning_rate=2.0e-5,
        )

        start = learning_rate_for_step(args, 0)
        peak = learning_rate_for_step(args, 10)
        end = learning_rate_for_step(args, 110)

        self.assertGreater(peak, start)
        self.assertAlmostEqual(peak, args.learning_rate)
        self.assertAlmostEqual(end, args.min_learning_rate)


if __name__ == "__main__":
    unittest.main()
