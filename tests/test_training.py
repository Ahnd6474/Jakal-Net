import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from progressive_b_example import (  # noqa: E402
    ProgressiveBExampleLM,
    build_char_vocab,
    compute_next_token_loss,
    estimate_next_token_loss,
    generate_next_tokens,
    sample_next_token_batch,
    split_train_val,
    train_next_token_model,
)


class TrainingTests(unittest.TestCase):
    def test_char_vocab_roundtrip(self) -> None:
        vocab = build_char_vocab("abca")
        encoded = vocab.encode("caba")
        decoded = vocab.decode(encoded.tolist())

        self.assertEqual(vocab.size, 3)
        self.assertEqual(decoded, "caba")

    def test_sampling_returns_context_and_target(self) -> None:
        tokens = torch.arange(20, dtype=torch.long)

        batch = sample_next_token_batch(tokens, seq_len=5, batch_size=4, device="cpu")

        self.assertEqual(batch.context.shape, (4, 5))
        self.assertEqual(batch.target.shape, (4,))

    def test_training_loop_runs_and_returns_history(self) -> None:
        torch.manual_seed(30)
        text = "progressive b activation helps stable training. " * 12
        vocab = build_char_vocab(text)
        tokens = vocab.encode(text)
        train_tokens, val_tokens = split_train_val(tokens, train_fraction=0.8)
        model = ProgressiveBExampleLM(
            vocab_size=vocab.size,
            dim=12,
            seq_nodes=8,
            warmup_layers=1,
            final_refine_layers=1,
        )

        history = train_next_token_model(
            model,
            train_tokens,
            val_tokens,
            seq_len=8,
            batch_size=4,
            device="cpu",
            steps=4,
            eval_interval=2,
            eval_steps=2,
            learning_rate=1e-3,
        )

        self.assertGreaterEqual(len(history.train_losses), 2)
        self.assertEqual(len(history.train_losses), len(history.val_losses))
        self.assertTrue(all(torch.isfinite(torch.tensor(history.train_losses))))
        self.assertTrue(all(torch.isfinite(torch.tensor(history.val_losses))))

        batch = sample_next_token_batch(train_tokens, seq_len=8, batch_size=2, device="cpu")
        loss, logits = compute_next_token_loss(model, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(logits.shape, (2, vocab.size))

        estimated = estimate_next_token_loss(
            model,
            val_tokens,
            seq_len=8,
            batch_size=2,
            device="cpu",
            eval_steps=2,
        )
        self.assertTrue(torch.isfinite(torch.tensor(estimated)))

        prompt = train_tokens[:8]
        generated = generate_next_tokens(
            model,
            prompt,
            max_new_tokens=5,
            seq_len=8,
            device="cpu",
        )
        self.assertEqual(generated.shape, (13,))


if __name__ == "__main__":
    unittest.main()
