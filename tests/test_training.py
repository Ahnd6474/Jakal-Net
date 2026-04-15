import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
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


from train_progressive_b_lm import (  # noqa: E402
    ASSISTANT_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    USER_TOKEN,
    build_tokenizer,
)

class TrainingTests(unittest.TestCase):
    def test_build_tokenizer_byte_bpe_with_special_tokens(self) -> None:
        text = (
            f"{USER_TOKEN}\nhello\n{ASSISTANT_TOKEN}\nworld\n{EOS_TOKEN}\n"
            f"{USER_TOKEN}\nbyte bpe test\n{ASSISTANT_TOKEN}\nworks\n{PAD_TOKEN}\n"
        )
        with TemporaryDirectory() as tmpdir:
            vocab, tokenizer_label, tokenizer_model_path = build_tokenizer(
                text,
                text_path=None,
                tokenizer="byte_bpe",
                subword_vocab_size=128,
                subword_model_type="bpe",
                tokenizer_prefix=str(Path(tmpdir) / "byte_bpe_test"),
                subword_character_coverage=0.9995,
                user_defined_symbols=(USER_TOKEN, ASSISTANT_TOKEN, EOS_TOKEN, PAD_TOKEN),
            )

            self.assertEqual(tokenizer_label, "byte_bpe")
            self.assertIsNotNone(tokenizer_model_path)
            assert tokenizer_model_path is not None
            self.assertTrue(tokenizer_model_path.exists())
            encoded = vocab.encode(text)
            self.assertGreater(encoded.numel(), 0)
            self.assertGreaterEqual(vocab.token_id(USER_TOKEN), 0)
            decoded = vocab.decode(encoded[:16].tolist())
            self.assertIsInstance(decoded, str)

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

    def test_full_sequence_causal_sampling_and_loss(self) -> None:
        torch.manual_seed(7)
        text = "full sequence causal objective keeps every position supervised. " * 8
        vocab = build_char_vocab(text)
        tokens = vocab.encode(text)
        model = ProgressiveBExampleLM(
            vocab_size=vocab.size,
            dim=12,
            seq_nodes=8,
            warmup_layers=1,
            final_refine_layers=1,
        )

        batch = sample_next_token_batch(
            tokens,
            seq_len=8,
            batch_size=2,
            device="cpu",
            full_sequence_causal=True,
        )
        loss, logits = compute_next_token_loss(
            model,
            batch,
            full_sequence_causal=True,
        )

        self.assertEqual(batch.context.shape, (2, 8))
        self.assertEqual(batch.target.shape, (2, 8))
        self.assertEqual(logits.shape, (2, 8, vocab.size))
        self.assertTrue(torch.isfinite(loss))

    def test_teacher_forcing_chunked_matches_full_teacher_forcing(self) -> None:
        torch.manual_seed(11)
        text = "teacher forcing with chunking should preserve logits exactly. " * 8
        vocab = build_char_vocab(text)
        tokens = vocab.encode(text)
        model = ProgressiveBExampleLM(
            vocab_size=vocab.size,
            dim=12,
            seq_nodes=8,
            warmup_layers=1,
            final_refine_layers=1,
        )

        batch = sample_next_token_batch(
            tokens,
            seq_len=8,
            batch_size=2,
            device="cpu",
            teacher_forcing=True,
        )
        full_loss, full_logits = compute_next_token_loss(
            model,
            batch,
            teacher_forcing=True,
        )
        chunked_loss, chunked_logits = compute_next_token_loss(
            model,
            batch,
            teacher_forcing=True,
            teacher_forcing_chunk_size=3,
        )

        torch.testing.assert_close(chunked_logits, full_logits)
        torch.testing.assert_close(chunked_loss, full_loss)

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

    def test_teacher_forcing_chunked_training_loop_runs(self) -> None:
        torch.manual_seed(32)
        text = "chunked teacher forcing should keep the B path active while limiting memory. " * 8
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
            steps=3,
            eval_interval=2,
            eval_steps=1,
            learning_rate=1e-3,
            teacher_forcing=True,
            teacher_forcing_chunk_size=3,
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

    def test_full_sequence_causal_training_loop_runs(self) -> None:
        torch.manual_seed(31)
        text = "causal sequence objective should train without prefix expansion. " * 10
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
            steps=3,
            eval_interval=2,
            eval_steps=1,
            learning_rate=1e-3,
            full_sequence_causal=True,
        )

        self.assertGreaterEqual(len(history.train_losses), 2)
        self.assertEqual(len(history.train_losses), len(history.val_losses))
        self.assertTrue(all(torch.isfinite(torch.tensor(history.train_losses))))
        self.assertTrue(all(torch.isfinite(torch.tensor(history.val_losses))))


if __name__ == "__main__":
    unittest.main()
