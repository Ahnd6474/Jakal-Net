import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_causal_memory_lm import StreamingTokenBatcher  # noqa: E402

from jakal_net.causal_memory_lm import CausalHierarchicalMemoryLM, MemoryScanOutput  # noqa: E402


class CausalMemoryLMTests(unittest.TestCase):
    def test_forward_returns_sequence_logits_and_memory(self) -> None:
        torch.manual_seed(7)
        model = CausalHierarchicalMemoryLM(
            vocab_size=32,
            dim=8,
            max_seq_len=12,
            s_layers=1,
            memory_slots=(6, 4, 2),
            prediction_layers=1,
            s_window=12,
            prediction_window=4,
            memory_topk=2,
            pairwise_rank=4,
            route_rank=4,
        )
        token_ids = torch.randint(0, 32, (2, 5))

        output = model(token_ids, return_memory_state=True, return_layers=True)

        self.assertIsInstance(output, MemoryScanOutput)
        self.assertEqual(output.logits.shape, (2, 5, 32))
        self.assertEqual(len(output.memory_state), 3)
        self.assertEqual(output.memory_state[0].val.shape, (2, 6, 8))
        assert output.sequence_layer is not None
        assert output.query_layer is not None
        self.assertEqual(output.sequence_layer.val.shape, (2, 6, 8))
        self.assertEqual(output.query_layer.val.shape, (2, 5, 8))

    def test_reset_mask_preserves_fresh_path_for_selected_items(self) -> None:
        torch.manual_seed(8)
        model = CausalHierarchicalMemoryLM(
            vocab_size=24,
            dim=8,
            max_seq_len=12,
            s_layers=1,
            memory_slots=(5, 3),
            prediction_layers=1,
            s_window=12,
            prediction_window=4,
            memory_topk=2,
            pairwise_rank=4,
            route_rank=4,
        )
        token_ids = torch.randint(0, 24, (2, 4))

        fresh_output = model(token_ids, return_memory_state=True)
        assert isinstance(fresh_output, MemoryScanOutput)
        carried_output = model(
            token_ids,
            memory_state=fresh_output.memory_state,
            reset_mask=torch.tensor([True, False]),
            return_memory_state=True,
        )
        assert isinstance(carried_output, MemoryScanOutput)

        self.assertTrue(torch.allclose(carried_output.logits[0], fresh_output.logits[0]))
        self.assertFalse(torch.allclose(carried_output.logits[1], fresh_output.logits[1]))

    def test_streaming_batcher_emits_continuation_flags(self) -> None:
        tokens = torch.arange(30, dtype=torch.long)
        batcher = StreamingTokenBatcher(
            tokens,
            seq_len=4,
            batch_size=2,
            device=torch.device("cpu"),
            random_starts=False,
        )

        batch1 = batcher.next_batch()
        batch2 = batcher.next_batch()

        self.assertTrue(torch.equal(batch1.reset_mask, torch.tensor([True, True])))
        self.assertTrue(torch.equal(batch2.reset_mask, torch.tensor([False, False])))
        self.assertTrue(torch.equal(batch1.context[0], torch.tensor([0, 1, 2, 3])))
        self.assertTrue(torch.equal(batch2.context[0], torch.tensor([4, 5, 6, 7])))


if __name__ == "__main__":
    unittest.main()
