import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_causal_memory_lm import (  # noqa: E402
    DocumentChunk,
    DocumentChunkBatcher,
    TokenizedDocument,
    make_document_chunks,
)

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

    def test_document_chunks_insert_continuation_prefix(self) -> None:
        chunks = make_document_chunks(
            content_ids=torch.tensor([10, 11, 12, 13, 14, 15, 16], dtype=torch.long),
            mode_token_id=2,
            seq_len=5,
            bos_token_id=1,
            cont_token_id=3,
            eos_token_id=4,
            pad_token_id=0,
        )

        self.assertEqual(len(chunks), 3)
        self.assertTrue(torch.equal(chunks[0].context[:5], torch.tensor([1, 2, 10, 11, 12])))
        self.assertTrue(torch.equal(chunks[1].context[:5], torch.tensor([3, 2, 13, 14, 15])))
        self.assertTrue(torch.equal(chunks[2].context[:4], torch.tensor([3, 2, 16, 0])))
        self.assertEqual(int(chunks[0].target[4].item()), 3)
        self.assertEqual(int(chunks[2].target[2].item()), 4)
        self.assertFalse(chunks[0].is_continuation)
        self.assertTrue(chunks[1].is_continuation)

    def test_document_batcher_carries_within_document_and_resets_on_new_document(self) -> None:
        doc1 = TokenizedDocument(
            kind="text",
            source="doc1",
            token_count=5,
            chunks=(
                DocumentChunk(
                    context=torch.tensor([1, 2, 10, 0], dtype=torch.long),
                    target=torch.tensor([2, 10, 3, 0], dtype=torch.long),
                    loss_mask=torch.tensor([1.0, 1.0, 1.0, 0.0]),
                    is_continuation=False,
                ),
                DocumentChunk(
                    context=torch.tensor([3, 2, 11, 0], dtype=torch.long),
                    target=torch.tensor([2, 11, 4, 0], dtype=torch.long),
                    loss_mask=torch.tensor([1.0, 1.0, 1.0, 0.0]),
                    is_continuation=True,
                ),
            ),
        )
        doc2 = TokenizedDocument(
            kind="text",
            source="doc2",
            token_count=2,
            chunks=(
                DocumentChunk(
                    context=torch.tensor([1, 2, 20, 0], dtype=torch.long),
                    target=torch.tensor([2, 20, 4, 0], dtype=torch.long),
                    loss_mask=torch.tensor([1.0, 1.0, 1.0, 0.0]),
                    is_continuation=False,
                ),
            ),
        )
        batcher = DocumentChunkBatcher((doc1, doc2), batch_size=1, device=torch.device("cpu"))
        batcher.current_doc[0] = 0
        batcher.current_chunk[0] = 0
        batcher.needs_reset[0] = False

        batch1 = batcher.next_batch()
        batch2 = batcher.next_batch()
        batch3 = batcher.next_batch()

        self.assertTrue(torch.equal(batch1.reset_mask, torch.tensor([False])))
        self.assertTrue(torch.equal(batch2.reset_mask, torch.tensor([False])))
        self.assertTrue(torch.equal(batch1.context[0], doc1.chunks[0].context))
        self.assertTrue(torch.equal(batch2.context[0], doc1.chunks[1].context))
        self.assertTrue(torch.equal(batch3.reset_mask, torch.tensor([True])))


if __name__ == "__main__":
    unittest.main()
