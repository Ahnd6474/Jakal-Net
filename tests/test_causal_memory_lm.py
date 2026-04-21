import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_causal_memory_lm import (  # noqa: E402
    DocumentChunk,
    DocumentChunkBatcher,
    TokenizedDocument,
    load_pretokenized_bundle,
    make_document_chunks,
    save_pretokenized_bundle,
)

from jakal_net.causal_memory_lm import CausalHierarchicalMemoryLM, MemoryScanOutput, ModelRecurrentState  # noqa: E402
from jakal_net.latent_graph import KModule  # noqa: E402


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

    def test_s_microbatch_matches_full_batch_forward(self) -> None:
        torch.manual_seed(9)
        base_model = CausalHierarchicalMemoryLM(
            vocab_size=24,
            dim=8,
            max_seq_len=12,
            s_layers=1,
            memory_slots=(5, 3),
            prediction_layers=1,
            s_window=8,
            prediction_window=4,
            memory_topk=2,
            pairwise_rank=4,
            route_rank=4,
        )
        microbatched_model = CausalHierarchicalMemoryLM(
            vocab_size=24,
            dim=8,
            max_seq_len=12,
            s_layers=1,
            memory_slots=(5, 3),
            prediction_layers=1,
            s_window=8,
            s_microbatch_size=2,
            prediction_window=4,
            memory_topk=2,
            pairwise_rank=4,
            route_rank=4,
        )
        microbatched_model.load_state_dict(base_model.state_dict())
        token_ids = torch.randint(0, 24, (4, 5))

        base_output = base_model(token_ids, return_memory_state=True)
        micro_output = microbatched_model(token_ids, return_memory_state=True)

        assert isinstance(base_output, MemoryScanOutput)
        assert isinstance(micro_output, MemoryScanOutput)
        self.assertTrue(torch.allclose(base_output.logits, micro_output.logits))
        self.assertEqual(len(base_output.memory_state), len(micro_output.memory_state))
        for base_layer, micro_layer in zip(base_output.memory_state, micro_output.memory_state):
            self.assertTrue(torch.allclose(base_layer.state, micro_layer.state))
            self.assertTrue(torch.allclose(base_layer.val, micro_layer.val))

    def test_forward_logits_remain_finite(self) -> None:
        torch.manual_seed(10)
        model = CausalHierarchicalMemoryLM(
            vocab_size=64,
            dim=16,
            max_seq_len=16,
            s_layers=2,
            memory_slots=(8, 4, 2),
            prediction_layers=2,
            s_window=8,
            s_microbatch_size=1,
            prediction_window=4,
            memory_topk=2,
            pairwise_rank=8,
            route_rank=8,
        )
        token_ids = torch.randint(0, 64, (2, 8))

        logits = model(token_ids)

        self.assertTrue(torch.isfinite(logits).all().item())

    def test_constructor_accepts_scan_backend_compatibility_args(self) -> None:
        model = CausalHierarchicalMemoryLM(
            vocab_size=64,
            dim=16,
            max_seq_len=16,
            s_layers=1,
            memory_slots=(8, 4, 2),
            prediction_layers=1,
            memory_topk=2,
            pairwise_rank=8,
            route_rank=8,
            scan_backend="native",
            scan_checkpoint_chunk_size=32,
        )

        self.assertEqual(model.scan_backend, "native")
        self.assertEqual(model.scan_checkpoint_chunk_size, 32)

    def test_forward_supports_optional_knowledge_module(self) -> None:
        torch.manual_seed(11)
        knowledge_module = KModule(
            dim=8,
            num_nodes=10,
            route_rank=4,
            pairwise_rank=4,
            route_topk=4,
            propagation_topk=3,
            implementation="reference",
        )
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
            knowledge_module=knowledge_module,
        )
        token_ids = torch.randint(0, 32, (2, 5))

        output = model(token_ids, return_memory_state=True, return_layers=True)

        self.assertIsInstance(output, MemoryScanOutput)
        self.assertEqual(output.logits.shape, (2, 5, 32))
        self.assertIsNotNone(output.knowledge_state)
        assert output.knowledge_state is not None
        self.assertEqual(output.knowledge_state.val.shape, (2, 10, 8))
        recurrent_state = output.recurrent_state
        self.assertIsInstance(recurrent_state, ModelRecurrentState)
        self.assertEqual(recurrent_state.memory_state[0].val.shape, (2, 6, 8))
        assert recurrent_state.knowledge_state is not None
        self.assertEqual(recurrent_state.knowledge_state.val.shape, (2, 10, 8))

    def test_model_can_build_internal_knowledge_module(self) -> None:
        torch.manual_seed(12)
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
            knowledge_nodes=9,
            knowledge_route_topk=3,
            knowledge_propagation_topk=3,
            knowledge_propagation_layers=2,
        )
        token_ids = torch.randint(0, 32, (2, 5))

        output = model(token_ids, return_memory_state=True)

        self.assertIsNotNone(model.knowledge_module)
        self.assertIsInstance(output, MemoryScanOutput)
        self.assertIsNotNone(output.knowledge_state)

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

    def test_pretokenized_bundle_roundtrip_uses_flat_storage(self) -> None:
        documents = (
            TokenizedDocument(
                kind="dialogue",
                source="doc1",
                token_count=6,
                chunks=(
                    DocumentChunk(
                        context=torch.tensor([1, 2, 10, 11, 0, 0], dtype=torch.long),
                        target=torch.tensor([2, 10, 11, 3, 0, 0], dtype=torch.long),
                        loss_mask=torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
                        is_continuation=False,
                    ),
                    DocumentChunk(
                        context=torch.tensor([3, 2, 12, 13, 14, 0], dtype=torch.long),
                        target=torch.tensor([2, 12, 13, 14, 4, 0], dtype=torch.long),
                        loss_mask=torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.0], dtype=torch.float32),
                        is_continuation=True,
                    ),
                ),
            ),
            TokenizedDocument(
                kind="text",
                source="doc2",
                token_count=2,
                chunks=(
                    DocumentChunk(
                        context=torch.tensor([1, 5, 20, 0, 0, 0], dtype=torch.long),
                        target=torch.tensor([5, 20, 4, 0, 0, 0], dtype=torch.long),
                        loss_mask=torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                        is_continuation=False,
                    ),
                ),
            ),
        )
        corpus_info = {
            "special_tokens": {"pad": 0, "eos": 4, "cont": 3},
            "tokenized_summary": {"documents": 2, "chunks": 3, "tokens": 8},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "bundle.pt"
            save_pretokenized_bundle(
                bundle_path,
                documents=documents,
                vocab_size=16384,
                tokenizer_label="test",
                tokenizer_model_path=None,
                corpus_info=corpus_info,
            )
            raw_bundle = torch.load(bundle_path, map_location="cpu")
            self.assertEqual(raw_bundle["storage_format"], "flat_v2")
            self.assertEqual(int(raw_bundle["seq_len"]), 6)
            self.assertNotIn("target", str(raw_bundle))
            restored = load_pretokenized_bundle(bundle_path)

        self.assertEqual(len(restored["documents"]), len(documents))
        for restored_document, original_document in zip(restored["documents"], documents):
            self.assertEqual(restored_document.kind, original_document.kind)
            self.assertEqual(restored_document.source, original_document.source)
            self.assertEqual(restored_document.token_count, original_document.token_count)
            self.assertEqual(len(restored_document.chunks), len(original_document.chunks))
            for restored_chunk, original_chunk in zip(restored_document.chunks, original_document.chunks):
                self.assertTrue(torch.equal(restored_chunk.context, original_chunk.context))
                self.assertTrue(torch.equal(restored_chunk.target, original_chunk.target))
                self.assertTrue(torch.equal(restored_chunk.loss_mask, original_chunk.loss_mask))
                self.assertEqual(restored_chunk.is_continuation, original_chunk.is_continuation)


if __name__ == "__main__":
    unittest.main()
