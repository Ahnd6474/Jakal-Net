import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_causal_memory_lm import initialize_model_embedding_from_hf  # noqa: E402

from jakal_net.causal_memory_lm import (  # noqa: E402
    CausalMemoryLM,
    MemoryScanOutput,
    ModelRecurrentState,
    ValueNormStateProjection,
)


class CausalMemoryLMTests(unittest.TestCase):
    def test_forward_returns_logits_layers_and_trace_state(self) -> None:
        torch.manual_seed(7)
        model = CausalMemoryLM(
            vocab_size=32,
            dim=16,
            max_seq_len=12,
            transformer_layers=2,
            transformer_heads=4,
            knowledge_memory_size=24,
            knowledge_beta_dim=8,
            knowledge_relation_rank=6,
        )
        token_ids = torch.randint(0, 32, (2, 5))

        output = model(token_ids, return_memory_state=True, return_layers=True)

        self.assertIsInstance(output, MemoryScanOutput)
        self.assertEqual(output.logits.shape, (2, 5, 32))
        self.assertEqual(len(output.memory_state), 1)
        self.assertEqual(output.memory_state[0].shape, (2, 2, 4, 24))
        self.assertEqual(model.knowledge_block.relation.shape, (24, 24))
        self.assertEqual(model.knowledge_block.value.shape, (2, 24, 16))
        self.assertIsNotNone(output.knowledge_state)
        assert output.sequence_layer is not None
        assert output.query_layer is not None
        self.assertEqual(output.sequence_layer.val.shape, (2, 5, 16))
        self.assertEqual(output.query_layer.val.shape, (2, 5, 16))
        recurrent_state = output.recurrent_state
        self.assertIsInstance(recurrent_state, ModelRecurrentState)
        self.assertEqual(recurrent_state.memory_state[0].shape, (2, 2, 4, 24))

    def test_initialize_memory_state_returns_zero_trace(self) -> None:
        model = CausalMemoryLM(
            vocab_size=16,
            dim=8,
            max_seq_len=8,
            transformer_heads=1,
            knowledge_memory_size=10,
            knowledge_beta_dim=4,
            knowledge_relation_rank=4,
        )

        state = model.initialize_memory_state(3, device=torch.device("cpu"), dtype=torch.float32)

        self.assertEqual(len(state), 1)
        self.assertEqual(state[0].shape, (3, 6, 1, 10))
        self.assertTrue(torch.equal(state[0], torch.zeros_like(state[0])))

    def test_carried_trace_changes_output_and_reset_mask_restores_fresh_path(self) -> None:
        torch.manual_seed(11)
        model = CausalMemoryLM(
            vocab_size=24,
            dim=12,
            max_seq_len=10,
            transformer_layers=2,
            transformer_heads=3,
            knowledge_memory_size=20,
            knowledge_beta_dim=6,
            knowledge_relation_rank=5,
        )
        token_ids = torch.randint(0, 24, (2, 6))

        fresh = model(token_ids, return_memory_state=True)
        assert isinstance(fresh, MemoryScanOutput)
        carried = model(
            token_ids,
            memory_state=fresh.memory_state,
            reset_mask=torch.tensor([True, False], dtype=torch.bool),
            return_memory_state=True,
        )
        assert isinstance(carried, MemoryScanOutput)

        self.assertTrue(torch.allclose(carried.logits[0], fresh.logits[0]))
        self.assertFalse(torch.allclose(carried.logits[1], fresh.logits[1]))

    def test_collect_internal_stats_reports_trace_metrics_when_enabled(self) -> None:
        torch.manual_seed(13)
        model = CausalMemoryLM(
            vocab_size=16,
            dim=8,
            max_seq_len=8,
            knowledge_memory_size=12,
            knowledge_beta_dim=4,
            knowledge_relation_rank=4,
        )
        token_ids = torch.randint(0, 16, (2, 4))

        model.set_track_stats(True)
        _ = model(token_ids)
        stats = model.collect_internal_stats()

        self.assertIn("knowledge/gate_mean", stats)
        self.assertIn("knowledge/beta_entropy", stats)
        self.assertIn("knowledge/trace_rms", stats)
        self.assertIn("knowledge/relation_rms", stats)

    def test_residual_two_hop_memory_path_runs_and_exposes_hop_gate(self) -> None:
        torch.manual_seed(15)
        model = CausalMemoryLM(
            vocab_size=16,
            dim=8,
            max_seq_len=8,
            transformer_layers=2,
            transformer_heads=2,
            knowledge_memory_size=12,
            knowledge_beta_dim=4,
            knowledge_hops=2,
            knowledge_relation_rank=4,
        )
        token_ids = torch.randint(0, 16, (2, 4))

        model.set_track_stats(True)
        output = model(token_ids, return_memory_state=True)
        stats = model.collect_internal_stats()

        self.assertIsInstance(output, MemoryScanOutput)
        self.assertEqual(model.knowledge_block.hop_residual_logit.shape, (2, 1))
        self.assertIn("knowledge/hop_gate_mean", stats)

    def test_memory_lookup_uses_attention_query_projection(self) -> None:
        torch.manual_seed(17)
        model = CausalMemoryLM(
            vocab_size=16,
            dim=8,
            max_seq_len=8,
            transformer_layers=1,
            transformer_heads=2,
            knowledge_memory_size=12,
            knowledge_beta_dim=4,
            knowledge_relation_rank=4,
        )
        token_ids = torch.randint(0, 16, (2, 4))
        batch_size, seq_len = token_ids.shape
        hidden_after_attention = torch.randn(batch_size, seq_len, model.dim)
        attention_input = torch.randn(batch_size, seq_len, model.dim)
        query_states = torch.randn(batch_size, seq_len, model.dim)
        attention_probs = torch.softmax(
            torch.randn(batch_size, model.transformer_heads, seq_len, seq_len),
            dim=-1,
        )

        model.encoder_layers[0].attend = MagicMock(  # type: ignore[method-assign]
            return_value=(hidden_after_attention, attention_input, attention_probs, query_states)
        )
        original_lookup = model.knowledge_block.lookup
        model.knowledge_block.lookup = MagicMock(side_effect=original_lookup)  # type: ignore[method-assign]

        _ = model(token_ids)

        lookup_arg = model.knowledge_block.lookup.call_args[0][0]
        self.assertTrue(torch.equal(lookup_arg, query_states))

    def test_value_norm_state_projection_uses_vector_norm(self) -> None:
        projection = ValueNormStateProjection()
        val = torch.tensor([[[3.0, 4.0], [5.0, 12.0]]])

        state = projection(val)

        self.assertTrue(torch.allclose(state, torch.tensor([[[5.0], [13.0]]])))

    def test_hf_embedding_init_copies_into_shared_embedding_table(self) -> None:
        model = CausalMemoryLM(
            vocab_size=4,
            dim=3,
            max_seq_len=8,
            transformer_heads=1,
            knowledge_memory_size=6,
            knowledge_beta_dim=4,
            knowledge_relation_rank=4,
        )

        class FakeTokenizer:
            def get_vocab(self) -> dict[str, int]:
                return {"tok0": 0, "tok1": 1, "tok2": 2, "tok3": 3}

            def convert_tokens_to_ids(self, token: str) -> int:
                return {"tok0": 0, "tok1": 1, "tok2": 2, "tok3": 3}[token]

        class FakeVocab:
            tokenizer = FakeTokenizer()

        class FakeHFModel:
            def __init__(self, weight: torch.Tensor) -> None:
                self._weight = weight

            def get_input_embeddings(self) -> object:
                return type("Embeddings", (), {"weight": self._weight})()

        hf_weight = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )

        with patch("train_causal_memory_lm.AutoModelForCausalLM.from_pretrained", return_value=FakeHFModel(hf_weight)):
            stats = initialize_model_embedding_from_hf(
                model=model,
                vocab=FakeVocab(),
                model_name_or_path="fake-model",
            )

        self.assertEqual(stats, {"copied": 4, "skipped": 0})
        self.assertTrue(torch.allclose(model.s_module.token_embedding.weight, hf_weight))
        self.assertEqual(
            model.lm_head.weight.data_ptr(),
            model.s_module.token_embedding.weight.data_ptr(),
        )


if __name__ == "__main__":
    unittest.main()
