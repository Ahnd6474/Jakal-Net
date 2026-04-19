import unittest
from pathlib import Path
import sys

import torch

from jakal_net import Layer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import progressive_b_example as progressive_b_module  # noqa: E402

from progressive_b_example import (  # noqa: E402
    ProgressiveBExampleLM,
    ProgressiveBJointBlock,
    ProgressiveBStageSpec,
    build_progressive_b_stage_specs,
)


class ProgressiveBArchitectureTests(unittest.TestCase):
    def test_stage_builder_matches_progressive_ratios(self) -> None:
        specs = build_progressive_b_stage_specs(seq_nodes=20, lite_layers=2, mid_layers=3, full_layers=1)

        self.assertEqual(len(specs), 3)
        self.assertEqual(specs[0].expanded_nodes, 21)
        self.assertEqual(specs[0].compressed_nodes, 18)
        self.assertEqual(specs[1].expanded_nodes, 22)
        self.assertEqual(specs[1].compressed_nodes, 16)
        self.assertEqual(specs[2].expanded_nodes, 24)
        self.assertEqual(specs[2].compressed_nodes, 14)
        self.assertLess(specs[0].alpha_b, specs[1].alpha_b)
        self.assertLess(specs[1].alpha_b, specs[2].alpha_b)

    def test_joint_block_updates_s_and_returns_compressed_b(self) -> None:
        torch.manual_seed(20)
        s_layer = Layer(
            dim=6,
            num_nodes=10,
            state=torch.randn(2, 10),
            val=torch.randn(2, 10, 6),
        )
        block = ProgressiveBJointBlock(
            dim=6,
            seq_nodes=10,
            expanded_nodes=11,
            compressed_nodes=8,
            alpha_b=0.5,
            beta_s_to_b=0.4,
            beta_b_to_s=0.2,
            s_window=3,
            route_topk=3,
            expanded_topk=3,
            compressed_topk=2,
            implementation="streaming",
        )

        next_s, compressed_b = block(s_layer)

        self.assertEqual(next_s.state.shape, (2, 10))
        self.assertEqual(next_s.val.shape, (2, 10, 6))
        self.assertEqual(compressed_b.state.shape, (2, 8))
        self.assertEqual(compressed_b.val.shape, (2, 8, 6))
        self.assertFalse(torch.allclose(next_s.state, s_layer.state))

    def test_example_lm_runs_end_to_end(self) -> None:
        torch.manual_seed(21)
        stage_specs = [
            ProgressiveBStageSpec(
                num_layers=1,
                expanded_nodes=9,
                compressed_nodes=7,
                alpha_b=0.3,
                beta_s_to_b=0.25,
                beta_b_to_s=0.15,
            ),
            ProgressiveBStageSpec(
                num_layers=1,
                expanded_nodes=10,
                compressed_nodes=6,
                alpha_b=0.7,
                beta_s_to_b=0.55,
                beta_b_to_s=0.35,
            ),
        ]
        model = ProgressiveBExampleLM(
            vocab_size=32,
            dim=8,
            seq_nodes=8,
            warmup_layers=1,
            stage_specs=stage_specs,
            final_refine_layers=1,
            s_window=2,
            route_topk=3,
            expanded_topk=3,
            compressed_topk=2,
            implementation="streaming",
        )
        token_ids = torch.randint(0, 32, (4, 8))

        logits, s_layer, compressed_b = model(token_ids, return_layers=True)

        self.assertEqual(logits.shape, (4, 32))
        self.assertEqual(s_layer.state.shape, (4, 8))
        self.assertIsNotNone(compressed_b)
        assert compressed_b is not None
        self.assertEqual(compressed_b.num_nodes, 6)
        self.assertEqual(compressed_b.val.shape, (4, 6, 8))

    def test_joint_block_initializes_b_from_learnable_slots(self) -> None:
        torch.manual_seed(22)
        s_layer = Layer(
            dim=4,
            num_nodes=9,
            state=torch.randn(2, 9),
            val=torch.randn(2, 9, 4),
        )
        block = ProgressiveBJointBlock(
            dim=4,
            seq_nodes=8,
            expanded_nodes=10,
            compressed_nodes=6,
            alpha_b=0.5,
            beta_s_to_b=0.4,
            beta_b_to_s=0.2,
            implementation="streaming",
        )

        prepared = block._prepare_compressed_layer(s_layer, None, 7)

        self.assertTrue(block.compressed_slot_state.requires_grad)
        self.assertTrue(block.compressed_slot_val.requires_grad)
        self.assertGreater(float(prepared.state.abs().sum().item()), 0.0)
        self.assertTrue(torch.allclose(prepared.state[0], prepared.state[1]))
        self.assertEqual(prepared.val.shape, (2, 7, 4))
        self.assertTrue(torch.allclose(prepared.val[0], prepared.val[1]))
        self.assertFalse(torch.allclose(prepared.val[:, 0, :], prepared.val[:, -1, :]))

    def test_example_lm_accepts_longer_sequence_than_reference_length(self) -> None:
        torch.manual_seed(23)
        model = ProgressiveBExampleLM(
            vocab_size=32,
            dim=8,
            seq_nodes=8,
            warmup_layers=1,
            final_refine_layers=1,
            implementation="streaming",
        )
        token_ids = torch.randint(0, 32, (2, 12))

        logits, s_layer, compressed_b = model(token_ids, return_layers=True)

        self.assertEqual(logits.shape, (2, 32))
        self.assertEqual(s_layer.state.shape, (2, 12))
        self.assertIsNotNone(compressed_b)
        assert compressed_b is not None
        self.assertEqual(compressed_b.num_nodes, 9)
        self.assertEqual(compressed_b.val.shape, (2, 9, 8))

    def test_example_lm_shares_pairwise_by_role(self) -> None:
        model = ProgressiveBExampleLM(
            vocab_size=32,
            dim=8,
            seq_nodes=8,
            warmup_layers=2,
            final_refine_layers=2,
        )

        self.assertIs(model.s_warmup[0].pairwise_fn, model.s_warmup[1].pairwise_fn)
        self.assertIs(model.s_warmup[0].pairwise_fn, model.joint_blocks[0].s_propagation.pairwise_fn)
        self.assertIs(model.s_warmup[0].pairwise_fn, model.s_refine[0].pairwise_fn)
        self.assertIs(
            model.joint_blocks[0].expanded_propagation.pairwise_fn,
            model.joint_blocks[1].expanded_propagation.pairwise_fn,
        )
        self.assertIs(
            model.joint_blocks[0].compressed_propagation.pairwise_fn,
            model.joint_blocks[1].compressed_propagation.pairwise_fn,
        )
        self.assertIsNot(
            model.joint_blocks[0].s_propagation.pairwise_fn,
            model.joint_blocks[0].expanded_propagation.pairwise_fn,
        )
        self.assertIsNot(
            model.joint_blocks[0].expanded_propagation.pairwise_fn,
            model.joint_blocks[0].compressed_propagation.pairwise_fn,
        )

    def test_query_block_batches_transition_and_limits_propagation_to_queries(self) -> None:
        torch.manual_seed(24)
        model = ProgressiveBExampleLM(
            vocab_size=32,
            dim=8,
            seq_nodes=8,
            warmup_layers=1,
            final_refine_layers=1,
            implementation="streaming",
        )
        token_ids = torch.randint(0, 32, (2, 8))

        transition_calls: list[tuple[int, int]] = []
        propagation_calls: list[int] = []
        original_transition = model.query_transition.compute_delta
        original_query_helper = progressive_b_module._compute_dense_causal_propagation_delta

        def record_transition(src_layer, dst_layer):
            transition_calls.append((src_layer.num_nodes, dst_layer.num_nodes))
            return original_transition(src_layer, dst_layer)

        def record_propagation(propagation, query_layer):
            propagation_calls.append(query_layer.num_nodes)
            return original_query_helper(propagation, query_layer)

        model.query_transition.compute_delta = record_transition
        progressive_b_module._compute_dense_causal_propagation_delta = record_propagation

        try:
            logits = model.forward_query_block(token_ids, target_len=4)
        finally:
            progressive_b_module._compute_dense_causal_propagation_delta = original_query_helper

        self.assertEqual(logits.shape, (2, 4, 32))
        self.assertEqual(transition_calls, [(8, 4)] * 2)
        self.assertEqual(propagation_calls, [4] * 2)

    def test_query_block_accepts_seed_tokens_for_structural_slots(self) -> None:
        torch.manual_seed(25)
        model = ProgressiveBExampleLM(
            vocab_size=32,
            dim=8,
            seq_nodes=8,
            warmup_layers=1,
            final_refine_layers=1,
        )
        token_ids = torch.randint(0, 32, (2, 8))
        seed_tokens = torch.full((2, 1), 31, dtype=torch.long)

        logits = model.forward_query_block(
            token_ids,
            target_len=5,
            query_seed_token_ids=seed_tokens,
        )

        self.assertEqual(logits.shape, (2, 5, 32))

    def test_query_block_accepts_parallel_feedback_tokens(self) -> None:
        torch.manual_seed(26)
        model = ProgressiveBExampleLM(
            vocab_size=32,
            dim=8,
            seq_nodes=8,
            warmup_layers=1,
            final_refine_layers=1,
        )
        token_ids = torch.randint(0, 32, (2, 8))
        feedback_tokens = torch.randint(0, 32, (2, 5))

        logits = model.forward_query_block(
            token_ids,
            target_len=5,
            query_feedback_token_ids=feedback_tokens,
        )

        self.assertEqual(logits.shape, (2, 5, 32))



if __name__ == "__main__":
    unittest.main()
