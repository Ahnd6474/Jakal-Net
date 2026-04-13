import unittest
from pathlib import Path
import sys

import torch

from jakal_net import Layer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

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


if __name__ == "__main__":
    unittest.main()
