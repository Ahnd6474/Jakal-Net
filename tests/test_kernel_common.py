import unittest

import torch

from jakal_net.core import allocate_accumulator, iter_block_spans, iter_blocks
from jakal_net.kernel_common import (
    OnlineSoftmaxState,
    OnlineSoftmaxStats,
    apply_slot_mask_to_state,
    apply_slot_mask_to_val,
    build_topk_mask,
    finalize_online_softmax,
    gather_state_by_indices,
    gather_val_by_indices,
    masked_softmax,
    normalize_slot_mask,
    normalize_with_online_softmax,
    online_softmax_reduce_step,
    online_softmax_stats_step,
    pairwise_scores_dense,
    pairwise_slot_mask,
    prepare_route_context,
    route_block_logits,
    route_logits,
    route_slot_mask,
    select_topk,
    supports_pairwise_kernel,
    supports_pairwise_route_kernel,
    supports_route_kernel,
)
from jakal_net.modules import (
    BilinearPairwise,
    BilinearPairwiseRoute,
    DiagonalBilinearPairwise,
    HadamardMLPPairwise,
    LearnedPositionEncoding,
    LinearRoute,
    MLPRoute,
    SourceTargetHadamardMLPRoute,
)


class KernelCommonTests(unittest.TestCase):
    def test_iter_block_spans_matches_tuple_scheduler(self) -> None:
        spans = list(iter_block_spans(10, 4, name="block"))
        tuples = list(iter_blocks(10, 4, name="block"))

        self.assertEqual([(span.start, span.end) for span in spans], tuples)
        self.assertEqual([span.size for span in spans], [4, 4, 2])

    def test_allocate_accumulator_promotes_low_precision(self) -> None:
        tensor = allocate_accumulator((2, 3), device="cpu", tensor_dtype=torch.float16)

        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.equal(tensor, torch.zeros_like(tensor)))

    def test_pairwise_kernel_family_matches_module_forward(self) -> None:
        torch.manual_seed(0)
        target = torch.randn(2, 5, 4)
        source = torch.randn(2, 7, 4)

        for module in (
            DiagonalBilinearPairwise(dim=4),
            BilinearPairwise(dim=4),
            HadamardMLPPairwise(dim=4, hidden_dim=6),
        ):
            expected = module(target, source)
            actual = pairwise_scores_dense(module, target, source)
            self.assertTrue(torch.allclose(expected, actual))
            self.assertTrue(supports_pairwise_kernel(module))

    def test_route_kernel_family_matches_linear_and_mlp(self) -> None:
        torch.manual_seed(1)
        src_val = torch.randn(3, 6, 5)

        for module in (
            LinearRoute(src_dim=5, dst_nodes=4),
            MLPRoute(src_dim=5, dst_nodes=4, hidden_dim=7),
        ):
            expected = module(src_val)
            actual = route_logits(module, src_val)
            self.assertTrue(torch.allclose(expected, actual))
            self.assertTrue(supports_route_kernel(module))

        self.assertFalse(
            supports_route_kernel(BilinearPairwiseRoute(src_dim=5, dst_dim=5))
        )
        self.assertFalse(
            supports_route_kernel(SourceTargetHadamardMLPRoute(src_dim=5, dst_dim=5))
        )
        self.assertTrue(
            supports_pairwise_route_kernel(
                SourceTargetHadamardMLPRoute(src_dim=5, dst_dim=5)
            )
        )

    def test_route_block_logits_match_full_route_logits(self) -> None:
        torch.manual_seed(2)
        src_val = torch.randn(2, 4, 5)

        for module in (
            LinearRoute(src_dim=5, dst_nodes=6),
            MLPRoute(src_dim=5, dst_nodes=6, hidden_dim=7),
        ):
            full = route_logits(module, src_val)
            context = prepare_route_context(module, src_val)
            pieces = [
                route_block_logits(module, context, start=start, end=end)
                for start, end in ((0, 2), (2, 5), (5, 6))
            ]
            rebuilt = torch.cat(pieces, dim=-1)
            self.assertTrue(torch.allclose(full, rebuilt, atol=1e-6, rtol=1e-6))

    def test_learned_position_encoding_generates_variable_lengths(self) -> None:
        module = LearnedPositionEncoding(dim=4)

        short = module(3)
        long = module(7)

        self.assertEqual(short.shape, (3, 4))
        self.assertEqual(long.shape, (7, 4))
        self.assertFalse(torch.allclose(long[0], long[-1]))

    def test_masked_softmax_only_uses_active_entries(self) -> None:
        logits = torch.tensor([[3.0, 1.0, -4.0]])
        mask = torch.tensor([[True, False, True]])

        routes = masked_softmax(logits, mask, dim=-1)

        self.assertTrue(torch.allclose(routes[:, 1], torch.zeros(1)))
        self.assertTrue(torch.allclose(routes.sum(dim=-1), torch.ones(1)))

    def test_topk_selection_and_mask_agree(self) -> None:
        scores = torch.tensor([[1.0, 5.0, 3.0, -2.0]])

        selected = select_topk(scores, 2, dim=-1)
        mask = build_topk_mask(scores, 2, dim=-1)

        self.assertTrue(torch.equal(selected.indices, torch.tensor([[1, 2]])))
        self.assertTrue(torch.equal(mask, torch.tensor([[False, True, True, False]])))

    def test_slot_mask_utilities_normalize_and_combine(self) -> None:
        slot_mask = torch.tensor([True, False, True])
        state = torch.tensor([[1.0, 2.0, 3.0]])
        val = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])

        normalized = normalize_slot_mask(slot_mask, batch_shape=(1,), num_nodes=3, device=state.device)
        masked_state = apply_slot_mask_to_state(state, slot_mask)
        masked_val = apply_slot_mask_to_val(val, slot_mask)

        self.assertTrue(torch.equal(normalized, torch.tensor([[True, False, True]])))
        self.assertTrue(torch.equal(masked_state, torch.tensor([[1.0, 0.0, 3.0]])))
        self.assertTrue(
            torch.equal(
                masked_val,
                torch.tensor([[[1.0, 10.0], [0.0, 0.0], [3.0, 30.0]]]),
            )
        )

        source_mask = torch.tensor([[True, False, True]])
        target_mask = torch.tensor([[True, True]])
        self.assertTrue(
            torch.equal(
                pairwise_slot_mask(target_mask, source_mask),
                torch.tensor([[[True, False, True], [True, False, True]]]),
            )
        )
        self.assertTrue(
            torch.equal(
                route_slot_mask(source_mask, target_mask),
                torch.tensor([[[True, True], [False, False], [True, True]]]),
            )
        )

    def test_gather_helpers_match_manual_selection(self) -> None:
        projected_state = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        projected_val = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        indices = torch.tensor([[[3, 1], [2, 0]]])

        gathered_state = gather_state_by_indices(projected_state, indices)
        gathered_val = gather_val_by_indices(projected_val, indices)

        self.assertTrue(torch.equal(gathered_state, torch.tensor([[[40.0, 20.0], [30.0, 10.0]]])))
        self.assertTrue(torch.equal(gathered_val, torch.tensor([[[[4.0], [2.0]], [[3.0], [1.0]]]])))

    def test_online_softmax_reduction_matches_dense_softmax(self) -> None:
        logits = torch.tensor([[[1.0, 0.5, 2.0, 1.5], [3.0, -1.0, 4.0, 0.0]]])
        values = torch.tensor(
            [[[[2.0], [4.0], [6.0], [8.0]], [[1.0], [3.0], [5.0], [7.0]]]]
        )

        state: OnlineSoftmaxState | None = None
        for start in (0, 2):
            state = online_softmax_reduce_step(
                state,
                logits[..., start : start + 2],
                values[..., start : start + 2, :],
            )

        self.assertIsNotNone(state)
        result = finalize_online_softmax(state)
        expected_weights = torch.softmax(logits, dim=-1)
        expected = (expected_weights.unsqueeze(-1) * values).sum(dim=-2)
        self.assertTrue(torch.allclose(result, expected, atol=1e-6, rtol=1e-6))

    def test_online_softmax_stats_match_dense_softmax(self) -> None:
        logits = torch.tensor([[[1.0, 0.5, 2.0, 1.5], [3.0, -1.0, 4.0, 0.0]]])

        stats: OnlineSoftmaxStats | None = None
        pieces = []
        for start in (0, 2):
            block = logits[..., start : start + 2]
            pieces.append(block)
            stats = online_softmax_stats_step(stats, block)

        self.assertIsNotNone(stats)
        rebuilt = torch.cat(
            [normalize_with_online_softmax(block, stats) for block in pieces], dim=-1
        )
        expected = torch.softmax(logits, dim=-1)
        self.assertTrue(torch.allclose(rebuilt, expected, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
