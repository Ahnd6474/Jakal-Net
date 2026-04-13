import unittest

import torch

from jakal_net import (
    BilinearPairwise,
    DiagonalBilinearPairwise,
    Layer,
    LinearRoute,
    MLPRoute,
    Propagation,
    SparsePropagation,
    SparseTransition,
    Transition,
)


def _pairwise_fn(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    base = torch.einsum("...id,...jd->...ij", target, source)
    target_bias = target.mean(dim=-1, keepdim=True)
    source_bias = source.mean(dim=-1).unsqueeze(-2)
    return base + 0.25 * target_bias - 0.1 * source_bias


def _state_proj_fn(state: torch.Tensor) -> torch.Tensor:
    return state * 1.5 - 0.25


def _val_proj_fn(val: torch.Tensor) -> torch.Tensor:
    return val * 0.75 + 0.5


def _make_route_fn(src_dim: int, dst_nodes: int):
    weight = torch.arange(1, src_dim * dst_nodes + 1, dtype=torch.float32).view(
        src_dim, dst_nodes
    )
    bias = torch.linspace(-0.3, 0.4, steps=dst_nodes)

    def route_fn(src_val: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...jd,dk->...jk", src_val, weight) + bias

    return route_fn


class JakalNetModuleTests(unittest.TestCase):
    def assert_delta_close(
        self, left, right, *, atol: float = 1e-5, rtol: float = 1e-5
    ) -> None:
        self.assertTrue(
            torch.allclose(left.delta_state, right.delta_state, atol=atol, rtol=rtol)
        )
        self.assertTrue(torch.allclose(left.delta_val, right.delta_val, atol=atol, rtol=rtol))

    def test_layer_zeros_supports_prefix_dims(self) -> None:
        layer = Layer.zeros(dim=3, num_nodes=4, batch_shape=(2, 5))

        self.assertEqual(layer.state.shape, (2, 5, 4))
        self.assertEqual(layer.val.shape, (2, 5, 4, 3))
        self.assertTrue(torch.equal(layer.state, torch.zeros_like(layer.state)))
        self.assertTrue(torch.equal(layer.val, torch.zeros_like(layer.val)))

    def test_dense_propagation_streaming_matches_reference(self) -> None:
        torch.manual_seed(0)
        layer = Layer(
            dim=4,
            num_nodes=7,
            state=torch.randn(2, 7),
            val=torch.randn(2, 7, 4),
        )

        reference = Propagation(
            pairwise_fn=_pairwise_fn,
            state_proj_fn=_state_proj_fn,
            val_proj_fn=_val_proj_fn,
            implementation="reference",
        )
        streaming = Propagation(
            pairwise_fn=_pairwise_fn,
            state_proj_fn=_state_proj_fn,
            val_proj_fn=_val_proj_fn,
            implementation="streaming",
            target_block_size=3,
            source_block_size=2,
        )

        self.assert_delta_close(reference(layer), streaming(layer))

    def test_dense_propagation_kernel_matches_reference(self) -> None:
        torch.manual_seed(10)
        layer = Layer(
            dim=4,
            num_nodes=7,
            state=torch.randn(2, 7),
            val=torch.randn(2, 7, 4),
        )

        reference = Propagation(
            pairwise_fn=DiagonalBilinearPairwise(dim=4),
            state_proj_fn=_state_proj_fn,
            val_proj_fn=_val_proj_fn,
            implementation="reference",
        )
        kernel = Propagation(
            pairwise_fn=DiagonalBilinearPairwise(dim=4),
            state_proj_fn=_state_proj_fn,
            val_proj_fn=_val_proj_fn,
            implementation="kernel",
        )
        kernel.pairwise_fn.load_state_dict(reference.pairwise_fn.state_dict())

        self.assert_delta_close(reference(layer), kernel(layer))

    def test_sparse_window_propagation_streaming_matches_reference(self) -> None:
        torch.manual_seed(1)
        layer = Layer(
            dim=3,
            num_nodes=8,
            state=torch.randn(2, 8),
            val=torch.randn(2, 8, 3),
        )

        reference = SparsePropagation(
            pairwise_fn=_pairwise_fn,
            state_proj_fn=_state_proj_fn,
            val_proj_fn=_val_proj_fn,
            sparse_type="window",
            window=2,
            implementation="reference",
        )
        streaming = SparsePropagation(
            pairwise_fn=_pairwise_fn,
            state_proj_fn=_state_proj_fn,
            val_proj_fn=_val_proj_fn,
            sparse_type="window",
            window=2,
            implementation="streaming",
            target_block_size=3,
            source_block_size=2,
        )

        self.assert_delta_close(reference(layer), streaming(layer))

    def test_sparse_window_propagation_kernel_matches_reference(self) -> None:
        torch.manual_seed(11)
        layer = Layer(
            dim=3,
            num_nodes=8,
            state=torch.randn(2, 8),
            val=torch.randn(2, 8, 3),
        )

        reference = SparsePropagation(
            pairwise_fn=DiagonalBilinearPairwise(dim=3),
            state_proj_fn=_state_proj_fn,
            val_proj_fn=_val_proj_fn,
            sparse_type="window",
            window=2,
            implementation="reference",
        )
        kernel = SparsePropagation(
            pairwise_fn=DiagonalBilinearPairwise(dim=3),
            state_proj_fn=_state_proj_fn,
            val_proj_fn=_val_proj_fn,
            sparse_type="window",
            window=2,
            implementation="kernel",
        )
        kernel.pairwise_fn.load_state_dict(reference.pairwise_fn.state_dict())

        self.assert_delta_close(reference(layer), kernel(layer))

    def test_sparse_topk_propagation_streaming_matches_reference(self) -> None:
        torch.manual_seed(2)
        layer = Layer(
            dim=5,
            num_nodes=9,
            state=torch.randn(2, 9),
            val=torch.randn(2, 9, 5),
        )

        reference = SparsePropagation(
            pairwise_fn=_pairwise_fn,
            state_proj_fn=_state_proj_fn,
            val_proj_fn=_val_proj_fn,
            sparse_type="topk",
            topk=3,
            implementation="reference",
        )
        streaming = SparsePropagation(
            pairwise_fn=_pairwise_fn,
            state_proj_fn=_state_proj_fn,
            val_proj_fn=_val_proj_fn,
            sparse_type="topk",
            topk=3,
            implementation="streaming",
            target_block_size=4,
            source_block_size=2,
        )

        self.assert_delta_close(reference(layer), streaming(layer))

    def test_sparse_topk_propagation_kernel_matches_reference(self) -> None:
        torch.manual_seed(12)
        layer = Layer(
            dim=5,
            num_nodes=9,
            state=torch.randn(2, 9),
            val=torch.randn(2, 9, 5),
        )

        reference = SparsePropagation(
            pairwise_fn=BilinearPairwise(dim=5),
            state_proj_fn=_state_proj_fn,
            val_proj_fn=_val_proj_fn,
            sparse_type="topk",
            topk=3,
            implementation="reference",
        )
        kernel = SparsePropagation(
            pairwise_fn=BilinearPairwise(dim=5),
            state_proj_fn=_state_proj_fn,
            val_proj_fn=_val_proj_fn,
            sparse_type="topk",
            topk=3,
            implementation="kernel",
        )
        kernel.pairwise_fn.load_state_dict(reference.pairwise_fn.state_dict())

        self.assert_delta_close(reference(layer), kernel(layer))

    def test_propagation_has_no_state_activation(self) -> None:
        layer = Layer(
            dim=1,
            num_nodes=2,
            state=torch.tensor([[-1.0, 2.0]]),
            val=torch.tensor([[[1.0], [3.0]]]),
        )

        def pairwise_fn(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
            return torch.ones((*target.shape[:-2], target.shape[-2], source.shape[-2]))

        op = Propagation(
            pairwise_fn=pairwise_fn,
            implementation="streaming",
            target_block_size=1,
            source_block_size=1,
        )
        delta = op(layer)

        self.assertTrue(torch.allclose(delta.delta_state, torch.tensor([[0.5, 0.5]])))
        self.assertTrue(torch.allclose(delta.delta_val, torch.tensor([[[2.0], [2.0]]])))

    def test_dense_transition_streaming_matches_reference(self) -> None:
        torch.manual_seed(3)
        src = Layer(
            dim=4,
            num_nodes=7,
            state=torch.randn(2, 7),
            val=torch.randn(2, 7, 4),
        )
        dst = Layer.zeros(dim=6, num_nodes=5, batch_shape=(2,))
        route_fn = _make_route_fn(src_dim=4, dst_nodes=5)

        reference = Transition(
            route_fn=route_fn,
            state_activation_fn=lambda x: torch.nn.functional.softplus(x) + 0.1,
            val_proj_fn=lambda val: val[..., :3].repeat_interleave(2, dim=-1),
            state_proj_fn=_state_proj_fn,
            implementation="reference",
        )
        streaming = Transition(
            route_fn=route_fn,
            state_activation_fn=lambda x: torch.nn.functional.softplus(x) + 0.1,
            val_proj_fn=lambda val: val[..., :3].repeat_interleave(2, dim=-1),
            state_proj_fn=_state_proj_fn,
            implementation="streaming",
            src_block_size=3,
        )

        self.assert_delta_close(reference.compute_delta(src, dst), streaming.compute_delta(src, dst))

    def test_dense_transition_kernel_matches_reference(self) -> None:
        torch.manual_seed(13)
        src = Layer(
            dim=4,
            num_nodes=7,
            state=torch.randn(2, 7),
            val=torch.randn(2, 7, 4),
        )
        dst = Layer.zeros(dim=6, num_nodes=5, batch_shape=(2,))
        route_fn = LinearRoute(src_dim=4, dst_nodes=5)

        reference = Transition(
            route_fn=route_fn,
            state_activation_fn=lambda x: torch.nn.functional.softplus(x) + 0.1,
            val_proj_fn=lambda val: val[..., :3].repeat_interleave(2, dim=-1),
            state_proj_fn=_state_proj_fn,
            implementation="reference",
        )
        kernel = Transition(
            route_fn=LinearRoute(src_dim=4, dst_nodes=5),
            state_activation_fn=lambda x: torch.nn.functional.softplus(x) + 0.1,
            val_proj_fn=lambda val: val[..., :3].repeat_interleave(2, dim=-1),
            state_proj_fn=_state_proj_fn,
            implementation="kernel",
        )
        kernel.route_fn.load_state_dict(reference.route_fn.state_dict())

        self.assert_delta_close(reference.compute_delta(src, dst), kernel.compute_delta(src, dst))

    def test_dense_transition_kernel_matches_reference_with_mlp_route(self) -> None:
        torch.manual_seed(15)
        src = Layer(
            dim=4,
            num_nodes=7,
            state=torch.randn(2, 7),
            val=torch.randn(2, 7, 4),
        )
        dst = Layer.zeros(dim=6, num_nodes=5, batch_shape=(2,))
        route_fn = MLPRoute(src_dim=4, dst_nodes=5, hidden_dim=9)

        reference = Transition(
            route_fn=route_fn,
            state_activation_fn=lambda x: torch.nn.functional.softplus(x) + 0.1,
            val_proj_fn=lambda val: val[..., :3].repeat_interleave(2, dim=-1),
            state_proj_fn=_state_proj_fn,
            implementation="reference",
        )
        kernel = Transition(
            route_fn=MLPRoute(src_dim=4, dst_nodes=5, hidden_dim=9),
            state_activation_fn=lambda x: torch.nn.functional.softplus(x) + 0.1,
            val_proj_fn=lambda val: val[..., :3].repeat_interleave(2, dim=-1),
            state_proj_fn=_state_proj_fn,
            implementation="kernel",
        )
        kernel.route_fn.load_state_dict(reference.route_fn.state_dict())

        self.assert_delta_close(reference.compute_delta(src, dst), kernel.compute_delta(src, dst))

    def test_sparse_transition_streaming_matches_reference(self) -> None:
        torch.manual_seed(4)
        src = Layer(
            dim=3,
            num_nodes=8,
            state=torch.randn(2, 8),
            val=torch.randn(2, 8, 3),
        )
        dst = Layer.zeros(dim=4, num_nodes=6, batch_shape=(2,))
        route_fn = _make_route_fn(src_dim=3, dst_nodes=6)

        reference = SparseTransition(
            route_fn=route_fn,
            topk=2,
            state_activation_fn=lambda x: x + 1.25,
            val_proj_fn=lambda val: torch.cat((val, val[..., :1]), dim=-1),
            state_proj_fn=_state_proj_fn,
            implementation="reference",
        )
        streaming = SparseTransition(
            route_fn=route_fn,
            topk=2,
            state_activation_fn=lambda x: x + 1.25,
            val_proj_fn=lambda val: torch.cat((val, val[..., :1]), dim=-1),
            state_proj_fn=_state_proj_fn,
            implementation="streaming",
            src_block_size=3,
        )

        self.assert_delta_close(reference.compute_delta(src, dst), streaming.compute_delta(src, dst))

    def test_sparse_transition_kernel_matches_reference(self) -> None:
        torch.manual_seed(14)
        src = Layer(
            dim=3,
            num_nodes=8,
            state=torch.randn(2, 8),
            val=torch.randn(2, 8, 3),
        )
        dst = Layer.zeros(dim=4, num_nodes=6, batch_shape=(2,))
        route_fn = LinearRoute(src_dim=3, dst_nodes=6)

        reference = SparseTransition(
            route_fn=route_fn,
            topk=2,
            state_activation_fn=lambda x: x + 1.25,
            val_proj_fn=lambda val: torch.cat((val, val[..., :1]), dim=-1),
            state_proj_fn=_state_proj_fn,
            implementation="reference",
        )
        kernel = SparseTransition(
            route_fn=LinearRoute(src_dim=3, dst_nodes=6),
            topk=2,
            state_activation_fn=lambda x: x + 1.25,
            val_proj_fn=lambda val: torch.cat((val, val[..., :1]), dim=-1),
            state_proj_fn=_state_proj_fn,
            implementation="kernel",
        )
        kernel.route_fn.load_state_dict(reference.route_fn.state_dict())

        self.assert_delta_close(reference.compute_delta(src, dst), kernel.compute_delta(src, dst))

    def test_sparse_transition_kernel_matches_reference_with_mlp_route(self) -> None:
        torch.manual_seed(16)
        src = Layer(
            dim=3,
            num_nodes=8,
            state=torch.randn(2, 8),
            val=torch.randn(2, 8, 3),
        )
        dst = Layer.zeros(dim=4, num_nodes=6, batch_shape=(2,))
        route_fn = MLPRoute(src_dim=3, dst_nodes=6, hidden_dim=7)

        reference = SparseTransition(
            route_fn=route_fn,
            topk=2,
            state_activation_fn=lambda x: x + 1.25,
            val_proj_fn=lambda val: torch.cat((val, val[..., :1]), dim=-1),
            state_proj_fn=_state_proj_fn,
            implementation="reference",
        )
        kernel = SparseTransition(
            route_fn=MLPRoute(src_dim=3, dst_nodes=6, hidden_dim=7),
            topk=2,
            state_activation_fn=lambda x: x + 1.25,
            val_proj_fn=lambda val: torch.cat((val, val[..., :1]), dim=-1),
            state_proj_fn=_state_proj_fn,
            implementation="kernel",
        )
        kernel.route_fn.load_state_dict(reference.route_fn.state_dict())

        self.assert_delta_close(reference.compute_delta(src, dst), kernel.compute_delta(src, dst))

    def test_transition_routes_with_state_activation_then_adds_to_dst(self) -> None:
        src = Layer(
            dim=2,
            num_nodes=2,
            state=torch.tensor([[1.0, 2.0]]),
            val=torch.tensor([[[1.0, 0.0], [0.0, 2.0]]]),
        )
        dst = Layer.zeros(dim=2, num_nodes=2, batch_shape=(1,))

        def route_fn(src_val: torch.Tensor) -> torch.Tensor:
            return torch.zeros((*src_val.shape[:-1], 2))

        op = Transition(
            route_fn=route_fn,
            state_activation_fn=lambda x: x + 1.0,
            implementation="streaming",
            src_block_size=1,
        )
        updated = op(src, dst)

        self.assertTrue(torch.allclose(updated.state, torch.tensor([[4.0, 4.0]])))
        self.assertTrue(torch.allclose(updated.val, torch.tensor([[[1.0, 3.0], [1.0, 3.0]]])))

    def test_sparse_transition_uses_topk_before_softmax(self) -> None:
        src = Layer(
            dim=1,
            num_nodes=2,
            state=torch.tensor([[1.0, 2.0]]),
            val=torch.tensor([[[10.0], [20.0]]]),
        )
        dst = Layer(
            dim=1,
            num_nodes=3,
            state=torch.tensor([[5.0, 5.0, 5.0]]),
            val=torch.tensor([[[1.0], [1.0], [1.0]]]),
        )

        logits = torch.tensor([[[3.0, 1.0, 0.0], [0.0, 2.0, 4.0]]])

        def route_fn(src_val: torch.Tensor) -> torch.Tensor:
            return logits.expand(src_val.shape[0], -1, -1)

        op = SparseTransition(
            route_fn=route_fn,
            topk=1,
            state_activation_fn=lambda x: x + 1.0,
            implementation="streaming",
            src_block_size=2,
        )
        updated = op(src, dst)

        self.assertTrue(torch.allclose(updated.state, torch.tensor([[7.0, 5.0, 11.0]])))
        self.assertTrue(torch.allclose(updated.val, torch.tensor([[[21.0], [1.0], [61.0]]])))


if __name__ == "__main__":
    unittest.main()
