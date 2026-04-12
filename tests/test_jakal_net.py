import unittest

import torch

from jakal_net import Layer, Propagation, SparsePropagation, SparseTransition, Transition


class JakalNetModuleTests(unittest.TestCase):
    def test_layer_zeros_supports_prefix_dims(self) -> None:
        layer = Layer.zeros(dim=3, num_nodes=4, batch_shape=(2, 5))

        self.assertEqual(layer.state.shape, (2, 5, 4))
        self.assertEqual(layer.val.shape, (2, 5, 4, 3))
        self.assertTrue(torch.equal(layer.state, torch.zeros_like(layer.state)))
        self.assertTrue(torch.equal(layer.val, torch.zeros_like(layer.val)))

    def test_propagation_returns_delta_without_state_activation(self) -> None:
        layer = Layer(
            dim=1,
            num_nodes=2,
            state=torch.tensor([[-1.0, 2.0]]),
            val=torch.tensor([[[1.0], [3.0]]]),
        )

        def pairwise_fn(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
            return torch.ones((*target.shape[:-2], target.shape[-2], source.shape[-2]))

        op = Propagation(pairwise_fn=pairwise_fn)
        delta = op(layer)

        self.assertTrue(torch.allclose(delta.delta_state, torch.tensor([[0.5, 0.5]])))
        self.assertTrue(torch.allclose(delta.delta_val, torch.tensor([[[2.0], [2.0]]])))

    def test_sparse_propagation_window_is_causal(self) -> None:
        layer = Layer(
            dim=1,
            num_nodes=4,
            state=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            val=torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]),
        )

        def pairwise_fn(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
            return torch.ones((*target.shape[:-2], target.shape[-2], source.shape[-2]))

        op = SparsePropagation(pairwise_fn=pairwise_fn, sparse_type="window", window=1)
        delta = op(layer)

        expected = torch.tensor([[0.5, 1.5, 2.5, 3.5]])
        self.assertTrue(torch.allclose(delta.delta_state, expected))

    def test_sparse_propagation_topk_keeps_best_sources_per_target(self) -> None:
        layer = Layer(
            dim=1,
            num_nodes=3,
            state=torch.tensor([[1.0, 2.0, 3.0]]),
            val=torch.tensor([[[1.0], [5.0], [2.0]]]),
        )

        def pairwise_fn(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
            source_scores = source.squeeze(-1).unsqueeze(-2)
            return source_scores.expand(*target.shape[:-2], target.shape[-2], source.shape[-2])

        op = SparsePropagation(pairwise_fn=pairwise_fn, sparse_type="topk", topk=1)
        delta = op(layer)

        expected = torch.full((1, 3), 10.0 / 6.0)
        self.assertTrue(torch.allclose(delta.delta_state, expected))

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
        )
        updated = op(src, dst)

        self.assertTrue(torch.allclose(updated.state, torch.tensor([[7.0, 5.0, 11.0]])))
        self.assertTrue(torch.allclose(updated.val, torch.tensor([[[21.0], [1.0], [61.0]]])))


if __name__ == "__main__":
    unittest.main()
