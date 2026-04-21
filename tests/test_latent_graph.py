import unittest

import torch

from jakal_net import KModule, Layer


class KModuleTests(unittest.TestCase):
    def test_initialize_state_matches_configured_shape(self) -> None:
        module = KModule(dim=6, num_nodes=10, route_topk=4, propagation_topk=3)

        layer = module.initialize_state(batch_size=3, device="cpu", dtype=torch.float32)

        self.assertEqual(layer.state.shape, (3, 10))
        self.assertEqual(layer.val.shape, (3, 10, 6))
        self.assertEqual(layer.dim, 6)
        self.assertEqual(layer.num_nodes, 10)

    def test_forward_returns_k_layers_and_b_delta(self) -> None:
        torch.manual_seed(0)
        module = KModule(
            dim=8,
            num_nodes=12,
            route_rank=4,
            pairwise_rank=4,
            route_topk=5,
            propagation_topk=4,
            propagation_layers=2,
            implementation="reference",
        )
        b_layer = Layer(
            dim=8,
            num_nodes=7,
            state=torch.randn(2, 7),
            val=torch.randn(2, 7, 8),
        )

        output = module(b_layer, update_b=False)

        self.assertEqual(output.routed_k_layer.state.shape, (2, 12))
        self.assertEqual(output.propagated_k_layer.val.shape, (2, 12, 8))
        self.assertEqual(output.b_delta.delta_state.shape, (2, 7))
        self.assertEqual(output.b_delta.delta_val.shape, (2, 7, 8))
        self.assertIsNone(output.updated_b_layer)

    def test_forward_can_update_b_layer(self) -> None:
        torch.manual_seed(1)
        module = KModule(
            dim=4,
            num_nodes=9,
            route_rank=3,
            pairwise_rank=3,
            route_topk=4,
            propagation_topk=3,
            implementation="reference",
        )
        b_layer = Layer(
            dim=4,
            num_nodes=5,
            state=torch.randn(2, 5),
            val=torch.randn(2, 5, 4),
        )

        output = module(b_layer, update_b=True)

        self.assertIsNotNone(output.updated_b_layer)
        assert output.updated_b_layer is not None
        self.assertEqual(output.updated_b_layer.state.shape, b_layer.state.shape)
        self.assertEqual(output.updated_b_layer.val.shape, b_layer.val.shape)

    def test_reset_state_replaces_only_selected_batch_rows(self) -> None:
        torch.manual_seed(2)
        module = KModule(dim=5, num_nodes=6, route_topk=3, propagation_topk=3)
        base = module.initialize_state(batch_size=2, device="cpu", dtype=torch.float32)
        modified = base.with_tensors(
            state=base.state + 0.5,
            val=base.val + 0.25,
        )

        reset = module.reset_state(modified, reset_mask=torch.tensor([True, False]))

        self.assertTrue(torch.allclose(reset.state[0], base.state[0]))
        self.assertTrue(torch.allclose(reset.val[0], base.val[0]))
        self.assertTrue(torch.allclose(reset.state[1], modified.state[1]))
        self.assertTrue(torch.allclose(reset.val[1], modified.val[1]))

    def test_forward_reset_mask_applies_once_to_each_call(self) -> None:
        torch.manual_seed(3)
        module = KModule(dim=4, num_nodes=5, route_topk=3, propagation_topk=3)
        b_layer = Layer(
            dim=4,
            num_nodes=1,
            state=torch.randn(2, 1),
            val=torch.randn(2, 1, 4),
        )
        initial = module.initialize_state(batch_size=2, device="cpu", dtype=torch.float32)
        modified = initial.with_tensors(
            state=initial.state + 0.5,
            val=initial.val + 0.25,
        )

        output = module(
            b_layer,
            k_layer=modified,
            reset_mask=torch.tensor([True, False]),
            update_b=False,
        )

        expected_first = module.route_from_b(
            Layer(dim=4, num_nodes=1, state=b_layer.state[:1], val=b_layer.val[:1]),
            Layer(dim=4, num_nodes=5, state=initial.state[:1], val=initial.val[:1]),
        )
        expected_second = module.route_from_b(
            Layer(dim=4, num_nodes=1, state=b_layer.state[1:], val=b_layer.val[1:]),
            Layer(dim=4, num_nodes=5, state=modified.state[1:], val=modified.val[1:]),
        )

        self.assertTrue(torch.allclose(output.routed_k_layer.state[:1], expected_first.state, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(output.routed_k_layer.val[:1], expected_first.val, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(output.routed_k_layer.state[1:], expected_second.state, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(output.routed_k_layer.val[1:], expected_second.val, atol=1e-5, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
