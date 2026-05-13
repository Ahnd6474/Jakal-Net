import unittest

import torch
from torch import nn

from jakal_net import BModule, KModule, SModule
from jakal_net._architectural_common import signed_softmax_state
from jakal_net.core import Layer, LayerDelta
from jakal_net.modules import ResidualFeedForward, make_norm
from jakal_net.native_backend import _native_scan_signed_softmax_state
from jakal_net.propagation_stack import PropagationLayer


class _EchoDeltaPropagation(nn.Module):
    def __init__(self, *, window: int) -> None:
        super().__init__()
        self.sparse_type = "window"
        self.window = int(window)

    def compute_delta(self, layer: Layer) -> LayerDelta:
        return LayerDelta(
            delta_state=torch.zeros_like(layer.state),
            delta_val=layer.val.clone(),
        )


class ArchitecturalModuleTests(unittest.TestCase):
    def test_signed_softmax_state_matches_unsigned_softmax_without_extra_layer_norm(self) -> None:
        state = torch.tensor([[2.0, -1.0, 0.0, -3.0]], dtype=torch.float32)

        expected = torch.sign(state) * torch.softmax(state.abs(), dim=-1) * state.shape[-1]
        actual = signed_softmax_state(state)
        native_actual = _native_scan_signed_softmax_state(state)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(native_actual, expected, atol=1e-6, rtol=1e-6))

    def test_s_module_encodes_sequence_layer(self) -> None:
        torch.manual_seed(0)
        s_module = SModule(
            vocab_size=32,
            dim=8,
            max_seq_len=12,
            s_layers=1,
            pairwise_kind="low_rank_bilinear",
            pairwise_rank=4,
            implementation="reference",
            s_window=12,
            norm_kind="rmsnorm",
        )
        state_projection = torch.nn.Linear(8, 1)
        input_ids = torch.randint(0, 32, (2, 5))

        layer = s_module.encode(input_ids, state_projection=state_projection)

        self.assertEqual(layer.state.shape, (2, 6))
        self.assertEqual(layer.val.shape, (2, 6, 8))
        self.assertAlmostEqual(
            float(s_module.sequence_stack.blocks[0].residual_scale.detach().item()),
            1.0,
        )

    def test_b_module_scan_returns_query_and_memory(self) -> None:
        torch.manual_seed(1)
        b_module = BModule(
            dim=8,
            memory_slots=(6, 3),
            memory_topk=2,
            pairwise_kind="low_rank_bilinear",
            route_kind="low_rank_bilinear",
            pairwise_rank=4,
            route_rank=4,
            implementation="reference",
        )
        state_projection = torch.nn.Linear(8, 1)
        query_projection = torch.nn.Linear(8, 8, bias=False)
        query_input_norm = torch.nn.LayerNorm(8)
        aligned_s = torch.randn(2, 5, 8)
        memory_state = b_module.initialize_state(2, device="cpu", dtype=aligned_s.dtype)

        output = b_module.scan(
            aligned_s,
            memory_state,
            state_projection=state_projection,
            query_projection=query_projection,
            query_input_norm=query_input_norm,
        )

        self.assertEqual(output.query_layer.state.shape, (2, 5))
        self.assertEqual(output.query_layer.val.shape, (2, 5, 8))
        self.assertEqual(len(output.memory_state), 2)
        self.assertEqual(output.memory_state[0].val.shape, (2, 6, 8))

    def test_b_module_bridge_layer_and_injection_work(self) -> None:
        torch.manual_seed(2)
        b_module = BModule(
            dim=8,
            memory_slots=(6, 3),
            memory_topk=2,
            pairwise_kind="low_rank_bilinear",
            route_kind="low_rank_bilinear",
            pairwise_rank=4,
            route_rank=4,
            implementation="reference",
        )
        state_projection = torch.nn.Linear(8, 1)
        memory_state = b_module.initialize_state(2, device="cpu", dtype=torch.float32)

        bridge = b_module.build_bridge_layer(memory_state, state_projection=state_projection)
        injected = b_module.inject_bridge(bridge, memory_state)

        self.assertEqual(bridge.state.shape, (2, 1))
        self.assertEqual(bridge.val.shape, (2, 1, 8))
        self.assertEqual(len(injected), 2)
        self.assertEqual(injected[0].val.shape, (2, 6, 8))

    def test_b_module_scan_can_attach_k_module(self) -> None:
        torch.manual_seed(3)
        b_module = BModule(
            dim=8,
            memory_slots=(6, 3),
            memory_topk=2,
            pairwise_kind="low_rank_bilinear",
            route_kind="low_rank_bilinear",
            pairwise_rank=4,
            route_rank=4,
            implementation="reference",
        )
        k_module = KModule(
            dim=8,
            num_nodes=10,
            route_rank=4,
            pairwise_rank=4,
            route_topk=4,
            propagation_topk=3,
            implementation="reference",
        )
        state_projection = torch.nn.Linear(8, 1)
        query_projection = torch.nn.Linear(8, 8, bias=False)
        query_input_norm = torch.nn.LayerNorm(8)
        aligned_s = torch.randn(2, 4, 8)
        memory_state = b_module.initialize_state(2, device="cpu", dtype=aligned_s.dtype)

        output = b_module.scan(
            aligned_s,
            memory_state,
            state_projection=state_projection,
            query_projection=query_projection,
            query_input_norm=query_input_norm,
            knowledge_module=k_module,
        )

        self.assertIsNotNone(output.bridge_layer)
        self.assertIsNotNone(output.knowledge_state)
        self.assertIsNotNone(output.knowledge_output)
        assert output.bridge_layer is not None
        assert output.knowledge_state is not None
        self.assertEqual(output.bridge_layer.val.shape, (2, 1, 8))
        self.assertEqual(output.knowledge_state.val.shape, (2, 10, 8))

    def test_b_module_scan_supports_multi_level_multi_head_configuration(self) -> None:
        torch.manual_seed(4)
        b_module = BModule(
            dim=8,
            memory_slots=(8, 4, 2, 2),
            memory_topk=2,
            pairwise_kind="low_rank_bilinear",
            route_kind="low_rank_bilinear",
            pairwise_rank=4,
            route_rank=4,
            pairwise_heads=2,
            route_heads=2,
            implementation="native",
        )
        state_projection = torch.nn.Linear(8, 1)
        query_projection = torch.nn.Linear(8, 8, bias=False)
        query_input_norm = torch.nn.LayerNorm(8)
        aligned_s = torch.randn(2, 3, 8)
        memory_state = b_module.initialize_state(2, device="cpu", dtype=aligned_s.dtype)

        output = b_module.scan(
            aligned_s,
            memory_state,
            state_projection=state_projection,
            query_projection=query_projection,
            query_input_norm=query_input_norm,
        )

        self.assertEqual(len(output.memory_state), 4)
        self.assertEqual(output.memory_state[0].val.shape, (2, 8, 8))
        self.assertEqual(output.memory_state[3].val.shape, (2, 2, 8))

    def test_propagation_layer_applies_post_norm_on_dense_fastpath(self) -> None:
        layer = Layer(
            dim=4,
            num_nodes=2,
            state=torch.tensor([[0.1, -0.2]], dtype=torch.float32),
            val=torch.tensor(
                [[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]],
                dtype=torch.float32,
            ),
        )
        norm = make_norm(4, "rmsnorm")
        propagation_layer = PropagationLayer(
            propagation=_EchoDeltaPropagation(window=1),
            norm=norm,
            ffn=nn.Identity(),
            unit_norm_values=False,
            state_activation_kind="signed_softmax",
            residual_scale_style="rezero",
            residual_scale_init=1.0,
        )

        actual = propagation_layer(layer)
        expected = norm(layer.val + layer.val)

        self.assertTrue(torch.allclose(actual.val, expected, atol=1e-6, rtol=1e-6))

    def test_propagation_layer_applies_post_norm_on_general_path(self) -> None:
        layer = Layer(
            dim=4,
            num_nodes=3,
            state=torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float32),
            val=torch.tensor(
                [[[1.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 4.0], [5.0, 6.0, 7.0, 8.0]]],
                dtype=torch.float32,
            ),
        )
        norm = make_norm(4, "rmsnorm")
        propagation_layer = PropagationLayer(
            propagation=_EchoDeltaPropagation(window=1),
            norm=norm,
            ffn=nn.Identity(),
            unit_norm_values=False,
            state_activation_kind="signed_softmax",
            residual_scale_style="rezero",
            residual_scale_init=1.0,
        )

        actual = propagation_layer(layer)
        expected = norm(layer.val + layer.val)

        self.assertTrue(torch.allclose(actual.val, expected, atol=1e-6, rtol=1e-6))

    def test_residual_feed_forward_without_input_norm_keeps_identity_when_zeroed(self) -> None:
        ffn = ResidualFeedForward(
            dim=4,
            hidden_mult=1.0,
            residual_scale=1.0,
            learnable_residual_scale=False,
            use_input_norm=False,
        )
        first = ffn.net[0]
        second = ffn.net[3]
        assert isinstance(first, nn.Linear)
        assert isinstance(second, nn.Linear)
        with torch.no_grad():
            first.weight.zero_()
            first.bias.zero_()
            second.weight.zero_()
            second.bias.zero_()

        val = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]],
            dtype=torch.float32,
        )
        actual = ffn(val)
        expected = val

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
