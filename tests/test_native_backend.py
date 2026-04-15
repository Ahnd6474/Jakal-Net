import os
import types
import unittest
from unittest import mock

import torch

from jakal_net import (
    BilinearPairwise,
    DiagonalBilinearPairwise,
    DiagonalBilinearRoute,
    Layer,
    LayerDelta,
    LinearRoute,
    LowRankBilinearRoute,
    MLPRoute,
    Propagation,
    SparsePropagation,
    SparseTransition,
    Transition,
    native_status,
)
from jakal_net import native_available
from jakal_net.native_backend import (
    DISABLE_NATIVE_ENV,
    propagation_query_topk_native,
    transition_query_topk_native,
)
import jakal_net.native_backend as native_backend


def _state_proj_fn(state: torch.Tensor) -> torch.Tensor:
    return state * 1.25 - 0.5


def _val_proj_fn(val: torch.Tensor) -> torch.Tensor:
    return val * 0.5 + 0.25


def _gather_state(projected_state: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    source_nodes = projected_state.shape[-1]
    expanded = projected_state.unsqueeze(-2).expand(*indices.shape[:-1], source_nodes)
    return torch.take_along_dim(expanded, indices, dim=-1)


def _gather_val(projected_val: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    source_nodes = projected_val.shape[-2]
    out_dim = projected_val.shape[-1]
    expanded = projected_val.unsqueeze(-3).expand(*indices.shape[:-1], source_nodes, out_dim)
    return torch.take_along_dim(
        expanded,
        indices.unsqueeze(-1).expand(*indices.shape, out_dim),
        dim=-2,
    )


class NativeBackendTests(unittest.TestCase):
    def tearDown(self) -> None:
        native_status(force_reload=True)

    def test_native_status_reports_disabled_env(self) -> None:
        with mock.patch.dict(os.environ, {DISABLE_NATIVE_ENV: "1"}, clear=False):
            status = native_status(force_reload=True)
            self.assertFalse(status.available)
            self.assertIn(DISABLE_NATIVE_ENV, status.error or "")

        native_status(force_reload=True)

    def test_native_loader_uses_fake_extension_when_available(self) -> None:
        fake_module = types.SimpleNamespace(
            supported_ops=lambda: ["propagation_dense"],
            supported_devices=lambda: ["cpu"],
            backend_name=lambda: "fake_native",
            propagation_dense=lambda *args: (
                torch.full((2, 7), 3.0),
                torch.full((2, 7, 4), 5.0),
            ),
        )

        with mock.patch.object(native_backend.importlib, "import_module", return_value=fake_module):
            status = native_status(force_reload=True)
            self.assertTrue(status.available)
            self.assertEqual(status.backend_name, "fake_native")
            self.assertTrue(native_available())

            layer = Layer(
                dim=4,
                num_nodes=7,
                state=torch.randn(2, 7),
                val=torch.randn(2, 7, 4),
            )
            op = Propagation(
                pairwise_fn=DiagonalBilinearPairwise(dim=4),
                implementation="native",
            )
            delta = op.compute_delta(layer)

        self.assertTrue(torch.equal(delta.delta_state, torch.full((2, 7), 3.0)))
        self.assertTrue(torch.equal(delta.delta_val, torch.full((2, 7, 4), 5.0)))

    def test_propagation_native_falls_back_to_kernel_when_extension_missing(self) -> None:
        torch.manual_seed(0)
        layer = Layer(
            dim=4,
            num_nodes=7,
            state=torch.randn(2, 7),
            val=torch.randn(2, 7, 4),
        )

        with mock.patch.dict(os.environ, {DISABLE_NATIVE_ENV: "1"}, clear=False):
            native_status(force_reload=True)
            reference = Propagation(
                pairwise_fn=DiagonalBilinearPairwise(dim=4),
                state_proj_fn=_state_proj_fn,
                val_proj_fn=_val_proj_fn,
                implementation="kernel",
            )
            native = Propagation(
                pairwise_fn=DiagonalBilinearPairwise(dim=4),
                state_proj_fn=_state_proj_fn,
                val_proj_fn=_val_proj_fn,
                implementation="native",
            )
            native.pairwise_fn.load_state_dict(reference.pairwise_fn.state_dict())
            kernel_delta = reference.compute_delta(layer)
            native_delta = native.compute_delta(layer)

        native_status(force_reload=True)
        self.assertTrue(torch.allclose(kernel_delta.delta_state, native_delta.delta_state))
        self.assertTrue(torch.allclose(kernel_delta.delta_val, native_delta.delta_val))

    def test_transition_native_falls_back_to_kernel_when_extension_missing(self) -> None:
        torch.manual_seed(1)
        src = Layer(
            dim=4,
            num_nodes=6,
            state=torch.randn(2, 6),
            val=torch.randn(2, 6, 4),
        )
        dst = Layer.zeros(dim=5, num_nodes=4, batch_shape=(2,))

        with mock.patch.dict(os.environ, {DISABLE_NATIVE_ENV: "1"}, clear=False):
            native_status(force_reload=True)
            reference = Transition(
                route_fn=MLPRoute(src_dim=4, dst_nodes=4, hidden_dim=7),
                state_proj_fn=_state_proj_fn,
                val_proj_fn=lambda val: val[..., :1].repeat(1, 1, 5),
                implementation="kernel",
            )
            native = Transition(
                route_fn=MLPRoute(src_dim=4, dst_nodes=4, hidden_dim=7),
                state_proj_fn=_state_proj_fn,
                val_proj_fn=lambda val: val[..., :1].repeat(1, 1, 5),
                implementation="native",
            )
            native.route_fn.load_state_dict(reference.route_fn.state_dict())
            kernel_delta = reference.compute_delta(src, dst)
            native_delta = native.compute_delta(src, dst)

        native_status(force_reload=True)
        self.assertTrue(torch.allclose(kernel_delta.delta_state, native_delta.delta_state))
        self.assertTrue(torch.allclose(kernel_delta.delta_val, native_delta.delta_val))

    def test_sparse_transition_native_falls_back_without_extension(self) -> None:
        torch.manual_seed(2)
        src = Layer(
            dim=3,
            num_nodes=8,
            state=torch.randn(2, 8),
            val=torch.randn(2, 8, 3),
        )
        dst = Layer.zeros(dim=4, num_nodes=6, batch_shape=(2,))

        with mock.patch.dict(os.environ, {DISABLE_NATIVE_ENV: "1"}, clear=False):
            native_status(force_reload=True)
            reference = SparseTransition(
                route_fn=LinearRoute(src_dim=3, dst_nodes=6),
                topk=2,
                state_proj_fn=_state_proj_fn,
                val_proj_fn=lambda val: torch.cat((val, val[..., :1]), dim=-1),
                implementation="kernel",
            )
            native = SparseTransition(
                route_fn=LinearRoute(src_dim=3, dst_nodes=6),
                topk=2,
                state_proj_fn=_state_proj_fn,
                val_proj_fn=lambda val: torch.cat((val, val[..., :1]), dim=-1),
                implementation="native",
            )
            native.route_fn.load_state_dict(reference.route_fn.state_dict())
            kernel_delta = reference.compute_delta(src, dst)
            native_delta = native.compute_delta(src, dst)

        native_status(force_reload=True)
        self.assertTrue(torch.allclose(kernel_delta.delta_state, native_delta.delta_state))
        self.assertTrue(torch.allclose(kernel_delta.delta_val, native_delta.delta_val))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable.")
class CudaNativeBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        status = native_status(force_reload=True)
        if not status.available:
            self.skipTest("Native backend is unavailable.")
        if "cuda" not in status.supported_devices:
            self.skipTest("Native backend does not report CUDA support.")
        self.device = torch.device("cuda")

    def tearDown(self) -> None:
        native_status(force_reload=True)

    def assert_delta_close(
        self, left, right, *, atol: float = 1e-5, rtol: float = 1e-5
    ) -> None:
        self.assertTrue(
            torch.allclose(left.delta_state, right.delta_state, atol=atol, rtol=rtol)
        )
        self.assertTrue(
            torch.allclose(left.delta_val, right.delta_val, atol=atol, rtol=rtol)
        )

    def test_dense_propagation_native_uses_cuda_backend_and_matches_reference(self) -> None:
        torch.manual_seed(30)
        layer = Layer(
            dim=4,
            num_nodes=7,
            state=torch.randn(2, 7, device=self.device),
            val=torch.randn(2, 7, 4, device=self.device),
        )
        reference = Propagation(
            pairwise_fn=DiagonalBilinearPairwise(dim=4).to(self.device),
            state_proj_fn=_state_proj_fn,
            val_proj_fn=_val_proj_fn,
            implementation="reference",
        )
        kernel = Propagation(
            pairwise_fn=DiagonalBilinearPairwise(dim=4).to(self.device),
            state_proj_fn=_state_proj_fn,
            val_proj_fn=_val_proj_fn,
            implementation="kernel",
        )
        native = Propagation(
            pairwise_fn=DiagonalBilinearPairwise(dim=4).to(self.device),
            state_proj_fn=_state_proj_fn,
            val_proj_fn=_val_proj_fn,
            implementation="native",
        )
        kernel.pairwise_fn.load_state_dict(reference.pairwise_fn.state_dict())
        native.pairwise_fn.load_state_dict(reference.pairwise_fn.state_dict())

        reference_delta = reference.compute_delta(layer)
        self.assert_delta_close(reference_delta, kernel.compute_delta(layer))

        module = native_backend._native_module()
        with mock.patch.object(module, "propagation_dense", wraps=module.propagation_dense) as wrapped:
            native_delta = native.compute_delta(layer)
        self.assertGreater(wrapped.call_count, 0)
        self.assert_delta_close(reference_delta, native_delta)

    def test_sparse_propagation_native_window_and_topk_match_reference_on_cuda(self) -> None:
        torch.manual_seed(31)
        layer = Layer(
            dim=5,
            num_nodes=9,
            state=torch.randn(2, 9, device=self.device),
            val=torch.randn(2, 9, 5, device=self.device),
        )

        window_reference = SparsePropagation(
            pairwise_fn=DiagonalBilinearPairwise(dim=5).to(self.device),
            state_proj_fn=_state_proj_fn,
            val_proj_fn=_val_proj_fn,
            sparse_type="window",
            window=3,
            implementation="reference",
        )
        window_native = SparsePropagation(
            pairwise_fn=DiagonalBilinearPairwise(dim=5).to(self.device),
            state_proj_fn=_state_proj_fn,
            val_proj_fn=_val_proj_fn,
            sparse_type="window",
            window=3,
            implementation="native",
        )
        window_native.pairwise_fn.load_state_dict(window_reference.pairwise_fn.state_dict())

        module = native_backend._native_module()
        with mock.patch.object(module, "propagation_window", wraps=module.propagation_window) as wrapped_window:
            window_delta = window_native.compute_delta(layer)
        self.assertGreater(wrapped_window.call_count, 0)
        self.assert_delta_close(window_reference.compute_delta(layer), window_delta)

        topk_reference = SparsePropagation(
            pairwise_fn=BilinearPairwise(dim=5).to(self.device),
            state_proj_fn=_state_proj_fn,
            val_proj_fn=_val_proj_fn,
            sparse_type="topk",
            topk=3,
            implementation="reference",
        )
        topk_kernel = SparsePropagation(
            pairwise_fn=BilinearPairwise(dim=5).to(self.device),
            state_proj_fn=_state_proj_fn,
            val_proj_fn=_val_proj_fn,
            sparse_type="topk",
            topk=3,
            implementation="kernel",
        )
        topk_native = SparsePropagation(
            pairwise_fn=BilinearPairwise(dim=5).to(self.device),
            state_proj_fn=_state_proj_fn,
            val_proj_fn=_val_proj_fn,
            sparse_type="topk",
            topk=3,
            implementation="native",
        )
        topk_kernel.pairwise_fn.load_state_dict(topk_reference.pairwise_fn.state_dict())
        topk_native.pairwise_fn.load_state_dict(topk_reference.pairwise_fn.state_dict())

        topk_reference_delta = topk_reference.compute_delta(layer)
        self.assert_delta_close(topk_reference_delta, topk_kernel.compute_delta(layer))
        with mock.patch.object(module, "propagation_topk", wraps=module.propagation_topk) as wrapped_topk:
            topk_native_delta = topk_native.compute_delta(layer)
        self.assertGreater(wrapped_topk.call_count, 0)
        self.assert_delta_close(topk_reference_delta, topk_native_delta)

    def test_dense_transition_native_matches_reference_with_mlp_route_on_cuda(self) -> None:
        torch.manual_seed(32)
        src = Layer(
            dim=4,
            num_nodes=7,
            state=torch.randn(2, 7, device=self.device),
            val=torch.randn(2, 7, 4, device=self.device),
        )
        dst = Layer.zeros(dim=6, num_nodes=5, batch_shape=(2,), device=self.device)
        reference = Transition(
            route_fn=MLPRoute(src_dim=4, dst_nodes=5, hidden_dim=9).to(self.device),
            state_activation_fn=lambda x: torch.nn.functional.softplus(x) + 0.1,
            val_proj_fn=lambda val: val[..., :3].repeat_interleave(2, dim=-1),
            state_proj_fn=_state_proj_fn,
            implementation="reference",
        )
        kernel = Transition(
            route_fn=MLPRoute(src_dim=4, dst_nodes=5, hidden_dim=9).to(self.device),
            state_activation_fn=lambda x: torch.nn.functional.softplus(x) + 0.1,
            val_proj_fn=lambda val: val[..., :3].repeat_interleave(2, dim=-1),
            state_proj_fn=_state_proj_fn,
            implementation="kernel",
        )
        native = Transition(
            route_fn=MLPRoute(src_dim=4, dst_nodes=5, hidden_dim=9).to(self.device),
            state_activation_fn=lambda x: torch.nn.functional.softplus(x) + 0.1,
            val_proj_fn=lambda val: val[..., :3].repeat_interleave(2, dim=-1),
            state_proj_fn=_state_proj_fn,
            implementation="native",
        )
        kernel.route_fn.load_state_dict(reference.route_fn.state_dict())
        native.route_fn.load_state_dict(reference.route_fn.state_dict())

        reference_delta = reference.compute_delta(src, dst)
        self.assert_delta_close(reference_delta, kernel.compute_delta(src, dst))

        module = native_backend._native_module()
        with mock.patch.object(module, "transition_dense", wraps=module.transition_dense) as wrapped:
            native_delta = native.compute_delta(src, dst)
        self.assertGreater(wrapped.call_count, 0)
        self.assert_delta_close(reference_delta, native_delta)

    def test_sparse_transition_native_matches_reference_on_cuda(self) -> None:
        torch.manual_seed(33)
        src = Layer(
            dim=3,
            num_nodes=8,
            state=torch.randn(2, 8, device=self.device),
            val=torch.randn(2, 8, 3, device=self.device),
        )
        dst = Layer.zeros(dim=4, num_nodes=6, batch_shape=(2,), device=self.device)
        reference = SparseTransition(
            route_fn=LinearRoute(src_dim=3, dst_nodes=6).to(self.device),
            topk=2,
            state_activation_fn=lambda x: x + 1.25,
            val_proj_fn=lambda val: torch.cat((val, val[..., :1]), dim=-1),
            state_proj_fn=_state_proj_fn,
            implementation="reference",
        )
        kernel = SparseTransition(
            route_fn=LinearRoute(src_dim=3, dst_nodes=6).to(self.device),
            topk=2,
            state_activation_fn=lambda x: x + 1.25,
            val_proj_fn=lambda val: torch.cat((val, val[..., :1]), dim=-1),
            state_proj_fn=_state_proj_fn,
            implementation="kernel",
        )
        native = SparseTransition(
            route_fn=LinearRoute(src_dim=3, dst_nodes=6).to(self.device),
            topk=2,
            state_activation_fn=lambda x: x + 1.25,
            val_proj_fn=lambda val: torch.cat((val, val[..., :1]), dim=-1),
            state_proj_fn=_state_proj_fn,
            implementation="native",
        )
        kernel.route_fn.load_state_dict(reference.route_fn.state_dict())
        native.route_fn.load_state_dict(reference.route_fn.state_dict())

        reference_delta = reference.compute_delta(src, dst)
        self.assert_delta_close(reference_delta, kernel.compute_delta(src, dst))

        module = native_backend._native_module()
        with mock.patch.object(module, "transition_topk", wraps=module.transition_topk) as wrapped:
            native_delta = native.compute_delta(src, dst)
        self.assertGreater(wrapped.call_count, 0)
        self.assert_delta_close(reference_delta, native_delta)

    def test_sparse_transition_native_pairwise_routes_match_reference_on_cuda(self) -> None:
        torch.manual_seed(34)
        src = Layer(
            dim=4,
            num_nodes=7,
            state=torch.randn(2, 7, device=self.device),
            val=torch.randn(2, 7, 4, device=self.device),
        )
        dst = Layer(
            dim=4,
            num_nodes=6,
            state=torch.randn(2, 6, device=self.device),
            val=torch.randn(2, 6, 4, device=self.device),
        )

        module = native_backend._native_module()
        for route_fn in (
            DiagonalBilinearRoute(src_dim=4, dst_dim=4).to(self.device),
            LowRankBilinearRoute(src_dim=4, dst_dim=4, rank=3).to(self.device),
        ):
            reference = SparseTransition(
                route_fn=route_fn,
                topk=2,
                state_activation_fn=lambda x: x + 1.25,
                val_proj_fn=_val_proj_fn,
                state_proj_fn=_state_proj_fn,
                implementation="reference",
            )
            native = SparseTransition(
                route_fn=type(route_fn)(src_dim=4, dst_dim=4, **({"rank": 3} if isinstance(route_fn, LowRankBilinearRoute) else {})).to(self.device),
                topk=2,
                state_activation_fn=lambda x: x + 1.25,
                val_proj_fn=_val_proj_fn,
                state_proj_fn=_state_proj_fn,
                implementation="native",
                src_block_size=3,
                dst_block_size=2,
            )
            native.route_fn.load_state_dict(reference.route_fn.state_dict())

            reference_delta = reference.compute_delta(src, dst)
            with mock.patch.object(
                module,
                "transition_pairwise_topk",
                wraps=module.transition_pairwise_topk,
            ) as wrapped:
                native_delta = native.compute_delta(src, dst)
            self.assertGreater(wrapped.call_count, 0)
            self.assert_delta_close(reference_delta, native_delta)

    def test_query_topk_reduce_cuda_matches_reference(self) -> None:
        torch.manual_seed(35)
        module = native_backend._native_module()
        if native_status().backend_name != "aten_cpp_cuda":
            self.skipTest("Native CUDA source was not compiled.")

        edges = torch.randn(2, 3, 4, device=self.device)
        indices = torch.tensor(
            [[[0, 3, 5, 1], [2, 1, 4, 0], [5, 4, 3, 2]],
             [[1, 2, 3, 4], [0, 5, 4, 2], [3, 1, 0, 5]]],
            device=self.device,
            dtype=torch.long,
        )
        projected_state = torch.randn(2, 6, device=self.device)
        projected_val = torch.randn(2, 6, 5, device=self.device)

        expected_state = (edges * _gather_state(projected_state, indices)).sum(dim=-1)
        expected_val = (edges.unsqueeze(-1) * _gather_val(projected_val, indices)).sum(dim=-2)
        actual_state, actual_val = module.query_topk_reduce_cuda(
            edges.contiguous(),
            indices.contiguous(),
            projected_state.contiguous(),
            projected_val.contiguous(),
        )

        self.assertTrue(torch.allclose(expected_state, actual_state, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(expected_val, actual_val, atol=1e-5, rtol=1e-5))

    def test_query_only_native_ops_match_reference_on_cuda(self) -> None:
        torch.manual_seed(36)
        source = Layer(
            dim=4,
            num_nodes=9,
            state=torch.randn(2, 9, device=self.device),
            val=torch.randn(2, 9, 4, device=self.device),
        )
        query_val = torch.randn(2, 3, 4, device=self.device)
        projected_state = _state_proj_fn(source.state)
        projected_val = _val_proj_fn(source.val)

        pairwise = DiagonalBilinearPairwise(dim=4).to(self.device)
        scores = pairwise(query_val, source.val)
        selected = torch.topk(scores, k=4, dim=-1)
        edges = torch.nn.functional.softsign(selected.values)
        expected_propagation = LayerDelta(
            delta_state=(edges * _gather_state(projected_state, selected.indices)).sum(dim=-1),
            delta_val=(
                edges.unsqueeze(-1) * _gather_val(projected_val, selected.indices)
            ).sum(dim=-2),
        )

        module = native_backend._native_module()
        with mock.patch.object(
            module,
            "propagation_query_topk",
            wraps=module.propagation_query_topk,
        ) as wrapped:
            actual_propagation = propagation_query_topk_native(
                pairwise_fn=pairwise,
                edge_compress_name="softsign",
                query_val=query_val,
                source_val=source.val,
                projected_state=projected_state,
                projected_val=projected_val,
                topk=4,
                query_block_size=2,
                source_block_size=3,
            )
        self.assertGreater(wrapped.call_count, 0)
        self.assert_delta_close(expected_propagation, actual_propagation)

        route_fn = LowRankBilinearRoute(src_dim=4, dst_dim=4, rank=3).to(self.device)
        sender_strength = torch.nn.functional.softplus(source.state) + 0.1
        route_logits = route_fn(source.val, query_val).transpose(-1, -2)
        selected_route = torch.topk(route_logits, k=4, dim=-1)
        routes = torch.softmax(selected_route.values, dim=-1)
        weighted_state = sender_strength * projected_state
        weighted_val = sender_strength.unsqueeze(-1) * projected_val
        expected_transition = LayerDelta(
            delta_state=(routes * _gather_state(weighted_state, selected_route.indices)).sum(dim=-1),
            delta_val=(
                routes.unsqueeze(-1) * _gather_val(weighted_val, selected_route.indices)
            ).sum(dim=-2),
        )

        with mock.patch.object(
            module,
            "transition_query_topk",
            wraps=module.transition_query_topk,
        ) as wrapped:
            actual_transition = transition_query_topk_native(
                route_fn=route_fn,
                sender_strength=sender_strength,
                src_val=source.val,
                query_val=query_val,
                projected_state=projected_state,
                projected_val=projected_val,
                topk=4,
                query_block_size=2,
                source_block_size=3,
            )
        self.assertGreater(wrapped.call_count, 0)
        self.assert_delta_close(expected_transition, actual_transition)


if __name__ == "__main__":
    unittest.main()
