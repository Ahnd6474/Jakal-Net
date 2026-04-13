import os
import types
import unittest
from unittest import mock

import torch

from jakal_net import (
    DiagonalBilinearPairwise,
    Layer,
    LinearRoute,
    MLPRoute,
    Propagation,
    SparseTransition,
    Transition,
    native_status,
)
from jakal_net import native_available
from jakal_net.native_backend import DISABLE_NATIVE_ENV
import jakal_net.native_backend as native_backend


def _state_proj_fn(state: torch.Tensor) -> torch.Tensor:
    return state * 1.25 - 0.5


def _val_proj_fn(val: torch.Tensor) -> torch.Tensor:
    return val * 0.5 + 0.25


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


if __name__ == "__main__":
    unittest.main()
