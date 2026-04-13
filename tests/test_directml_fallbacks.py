import os
import unittest
from unittest import mock

import torch

from jakal_net import (
    DiagonalBilinearPairwise,
    Layer,
    LinearRoute,
    Propagation,
    SparsePropagation,
    SparseTransition,
    Transition,
    native_status,
    resolve_device,
)
from jakal_net.native_backend import DISABLE_NATIVE_ENV


def _directml_device():
    try:
        return resolve_device("directml")
    except Exception:  # noqa: BLE001
        return None


DIRECTML_DEVICE = _directml_device()


def _state_proj_fn(state: torch.Tensor) -> torch.Tensor:
    return state * 1.25 - 0.5


def _val_proj_fn(val: torch.Tensor) -> torch.Tensor:
    return val * 0.5 + 0.25


@unittest.skipUnless(DIRECTML_DEVICE is not None, "DirectML is unavailable.")
class DirectMLFallbackTests(unittest.TestCase):
    def assert_delta_close(
        self, left, right, *, atol: float = 1e-5, rtol: float = 1e-5
    ) -> None:
        self.assertTrue(
            torch.allclose(left.delta_state, right.delta_state, atol=atol, rtol=rtol)
        )
        self.assertTrue(
            torch.allclose(left.delta_val, right.delta_val, atol=atol, rtol=rtol)
        )

    def tearDown(self) -> None:
        native_status(force_reload=True)

    def test_dense_propagation_streaming_matches_reference_on_directml(self) -> None:
        torch.manual_seed(20)
        layer = Layer(
            dim=4,
            num_nodes=7,
            state=torch.randn(2, 7, device=DIRECTML_DEVICE),
            val=torch.randn(2, 7, 4, device=DIRECTML_DEVICE),
        )
        pairwise = DiagonalBilinearPairwise(dim=4).to(DIRECTML_DEVICE)

        reference = Propagation(
            pairwise_fn=pairwise,
            state_proj_fn=_state_proj_fn,
            val_proj_fn=_val_proj_fn,
            implementation="reference",
        )
        streaming = Propagation(
            pairwise_fn=DiagonalBilinearPairwise(dim=4).to(DIRECTML_DEVICE),
            state_proj_fn=_state_proj_fn,
            val_proj_fn=_val_proj_fn,
            implementation="streaming",
            target_block_size=3,
            source_block_size=2,
        )
        streaming.pairwise_fn.load_state_dict(reference.pairwise_fn.state_dict())

        self.assert_delta_close(reference.compute_delta(layer), streaming.compute_delta(layer))

    def test_sparse_topk_propagation_modes_match_kernel_on_directml(self) -> None:
        torch.manual_seed(21)
        layer = Layer(
            dim=4,
            num_nodes=8,
            state=torch.randn(2, 8, device=DIRECTML_DEVICE),
            val=torch.randn(2, 8, 4, device=DIRECTML_DEVICE),
        )

        kernel = SparsePropagation(
            pairwise_fn=DiagonalBilinearPairwise(dim=4).to(DIRECTML_DEVICE),
            state_proj_fn=_state_proj_fn,
            val_proj_fn=_val_proj_fn,
            sparse_type="topk",
            topk=3,
            implementation="kernel",
            target_block_size=4,
            source_block_size=2,
        )
        reference = SparsePropagation(
            pairwise_fn=DiagonalBilinearPairwise(dim=4).to(DIRECTML_DEVICE),
            state_proj_fn=_state_proj_fn,
            val_proj_fn=_val_proj_fn,
            sparse_type="topk",
            topk=3,
            implementation="reference",
            target_block_size=4,
            source_block_size=2,
        )
        streaming = SparsePropagation(
            pairwise_fn=DiagonalBilinearPairwise(dim=4).to(DIRECTML_DEVICE),
            state_proj_fn=_state_proj_fn,
            val_proj_fn=_val_proj_fn,
            sparse_type="topk",
            topk=3,
            implementation="streaming",
            target_block_size=4,
            source_block_size=2,
        )
        native = SparsePropagation(
            pairwise_fn=DiagonalBilinearPairwise(dim=4).to(DIRECTML_DEVICE),
            state_proj_fn=_state_proj_fn,
            val_proj_fn=_val_proj_fn,
            sparse_type="topk",
            topk=3,
            implementation="native",
            target_block_size=4,
            source_block_size=2,
        )
        reference.pairwise_fn.load_state_dict(kernel.pairwise_fn.state_dict())
        streaming.pairwise_fn.load_state_dict(kernel.pairwise_fn.state_dict())
        native.pairwise_fn.load_state_dict(kernel.pairwise_fn.state_dict())

        kernel_delta = kernel.compute_delta(layer)
        self.assert_delta_close(kernel_delta, reference.compute_delta(layer))
        self.assert_delta_close(kernel_delta, streaming.compute_delta(layer))
        with mock.patch.dict(os.environ, {DISABLE_NATIVE_ENV: "1"}, clear=False):
            native_status(force_reload=True)
            self.assert_delta_close(kernel_delta, native.compute_delta(layer))
        native_status(force_reload=True)

    def test_dense_transition_modes_match_reference_on_directml(self) -> None:
        torch.manual_seed(22)
        src = Layer(
            dim=4,
            num_nodes=7,
            state=torch.randn(2, 7, device=DIRECTML_DEVICE),
            val=torch.randn(2, 7, 4, device=DIRECTML_DEVICE),
        )
        dst = Layer.zeros(dim=5, num_nodes=4, batch_shape=(2,), device=DIRECTML_DEVICE)

        reference = Transition(
            route_fn=LinearRoute(src_dim=4, dst_nodes=4).to(DIRECTML_DEVICE),
            state_proj_fn=_state_proj_fn,
            val_proj_fn=lambda val: val[..., :1].repeat(1, 1, 5),
            implementation="reference",
            src_block_size=3,
            dst_block_size=2,
        )
        kernel = Transition(
            route_fn=LinearRoute(src_dim=4, dst_nodes=4).to(DIRECTML_DEVICE),
            state_proj_fn=_state_proj_fn,
            val_proj_fn=lambda val: val[..., :1].repeat(1, 1, 5),
            implementation="kernel",
            src_block_size=3,
            dst_block_size=2,
        )
        native = Transition(
            route_fn=LinearRoute(src_dim=4, dst_nodes=4).to(DIRECTML_DEVICE),
            state_proj_fn=_state_proj_fn,
            val_proj_fn=lambda val: val[..., :1].repeat(1, 1, 5),
            implementation="native",
            src_block_size=3,
            dst_block_size=2,
        )
        kernel.route_fn.load_state_dict(reference.route_fn.state_dict())
        native.route_fn.load_state_dict(reference.route_fn.state_dict())

        reference_delta = reference.compute_delta(src, dst)
        self.assert_delta_close(reference_delta, kernel.compute_delta(src, dst))
        with mock.patch.dict(os.environ, {DISABLE_NATIVE_ENV: "1"}, clear=False):
            native_status(force_reload=True)
            self.assert_delta_close(reference_delta, native.compute_delta(src, dst))
        native_status(force_reload=True)

    def test_sparse_transition_modes_match_reference_on_directml(self) -> None:
        torch.manual_seed(23)
        src = Layer(
            dim=3,
            num_nodes=8,
            state=torch.randn(2, 8, device=DIRECTML_DEVICE),
            val=torch.randn(2, 8, 3, device=DIRECTML_DEVICE),
        )
        dst = Layer.zeros(dim=4, num_nodes=6, batch_shape=(2,), device=DIRECTML_DEVICE)

        reference = SparseTransition(
            route_fn=LinearRoute(src_dim=3, dst_nodes=6).to(DIRECTML_DEVICE),
            topk=2,
            state_proj_fn=_state_proj_fn,
            val_proj_fn=lambda val: torch.cat((val, val[..., :1]), dim=-1),
            implementation="reference",
            src_block_size=3,
            dst_block_size=2,
        )
        kernel = SparseTransition(
            route_fn=LinearRoute(src_dim=3, dst_nodes=6).to(DIRECTML_DEVICE),
            topk=2,
            state_proj_fn=_state_proj_fn,
            val_proj_fn=lambda val: torch.cat((val, val[..., :1]), dim=-1),
            implementation="kernel",
            src_block_size=3,
            dst_block_size=2,
        )
        streaming = SparseTransition(
            route_fn=LinearRoute(src_dim=3, dst_nodes=6).to(DIRECTML_DEVICE),
            topk=2,
            state_proj_fn=_state_proj_fn,
            val_proj_fn=lambda val: torch.cat((val, val[..., :1]), dim=-1),
            implementation="streaming",
            src_block_size=3,
            dst_block_size=2,
        )
        native = SparseTransition(
            route_fn=LinearRoute(src_dim=3, dst_nodes=6).to(DIRECTML_DEVICE),
            topk=2,
            state_proj_fn=_state_proj_fn,
            val_proj_fn=lambda val: torch.cat((val, val[..., :1]), dim=-1),
            implementation="native",
            src_block_size=3,
            dst_block_size=2,
        )
        kernel.route_fn.load_state_dict(reference.route_fn.state_dict())
        streaming.route_fn.load_state_dict(reference.route_fn.state_dict())
        native.route_fn.load_state_dict(reference.route_fn.state_dict())

        reference_delta = reference.compute_delta(src, dst)
        self.assert_delta_close(reference_delta, kernel.compute_delta(src, dst))
        self.assert_delta_close(reference_delta, streaming.compute_delta(src, dst))
        with mock.patch.dict(os.environ, {DISABLE_NATIVE_ENV: "1"}, clear=False):
            native_status(force_reload=True)
            self.assert_delta_close(reference_delta, native.compute_delta(src, dst))
        native_status(force_reload=True)


if __name__ == "__main__":
    unittest.main()
