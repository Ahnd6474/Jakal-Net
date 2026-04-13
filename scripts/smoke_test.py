from __future__ import annotations

import argparse

import torch

from jakal_net import (
    DiagonalBilinearPairwise,
    describe_device,
    Layer,
    LinearRoute,
    native_status,
    Propagation,
    resolve_device,
    ScalarAffine,
    SparsePropagation,
    SparseTransition,
    Transition,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = resolve_device(args.device)
    print(f"using device: {describe_device(args.device)}")
    print(f"native backend: {native_status().backend_name or 'unavailable'}")

    same_level = Layer.zeros(dim=8, num_nodes=16, batch_shape=(2,), device=device)
    same_level = same_level.with_tensors(
        state=torch.randn_like(same_level.state),
        val=torch.randn_like(same_level.val),
    )

    propagation = Propagation(
        pairwise_fn=DiagonalBilinearPairwise(dim=8),
        state_proj_fn=ScalarAffine(),
    ).to(device)
    dense_delta = propagation(same_level)
    assert dense_delta.delta_state.shape == same_level.state.shape
    assert dense_delta.delta_val.shape == same_level.val.shape

    sparse_propagation = SparsePropagation(
        pairwise_fn=DiagonalBilinearPairwise(dim=8),
        sparse_type="window",
        window=3,
    ).to(device)
    sparse_delta = sparse_propagation(same_level)
    assert sparse_delta.delta_state.shape == same_level.state.shape
    assert sparse_delta.delta_val.shape == same_level.val.shape

    kernel_propagation = Propagation(
        pairwise_fn=DiagonalBilinearPairwise(dim=8),
        state_proj_fn=ScalarAffine(),
        implementation="kernel",
    ).to(device)
    kernel_propagation.pairwise_fn.load_state_dict(propagation.pairwise_fn.state_dict())
    kernel_propagation.state_proj_fn.load_state_dict(propagation.state_proj_fn.state_dict())
    kernel_delta = kernel_propagation(same_level)
    assert kernel_delta.delta_state.shape == same_level.state.shape
    assert kernel_delta.delta_val.shape == same_level.val.shape

    native_propagation = Propagation(
        pairwise_fn=DiagonalBilinearPairwise(dim=8),
        state_proj_fn=ScalarAffine(),
        implementation="native",
    ).to(device)
    native_propagation.pairwise_fn.load_state_dict(propagation.pairwise_fn.state_dict())
    native_propagation.state_proj_fn.load_state_dict(propagation.state_proj_fn.state_dict())
    native_delta = native_propagation(same_level)
    assert native_delta.delta_state.shape == same_level.state.shape
    assert native_delta.delta_val.shape == same_level.val.shape

    src = Layer.zeros(dim=8, num_nodes=16, batch_shape=(2,), device=device)
    src = src.with_tensors(
        state=torch.randn_like(src.state),
        val=torch.randn_like(src.val),
    )
    dst = Layer.zeros(dim=8, num_nodes=6, batch_shape=(2,), device=device)

    transition = Transition(route_fn=LinearRoute(src_dim=8, dst_nodes=6)).to(device)
    dst_after_dense = transition(src, dst)
    assert dst_after_dense.state.shape == dst.state.shape
    assert dst_after_dense.val.shape == dst.val.shape

    sparse_transition = SparseTransition(
        route_fn=LinearRoute(src_dim=8, dst_nodes=6),
        topk=2,
    ).to(device)
    dst_after_sparse = sparse_transition(src, dst)
    assert dst_after_sparse.state.shape == dst.state.shape
    assert dst_after_sparse.val.shape == dst.val.shape

    kernel_transition = SparseTransition(
        route_fn=LinearRoute(src_dim=8, dst_nodes=6),
        topk=2,
        implementation="kernel",
    ).to(device)
    kernel_transition.route_fn.load_state_dict(sparse_transition.route_fn.state_dict())
    dst_after_kernel = kernel_transition(src, dst)
    assert dst_after_kernel.state.shape == dst.state.shape
    assert dst_after_kernel.val.shape == dst.val.shape

    native_transition = SparseTransition(
        route_fn=LinearRoute(src_dim=8, dst_nodes=6),
        topk=2,
        implementation="native",
    ).to(device)
    native_transition.route_fn.load_state_dict(sparse_transition.route_fn.state_dict())
    dst_after_native = native_transition(src, dst)
    assert dst_after_native.state.shape == dst.state.shape
    assert dst_after_native.val.shape == dst.val.shape

    print("smoke test passed")


if __name__ == "__main__":
    main()
