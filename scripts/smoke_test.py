from __future__ import annotations

import torch

from jakal_net import (
    DiagonalBilinearPairwise,
    Layer,
    LinearRoute,
    Propagation,
    ScalarAffine,
    SparsePropagation,
    SparseTransition,
    Transition,
)


def main() -> None:
    torch.manual_seed(0)

    same_level = Layer.zeros(dim=8, num_nodes=16, batch_shape=(2,))
    same_level = same_level.with_tensors(
        state=torch.randn_like(same_level.state),
        val=torch.randn_like(same_level.val),
    )

    propagation = Propagation(
        pairwise_fn=DiagonalBilinearPairwise(dim=8),
        state_proj_fn=ScalarAffine(),
    )
    dense_delta = propagation(same_level)
    assert dense_delta.delta_state.shape == same_level.state.shape
    assert dense_delta.delta_val.shape == same_level.val.shape

    sparse_propagation = SparsePropagation(
        pairwise_fn=DiagonalBilinearPairwise(dim=8),
        sparse_type="window",
        window=3,
    )
    sparse_delta = sparse_propagation(same_level)
    assert sparse_delta.delta_state.shape == same_level.state.shape
    assert sparse_delta.delta_val.shape == same_level.val.shape

    src = Layer.zeros(dim=8, num_nodes=16, batch_shape=(2,))
    src = src.with_tensors(
        state=torch.randn_like(src.state),
        val=torch.randn_like(src.val),
    )
    dst = Layer.zeros(dim=8, num_nodes=6, batch_shape=(2,))

    transition = Transition(route_fn=LinearRoute(src_dim=8, dst_nodes=6))
    dst_after_dense = transition(src, dst)
    assert dst_after_dense.state.shape == dst.state.shape
    assert dst_after_dense.val.shape == dst.val.shape

    sparse_transition = SparseTransition(
        route_fn=LinearRoute(src_dim=8, dst_nodes=6),
        topk=2,
    )
    dst_after_sparse = sparse_transition(src, dst)
    assert dst_after_sparse.state.shape == dst.state.shape
    assert dst_after_sparse.val.shape == dst.val.shape

    print("smoke test passed")


if __name__ == "__main__":
    main()
