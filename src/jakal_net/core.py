from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Literal

import torch
from torch import Tensor

MergeMode = Literal["add", "replace"]
SparsePropagationType = Literal["window", "topk"]
ImplementationMode = Literal["reference", "streaming", "kernel"]


@dataclass(slots=True)
class LayerDelta:
    delta_state: Tensor
    delta_val: Tensor

    def __post_init__(self) -> None:
        if self.delta_state.ndim < 1:
            raise ValueError("delta_state must end with the node dimension.")
        if self.delta_val.ndim < 2:
            raise ValueError("delta_val must end with [num_nodes, dim].")
        if self.delta_val.shape[:-1] != self.delta_state.shape:
            raise ValueError(
                "delta_val must match delta_state on every dimension except value dim."
            )

    @classmethod
    def zeros_like(cls, layer: "Layer") -> "LayerDelta":
        return cls(
            delta_state=torch.zeros_like(layer.state),
            delta_val=torch.zeros_like(layer.val),
        )


@dataclass(slots=True)
class Layer:
    dim: int
    num_nodes: int
    state: Tensor
    val: Tensor

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive.")
        if self.num_nodes <= 0:
            raise ValueError("num_nodes must be positive.")
        if self.state.ndim < 1:
            raise ValueError("state must end with [num_nodes].")
        if self.val.ndim < 2:
            raise ValueError("val must end with [num_nodes, dim].")
        if self.state.shape[-1] != self.num_nodes:
            raise ValueError(
                f"state.shape[-1] must be {self.num_nodes}, got {self.state.shape[-1]}."
            )
        if self.val.shape[-2] != self.num_nodes:
            raise ValueError(
                f"val.shape[-2] must be {self.num_nodes}, got {self.val.shape[-2]}."
            )
        if self.val.shape[-1] != self.dim:
            raise ValueError(f"val.shape[-1] must be {self.dim}, got {self.val.shape[-1]}.")
        if self.val.shape[:-1] != self.state.shape:
            raise ValueError(
                "val must match state on every dimension except the trailing value dim."
            )

    @classmethod
    def zeros(
        cls,
        dim: int,
        num_nodes: int,
        batch_shape: tuple[int, ...] = (),
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "Layer":
        state = torch.zeros(*batch_shape, num_nodes, device=device, dtype=dtype)
        val = torch.zeros(*batch_shape, num_nodes, dim, device=device, dtype=dtype)
        return cls(dim=dim, num_nodes=num_nodes, state=state, val=val)

    @property
    def batch_shape(self) -> torch.Size:
        return self.state.shape[:-1]

    def clone(self) -> "Layer":
        return Layer(
            dim=self.dim,
            num_nodes=self.num_nodes,
            state=self.state.clone(),
            val=self.val.clone(),
        )

    def with_tensors(self, *, state: Tensor | None = None, val: Tensor | None = None) -> "Layer":
        return Layer(
            dim=self.dim,
            num_nodes=self.num_nodes,
            state=self.state if state is None else state,
            val=self.val if val is None else val,
        )

    def apply_delta(self, delta: LayerDelta, merge_mode: MergeMode = "add") -> "Layer":
        validate_merge_mode(merge_mode)
        if self.state.shape != delta.delta_state.shape:
            raise ValueError("delta_state shape does not match the target layer.")
        if self.val.shape != delta.delta_val.shape:
            raise ValueError("delta_val shape does not match the target layer.")

        if merge_mode == "add":
            next_state = self.state + delta.delta_state
            next_val = self.val + delta.delta_val
        else:
            next_state = delta.delta_state
            next_val = delta.delta_val
        return self.with_tensors(state=next_state, val=next_val)


def validate_merge_mode(merge_mode: MergeMode) -> None:
    if merge_mode not in {"add", "replace"}:
        raise ValueError(f"Unsupported merge_mode: {merge_mode!r}.")


def validate_implementation(implementation: ImplementationMode) -> None:
    if implementation not in {"reference", "streaming", "kernel"}:
        raise ValueError(f"Unsupported implementation: {implementation!r}.")


def apply_optional_layer_fn(
    layer: Layer, layer_fn: Callable[[Layer], Layer] | None
) -> Layer:
    if layer_fn is None:
        return layer
    next_layer = layer_fn(layer)
    if not isinstance(next_layer, Layer):
        raise TypeError("layer_fn must return a Layer instance.")
    return next_layer


def validate_projected_state(projected_state: Tensor, reference: Layer) -> None:
    if projected_state.shape != reference.state.shape:
        raise ValueError(
            "state_proj_fn must return a tensor shaped like layer.state, "
            f"expected {tuple(reference.state.shape)}, got {tuple(projected_state.shape)}."
        )


def validate_projected_val(projected_val: Tensor, reference: Layer) -> None:
    if projected_val.shape != reference.val.shape:
        raise ValueError(
            "val_proj_fn must return a tensor shaped like layer.val, "
            f"expected {tuple(reference.val.shape)}, got {tuple(projected_val.shape)}."
        )


def validate_pairwise_scores(scores: Tensor, layer: Layer) -> None:
    validate_pairwise_block_scores(
        scores=scores,
        batch_shape=layer.batch_shape,
        target_nodes=layer.num_nodes,
        source_nodes=layer.num_nodes,
    )


def validate_pairwise_block_scores(
    scores: Tensor,
    *,
    batch_shape: torch.Size | tuple[int, ...],
    target_nodes: int,
    source_nodes: int,
) -> None:
    expected_shape = (*batch_shape, target_nodes, source_nodes)
    if tuple(scores.shape) != expected_shape:
        raise ValueError(
            "pairwise_fn must return scores shaped like [..., num_nodes, num_nodes], "
            f"expected {expected_shape}, got {tuple(scores.shape)}."
        )


def validate_route_logits(logits: Tensor, src_layer: Layer, dst_layer: Layer) -> None:
    validate_route_block_logits(
        logits=logits,
        batch_shape=src_layer.batch_shape,
        source_nodes=src_layer.num_nodes,
        dst_nodes=dst_layer.num_nodes,
    )


def validate_route_block_logits(
    logits: Tensor,
    *,
    batch_shape: torch.Size | tuple[int, ...],
    source_nodes: int,
    dst_nodes: int,
) -> None:
    expected_shape = (*batch_shape, source_nodes, dst_nodes)
    if tuple(logits.shape) != expected_shape:
        raise ValueError(
            "route_fn must return logits shaped like [..., src_nodes, dst_nodes], "
            f"expected {expected_shape}, got {tuple(logits.shape)}."
        )


def resolve_block_size(block_size: int | None, total_size: int, *, name: str) -> int:
    if total_size <= 0:
        raise ValueError("total_size must be positive.")
    if block_size is None:
        return total_size
    if block_size <= 0:
        raise ValueError(f"{name} must be positive when provided.")
    return min(block_size, total_size)


def iter_blocks(total_size: int, block_size: int | None, *, name: str) -> Iterator[tuple[int, int]]:
    resolved = resolve_block_size(block_size, total_size, name=name)
    for start in range(0, total_size, resolved):
        yield start, min(start + resolved, total_size)


def resolve_accumulator_dtype(
    tensor_dtype: torch.dtype, accumulator_dtype: torch.dtype | None
) -> torch.dtype:
    if accumulator_dtype is not None:
        return accumulator_dtype
    if tensor_dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return tensor_dtype
