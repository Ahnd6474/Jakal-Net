from __future__ import annotations

import torch
from torch import Tensor, nn

from jakal_net.core import Layer, LayerDelta
from jakal_net.modules import (
    AdditiveLowRankPairwise,
    AdditiveLowRankRoute,
    BilinearPairwise,
    BilinearPairwiseRoute,
    DiagonalBilinearPairwise,
    DiagonalBilinearRoute,
    LowRankBilinearPairwise,
    LowRankBilinearRoute,
    MultiHeadPairwise,
    MultiHeadRoute,
    QueryNormalizedDotRoute,
    ScaledCosinePairwise,
)

STATE_MASS_PER_NODE = 1.0
PARAM_INIT_STD = 0.02
LOW_RANK_SCALE_INIT = 0.1


def _make_single_pairwise(
    kind: str,
    *,
    dim: int,
    rank: int,
) -> nn.Module:
    if kind == "low_rank_bilinear":
        return LowRankBilinearPairwise(dim=dim, rank=rank)
    if kind == "diagonal_bilinear":
        return DiagonalBilinearPairwise(dim=dim)
    if kind == "bilinear":
        return BilinearPairwise(dim=dim)
    if kind == "additive_low_rank":
        return AdditiveLowRankPairwise(dim=dim, rank=rank)
    if kind == "scaled_cosine":
        return ScaledCosinePairwise(dim=dim)
    raise ValueError(f"Unsupported pairwise kind: {kind!r}.")


def make_pairwise(
    kind: str,
    *,
    dim: int,
    rank: int,
    heads: int = 1,
) -> nn.Module:
    if heads <= 0:
        raise ValueError("heads must be positive.")
    if heads == 1:
        return _make_single_pairwise(kind, dim=dim, rank=rank)
    return MultiHeadPairwise(
        [_make_single_pairwise(kind, dim=dim, rank=rank) for _ in range(heads)]
    )


def _make_single_route(
    kind: str,
    *,
    dim: int,
    rank: int,
) -> nn.Module:
    if kind == "low_rank_bilinear":
        return LowRankBilinearRoute(src_dim=dim, dst_dim=dim, rank=rank)
    if kind == "diagonal_bilinear":
        return DiagonalBilinearRoute(src_dim=dim, dst_dim=dim)
    if kind == "bilinear":
        return BilinearPairwiseRoute(src_dim=dim, dst_dim=dim, route_dim=dim)
    if kind == "additive_low_rank":
        return AdditiveLowRankRoute(src_dim=dim, dst_dim=dim, route_dim=rank)
    if kind == "query_norm_dot":
        return QueryNormalizedDotRoute(src_dim=dim, dst_dim=dim)
    raise ValueError(f"Unsupported route kind: {kind!r}.")


def make_route(
    kind: str,
    *,
    dim: int,
    rank: int,
    heads: int = 1,
) -> nn.Module:
    if heads <= 0:
        raise ValueError("heads must be positive.")
    if heads == 1:
        return _make_single_route(kind, dim=dim, rank=rank)
    return MultiHeadRoute(
        [_make_single_route(kind, dim=dim, rank=rank) for _ in range(heads)]
    )


def layer_with_val_norm(layer: Layer, norm: nn.LayerNorm) -> Layer:
    return layer.with_tensors(val=norm(layer.val))


def signed_softmax_state(state: Tensor) -> Tensor:
    clean_state = torch.nan_to_num(state)
    magnitude = torch.softmax(clean_state.abs(), dim=-1)
    state_mass = float(state.size(-1)) * STATE_MASS_PER_NODE
    return torch.sign(clean_state) * magnitude * state_mass


def signed_abs_softmax_edges(scores: Tensor) -> Tensor:
    clean_scores = torch.nan_to_num(scores)
    return torch.sign(clean_scores) * torch.softmax(clean_scores.abs(), dim=-1)


def identity_state_activation(state: Tensor) -> Tensor:
    return state


def apply_delta(
    layer: Layer,
    delta: LayerDelta,
    *,
    residual: bool = True,
    val_norm: nn.LayerNorm | None = None,
) -> Layer:
    updated = layer.apply_delta(delta, merge_mode="add" if residual else "replace")
    state = signed_softmax_state(updated.state)
    val = updated.val if val_norm is None else val_norm(updated.val)
    return updated.with_tensors(state=state, val=val)


def clone_layer(layer: Layer) -> Layer:
    return Layer(
        dim=layer.dim,
        num_nodes=layer.num_nodes,
        state=layer.state.clone(),
        val=layer.val.clone(),
    )


def init_linear(linear: nn.Linear, *, std: float = PARAM_INIT_STD) -> None:
    nn.init.normal_(linear.weight, mean=0.0, std=std)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def init_pairwise_or_route_scales(module: nn.Module) -> None:
    if isinstance(module, (LowRankBilinearPairwise, LowRankBilinearRoute)):
        module.weight.data.fill_(LOW_RANK_SCALE_INIT)
        init_linear(module.source_proj)
        init_linear(module.target_proj)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
        return
    if isinstance(module, (DiagonalBilinearPairwise, DiagonalBilinearRoute)):
        module.weight.data.fill_(LOW_RANK_SCALE_INIT)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
        return
    if isinstance(module, BilinearPairwise):
        nn.init.normal_(module.weight, mean=0.0, std=PARAM_INIT_STD)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return
    if isinstance(module, BilinearPairwiseRoute):
        init_linear(module.source_proj)
        init_linear(module.target_proj)
        nn.init.normal_(module.weight, mean=0.0, std=PARAM_INIT_STD)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return
    if isinstance(module, AdditiveLowRankPairwise):
        init_linear(module.target_proj)
        init_linear(module.source_proj)
        init_linear(module.target_out)
        init_linear(module.source_out)
        nn.init.normal_(module.interaction_weight, mean=0.0, std=PARAM_INIT_STD)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return
    if isinstance(module, AdditiveLowRankRoute):
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                init_linear(layer)
