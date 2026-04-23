from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from jakal_net.core import Layer, LayerDelta
from jakal_net.modules import (
    AdditiveLowRankPairwise,
    AdditiveLowRankRoute,
    BilinearPairwise,
    BilinearPairwiseRoute,
    DiagonalBilinearPairwise,
    DiagonalBilinearRoute,
    FixedProjectionRoute,
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
UNIT_NORM_EPS = 1e-6


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


def _make_anchor_pairwise(
    kind: str,
    *,
    dim: int,
    rank: int,
) -> nn.Module:
    if kind == "scaled_cosine":
        return ScaledCosinePairwise(dim=dim)
    raise ValueError(f"Unsupported pairwise anchor kind: {kind!r}.")


def _freeze_module_parameters(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def make_pairwise(
    kind: str,
    *,
    dim: int,
    rank: int,
    heads: int = 1,
    frozen_heads: int = 0,
    anchor_heads: int = 0,
    anchor_kind: str = "scaled_cosine",
) -> nn.Module:
    if heads <= 0:
        raise ValueError("heads must be positive.")
    if frozen_heads < 0 or frozen_heads > heads:
        raise ValueError("frozen_heads must be between 0 and heads.")
    if anchor_heads < 0 or anchor_heads > heads:
        raise ValueError("anchor_heads must be between 0 and heads.")
    if anchor_heads + frozen_heads > heads:
        raise ValueError("anchor_heads + frozen_heads must be <= heads.")
    if heads == 1:
        if anchor_heads == 1:
            return _make_anchor_pairwise(anchor_kind, dim=dim, rank=rank)
        module = _make_single_pairwise(kind, dim=dim, rank=rank)
        return _freeze_module_parameters(module) if frozen_heads == 1 else module
    head_modules = []
    for head_index in range(heads):
        if head_index < anchor_heads:
            module = _make_anchor_pairwise(anchor_kind, dim=dim, rank=rank)
        else:
            module = _make_single_pairwise(kind, dim=dim, rank=rank)
        if anchor_heads <= head_index < anchor_heads + frozen_heads:
            module = _freeze_module_parameters(module)
        head_modules.append(module)
    return MultiHeadPairwise(head_modules)


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


def _make_anchor_route(
    kind: str,
    *,
    dim: int,
    rank: int,
) -> nn.Module:
    if kind == "fixed_projection":
        return FixedProjectionRoute(src_dim=dim, dst_dim=dim, proj_dim=rank)
    if kind == "query_norm_dot":
        return QueryNormalizedDotRoute(src_dim=dim, dst_dim=dim)
    raise ValueError(f"Unsupported route anchor kind: {kind!r}.")


def make_route(
    kind: str,
    *,
    dim: int,
    rank: int,
    heads: int = 1,
    frozen_heads: int = 0,
    anchor_heads: int = 0,
    anchor_kind: str = "fixed_projection",
) -> nn.Module:
    if heads <= 0:
        raise ValueError("heads must be positive.")
    if frozen_heads < 0 or frozen_heads > heads:
        raise ValueError("frozen_heads must be between 0 and heads.")
    if anchor_heads < 0 or anchor_heads > heads:
        raise ValueError("anchor_heads must be between 0 and heads.")
    if anchor_heads + frozen_heads > heads:
        raise ValueError("anchor_heads + frozen_heads must be <= heads.")
    if heads == 1:
        if anchor_heads == 1:
            return _make_anchor_route(anchor_kind, dim=dim, rank=rank)
        module = _make_single_route(kind, dim=dim, rank=rank)
        return _freeze_module_parameters(module) if frozen_heads == 1 else module
    head_modules = []
    for head_index in range(heads):
        if head_index < anchor_heads:
            module = _make_anchor_route(anchor_kind, dim=dim, rank=rank)
        else:
            module = _make_single_route(kind, dim=dim, rank=rank)
        if anchor_heads <= head_index < anchor_heads + frozen_heads:
            module = _freeze_module_parameters(module)
        head_modules.append(module)
    return MultiHeadRoute(head_modules)


def unit_normalize_values(val: Tensor, *, eps: float = UNIT_NORM_EPS) -> Tensor:
    return F.normalize(torch.nan_to_num(val), p=2.0, dim=-1, eps=eps)


def layer_with_val_norm(
    layer: Layer,
    norm: nn.LayerNorm,
    *,
    unit_norm_values: bool = False,
) -> Layer:
    val = norm(layer.val)
    if unit_norm_values:
        val = unit_normalize_values(val)
    return layer.with_tensors(val=val)


def signed_softmax_state(state: Tensor) -> Tensor:
    clean_state = torch.nan_to_num(state)
    clean_state = F.layer_norm(clean_state, (clean_state.shape[-1],))
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
    unit_norm_values: bool = False,
) -> Layer:
    updated = layer.apply_delta(delta, merge_mode="add" if residual else "replace")
    state = signed_softmax_state(updated.state)
    val = updated.val if val_norm is None else val_norm(updated.val)
    if unit_norm_values:
        val = unit_normalize_values(val)
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
