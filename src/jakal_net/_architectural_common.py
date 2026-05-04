from __future__ import annotations

import os

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


def _maybe_compile(fn):
    if os.environ.get("JAKAL_COMPILE_SMALL_OPS", "0") != "1":
        return fn
    compiler = getattr(torch, "compile", None)
    if compiler is None:
        return fn
    try:
        return compiler(fn, fullgraph=False, dynamic=True)
    except Exception:
        return fn


def _normalize_value_directions_impl(val: Tensor, *, eps: float = UNIT_NORM_EPS) -> Tensor:
    return F.normalize(torch.nan_to_num(val), p=2.0, dim=-1, eps=eps)


def _signed_softmax_state_impl(state: Tensor) -> Tensor:
    clean_state = torch.nan_to_num(state)
    clean_state = F.layer_norm(clean_state, (clean_state.shape[-1],))
    magnitude = torch.softmax(clean_state.abs(), dim=-1)
    state_mass = float(state.size(-1)) * STATE_MASS_PER_NODE
    return torch.sign(clean_state) * magnitude * state_mass


def _softsign_state_impl(state: Tensor) -> Tensor:
    return F.softsign(torch.nan_to_num(state))


_COMPILED_UNIT_NORMALIZE_VALUES = _maybe_compile(_normalize_value_directions_impl)
_COMPILED_SIGNED_SOFTMAX_STATE = _maybe_compile(_signed_softmax_state_impl)
_COMPILED_SOFTSIGN_STATE = _maybe_compile(_softsign_state_impl)


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
    if kind == "diagonal_bilinear":
        return _freeze_module_parameters(DiagonalBilinearPairwise(dim=dim))
    if kind == "constant_one":
        module = LowRankBilinearPairwise(dim=dim, rank=rank, bias=True)
        module._constant_one_anchor = True
        with torch.no_grad():
            module.source_proj.weight.zero_()
            module.target_proj.weight.zero_()
            module.weight.fill_(1.0)
            assert module.bias is not None
            module.bias.fill_(1.0)
        return _freeze_module_parameters(module)
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
    if kind == "diagonal_bilinear":
        return _freeze_module_parameters(DiagonalBilinearRoute(src_dim=dim, dst_dim=dim))
    if kind == "fixed_projection":
        return FixedProjectionRoute(src_dim=dim, dst_dim=dim, proj_dim=rank)
    if kind == "query_norm_dot":
        return QueryNormalizedDotRoute(src_dim=dim, dst_dim=dim)
    if kind == "constant_one":
        module = LowRankBilinearRoute(src_dim=dim, dst_dim=dim, rank=rank, bias=True)
        module._constant_one_anchor = True
        with torch.no_grad():
            module.source_proj.weight.zero_()
            module.target_proj.weight.zero_()
            module.weight.fill_(1.0)
            assert module.bias is not None
            module.bias.fill_(1.0)
        return _freeze_module_parameters(module)
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


def normalize_value_directions(val: Tensor, *, eps: float = UNIT_NORM_EPS) -> Tensor:
    return _COMPILED_UNIT_NORMALIZE_VALUES(val, eps=eps)


def layer_with_val_norm(
    layer: Layer,
    norm: nn.LayerNorm,
    *,
    direction_only_values: bool = False,
) -> Layer:
    val = norm(layer.val)
    return layer.with_tensors(val=val)


def signed_softmax_state(state: Tensor) -> Tensor:
    return _COMPILED_SIGNED_SOFTMAX_STATE(state)


def softsign_state(state: Tensor) -> Tensor:
    return _COMPILED_SOFTSIGN_STATE(state)


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
    direction_only_values: bool = False,
) -> Layer:
    updated = layer.apply_delta(delta, merge_mode="add" if residual else "replace")
    state = softsign_state(updated.state) if direction_only_values else signed_softmax_state(updated.state)
    if val_norm is None:
        val = updated.val
    elif residual:
        touched = delta.delta_val.detach().abs().amax(dim=-1) > 0
        normalized_val = updated.val
        if val_norm is not None:
            normalized_val = val_norm(normalized_val)
        val = torch.where(touched.unsqueeze(-1), normalized_val, updated.val)
    else:
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
    linear_std = float(getattr(module, "_init_linear_std", PARAM_INIT_STD))
    low_rank_scale = float(getattr(module, "_init_low_rank_scale", LOW_RANK_SCALE_INIT))
    if getattr(module, "_constant_one_anchor", False):
        return
    if isinstance(module, (LowRankBilinearPairwise, LowRankBilinearRoute)):
        module.weight.data.fill_(low_rank_scale)
        init_linear(module.source_proj, std=linear_std)
        init_linear(module.target_proj, std=linear_std)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
        return
    if isinstance(module, (DiagonalBilinearPairwise, DiagonalBilinearRoute)):
        module.weight.data.fill_(low_rank_scale)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
        return
    if isinstance(module, BilinearPairwise):
        nn.init.normal_(module.weight, mean=0.0, std=linear_std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return
    if isinstance(module, BilinearPairwiseRoute):
        init_linear(module.source_proj, std=linear_std)
        init_linear(module.target_proj, std=linear_std)
        nn.init.normal_(module.weight, mean=0.0, std=linear_std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return
    if isinstance(module, AdditiveLowRankPairwise):
        init_linear(module.target_proj, std=linear_std)
        init_linear(module.source_proj, std=linear_std)
        init_linear(module.target_out, std=linear_std)
        init_linear(module.source_out, std=linear_std)
        nn.init.normal_(module.interaction_weight, mean=0.0, std=linear_std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return
    if isinstance(module, AdditiveLowRankRoute):
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                init_linear(layer, std=linear_std)
