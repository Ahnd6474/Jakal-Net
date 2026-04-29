from __future__ import annotations

from functools import lru_cache
import os

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None


@lru_cache(maxsize=1)
def triton_signed_smoothmax_available() -> bool:
    return triton is not None and tl is not None and torch.cuda.is_available()


def _parse_block_config(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != len(default):
        return default
    try:
        values = tuple(max(1, int(part)) for part in parts)
    except ValueError:
        return default
    return values


def _diag_backward_block_config() -> tuple[int, int, int, int]:
    return _parse_block_config(
        "JAKAL_NET_TRITON_SIGNED_SMOOTHMAX_DIAG_PASS2_BLOCKS",
        (32, 32, 32, 32),
    )


def _lowrank_backward_block_config() -> tuple[int, int, int, int]:
    return _parse_block_config(
        "JAKAL_NET_TRITON_SIGNED_SMOOTHMAX_LOWRANK_PASS2_BLOCKS",
        (32, 32, 32, 32),
    )


if triton is not None and tl is not None:
    @triton.jit
    def _signed_abs_softmax_edge_dot_tile_kernel(
        scores_ptr,
        grad_edges_ptr,
        row_max_ptr,
        row_denom_ptr,
        out_ptr,
        stride_scores_b,
        stride_scores_m,
        stride_scores_n,
        stride_grad_b,
        stride_grad_m,
        stride_grad_n,
        stride_row_b,
        stride_row_m,
        stride_out_b,
        stride_out_m,
        source_start,
        num_nodes,
        tile_nodes,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < num_nodes

        row_max = tl.load(
            row_max_ptr + pid_b * stride_row_b + offs_m * stride_row_m,
            mask=mask_m,
            other=-float("inf"),
        )
        row_denom = tl.load(
            row_denom_ptr + pid_b * stride_row_b + offs_m * stride_row_m,
            mask=mask_m,
            other=1.0,
        )
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for n_start in range(0, tile_nodes, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            source_idx = source_start + offs_n
            mask_n = offs_n < tile_nodes
            causal = source_idx[None, :] <= offs_m[:, None]
            full_mask = mask_m[:, None] & mask_n[None, :] & causal

            score_ptrs = (
                scores_ptr
                + pid_b * stride_scores_b
                + offs_m[:, None] * stride_scores_m
                + offs_n[None, :] * stride_scores_n
            )
            grad_ptrs = (
                grad_edges_ptr
                + pid_b * stride_grad_b
                + offs_m[:, None] * stride_grad_m
                + offs_n[None, :] * stride_grad_n
            )
            scores = tl.load(score_ptrs, mask=full_mask, other=0.0)
            grad_edges = tl.load(grad_ptrs, mask=full_mask, other=0.0)
            probs = tl.where(
                full_mask,
                tl.exp(tl.abs(scores) - row_max[:, None]) / tl.maximum(row_denom[:, None], 1.0e-20),
                0.0,
            )
            signs = tl.where(scores > 0, 1.0, tl.where(scores < 0, -1.0, 0.0))
            edges = signs * probs
            acc += tl.sum(grad_edges * edges, axis=1)

        out_ptrs = out_ptr + pid_b * stride_out_b + offs_m * stride_out_m
        tl.store(out_ptrs, acc, mask=mask_m)

    @triton.jit
    def _multihead_signed_smoothmax_scores_and_head_grads_tile_kernel(
        target_ptr,
        source_ptr,
        bias_ptr,
        score_out_ptr,
        grad_out_ptr,
        stride_target_b,
        stride_target_h,
        stride_target_n,
        stride_target_r,
        stride_source_b,
        stride_source_h,
        stride_source_n,
        stride_source_r,
        stride_score_out_b,
        stride_score_out_m,
        stride_score_out_n,
        stride_grad_out_b,
        stride_grad_out_h,
        stride_grad_out_m,
        stride_grad_out_n,
        source_start,
        num_nodes,
        tile_nodes,
        rank_dim,
        has_bias: tl.constexpr,
        heads: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        source_idx = source_start + offs_n
        mask_m = offs_m < num_nodes
        mask_n = offs_n < tile_nodes
        causal = source_idx[None, :] <= offs_m[:, None]
        full_mask = mask_m[:, None] & mask_n[None, :] & causal

        score_0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        score_1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        score_2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        score_3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, rank_dim, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < rank_dim

            if heads > 0:
                a_ptrs = target_ptr + pid_b * stride_target_b + 0 * stride_target_h + offs_m[:, None] * stride_target_n + offs_k[None, :] * stride_target_r
                b_ptrs = source_ptr + pid_b * stride_source_b + 0 * stride_source_h + offs_n[None, :] * stride_source_n + offs_k[:, None] * stride_source_r
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                score_0 += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
            if heads > 1:
                a_ptrs = target_ptr + pid_b * stride_target_b + 1 * stride_target_h + offs_m[:, None] * stride_target_n + offs_k[None, :] * stride_target_r
                b_ptrs = source_ptr + pid_b * stride_source_b + 1 * stride_source_h + offs_n[None, :] * stride_source_n + offs_k[:, None] * stride_source_r
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                score_1 += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
            if heads > 2:
                a_ptrs = target_ptr + pid_b * stride_target_b + 2 * stride_target_h + offs_m[:, None] * stride_target_n + offs_k[None, :] * stride_target_r
                b_ptrs = source_ptr + pid_b * stride_source_b + 2 * stride_source_h + offs_n[None, :] * stride_source_n + offs_k[:, None] * stride_source_r
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                score_2 += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
            if heads > 3:
                a_ptrs = target_ptr + pid_b * stride_target_b + 3 * stride_target_h + offs_m[:, None] * stride_target_n + offs_k[None, :] * stride_target_r
                b_ptrs = source_ptr + pid_b * stride_source_b + 3 * stride_source_h + offs_n[None, :] * stride_source_n + offs_k[:, None] * stride_source_r
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                score_3 += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)

        if has_bias:
            if heads > 0:
                score_0 += tl.load(bias_ptr + 0)
            if heads > 1:
                score_1 += tl.load(bias_ptr + 1)
            if heads > 2:
                score_2 += tl.load(bias_ptr + 2)
            if heads > 3:
                score_3 += tl.load(bias_ptr + 3)

        neg_large = -1.0e30
        score_0_masked = tl.where(full_mask, score_0, 0.0)
        score_1_masked = tl.where(full_mask, score_1, 0.0)
        score_2_masked = tl.where(full_mask, score_2, 0.0)
        score_3_masked = tl.where(full_mask, score_3, 0.0)

        max_abs = tl.where(full_mask, tl.abs(score_0), neg_large)
        if heads > 1:
            max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_1), neg_large))
        if heads > 2:
            max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_2), neg_large))
        if heads > 3:
            max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_3), neg_large))

        denom = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        numer = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        if heads > 0:
            p0 = tl.where(full_mask, tl.exp(tl.abs(score_0) - max_abs), 0.0)
            denom += p0
            numer += score_0_masked * p0
        if heads > 1:
            p1 = tl.where(full_mask, tl.exp(tl.abs(score_1) - max_abs), 0.0)
            denom += p1
            numer += score_1_masked * p1
        if heads > 2:
            p2 = tl.where(full_mask, tl.exp(tl.abs(score_2) - max_abs), 0.0)
            denom += p2
            numer += score_2_masked * p2
        if heads > 3:
            p3 = tl.where(full_mask, tl.exp(tl.abs(score_3) - max_abs), 0.0)
            denom += p3
            numer += score_3_masked * p3

        safe_denom = tl.maximum(denom, 1.0e-20)
        combined = numer / safe_denom
        score_out_ptrs = score_out_ptr + pid_b * stride_score_out_b + offs_m[:, None] * stride_score_out_m + offs_n[None, :] * stride_score_out_n
        tl.store(score_out_ptrs, tl.where(full_mask, combined, 0.0), mask=mask_m[:, None] & mask_n[None, :])

        if heads > 0:
            p0 = tl.where(full_mask, tl.exp(tl.abs(score_0) - max_abs) / safe_denom, 0.0)
            g0 = p0 * (1.0 + tl.where(score_0 > 0, 1.0, tl.where(score_0 < 0, -1.0, 0.0)) * (score_0 - combined))
            out_ptrs = grad_out_ptr + pid_b * stride_grad_out_b + 0 * stride_grad_out_h + offs_m[:, None] * stride_grad_out_m + offs_n[None, :] * stride_grad_out_n
            tl.store(out_ptrs, tl.where(full_mask, g0, 0.0), mask=mask_m[:, None] & mask_n[None, :])
        if heads > 1:
            p1 = tl.where(full_mask, tl.exp(tl.abs(score_1) - max_abs) / safe_denom, 0.0)
            g1 = p1 * (1.0 + tl.where(score_1 > 0, 1.0, tl.where(score_1 < 0, -1.0, 0.0)) * (score_1 - combined))
            out_ptrs = grad_out_ptr + pid_b * stride_grad_out_b + 1 * stride_grad_out_h + offs_m[:, None] * stride_grad_out_m + offs_n[None, :] * stride_grad_out_n
            tl.store(out_ptrs, tl.where(full_mask, g1, 0.0), mask=mask_m[:, None] & mask_n[None, :])
        if heads > 2:
            p2 = tl.where(full_mask, tl.exp(tl.abs(score_2) - max_abs) / safe_denom, 0.0)
            g2 = p2 * (1.0 + tl.where(score_2 > 0, 1.0, tl.where(score_2 < 0, -1.0, 0.0)) * (score_2 - combined))
            out_ptrs = grad_out_ptr + pid_b * stride_grad_out_b + 2 * stride_grad_out_h + offs_m[:, None] * stride_grad_out_m + offs_n[None, :] * stride_grad_out_n
            tl.store(out_ptrs, tl.where(full_mask, g2, 0.0), mask=mask_m[:, None] & mask_n[None, :])
        if heads > 3:
            p3 = tl.where(full_mask, tl.exp(tl.abs(score_3) - max_abs) / safe_denom, 0.0)
            g3 = p3 * (1.0 + tl.where(score_3 > 0, 1.0, tl.where(score_3 < 0, -1.0, 0.0)) * (score_3 - combined))
            out_ptrs = grad_out_ptr + pid_b * stride_grad_out_b + 3 * stride_grad_out_h + offs_m[:, None] * stride_grad_out_m + offs_n[None, :] * stride_grad_out_n
            tl.store(out_ptrs, tl.where(full_mask, g3, 0.0), mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def _multihead_signed_smoothmax_scores_tile_kernel(
        target_ptr,
        source_ptr,
        bias_ptr,
        out_ptr,
        stride_target_b,
        stride_target_h,
        stride_target_n,
        stride_target_r,
        stride_source_b,
        stride_source_h,
        stride_source_n,
        stride_source_r,
        stride_out_b,
        stride_out_m,
        stride_out_n,
        num_nodes,
        tile_nodes,
        rank_dim,
        has_bias: tl.constexpr,
        heads: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < num_nodes
        mask_n = offs_n < tile_nodes

        score_0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        score_1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        score_2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        score_3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, rank_dim, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < rank_dim

            if heads > 0:
                a_ptrs = (
                    target_ptr
                    + pid_b * stride_target_b
                    + 0 * stride_target_h
                    + offs_m[:, None] * stride_target_n
                    + offs_k[None, :] * stride_target_r
                )
                b_ptrs = (
                    source_ptr
                    + pid_b * stride_source_b
                    + 0 * stride_source_h
                    + offs_n[None, :] * stride_source_n
                    + offs_k[:, None] * stride_source_r
                )
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                score_0 += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
            if heads > 1:
                a_ptrs = (
                    target_ptr
                    + pid_b * stride_target_b
                    + 1 * stride_target_h
                    + offs_m[:, None] * stride_target_n
                    + offs_k[None, :] * stride_target_r
                )
                b_ptrs = (
                    source_ptr
                    + pid_b * stride_source_b
                    + 1 * stride_source_h
                    + offs_n[None, :] * stride_source_n
                    + offs_k[:, None] * stride_source_r
                )
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                score_1 += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
            if heads > 2:
                a_ptrs = (
                    target_ptr
                    + pid_b * stride_target_b
                    + 2 * stride_target_h
                    + offs_m[:, None] * stride_target_n
                    + offs_k[None, :] * stride_target_r
                )
                b_ptrs = (
                    source_ptr
                    + pid_b * stride_source_b
                    + 2 * stride_source_h
                    + offs_n[None, :] * stride_source_n
                    + offs_k[:, None] * stride_source_r
                )
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                score_2 += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
            if heads > 3:
                a_ptrs = (
                    target_ptr
                    + pid_b * stride_target_b
                    + 3 * stride_target_h
                    + offs_m[:, None] * stride_target_n
                    + offs_k[None, :] * stride_target_r
                )
                b_ptrs = (
                    source_ptr
                    + pid_b * stride_source_b
                    + 3 * stride_source_h
                    + offs_n[None, :] * stride_source_n
                    + offs_k[:, None] * stride_source_r
                )
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                score_3 += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)

        if has_bias:
            if heads > 0:
                score_0 += tl.load(bias_ptr + 0)
            if heads > 1:
                score_1 += tl.load(bias_ptr + 1)
            if heads > 2:
                score_2 += tl.load(bias_ptr + 2)
            if heads > 3:
                score_3 += tl.load(bias_ptr + 3)

        max_abs = tl.abs(score_0) if heads > 0 else tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if heads > 1:
            max_abs = tl.maximum(max_abs, tl.abs(score_1))
        if heads > 2:
            max_abs = tl.maximum(max_abs, tl.abs(score_2))
        if heads > 3:
            max_abs = tl.maximum(max_abs, tl.abs(score_3))

        denom = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        numer = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if heads > 0:
            p0 = tl.exp(tl.abs(score_0) - max_abs)
            denom += p0
            numer += score_0 * p0
        if heads > 1:
            p1 = tl.exp(tl.abs(score_1) - max_abs)
            denom += p1
            numer += score_1 * p1
        if heads > 2:
            p2 = tl.exp(tl.abs(score_2) - max_abs)
            denom += p2
            numer += score_2 * p2
        if heads > 3:
            p3 = tl.exp(tl.abs(score_3) - max_abs)
            denom += p3
            numer += score_3 * p3

        combined = numer / tl.maximum(denom, 1.0e-20)
        out_ptrs = out_ptr + pid_b * stride_out_b + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
        tl.store(out_ptrs, combined, mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def _multihead_signed_smoothmax_scores_kernel(
        target_ptr,
        source_ptr,
        bias_ptr,
        out_ptr,
        stride_target_b,
        stride_target_h,
        stride_target_n,
        stride_target_r,
        stride_source_b,
        stride_source_h,
        stride_source_n,
        stride_source_r,
        stride_out_b,
        stride_out_m,
        stride_out_n,
        num_nodes,
        rank_dim,
        has_bias: tl.constexpr,
        heads: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < num_nodes
        mask_n = offs_n < num_nodes

        score_0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        score_1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        score_2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        score_3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, rank_dim, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < rank_dim

            if heads > 0:
                a_ptrs = (
                    target_ptr
                    + pid_b * stride_target_b
                    + 0 * stride_target_h
                    + offs_m[:, None] * stride_target_n
                    + offs_k[None, :] * stride_target_r
                )
                b_ptrs = (
                    source_ptr
                    + pid_b * stride_source_b
                    + 0 * stride_source_h
                    + offs_n[None, :] * stride_source_n
                    + offs_k[:, None] * stride_source_r
                )
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                score_0 += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
            if heads > 1:
                a_ptrs = (
                    target_ptr
                    + pid_b * stride_target_b
                    + 1 * stride_target_h
                    + offs_m[:, None] * stride_target_n
                    + offs_k[None, :] * stride_target_r
                )
                b_ptrs = (
                    source_ptr
                    + pid_b * stride_source_b
                    + 1 * stride_source_h
                    + offs_n[None, :] * stride_source_n
                    + offs_k[:, None] * stride_source_r
                )
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                score_1 += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
            if heads > 2:
                a_ptrs = (
                    target_ptr
                    + pid_b * stride_target_b
                    + 2 * stride_target_h
                    + offs_m[:, None] * stride_target_n
                    + offs_k[None, :] * stride_target_r
                )
                b_ptrs = (
                    source_ptr
                    + pid_b * stride_source_b
                    + 2 * stride_source_h
                    + offs_n[None, :] * stride_source_n
                    + offs_k[:, None] * stride_source_r
                )
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                score_2 += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
            if heads > 3:
                a_ptrs = (
                    target_ptr
                    + pid_b * stride_target_b
                    + 3 * stride_target_h
                    + offs_m[:, None] * stride_target_n
                    + offs_k[None, :] * stride_target_r
                )
                b_ptrs = (
                    source_ptr
                    + pid_b * stride_source_b
                    + 3 * stride_source_h
                    + offs_n[None, :] * stride_source_n
                    + offs_k[:, None] * stride_source_r
                )
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                score_3 += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)

        if has_bias:
            if heads > 0:
                score_0 += tl.load(bias_ptr + 0)
            if heads > 1:
                score_1 += tl.load(bias_ptr + 1)
            if heads > 2:
                score_2 += tl.load(bias_ptr + 2)
            if heads > 3:
                score_3 += tl.load(bias_ptr + 3)

        max_abs = tl.abs(score_0) if heads > 0 else tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if heads > 1:
            max_abs = tl.maximum(max_abs, tl.abs(score_1))
        if heads > 2:
            max_abs = tl.maximum(max_abs, tl.abs(score_2))
        if heads > 3:
            max_abs = tl.maximum(max_abs, tl.abs(score_3))

        denom = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        numer = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        if heads > 0:
            p0 = tl.exp(tl.abs(score_0) - max_abs)
            denom += p0
            numer += score_0 * p0
        if heads > 1:
            p1 = tl.exp(tl.abs(score_1) - max_abs)
            denom += p1
            numer += score_1 * p1
        if heads > 2:
            p2 = tl.exp(tl.abs(score_2) - max_abs)
            denom += p2
            numer += score_2 * p2
        if heads > 3:
            p3 = tl.exp(tl.abs(score_3) - max_abs)
            denom += p3
            numer += score_3 * p3

        combined = numer / tl.maximum(denom, 1.0e-20)
        out_ptrs = out_ptr + pid_b * stride_out_b + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
        tl.store(out_ptrs, combined, mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def _diagonal_signed_smoothmax_scores_tile_kernel(
        flat_ptr,
        core_ptr,
        bias_ptr,
        out_ptr,
        stride_flat_b,
        stride_flat_n,
        stride_flat_d,
        stride_core_h,
        stride_core_d,
        stride_out_b,
        stride_out_m,
        stride_out_n,
        source_start,
        num_nodes,
        tile_nodes,
        dim,
        has_bias: tl.constexpr,
        heads: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        source_idx = source_start + offs_n
        mask_m = offs_m < num_nodes
        mask_n = offs_n < tile_nodes
        causal = source_idx[None, :] <= offs_m[:, None]
        full_mask = mask_m[:, None] & mask_n[None, :] & causal

        score_0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        score_1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        score_2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        score_3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, dim, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < dim
            target_ptrs = (
                flat_ptr
                + pid_b * stride_flat_b
                + offs_m[:, None] * stride_flat_n
                + offs_k[None, :] * stride_flat_d
            )
            source_ptrs = (
                flat_ptr
                + pid_b * stride_flat_b
                + source_idx[None, :] * stride_flat_n
                + offs_k[:, None] * stride_flat_d
            )
            target = tl.load(target_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            source = tl.load(source_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            if heads > 0:
                core0 = tl.load(core_ptr + 0 * stride_core_h + offs_k, mask=mask_k, other=0.0)
                score_0 += tl.dot(target * core0[None, :], source, allow_tf32=False, out_dtype=tl.float32)
            if heads > 1:
                core1 = tl.load(core_ptr + 1 * stride_core_h + offs_k, mask=mask_k, other=0.0)
                score_1 += tl.dot(target * core1[None, :], source, allow_tf32=False, out_dtype=tl.float32)
            if heads > 2:
                core2 = tl.load(core_ptr + 2 * stride_core_h + offs_k, mask=mask_k, other=0.0)
                score_2 += tl.dot(target * core2[None, :], source, allow_tf32=False, out_dtype=tl.float32)
            if heads > 3:
                core3 = tl.load(core_ptr + 3 * stride_core_h + offs_k, mask=mask_k, other=0.0)
                score_3 += tl.dot(target * core3[None, :], source, allow_tf32=False, out_dtype=tl.float32)

        if has_bias:
            if heads > 0:
                score_0 += tl.load(bias_ptr + 0)
            if heads > 1:
                score_1 += tl.load(bias_ptr + 1)
            if heads > 2:
                score_2 += tl.load(bias_ptr + 2)
            if heads > 3:
                score_3 += tl.load(bias_ptr + 3)

        neg_large = -1.0e30
        score_0_masked = tl.where(full_mask, score_0, 0.0)
        score_1_masked = tl.where(full_mask, score_1, 0.0)
        score_2_masked = tl.where(full_mask, score_2, 0.0)
        score_3_masked = tl.where(full_mask, score_3, 0.0)
        max_abs = tl.where(full_mask, tl.abs(score_0), neg_large)
        if heads > 1:
            max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_1), neg_large))
        if heads > 2:
            max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_2), neg_large))
        if heads > 3:
            max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_3), neg_large))

        denom = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        numer = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if heads > 0:
            p0 = tl.where(full_mask, tl.exp(tl.abs(score_0) - max_abs), 0.0)
            denom += p0
            numer += score_0_masked * p0
        if heads > 1:
            p1 = tl.where(full_mask, tl.exp(tl.abs(score_1) - max_abs), 0.0)
            denom += p1
            numer += score_1_masked * p1
        if heads > 2:
            p2 = tl.where(full_mask, tl.exp(tl.abs(score_2) - max_abs), 0.0)
            denom += p2
            numer += score_2_masked * p2
        if heads > 3:
            p3 = tl.where(full_mask, tl.exp(tl.abs(score_3) - max_abs), 0.0)
            denom += p3
            numer += score_3_masked * p3

        out_ptrs = out_ptr + pid_b * stride_out_b + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
        tl.store(out_ptrs, tl.where(full_mask, numer / tl.maximum(denom, 1.0e-20), 0.0), mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def _diagonal_signed_smoothmax_scores_and_head_grads_tile_kernel(
        flat_ptr,
        core_ptr,
        bias_ptr,
        score_out_ptr,
        grad_out_ptr,
        stride_flat_b,
        stride_flat_n,
        stride_flat_d,
        stride_core_h,
        stride_core_d,
        stride_score_out_b,
        stride_score_out_m,
        stride_score_out_n,
        stride_grad_out_b,
        stride_grad_out_h,
        stride_grad_out_m,
        stride_grad_out_n,
        source_start,
        num_nodes,
        tile_nodes,
        dim,
        has_bias: tl.constexpr,
        heads: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        source_idx = source_start + offs_n
        mask_m = offs_m < num_nodes
        mask_n = offs_n < tile_nodes
        causal = source_idx[None, :] <= offs_m[:, None]
        full_mask = mask_m[:, None] & mask_n[None, :] & causal

        score_0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        score_1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        score_2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        score_3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, dim, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < dim
            target_ptrs = (
                flat_ptr
                + pid_b * stride_flat_b
                + offs_m[:, None] * stride_flat_n
                + offs_k[None, :] * stride_flat_d
            )
            source_ptrs = (
                flat_ptr
                + pid_b * stride_flat_b
                + source_idx[None, :] * stride_flat_n
                + offs_k[:, None] * stride_flat_d
            )
            target = tl.load(target_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            source = tl.load(source_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            if heads > 0:
                core0 = tl.load(core_ptr + 0 * stride_core_h + offs_k, mask=mask_k, other=0.0)
                score_0 += tl.dot(target * core0[None, :], source, allow_tf32=False, out_dtype=tl.float32)
            if heads > 1:
                core1 = tl.load(core_ptr + 1 * stride_core_h + offs_k, mask=mask_k, other=0.0)
                score_1 += tl.dot(target * core1[None, :], source, allow_tf32=False, out_dtype=tl.float32)
            if heads > 2:
                core2 = tl.load(core_ptr + 2 * stride_core_h + offs_k, mask=mask_k, other=0.0)
                score_2 += tl.dot(target * core2[None, :], source, allow_tf32=False, out_dtype=tl.float32)
            if heads > 3:
                core3 = tl.load(core_ptr + 3 * stride_core_h + offs_k, mask=mask_k, other=0.0)
                score_3 += tl.dot(target * core3[None, :], source, allow_tf32=False, out_dtype=tl.float32)

        if has_bias:
            if heads > 0:
                score_0 += tl.load(bias_ptr + 0)
            if heads > 1:
                score_1 += tl.load(bias_ptr + 1)
            if heads > 2:
                score_2 += tl.load(bias_ptr + 2)
            if heads > 3:
                score_3 += tl.load(bias_ptr + 3)

        neg_large = -1.0e30
        score_0_masked = tl.where(full_mask, score_0, 0.0)
        score_1_masked = tl.where(full_mask, score_1, 0.0)
        score_2_masked = tl.where(full_mask, score_2, 0.0)
        score_3_masked = tl.where(full_mask, score_3, 0.0)
        max_abs = tl.where(full_mask, tl.abs(score_0), neg_large)
        if heads > 1:
            max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_1), neg_large))
        if heads > 2:
            max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_2), neg_large))
        if heads > 3:
            max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_3), neg_large))

        denom = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        numer = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if heads > 0:
            p0 = tl.where(full_mask, tl.exp(tl.abs(score_0) - max_abs), 0.0)
            denom += p0
            numer += score_0_masked * p0
        if heads > 1:
            p1 = tl.where(full_mask, tl.exp(tl.abs(score_1) - max_abs), 0.0)
            denom += p1
            numer += score_1_masked * p1
        if heads > 2:
            p2 = tl.where(full_mask, tl.exp(tl.abs(score_2) - max_abs), 0.0)
            denom += p2
            numer += score_2_masked * p2
        if heads > 3:
            p3 = tl.where(full_mask, tl.exp(tl.abs(score_3) - max_abs), 0.0)
            denom += p3
            numer += score_3_masked * p3

        safe_denom = tl.maximum(denom, 1.0e-20)
        combined = numer / safe_denom
        score_out_ptrs = score_out_ptr + pid_b * stride_score_out_b + offs_m[:, None] * stride_score_out_m + offs_n[None, :] * stride_score_out_n
        tl.store(score_out_ptrs, tl.where(full_mask, combined, 0.0), mask=mask_m[:, None] & mask_n[None, :])

        if heads > 0:
            p0 = tl.where(full_mask, tl.exp(tl.abs(score_0) - max_abs) / safe_denom, 0.0)
            g0 = p0 * (1.0 + tl.where(score_0 > 0, 1.0, tl.where(score_0 < 0, -1.0, 0.0)) * (score_0 - combined))
            out_ptrs = grad_out_ptr + pid_b * stride_grad_out_b + 0 * stride_grad_out_h + offs_m[:, None] * stride_grad_out_m + offs_n[None, :] * stride_grad_out_n
            tl.store(out_ptrs, tl.where(full_mask, g0, 0.0), mask=mask_m[:, None] & mask_n[None, :])
        if heads > 1:
            p1 = tl.where(full_mask, tl.exp(tl.abs(score_1) - max_abs) / safe_denom, 0.0)
            g1 = p1 * (1.0 + tl.where(score_1 > 0, 1.0, tl.where(score_1 < 0, -1.0, 0.0)) * (score_1 - combined))
            out_ptrs = grad_out_ptr + pid_b * stride_grad_out_b + 1 * stride_grad_out_h + offs_m[:, None] * stride_grad_out_m + offs_n[None, :] * stride_grad_out_n
            tl.store(out_ptrs, tl.where(full_mask, g1, 0.0), mask=mask_m[:, None] & mask_n[None, :])
        if heads > 2:
            p2 = tl.where(full_mask, tl.exp(tl.abs(score_2) - max_abs) / safe_denom, 0.0)
            g2 = p2 * (1.0 + tl.where(score_2 > 0, 1.0, tl.where(score_2 < 0, -1.0, 0.0)) * (score_2 - combined))
            out_ptrs = grad_out_ptr + pid_b * stride_grad_out_b + 2 * stride_grad_out_h + offs_m[:, None] * stride_grad_out_m + offs_n[None, :] * stride_grad_out_n
            tl.store(out_ptrs, tl.where(full_mask, g2, 0.0), mask=mask_m[:, None] & mask_n[None, :])
        if heads > 3:
            p3 = tl.where(full_mask, tl.exp(tl.abs(score_3) - max_abs) / safe_denom, 0.0)
            g3 = p3 * (1.0 + tl.where(score_3 > 0, 1.0, tl.where(score_3 < 0, -1.0, 0.0)) * (score_3 - combined))
            out_ptrs = grad_out_ptr + pid_b * stride_grad_out_b + 3 * stride_grad_out_h + offs_m[:, None] * stride_grad_out_m + offs_n[None, :] * stride_grad_out_n
            tl.store(out_ptrs, tl.where(full_mask, g3, 0.0), mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def _multihead_signed_smoothmax_head_grads_kernel(
        target_ptr,
        source_ptr,
        bias_ptr,
        out_ptr,
        stride_target_b,
        stride_target_h,
        stride_target_n,
        stride_target_r,
        stride_source_b,
        stride_source_h,
        stride_source_n,
        stride_source_r,
        stride_out_b,
        stride_out_h,
        stride_out_m,
        stride_out_n,
        source_start,
        num_nodes,
        tile_nodes,
        rank_dim,
        has_bias: tl.constexpr,
        heads: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        source_idx = source_start + offs_n
        mask_m = offs_m < num_nodes
        mask_n = offs_n < tile_nodes
        causal = source_idx[None, :] <= offs_m[:, None]
        full_mask = mask_m[:, None] & mask_n[None, :] & causal

        score_0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        score_1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        score_2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        score_3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, rank_dim, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < rank_dim

            if heads > 0:
                a_ptrs = target_ptr + pid_b * stride_target_b + 0 * stride_target_h + offs_m[:, None] * stride_target_n + offs_k[None, :] * stride_target_r
                b_ptrs = source_ptr + pid_b * stride_source_b + 0 * stride_source_h + offs_n[None, :] * stride_source_n + offs_k[:, None] * stride_source_r
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                score_0 += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
            if heads > 1:
                a_ptrs = target_ptr + pid_b * stride_target_b + 1 * stride_target_h + offs_m[:, None] * stride_target_n + offs_k[None, :] * stride_target_r
                b_ptrs = source_ptr + pid_b * stride_source_b + 1 * stride_source_h + offs_n[None, :] * stride_source_n + offs_k[:, None] * stride_source_r
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                score_1 += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
            if heads > 2:
                a_ptrs = target_ptr + pid_b * stride_target_b + 2 * stride_target_h + offs_m[:, None] * stride_target_n + offs_k[None, :] * stride_target_r
                b_ptrs = source_ptr + pid_b * stride_source_b + 2 * stride_source_h + offs_n[None, :] * stride_source_n + offs_k[:, None] * stride_source_r
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                score_2 += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
            if heads > 3:
                a_ptrs = target_ptr + pid_b * stride_target_b + 3 * stride_target_h + offs_m[:, None] * stride_target_n + offs_k[None, :] * stride_target_r
                b_ptrs = source_ptr + pid_b * stride_source_b + 3 * stride_source_h + offs_n[None, :] * stride_source_n + offs_k[:, None] * stride_source_r
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                score_3 += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)

        if has_bias:
            if heads > 0:
                score_0 += tl.load(bias_ptr + 0)
            if heads > 1:
                score_1 += tl.load(bias_ptr + 1)
            if heads > 2:
                score_2 += tl.load(bias_ptr + 2)
            if heads > 3:
                score_3 += tl.load(bias_ptr + 3)

        neg_large = -1.0e30
        score_0_masked = tl.where(full_mask, score_0, 0.0)
        score_1_masked = tl.where(full_mask, score_1, 0.0)
        score_2_masked = tl.where(full_mask, score_2, 0.0)
        score_3_masked = tl.where(full_mask, score_3, 0.0)

        max_abs = tl.where(full_mask, tl.abs(score_0), neg_large)
        if heads > 1:
            max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_1), neg_large))
        if heads > 2:
            max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_2), neg_large))
        if heads > 3:
            max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_3), neg_large))

        denom = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        numer = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        if heads > 0:
            p0 = tl.where(full_mask, tl.exp(tl.abs(score_0) - max_abs), 0.0)
            denom += p0
            numer += score_0_masked * p0
        if heads > 1:
            p1 = tl.where(full_mask, tl.exp(tl.abs(score_1) - max_abs), 0.0)
            denom += p1
            numer += score_1_masked * p1
        if heads > 2:
            p2 = tl.where(full_mask, tl.exp(tl.abs(score_2) - max_abs), 0.0)
            denom += p2
            numer += score_2_masked * p2
        if heads > 3:
            p3 = tl.where(full_mask, tl.exp(tl.abs(score_3) - max_abs), 0.0)
            denom += p3
            numer += score_3_masked * p3

        combined = numer / tl.maximum(denom, 1.0e-20)

        if heads > 0:
            p0 = tl.where(full_mask, tl.exp(tl.abs(score_0) - max_abs) / tl.maximum(denom, 1.0e-20), 0.0)
            g0 = p0 * (1.0 + tl.where(score_0 > 0, 1.0, tl.where(score_0 < 0, -1.0, 0.0)) * (score_0 - combined))
            out_ptrs = out_ptr + pid_b * stride_out_b + 0 * stride_out_h + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
            tl.store(out_ptrs, tl.where(full_mask, g0, 0.0), mask=mask_m[:, None] & mask_n[None, :])
        if heads > 1:
            p1 = tl.where(full_mask, tl.exp(tl.abs(score_1) - max_abs) / tl.maximum(denom, 1.0e-20), 0.0)
            g1 = p1 * (1.0 + tl.where(score_1 > 0, 1.0, tl.where(score_1 < 0, -1.0, 0.0)) * (score_1 - combined))
            out_ptrs = out_ptr + pid_b * stride_out_b + 1 * stride_out_h + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
            tl.store(out_ptrs, tl.where(full_mask, g1, 0.0), mask=mask_m[:, None] & mask_n[None, :])
        if heads > 2:
            p2 = tl.where(full_mask, tl.exp(tl.abs(score_2) - max_abs) / tl.maximum(denom, 1.0e-20), 0.0)
            g2 = p2 * (1.0 + tl.where(score_2 > 0, 1.0, tl.where(score_2 < 0, -1.0, 0.0)) * (score_2 - combined))
            out_ptrs = out_ptr + pid_b * stride_out_b + 2 * stride_out_h + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
            tl.store(out_ptrs, tl.where(full_mask, g2, 0.0), mask=mask_m[:, None] & mask_n[None, :])
        if heads > 3:
            p3 = tl.where(full_mask, tl.exp(tl.abs(score_3) - max_abs) / tl.maximum(denom, 1.0e-20), 0.0)
            g3 = p3 * (1.0 + tl.where(score_3 > 0, 1.0, tl.where(score_3 < 0, -1.0, 0.0)) * (score_3 - combined))
            out_ptrs = out_ptr + pid_b * stride_out_b + 3 * stride_out_h + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
            tl.store(out_ptrs, tl.where(full_mask, g3, 0.0), mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def _diagonal_signed_smoothmax_backward_tile_kernel(
        flat_ptr,
        core_ptr,
        bias_ptr,
        grad_edges_ptr,
        row_max_ptr,
        row_denom_ptr,
        edge_dot_ptr,
        grad_layer_ptr,
        grad_weights_ptr,
        grad_bias_ptr,
        stride_flat_b,
        stride_flat_n,
        stride_flat_d,
        stride_core_h,
        stride_core_d,
        stride_grad_edges_b,
        stride_grad_edges_m,
        stride_grad_edges_n,
        stride_row_b,
        stride_row_m,
        stride_edge_dot_b,
        stride_edge_dot_m,
        stride_grad_layer_b,
        stride_grad_layer_n,
        stride_grad_layer_d,
        stride_grad_weights_h,
        stride_grad_weights_d,
        source_start,
        num_nodes,
        tile_nodes,
        dim,
        has_bias: tl.constexpr,
        heads: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        source_idx = source_start + offs_n
        mask_m = offs_m < num_nodes
        mask_n = offs_n < tile_nodes
        causal = source_idx[None, :] <= offs_m[:, None]
        full_mask = mask_m[:, None] & mask_n[None, :] & causal

        score_0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        score_1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        score_2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        score_3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, dim, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < dim
            target_ptrs = (
                flat_ptr
                + pid_b * stride_flat_b
                + offs_m[:, None] * stride_flat_n
                + offs_k[None, :] * stride_flat_d
            )
            source_ptrs = (
                flat_ptr
                + pid_b * stride_flat_b
                + source_idx[None, :] * stride_flat_n
                + offs_k[:, None] * stride_flat_d
            )
            target = tl.load(target_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            source = tl.load(source_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            if heads > 0:
                core0 = tl.load(core_ptr + 0 * stride_core_h + offs_k * stride_core_d, mask=mask_k, other=0.0)
                score_0 += tl.dot(target * core0[None, :], source, allow_tf32=False, out_dtype=tl.float32)
            if heads > 1:
                core1 = tl.load(core_ptr + 1 * stride_core_h + offs_k * stride_core_d, mask=mask_k, other=0.0)
                score_1 += tl.dot(target * core1[None, :], source, allow_tf32=False, out_dtype=tl.float32)
            if heads > 2:
                core2 = tl.load(core_ptr + 2 * stride_core_h + offs_k * stride_core_d, mask=mask_k, other=0.0)
                score_2 += tl.dot(target * core2[None, :], source, allow_tf32=False, out_dtype=tl.float32)
            if heads > 3:
                core3 = tl.load(core_ptr + 3 * stride_core_h + offs_k * stride_core_d, mask=mask_k, other=0.0)
                score_3 += tl.dot(target * core3[None, :], source, allow_tf32=False, out_dtype=tl.float32)

        if has_bias:
            if heads > 0:
                score_0 += tl.load(bias_ptr + 0)
            if heads > 1:
                score_1 += tl.load(bias_ptr + 1)
            if heads > 2:
                score_2 += tl.load(bias_ptr + 2)
            if heads > 3:
                score_3 += tl.load(bias_ptr + 3)

        neg_large = -1.0e30
        max_abs = tl.where(full_mask, tl.abs(score_0), neg_large)
        if heads > 1:
            max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_1), neg_large))
        if heads > 2:
            max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_2), neg_large))
        if heads > 3:
            max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_3), neg_large))

        denom_heads = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        numer = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        p0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        p1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        p2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        p3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if heads > 0:
            p0 = tl.where(full_mask, tl.exp(tl.abs(score_0) - max_abs), 0.0)
            denom_heads += p0
            numer += score_0 * p0
        if heads > 1:
            p1 = tl.where(full_mask, tl.exp(tl.abs(score_1) - max_abs), 0.0)
            denom_heads += p1
            numer += score_1 * p1
        if heads > 2:
            p2 = tl.where(full_mask, tl.exp(tl.abs(score_2) - max_abs), 0.0)
            denom_heads += p2
            numer += score_2 * p2
        if heads > 3:
            p3 = tl.where(full_mask, tl.exp(tl.abs(score_3) - max_abs), 0.0)
            denom_heads += p3
            numer += score_3 * p3

        safe_head_denom = tl.maximum(denom_heads, 1.0e-20)
        combined = numer / safe_head_denom
        row_max = tl.load(
            row_max_ptr + pid_b * stride_row_b + offs_m * stride_row_m,
            mask=mask_m,
            other=-float("inf"),
        )
        row_denom = tl.load(
            row_denom_ptr + pid_b * stride_row_b + offs_m * stride_row_m,
            mask=mask_m,
            other=1.0,
        )
        edge_dot = tl.load(
            edge_dot_ptr + pid_b * stride_edge_dot_b + offs_m * stride_edge_dot_m,
            mask=mask_m,
            other=0.0,
        )
        grad_edges = tl.load(
            grad_edges_ptr
            + pid_b * stride_grad_edges_b
            + offs_m[:, None] * stride_grad_edges_m
            + offs_n[None, :] * stride_grad_edges_n,
            mask=full_mask,
            other=0.0,
        )
        probs = tl.where(
            full_mask,
            tl.exp(tl.abs(combined) - row_max[:, None]) / tl.maximum(row_denom[:, None], 1.0e-20),
            0.0,
        )
        signs = tl.where(combined > 0, 1.0, tl.where(combined < 0, -1.0, 0.0))
        grad_scores = tl.where(
            full_mask,
            signs * probs * (signs * grad_edges - edge_dot[:, None]),
            0.0,
        )

        g0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        g1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        g2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        g3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if heads > 0:
            sign0 = tl.where(score_0 > 0, 1.0, tl.where(score_0 < 0, -1.0, 0.0))
            g0 = tl.where(full_mask, (p0 / safe_head_denom) * (1.0 + sign0 * (score_0 - combined)), 0.0)
        if heads > 1:
            sign1 = tl.where(score_1 > 0, 1.0, tl.where(score_1 < 0, -1.0, 0.0))
            g1 = tl.where(full_mask, (p1 / safe_head_denom) * (1.0 + sign1 * (score_1 - combined)), 0.0)
        if heads > 2:
            sign2 = tl.where(score_2 > 0, 1.0, tl.where(score_2 < 0, -1.0, 0.0))
            g2 = tl.where(full_mask, (p2 / safe_head_denom) * (1.0 + sign2 * (score_2 - combined)), 0.0)
        if heads > 3:
            sign3 = tl.where(score_3 > 0, 1.0, tl.where(score_3 < 0, -1.0, 0.0))
            g3 = tl.where(full_mask, (p3 / safe_head_denom) * (1.0 + sign3 * (score_3 - combined)), 0.0)

        bias_sum_0 = tl.sum(grad_scores * g0)
        bias_sum_1 = tl.sum(grad_scores * g1)
        bias_sum_2 = tl.sum(grad_scores * g2)
        bias_sum_3 = tl.sum(grad_scores * g3)
        if has_bias:
            if heads > 0:
                tl.atomic_add(grad_bias_ptr + 0, bias_sum_0)
            if heads > 1:
                tl.atomic_add(grad_bias_ptr + 1, bias_sum_1)
            if heads > 2:
                tl.atomic_add(grad_bias_ptr + 2, bias_sum_2)
            if heads > 3:
                tl.atomic_add(grad_bias_ptr + 3, bias_sum_3)

        for d_start in range(0, dim, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < dim
            target_block = tl.load(
                flat_ptr
                + pid_b * stride_flat_b
                + offs_m[:, None] * stride_flat_n
                + offs_d[None, :] * stride_flat_d,
                mask=mask_m[:, None] & mask_d[None, :],
                other=0.0,
            )
            source_block = tl.load(
                flat_ptr
                + pid_b * stride_flat_b
                + source_idx[:, None] * stride_flat_n
                + offs_d[None, :] * stride_flat_d,
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
            )
            target_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
            source_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
            if heads > 0:
                gs0 = grad_scores * g0
                source_head = tl.dot(tl.trans(gs0), target_block, allow_tf32=False, out_dtype=tl.float32)
                target_acc += tl.dot(gs0, source_block, allow_tf32=False, out_dtype=tl.float32)
                core0 = tl.load(core_ptr + 0 * stride_core_h + offs_d * stride_core_d, mask=mask_d, other=0.0)
                source_acc += source_head * core0[None, :]
                tl.atomic_add(
                    grad_weights_ptr + 0 * stride_grad_weights_h + offs_d * stride_grad_weights_d,
                    tl.sum(source_head * source_block, axis=0),
                    mask=mask_d,
                )
            if heads > 1:
                gs1 = grad_scores * g1
                source_head = tl.dot(tl.trans(gs1), target_block, allow_tf32=False, out_dtype=tl.float32)
                target_acc += tl.dot(gs1, source_block, allow_tf32=False, out_dtype=tl.float32)
                core1 = tl.load(core_ptr + 1 * stride_core_h + offs_d * stride_core_d, mask=mask_d, other=0.0)
                source_acc += source_head * core1[None, :]
                tl.atomic_add(
                    grad_weights_ptr + 1 * stride_grad_weights_h + offs_d * stride_grad_weights_d,
                    tl.sum(source_head * source_block, axis=0),
                    mask=mask_d,
                )
            if heads > 2:
                gs2 = grad_scores * g2
                source_head = tl.dot(tl.trans(gs2), target_block, allow_tf32=False, out_dtype=tl.float32)
                target_acc += tl.dot(gs2, source_block, allow_tf32=False, out_dtype=tl.float32)
                core2 = tl.load(core_ptr + 2 * stride_core_h + offs_d * stride_core_d, mask=mask_d, other=0.0)
                source_acc += source_head * core2[None, :]
                tl.atomic_add(
                    grad_weights_ptr + 2 * stride_grad_weights_h + offs_d * stride_grad_weights_d,
                    tl.sum(source_head * source_block, axis=0),
                    mask=mask_d,
                )
            if heads > 3:
                gs3 = grad_scores * g3
                source_head = tl.dot(tl.trans(gs3), target_block, allow_tf32=False, out_dtype=tl.float32)
                target_acc += tl.dot(gs3, source_block, allow_tf32=False, out_dtype=tl.float32)
                core3 = tl.load(core_ptr + 3 * stride_core_h + offs_d * stride_core_d, mask=mask_d, other=0.0)
                source_acc += source_head * core3[None, :]
                tl.atomic_add(
                    grad_weights_ptr + 3 * stride_grad_weights_h + offs_d * stride_grad_weights_d,
                    tl.sum(source_head * source_block, axis=0),
                    mask=mask_d,
                )

            grad_layer_target_ptrs = (
                grad_layer_ptr
                + pid_b * stride_grad_layer_b
                + offs_m[:, None] * stride_grad_layer_n
                + offs_d[None, :] * stride_grad_layer_d
            )
            tl.atomic_add(grad_layer_target_ptrs, target_acc, mask=mask_m[:, None] & mask_d[None, :])
            grad_layer_source_ptrs = (
                grad_layer_ptr
                + pid_b * stride_grad_layer_b
                + source_idx[:, None] * stride_grad_layer_n
                + offs_d[None, :] * stride_grad_layer_d
            )
            tl.atomic_add(grad_layer_source_ptrs, source_acc, mask=mask_n[:, None] & mask_d[None, :])

    @triton.jit
    def _lowrank_signed_smoothmax_backward_tile_kernel(
        target_ptr,
        source_proj_ptr,
        source_weighted_ptr,
        core_ptr,
        bias_ptr,
        grad_edges_ptr,
        row_max_ptr,
        row_denom_ptr,
        edge_dot_ptr,
        grad_target_ptr,
        grad_source_ptr,
        grad_core_ptr,
        grad_bias_ptr,
        stride_target_h,
        stride_target_b,
        stride_target_n,
        stride_target_r,
        stride_source_proj_h,
        stride_source_proj_b,
        stride_source_proj_n,
        stride_source_proj_r,
        stride_source_weighted_h,
        stride_source_weighted_b,
        stride_source_weighted_n,
        stride_source_weighted_r,
        stride_core_h,
        stride_core_r,
        stride_grad_edges_b,
        stride_grad_edges_m,
        stride_grad_edges_n,
        stride_row_b,
        stride_row_m,
        stride_edge_dot_b,
        stride_edge_dot_m,
        stride_grad_target_h,
        stride_grad_target_b,
        stride_grad_target_n,
        stride_grad_target_r,
        stride_grad_source_h,
        stride_grad_source_b,
        stride_grad_source_n,
        stride_grad_source_r,
        stride_grad_core_h,
        stride_grad_core_r,
        source_start,
        grad_source_row_offset,
        num_nodes,
        tile_nodes,
        rank_dim,
        has_bias: tl.constexpr,
        heads: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        source_idx = source_start + offs_n
        mask_m = offs_m < num_nodes
        mask_n = offs_n < tile_nodes
        causal = source_idx[None, :] <= offs_m[:, None]
        full_mask = mask_m[:, None] & mask_n[None, :] & causal

        score_0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        score_1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        score_2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        score_3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, rank_dim, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < rank_dim
            if heads > 0:
                a_ptrs = target_ptr + 0 * stride_target_h + pid_b * stride_target_b + offs_m[:, None] * stride_target_n + offs_k[None, :] * stride_target_r
                b_ptrs = source_weighted_ptr + 0 * stride_source_weighted_h + pid_b * stride_source_weighted_b + offs_n[None, :] * stride_source_weighted_n + offs_k[:, None] * stride_source_weighted_r
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                score_0 += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
            if heads > 1:
                a_ptrs = target_ptr + 1 * stride_target_h + pid_b * stride_target_b + offs_m[:, None] * stride_target_n + offs_k[None, :] * stride_target_r
                b_ptrs = source_weighted_ptr + 1 * stride_source_weighted_h + pid_b * stride_source_weighted_b + offs_n[None, :] * stride_source_weighted_n + offs_k[:, None] * stride_source_weighted_r
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                score_1 += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
            if heads > 2:
                a_ptrs = target_ptr + 2 * stride_target_h + pid_b * stride_target_b + offs_m[:, None] * stride_target_n + offs_k[None, :] * stride_target_r
                b_ptrs = source_weighted_ptr + 2 * stride_source_weighted_h + pid_b * stride_source_weighted_b + offs_n[None, :] * stride_source_weighted_n + offs_k[:, None] * stride_source_weighted_r
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                score_2 += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
            if heads > 3:
                a_ptrs = target_ptr + 3 * stride_target_h + pid_b * stride_target_b + offs_m[:, None] * stride_target_n + offs_k[None, :] * stride_target_r
                b_ptrs = source_weighted_ptr + 3 * stride_source_weighted_h + pid_b * stride_source_weighted_b + offs_n[None, :] * stride_source_weighted_n + offs_k[:, None] * stride_source_weighted_r
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                score_3 += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)

        if has_bias:
            if heads > 0:
                score_0 += tl.load(bias_ptr + 0)
            if heads > 1:
                score_1 += tl.load(bias_ptr + 1)
            if heads > 2:
                score_2 += tl.load(bias_ptr + 2)
            if heads > 3:
                score_3 += tl.load(bias_ptr + 3)

        neg_large = -1.0e30
        max_abs = tl.where(full_mask, tl.abs(score_0), neg_large)
        if heads > 1:
            max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_1), neg_large))
        if heads > 2:
            max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_2), neg_large))
        if heads > 3:
            max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_3), neg_large))

        denom_heads = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        numer = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        p0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        p1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        p2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        p3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if heads > 0:
            p0 = tl.where(full_mask, tl.exp(tl.abs(score_0) - max_abs), 0.0)
            denom_heads += p0
            numer += score_0 * p0
        if heads > 1:
            p1 = tl.where(full_mask, tl.exp(tl.abs(score_1) - max_abs), 0.0)
            denom_heads += p1
            numer += score_1 * p1
        if heads > 2:
            p2 = tl.where(full_mask, tl.exp(tl.abs(score_2) - max_abs), 0.0)
            denom_heads += p2
            numer += score_2 * p2
        if heads > 3:
            p3 = tl.where(full_mask, tl.exp(tl.abs(score_3) - max_abs), 0.0)
            denom_heads += p3
            numer += score_3 * p3

        safe_head_denom = tl.maximum(denom_heads, 1.0e-20)
        combined = numer / safe_head_denom
        row_max = tl.load(row_max_ptr + pid_b * stride_row_b + offs_m * stride_row_m, mask=mask_m, other=-float("inf"))
        row_denom = tl.load(row_denom_ptr + pid_b * stride_row_b + offs_m * stride_row_m, mask=mask_m, other=1.0)
        edge_dot = tl.load(edge_dot_ptr + pid_b * stride_edge_dot_b + offs_m * stride_edge_dot_m, mask=mask_m, other=0.0)
        grad_edges = tl.load(
            grad_edges_ptr
            + pid_b * stride_grad_edges_b
            + offs_m[:, None] * stride_grad_edges_m
            + offs_n[None, :] * stride_grad_edges_n,
            mask=full_mask,
            other=0.0,
        )
        probs = tl.where(
            full_mask,
            tl.exp(tl.abs(combined) - row_max[:, None]) / tl.maximum(row_denom[:, None], 1.0e-20),
            0.0,
        )
        signs = tl.where(combined > 0, 1.0, tl.where(combined < 0, -1.0, 0.0))
        grad_scores = tl.where(full_mask, signs * probs * (signs * grad_edges - edge_dot[:, None]), 0.0)

        g0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        g1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        g2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        g3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if heads > 0:
            sign0 = tl.where(score_0 > 0, 1.0, tl.where(score_0 < 0, -1.0, 0.0))
            g0 = tl.where(full_mask, (p0 / safe_head_denom) * (1.0 + sign0 * (score_0 - combined)), 0.0)
        if heads > 1:
            sign1 = tl.where(score_1 > 0, 1.0, tl.where(score_1 < 0, -1.0, 0.0))
            g1 = tl.where(full_mask, (p1 / safe_head_denom) * (1.0 + sign1 * (score_1 - combined)), 0.0)
        if heads > 2:
            sign2 = tl.where(score_2 > 0, 1.0, tl.where(score_2 < 0, -1.0, 0.0))
            g2 = tl.where(full_mask, (p2 / safe_head_denom) * (1.0 + sign2 * (score_2 - combined)), 0.0)
        if heads > 3:
            sign3 = tl.where(score_3 > 0, 1.0, tl.where(score_3 < 0, -1.0, 0.0))
            g3 = tl.where(full_mask, (p3 / safe_head_denom) * (1.0 + sign3 * (score_3 - combined)), 0.0)

        bias_sum_0 = tl.sum(grad_scores * g0)
        bias_sum_1 = tl.sum(grad_scores * g1)
        bias_sum_2 = tl.sum(grad_scores * g2)
        bias_sum_3 = tl.sum(grad_scores * g3)
        if has_bias:
            if heads > 0:
                tl.atomic_add(grad_bias_ptr + 0, bias_sum_0)
            if heads > 1:
                tl.atomic_add(grad_bias_ptr + 1, bias_sum_1)
            if heads > 2:
                tl.atomic_add(grad_bias_ptr + 2, bias_sum_2)
            if heads > 3:
                tl.atomic_add(grad_bias_ptr + 3, bias_sum_3)

        for r_start in range(0, rank_dim, BLOCK_R):
            offs_r = r_start + tl.arange(0, BLOCK_R)
            mask_r = offs_r < rank_dim
            if heads > 0:
                gs0 = grad_scores * g0
                source_weighted = tl.load(
                    source_weighted_ptr + 0 * stride_source_weighted_h + pid_b * stride_source_weighted_b + offs_n[:, None] * stride_source_weighted_n + offs_r[None, :] * stride_source_weighted_r,
                    mask=mask_n[:, None] & mask_r[None, :],
                    other=0.0,
                )
                target = tl.load(
                    target_ptr + 0 * stride_target_h + pid_b * stride_target_b + offs_m[:, None] * stride_target_n + offs_r[None, :] * stride_target_r,
                    mask=mask_m[:, None] & mask_r[None, :],
                    other=0.0,
                )
                source_proj = tl.load(
                    source_proj_ptr + 0 * stride_source_proj_h + pid_b * stride_source_proj_b + offs_n[:, None] * stride_source_proj_n + offs_r[None, :] * stride_source_proj_r,
                    mask=mask_n[:, None] & mask_r[None, :],
                    other=0.0,
                )
                tgt_part = tl.dot(gs0, source_weighted, allow_tf32=False, out_dtype=tl.float32)
                src_weighted_part = tl.dot(tl.trans(gs0), target, allow_tf32=False, out_dtype=tl.float32)
                core0 = tl.load(core_ptr + 0 * stride_core_h + offs_r * stride_core_r, mask=mask_r, other=0.0)
                tl.atomic_add(
                    grad_target_ptr + 0 * stride_grad_target_h + pid_b * stride_grad_target_b + offs_m[:, None] * stride_grad_target_n + offs_r[None, :] * stride_grad_target_r,
                    tgt_part,
                    mask=mask_m[:, None] & mask_r[None, :],
                )
                tl.atomic_add(
                    grad_source_ptr + 0 * stride_grad_source_h + pid_b * stride_grad_source_b + (grad_source_row_offset + offs_n)[:, None] * stride_grad_source_n + offs_r[None, :] * stride_grad_source_r,
                    src_weighted_part * core0[None, :],
                    mask=mask_n[:, None] & mask_r[None, :],
                )
                tl.atomic_add(
                    grad_core_ptr + 0 * stride_grad_core_h + offs_r * stride_grad_core_r,
                    tl.sum(src_weighted_part * source_proj, axis=0),
                    mask=mask_r,
                )
            if heads > 1:
                gs1 = grad_scores * g1
                source_weighted = tl.load(
                    source_weighted_ptr + 1 * stride_source_weighted_h + pid_b * stride_source_weighted_b + offs_n[:, None] * stride_source_weighted_n + offs_r[None, :] * stride_source_weighted_r,
                    mask=mask_n[:, None] & mask_r[None, :],
                    other=0.0,
                )
                target = tl.load(
                    target_ptr + 1 * stride_target_h + pid_b * stride_target_b + offs_m[:, None] * stride_target_n + offs_r[None, :] * stride_target_r,
                    mask=mask_m[:, None] & mask_r[None, :],
                    other=0.0,
                )
                source_proj = tl.load(
                    source_proj_ptr + 1 * stride_source_proj_h + pid_b * stride_source_proj_b + offs_n[:, None] * stride_source_proj_n + offs_r[None, :] * stride_source_proj_r,
                    mask=mask_n[:, None] & mask_r[None, :],
                    other=0.0,
                )
                tgt_part = tl.dot(gs1, source_weighted, allow_tf32=False, out_dtype=tl.float32)
                src_weighted_part = tl.dot(tl.trans(gs1), target, allow_tf32=False, out_dtype=tl.float32)
                core1 = tl.load(core_ptr + 1 * stride_core_h + offs_r * stride_core_r, mask=mask_r, other=0.0)
                tl.atomic_add(
                    grad_target_ptr + 1 * stride_grad_target_h + pid_b * stride_grad_target_b + offs_m[:, None] * stride_grad_target_n + offs_r[None, :] * stride_grad_target_r,
                    tgt_part,
                    mask=mask_m[:, None] & mask_r[None, :],
                )
                tl.atomic_add(
                    grad_source_ptr + 1 * stride_grad_source_h + pid_b * stride_grad_source_b + (grad_source_row_offset + offs_n)[:, None] * stride_grad_source_n + offs_r[None, :] * stride_grad_source_r,
                    src_weighted_part * core1[None, :],
                    mask=mask_n[:, None] & mask_r[None, :],
                )
                tl.atomic_add(
                    grad_core_ptr + 1 * stride_grad_core_h + offs_r * stride_grad_core_r,
                    tl.sum(src_weighted_part * source_proj, axis=0),
                    mask=mask_r,
                )
            if heads > 2:
                gs2 = grad_scores * g2
                source_weighted = tl.load(
                    source_weighted_ptr + 2 * stride_source_weighted_h + pid_b * stride_source_weighted_b + offs_n[:, None] * stride_source_weighted_n + offs_r[None, :] * stride_source_weighted_r,
                    mask=mask_n[:, None] & mask_r[None, :],
                    other=0.0,
                )
                target = tl.load(
                    target_ptr + 2 * stride_target_h + pid_b * stride_target_b + offs_m[:, None] * stride_target_n + offs_r[None, :] * stride_target_r,
                    mask=mask_m[:, None] & mask_r[None, :],
                    other=0.0,
                )
                source_proj = tl.load(
                    source_proj_ptr + 2 * stride_source_proj_h + pid_b * stride_source_proj_b + offs_n[:, None] * stride_source_proj_n + offs_r[None, :] * stride_source_proj_r,
                    mask=mask_n[:, None] & mask_r[None, :],
                    other=0.0,
                )
                tgt_part = tl.dot(gs2, source_weighted, allow_tf32=False, out_dtype=tl.float32)
                src_weighted_part = tl.dot(tl.trans(gs2), target, allow_tf32=False, out_dtype=tl.float32)
                core2 = tl.load(core_ptr + 2 * stride_core_h + offs_r * stride_core_r, mask=mask_r, other=0.0)
                tl.atomic_add(
                    grad_target_ptr + 2 * stride_grad_target_h + pid_b * stride_grad_target_b + offs_m[:, None] * stride_grad_target_n + offs_r[None, :] * stride_grad_target_r,
                    tgt_part,
                    mask=mask_m[:, None] & mask_r[None, :],
                )
                tl.atomic_add(
                    grad_source_ptr + 2 * stride_grad_source_h + pid_b * stride_grad_source_b + (grad_source_row_offset + offs_n)[:, None] * stride_grad_source_n + offs_r[None, :] * stride_grad_source_r,
                    src_weighted_part * core2[None, :],
                    mask=mask_n[:, None] & mask_r[None, :],
                )
                tl.atomic_add(
                    grad_core_ptr + 2 * stride_grad_core_h + offs_r * stride_grad_core_r,
                    tl.sum(src_weighted_part * source_proj, axis=0),
                    mask=mask_r,
                )
            if heads > 3:
                gs3 = grad_scores * g3
                source_weighted = tl.load(
                    source_weighted_ptr + 3 * stride_source_weighted_h + pid_b * stride_source_weighted_b + offs_n[:, None] * stride_source_weighted_n + offs_r[None, :] * stride_source_weighted_r,
                    mask=mask_n[:, None] & mask_r[None, :],
                    other=0.0,
                )
                target = tl.load(
                    target_ptr + 3 * stride_target_h + pid_b * stride_target_b + offs_m[:, None] * stride_target_n + offs_r[None, :] * stride_target_r,
                    mask=mask_m[:, None] & mask_r[None, :],
                    other=0.0,
                )
                source_proj = tl.load(
                    source_proj_ptr + 3 * stride_source_proj_h + pid_b * stride_source_proj_b + offs_n[:, None] * stride_source_proj_n + offs_r[None, :] * stride_source_proj_r,
                    mask=mask_n[:, None] & mask_r[None, :],
                    other=0.0,
                )
                tgt_part = tl.dot(gs3, source_weighted, allow_tf32=False, out_dtype=tl.float32)
                src_weighted_part = tl.dot(tl.trans(gs3), target, allow_tf32=False, out_dtype=tl.float32)
                core3 = tl.load(core_ptr + 3 * stride_core_h + offs_r * stride_core_r, mask=mask_r, other=0.0)
                tl.atomic_add(
                    grad_target_ptr + 3 * stride_grad_target_h + pid_b * stride_grad_target_b + offs_m[:, None] * stride_grad_target_n + offs_r[None, :] * stride_grad_target_r,
                    tgt_part,
                    mask=mask_m[:, None] & mask_r[None, :],
                )
                tl.atomic_add(
                    grad_source_ptr + 3 * stride_grad_source_h + pid_b * stride_grad_source_b + (grad_source_row_offset + offs_n)[:, None] * stride_grad_source_n + offs_r[None, :] * stride_grad_source_r,
                    src_weighted_part * core3[None, :],
                    mask=mask_n[:, None] & mask_r[None, :],
                )
                tl.atomic_add(
                    grad_core_ptr + 3 * stride_grad_core_h + offs_r * stride_grad_core_r,
                    tl.sum(src_weighted_part * source_proj, axis=0),
                    mask=mask_r,
                )


def multihead_signed_smoothmax_scores(
    projected_target_bhnr: torch.Tensor,
    weighted_source_bhnr: torch.Tensor,
    biases: torch.Tensor | None = None,
) -> torch.Tensor:
    if not triton_signed_smoothmax_available():
        raise RuntimeError("Triton is unavailable.")
    if projected_target_bhnr.ndim != 4 or weighted_source_bhnr.ndim != 4:
        raise ValueError("expected [batch, heads, nodes, rank] tensors")
    if projected_target_bhnr.shape != weighted_source_bhnr.shape:
        raise ValueError("projected_target and weighted_source must have the same shape")
    batch, heads, num_nodes, rank_dim = projected_target_bhnr.shape
    if heads <= 0 or heads > 4:
        raise ValueError("Triton signed_smoothmax path currently supports 1-4 heads.")
    if projected_target_bhnr.device.type != "cuda":
        raise ValueError("Triton signed_smoothmax path requires CUDA tensors.")

    target = projected_target_bhnr.contiguous().to(dtype=torch.float32)
    source = weighted_source_bhnr.contiguous().to(dtype=torch.float32)
    bias = None if biases is None or biases.numel() == 0 else biases.contiguous().to(dtype=torch.float32, device=target.device)
    out = torch.empty((batch, num_nodes, num_nodes), device=target.device, dtype=torch.float32)

    grid = (triton.cdiv(num_nodes, 32), triton.cdiv(num_nodes, 32), batch)
    _multihead_signed_smoothmax_scores_kernel[grid](
        target,
        source,
        bias if bias is not None else target,
        out,
        target.stride(0),
        target.stride(1),
        target.stride(2),
        target.stride(3),
        source.stride(0),
        source.stride(1),
        source.stride(2),
        source.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_nodes,
        rank_dim,
        has_bias=bias is not None,
        heads=heads,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
        num_stages=2,
    )
    return out


def multihead_signed_smoothmax_scores_tile(
    projected_target_bhnr: torch.Tensor,
    weighted_source_tile_bhnr: torch.Tensor,
    biases: torch.Tensor | None = None,
) -> torch.Tensor:
    if not triton_signed_smoothmax_available():
        raise RuntimeError("Triton is unavailable.")
    if projected_target_bhnr.ndim != 4 or weighted_source_tile_bhnr.ndim != 4:
        raise ValueError("expected [batch, heads, nodes, rank] tensors")
    batch, heads, num_nodes, rank_dim = projected_target_bhnr.shape
    batch2, heads2, tile_nodes, rank_dim2 = weighted_source_tile_bhnr.shape
    if (batch, heads, rank_dim) != (batch2, heads2, rank_dim2):
        raise ValueError("projected_target and weighted_source_tile must align")
    if heads <= 0 or heads > 4:
        raise ValueError("Triton signed_smoothmax path currently supports 1-4 heads.")
    if projected_target_bhnr.device.type != "cuda":
        raise ValueError("Triton signed_smoothmax path requires CUDA tensors.")

    target = projected_target_bhnr.contiguous().to(dtype=torch.float32)
    source = weighted_source_tile_bhnr.contiguous().to(dtype=torch.float32)
    bias = None if biases is None or biases.numel() == 0 else biases.contiguous().to(dtype=torch.float32, device=target.device)
    out = torch.empty((batch, num_nodes, tile_nodes), device=target.device, dtype=torch.float32)
    grid = (triton.cdiv(num_nodes, 32), triton.cdiv(tile_nodes, 32), batch)
    _multihead_signed_smoothmax_scores_tile_kernel[grid](
        target,
        source,
        bias if bias is not None else target,
        out,
        target.stride(0),
        target.stride(1),
        target.stride(2),
        target.stride(3),
        source.stride(0),
        source.stride(1),
        source.stride(2),
        source.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_nodes,
        tile_nodes,
        rank_dim,
        has_bias=bias is not None,
        heads=heads,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
        num_stages=2,
    )
    return out


def multihead_signed_smoothmax_scores_and_head_grads_tile(
    projected_target_bhnr: torch.Tensor,
    weighted_source_tile_bhnr: torch.Tensor,
    source_start: int,
    biases: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not triton_signed_smoothmax_available():
        raise RuntimeError("Triton is unavailable.")
    if projected_target_bhnr.ndim != 4 or weighted_source_tile_bhnr.ndim != 4:
        raise ValueError("expected [batch, heads, nodes, rank] tensors")
    batch, heads, num_nodes, rank_dim = projected_target_bhnr.shape
    batch2, heads2, tile_nodes, rank_dim2 = weighted_source_tile_bhnr.shape
    if (batch, heads, rank_dim) != (batch2, heads2, rank_dim2):
        raise ValueError("projected_target and weighted_source_tile must align")
    if heads <= 0 or heads > 4:
        raise ValueError("Triton signed_smoothmax path currently supports 1-4 heads.")
    if projected_target_bhnr.device.type != "cuda":
        raise ValueError("Triton signed_smoothmax path requires CUDA tensors.")

    target = projected_target_bhnr.contiguous().to(dtype=torch.float32)
    source = weighted_source_tile_bhnr.contiguous().to(dtype=torch.float32)
    bias = None if biases is None or biases.numel() == 0 else biases.contiguous().to(dtype=torch.float32, device=target.device)
    score_out = torch.empty((batch, num_nodes, tile_nodes), device=target.device, dtype=torch.float32)
    grad_out = torch.empty((batch, heads, num_nodes, tile_nodes), device=target.device, dtype=torch.float32)
    grid = (triton.cdiv(num_nodes, 32), triton.cdiv(tile_nodes, 32), batch)
    _multihead_signed_smoothmax_scores_and_head_grads_tile_kernel[grid](
        target,
        source,
        bias if bias is not None else target,
        score_out,
        grad_out,
        target.stride(0),
        target.stride(1),
        target.stride(2),
        target.stride(3),
        source.stride(0),
        source.stride(1),
        source.stride(2),
        source.stride(3),
        score_out.stride(0),
        score_out.stride(1),
        score_out.stride(2),
        grad_out.stride(0),
        grad_out.stride(1),
        grad_out.stride(2),
        grad_out.stride(3),
        source_start,
        num_nodes,
        tile_nodes,
        rank_dim,
        has_bias=bias is not None,
        heads=heads,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
        num_stages=2,
    )
    return score_out, grad_out.permute(1, 0, 2, 3).contiguous()


def multihead_signed_smoothmax_head_grads(
    projected_target_bhnr: torch.Tensor,
    weighted_source_tile_bhnr: torch.Tensor,
    source_start: int,
    biases: torch.Tensor | None = None,
) -> torch.Tensor:
    if not triton_signed_smoothmax_available():
        raise RuntimeError("Triton is unavailable.")
    if projected_target_bhnr.ndim != 4 or weighted_source_tile_bhnr.ndim != 4:
        raise ValueError("expected [batch, heads, nodes, rank] tensors")
    batch, heads, num_nodes, rank_dim = projected_target_bhnr.shape
    batch2, heads2, tile_nodes, rank_dim2 = weighted_source_tile_bhnr.shape
    if (batch, heads, rank_dim) != (batch2, heads2, rank_dim2):
        raise ValueError("projected_target and weighted_source_tile must align")
    if heads <= 0 or heads > 4:
        raise ValueError("Triton signed_smoothmax path currently supports 1-4 heads.")
    target = projected_target_bhnr.contiguous().to(dtype=torch.float32)
    source = weighted_source_tile_bhnr.contiguous().to(dtype=torch.float32)
    bias = None if biases is None or biases.numel() == 0 else biases.contiguous().to(dtype=torch.float32, device=target.device)
    out = torch.empty((batch, heads, num_nodes, tile_nodes), device=target.device, dtype=torch.float32)
    grid = (triton.cdiv(num_nodes, 32), triton.cdiv(tile_nodes, 32), batch)
    _multihead_signed_smoothmax_head_grads_kernel[grid](
        target,
        source,
        bias if bias is not None else target,
        out,
        target.stride(0),
        target.stride(1),
        target.stride(2),
        target.stride(3),
        source.stride(0),
        source.stride(1),
        source.stride(2),
        source.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        source_start,
        num_nodes,
        tile_nodes,
        rank_dim,
        has_bias=bias is not None,
        heads=heads,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
        num_stages=2,
    )
    return out.permute(1, 0, 2, 3).contiguous()


def signed_abs_softmax_edge_dot_tile(
    scores_bmn: torch.Tensor,
    grad_edges_bmn: torch.Tensor,
    row_max_bm: torch.Tensor,
    row_denom_bm: torch.Tensor,
    source_start: int,
) -> torch.Tensor:
    if not triton_signed_smoothmax_available():
        raise RuntimeError("Triton is unavailable.")
    if scores_bmn.ndim != 3 or grad_edges_bmn.ndim != 3:
        raise ValueError("expected scores and grad_edges [batch, nodes, tile] tensors")
    if row_max_bm.ndim != 2 or row_denom_bm.ndim != 2:
        raise ValueError("expected row_max and row_denom [batch, nodes] tensors")
    if scores_bmn.shape != grad_edges_bmn.shape:
        raise ValueError("scores and grad_edges must have the same shape")
    if scores_bmn.shape[:2] != row_max_bm.shape or row_max_bm.shape != row_denom_bm.shape:
        raise ValueError("row stats must match [batch, nodes]")
    if scores_bmn.device.type != "cuda":
        raise ValueError("Triton signed_smoothmax path requires CUDA tensors.")

    scores = scores_bmn.contiguous().to(dtype=torch.float32)
    grad_edges = grad_edges_bmn.contiguous().to(dtype=torch.float32)
    row_max = row_max_bm.contiguous().to(dtype=torch.float32)
    row_denom = row_denom_bm.contiguous().to(dtype=torch.float32)
    batch, num_nodes, tile_nodes = scores.shape
    out = torch.empty((batch, num_nodes), device=scores.device, dtype=torch.float32)
    grid = (triton.cdiv(num_nodes, 64), batch)
    _signed_abs_softmax_edge_dot_tile_kernel[grid](
        scores,
        grad_edges,
        row_max,
        row_denom,
        out,
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        grad_edges.stride(0),
        grad_edges.stride(1),
        grad_edges.stride(2),
        row_max.stride(0),
        row_max.stride(1),
        out.stride(0),
        out.stride(1),
        source_start,
        num_nodes,
        tile_nodes,
        BLOCK_M=64,
        BLOCK_N=32,
        num_warps=4,
        num_stages=2,
    )
    return out


def diagonal_signed_smoothmax_scores_tile(
    flat_val_bnd: torch.Tensor,
    core_hd: torch.Tensor,
    source_start: int,
    tile_nodes: int,
    biases: torch.Tensor | None = None,
) -> torch.Tensor:
    if not triton_signed_smoothmax_available():
        raise RuntimeError("Triton is unavailable.")
    if flat_val_bnd.ndim != 3 or core_hd.ndim != 2:
        raise ValueError("expected flat_val [batch, nodes, dim] and core [heads, dim]")
    batch, num_nodes, dim = flat_val_bnd.shape
    heads, dim2 = core_hd.shape
    if dim != dim2:
        raise ValueError("flat_val and core must align on dim")
    if heads <= 0 or heads > 4:
        raise ValueError("Triton signed_smoothmax path currently supports 1-4 heads.")
    if flat_val_bnd.device.type != "cuda":
        raise ValueError("Triton signed_smoothmax path requires CUDA tensors.")

    flat_val = flat_val_bnd.contiguous().to(dtype=torch.float32)
    core = core_hd.contiguous().to(dtype=torch.float32)
    bias = None if biases is None or biases.numel() == 0 else biases.contiguous().to(dtype=torch.float32, device=flat_val.device)
    out = torch.empty((batch, num_nodes, tile_nodes), device=flat_val.device, dtype=torch.float32)
    grid = (triton.cdiv(num_nodes, 32), triton.cdiv(tile_nodes, 32), batch)
    _diagonal_signed_smoothmax_scores_tile_kernel[grid](
        flat_val,
        core,
        bias if bias is not None else flat_val,
        out,
        flat_val.stride(0),
        flat_val.stride(1),
        flat_val.stride(2),
        core.stride(0),
        core.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        source_start,
        num_nodes,
        tile_nodes,
        dim,
        has_bias=bias is not None,
        heads=heads,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
        num_stages=2,
    )
    return out


def diagonal_signed_smoothmax_scores_and_head_grads_tile(
    flat_val_bnd: torch.Tensor,
    core_hd: torch.Tensor,
    source_start: int,
    tile_nodes: int,
    biases: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not triton_signed_smoothmax_available():
        raise RuntimeError("Triton is unavailable.")
    if flat_val_bnd.ndim != 3 or core_hd.ndim != 2:
        raise ValueError("expected flat_val [batch, nodes, dim] and core [heads, dim]")
    batch, num_nodes, dim = flat_val_bnd.shape
    heads, dim2 = core_hd.shape
    if dim != dim2:
        raise ValueError("flat_val and core must align on dim")
    if heads <= 0 or heads > 4:
        raise ValueError("Triton signed_smoothmax path currently supports 1-4 heads.")
    if flat_val_bnd.device.type != "cuda":
        raise ValueError("Triton signed_smoothmax path requires CUDA tensors.")

    flat_val = flat_val_bnd.contiguous().to(dtype=torch.float32)
    core = core_hd.contiguous().to(dtype=torch.float32)
    bias = None if biases is None or biases.numel() == 0 else biases.contiguous().to(dtype=torch.float32, device=flat_val.device)
    score_out = torch.empty((batch, num_nodes, tile_nodes), device=flat_val.device, dtype=torch.float32)
    grad_out = torch.empty((batch, heads, num_nodes, tile_nodes), device=flat_val.device, dtype=torch.float32)
    grid = (triton.cdiv(num_nodes, 32), triton.cdiv(tile_nodes, 32), batch)
    _diagonal_signed_smoothmax_scores_and_head_grads_tile_kernel[grid](
        flat_val,
        core,
        bias if bias is not None else flat_val,
        score_out,
        grad_out,
        flat_val.stride(0),
        flat_val.stride(1),
        flat_val.stride(2),
        core.stride(0),
        core.stride(1),
        score_out.stride(0),
        score_out.stride(1),
        score_out.stride(2),
        grad_out.stride(0),
        grad_out.stride(1),
        grad_out.stride(2),
        grad_out.stride(3),
        source_start,
        num_nodes,
        tile_nodes,
        dim,
        has_bias=bias is not None,
        heads=heads,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
        num_stages=2,
    )
    return score_out, grad_out.permute(1, 0, 2, 3).contiguous()


def diagonal_signed_smoothmax_backward_tile_accumulate(
    flat_val_bnd: torch.Tensor,
    core_hd: torch.Tensor,
    source_start: int,
    grad_edges_bmn: torch.Tensor,
    row_max_bm: torch.Tensor,
    row_denom_bm: torch.Tensor,
    edge_dot_bm: torch.Tensor,
    grad_layer_bnd: torch.Tensor,
    grad_weights_hd: torch.Tensor,
    grad_biases_h: torch.Tensor | None = None,
    biases: torch.Tensor | None = None,
) -> None:
    if not triton_signed_smoothmax_available():
        raise RuntimeError("Triton is unavailable.")
    if flat_val_bnd.ndim != 3 or core_hd.ndim != 2:
        raise ValueError("expected flat_val [batch, nodes, dim] and core [heads, dim]")
    batch, num_nodes, dim = flat_val_bnd.shape
    heads, dim2 = core_hd.shape
    if dim != dim2:
        raise ValueError("flat_val and core must align on dim")
    tile_nodes = int(grad_edges_bmn.shape[-1])
    if grad_edges_bmn.shape != (batch, num_nodes, tile_nodes):
        raise ValueError("grad_edges must be [batch, nodes, tile_nodes]")
    if row_max_bm.shape != (batch, num_nodes) or row_denom_bm.shape != (batch, num_nodes):
        raise ValueError("row stats must be [batch, nodes]")
    if edge_dot_bm.shape != (batch, num_nodes):
        raise ValueError("edge_dot must be [batch, nodes]")
    if grad_layer_bnd.shape != (batch, num_nodes, dim):
        raise ValueError("grad_layer must be [batch, nodes, dim]")
    if grad_weights_hd.shape != (heads, dim):
        raise ValueError("grad_weights must be [heads, dim]")
    if heads <= 0 or heads > 4:
        raise ValueError("Triton signed_smoothmax path currently supports 1-4 heads.")
    if flat_val_bnd.device.type != "cuda":
        raise ValueError("Triton signed_smoothmax path requires CUDA tensors.")

    flat_val = flat_val_bnd.contiguous().to(dtype=torch.float32)
    core = core_hd.contiguous().to(dtype=torch.float32)
    grad_edges = grad_edges_bmn.contiguous().to(dtype=torch.float32)
    row_max = row_max_bm.contiguous().to(dtype=torch.float32)
    row_denom = row_denom_bm.contiguous().to(dtype=torch.float32)
    edge_dot = edge_dot_bm.contiguous().to(dtype=torch.float32)
    grad_layer = grad_layer_bnd.contiguous().to(dtype=torch.float32)
    grad_weights = grad_weights_hd.contiguous().to(dtype=torch.float32)
    grad_bias = None if grad_biases_h is None or grad_biases_h.numel() == 0 else grad_biases_h.contiguous().to(dtype=torch.float32)
    bias = None if biases is None or biases.numel() == 0 else biases.contiguous().to(dtype=torch.float32, device=flat_val.device)
    block_m, block_n, block_k, block_d = _diag_backward_block_config()
    grid = (triton.cdiv(num_nodes, block_m), triton.cdiv(tile_nodes, block_n), batch)
    _diagonal_signed_smoothmax_backward_tile_kernel[grid](
        flat_val,
        core,
        bias if bias is not None else flat_val,
        grad_edges,
        row_max,
        row_denom,
        edge_dot,
        grad_layer,
        grad_weights,
        grad_bias if grad_bias is not None else grad_weights,
        flat_val.stride(0),
        flat_val.stride(1),
        flat_val.stride(2),
        core.stride(0),
        core.stride(1),
        grad_edges.stride(0),
        grad_edges.stride(1),
        grad_edges.stride(2),
        row_max.stride(0),
        row_max.stride(1),
        edge_dot.stride(0),
        edge_dot.stride(1),
        grad_layer.stride(0),
        grad_layer.stride(1),
        grad_layer.stride(2),
        grad_weights.stride(0),
        grad_weights.stride(1),
        source_start,
        num_nodes,
        tile_nodes,
        dim,
        has_bias=grad_bias is not None and bias is not None,
        heads=heads,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )


def lowrank_signed_smoothmax_backward_tile_accumulate(
    projected_target_hbnr: torch.Tensor,
    projected_source_tile_hbnr: torch.Tensor,
    weighted_source_tile_hbnr: torch.Tensor,
    core_hr: torch.Tensor,
    source_start: int,
    grad_edges_bmn: torch.Tensor,
    row_max_bm: torch.Tensor,
    row_denom_bm: torch.Tensor,
    edge_dot_bm: torch.Tensor,
    grad_projected_target_hbnr: torch.Tensor,
    grad_projected_source_hbnr: torch.Tensor,
    grad_core_weights_hr: torch.Tensor,
    grad_biases_h: torch.Tensor | None = None,
    biases: torch.Tensor | None = None,
    grad_source_row_offset: int | None = None,
) -> None:
    if not triton_signed_smoothmax_available():
        raise RuntimeError("Triton is unavailable.")
    if projected_target_hbnr.ndim != 4 or projected_source_tile_hbnr.ndim != 4 or weighted_source_tile_hbnr.ndim != 4:
        raise ValueError("expected [heads, batch, nodes, rank] tensors")
    heads, batch, num_nodes, rank_dim = projected_target_hbnr.shape
    if projected_source_tile_hbnr.shape[:2] != (heads, batch) or weighted_source_tile_hbnr.shape[:2] != (heads, batch):
        raise ValueError("source tensors must align on [heads, batch]")
    tile_nodes = int(projected_source_tile_hbnr.shape[2])
    if projected_source_tile_hbnr.shape != (heads, batch, tile_nodes, rank_dim):
        raise ValueError("projected_source_tile must be [heads, batch, tile_nodes, rank]")
    if weighted_source_tile_hbnr.shape != (heads, batch, tile_nodes, rank_dim):
        raise ValueError("weighted_source_tile must be [heads, batch, tile_nodes, rank]")
    if core_hr.shape != (heads, rank_dim):
        raise ValueError("core must be [heads, rank]")
    if grad_edges_bmn.shape != (batch, num_nodes, tile_nodes):
        raise ValueError("grad_edges must be [batch, nodes, tile_nodes]")
    if row_max_bm.shape != (batch, num_nodes) or row_denom_bm.shape != (batch, num_nodes):
        raise ValueError("row stats must be [batch, nodes]")
    if edge_dot_bm.shape != (batch, num_nodes):
        raise ValueError("edge_dot must be [batch, nodes]")
    if grad_projected_target_hbnr.shape != (heads, batch, num_nodes, rank_dim):
        raise ValueError("grad_projected_target must be [heads, batch, nodes, rank]")
    if grad_projected_source_hbnr.shape[0:2] != (heads, batch) or grad_projected_source_hbnr.shape[3] != rank_dim:
        raise ValueError("grad_projected_source must be [heads, batch, nodes, rank]")
    if grad_source_row_offset is None:
        grad_source_row_offset = source_start
    expected_source_rows = tile_nodes if grad_source_row_offset == 0 else num_nodes
    if grad_projected_source_hbnr.shape[2] != expected_source_rows:
        raise ValueError("grad_projected_source row dimension does not match grad_source_row_offset policy")
    if grad_core_weights_hr.shape != (heads, rank_dim):
        raise ValueError("grad_core_weights must be [heads, rank]")
    if heads <= 0 or heads > 4:
        raise ValueError("Triton signed_smoothmax path currently supports 1-4 heads.")
    if projected_target_hbnr.device.type != "cuda":
        raise ValueError("Triton signed_smoothmax path requires CUDA tensors.")

    target = projected_target_hbnr.contiguous()
    source_proj = projected_source_tile_hbnr.contiguous()
    source_weighted = weighted_source_tile_hbnr.contiguous()
    core = core_hr.contiguous()
    grad_edges = grad_edges_bmn.contiguous().to(dtype=torch.float32)
    row_max = row_max_bm.contiguous().to(dtype=torch.float32)
    row_denom = row_denom_bm.contiguous().to(dtype=torch.float32)
    edge_dot = edge_dot_bm.contiguous().to(dtype=torch.float32)
    grad_target = grad_projected_target_hbnr.contiguous()
    grad_source = grad_projected_source_hbnr.contiguous()
    grad_core = grad_core_weights_hr.contiguous()
    grad_bias = None if grad_biases_h is None or grad_biases_h.numel() == 0 else grad_biases_h.contiguous()
    bias = None if biases is None or biases.numel() == 0 else biases.contiguous().to(dtype=torch.float32, device=target.device)
    block_m, block_n, block_k, block_r = _lowrank_backward_block_config()
    grid = (triton.cdiv(num_nodes, block_m), triton.cdiv(tile_nodes, block_n), batch)
    _lowrank_signed_smoothmax_backward_tile_kernel[grid](
        target,
        source_proj,
        source_weighted,
        core,
        bias if bias is not None else target,
        grad_edges,
        row_max,
        row_denom,
        edge_dot,
        grad_target,
        grad_source,
        grad_core,
        grad_bias if grad_bias is not None else grad_core,
        target.stride(0),
        target.stride(1),
        target.stride(2),
        target.stride(3),
        source_proj.stride(0),
        source_proj.stride(1),
        source_proj.stride(2),
        source_proj.stride(3),
        source_weighted.stride(0),
        source_weighted.stride(1),
        source_weighted.stride(2),
        source_weighted.stride(3),
        core.stride(0),
        core.stride(1),
        grad_edges.stride(0),
        grad_edges.stride(1),
        grad_edges.stride(2),
        row_max.stride(0),
        row_max.stride(1),
        edge_dot.stride(0),
        edge_dot.stride(1),
        grad_target.stride(0),
        grad_target.stride(1),
        grad_target.stride(2),
        grad_target.stride(3),
        grad_source.stride(0),
        grad_source.stride(1),
        grad_source.stride(2),
        grad_source.stride(3),
        grad_core.stride(0),
        grad_core.stride(1),
        source_start,
        grad_source_row_offset,
        num_nodes,
        tile_nodes,
        rank_dim,
        has_bias=grad_bias is not None and bias is not None,
        heads=heads,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCK_R=block_r,
        num_warps=4,
        num_stages=2,
    )


if triton is not None and tl is not None:
    @triton.jit
    def _diagonal_signed_smoothmax_backward_target_owned_kernel(
        flat_ptr,
        core_ptr,
        bias_ptr,
        proj_state_ptr,
        proj_val_ptr,
        grad_state_ptr,
        grad_val_ptr,
        row_max_ptr,
        row_denom_ptr,
        edge_dot_ptr,
        grad_layer_ptr,
        grad_weights_partial_ptr,
        grad_bias_partial_ptr,
        stride_flat_b,
        stride_flat_n,
        stride_flat_d,
        stride_core_h,
        stride_core_d,
        stride_proj_state_b,
        stride_proj_state_n,
        stride_proj_val_b,
        stride_proj_val_n,
        stride_proj_val_d,
        stride_grad_state_b,
        stride_grad_state_n,
        stride_grad_val_b,
        stride_grad_val_n,
        stride_grad_val_d,
        stride_row_b,
        stride_row_m,
        stride_edge_dot_b,
        stride_edge_dot_m,
        stride_grad_layer_b,
        stride_grad_layer_n,
        stride_grad_layer_d,
        stride_grad_weights_partial_b,
        stride_grad_weights_partial_rb,
        stride_grad_weights_partial_sb,
        stride_grad_weights_partial_h,
        stride_grad_weights_partial_d,
        stride_grad_bias_partial_b,
        stride_grad_bias_partial_rb,
        stride_grad_bias_partial_sb,
        stride_grad_bias_partial_h,
        num_nodes,
        dim,
        has_bias: tl.constexpr,
        heads: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < num_nodes
        grad_state = tl.load(
            grad_state_ptr + pid_b * stride_grad_state_b + offs_m * stride_grad_state_n,
            mask=mask_m,
            other=0.0,
        )
        row_max = tl.load(row_max_ptr + pid_b * stride_row_b + offs_m * stride_row_m, mask=mask_m, other=-float("inf"))
        row_denom = tl.load(row_denom_ptr + pid_b * stride_row_b + offs_m * stride_row_m, mask=mask_m, other=1.0)
        edge_dot = tl.load(edge_dot_ptr + pid_b * stride_edge_dot_b + offs_m * stride_edge_dot_m, mask=mask_m, other=0.0)
        bias0 = tl.zeros((), dtype=tl.float32)
        bias1 = tl.zeros((), dtype=tl.float32)
        bias2 = tl.zeros((), dtype=tl.float32)
        bias3 = tl.zeros((), dtype=tl.float32)

        for source_base in range(0, num_nodes, BLOCK_N):
            source_block = source_base // BLOCK_N
            offs_n = source_base + tl.arange(0, BLOCK_N)
            mask_n = offs_n < num_nodes
            causal = offs_n[None, :] <= offs_m[:, None]
            full_mask = mask_m[:, None] & mask_n[None, :] & causal
            score_0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            score_1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            score_2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            score_3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            grad_edges = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for k_start in range(0, dim, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)
                mask_k = offs_k < dim
                target_k = tl.load(
                    flat_ptr + pid_b * stride_flat_b + offs_m[:, None] * stride_flat_n + offs_k[None, :] * stride_flat_d,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )
                source_k = tl.load(
                    flat_ptr + pid_b * stride_flat_b + offs_n[:, None] * stride_flat_n + offs_k[None, :] * stride_flat_d,
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                grad_val_k = tl.load(
                    grad_val_ptr + pid_b * stride_grad_val_b + offs_m[:, None] * stride_grad_val_n + offs_k[None, :] * stride_grad_val_d,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )
                proj_val_k = tl.load(
                    proj_val_ptr + pid_b * stride_proj_val_b + offs_n[:, None] * stride_proj_val_n + offs_k[None, :] * stride_proj_val_d,
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                grad_edges += tl.dot(grad_val_k, tl.trans(proj_val_k), allow_tf32=False, out_dtype=tl.float32)
                if heads > 0:
                    core0 = tl.load(core_ptr + 0 * stride_core_h + offs_k * stride_core_d, mask=mask_k, other=0.0)
                    score_0 += tl.dot(target_k * core0[None, :], tl.trans(source_k), allow_tf32=False, out_dtype=tl.float32)
                if heads > 1:
                    core1 = tl.load(core_ptr + 1 * stride_core_h + offs_k * stride_core_d, mask=mask_k, other=0.0)
                    score_1 += tl.dot(target_k * core1[None, :], tl.trans(source_k), allow_tf32=False, out_dtype=tl.float32)
                if heads > 2:
                    core2 = tl.load(core_ptr + 2 * stride_core_h + offs_k * stride_core_d, mask=mask_k, other=0.0)
                    score_2 += tl.dot(target_k * core2[None, :], tl.trans(source_k), allow_tf32=False, out_dtype=tl.float32)
                if heads > 3:
                    core3 = tl.load(core_ptr + 3 * stride_core_h + offs_k * stride_core_d, mask=mask_k, other=0.0)
                    score_3 += tl.dot(target_k * core3[None, :], tl.trans(source_k), allow_tf32=False, out_dtype=tl.float32)

            source_state = tl.load(
                proj_state_ptr + pid_b * stride_proj_state_b + offs_n * stride_proj_state_n,
                mask=mask_n,
                other=0.0,
            )
            grad_edges += grad_state[:, None] * source_state[None, :]

            if has_bias:
                if heads > 0:
                    score_0 += tl.load(bias_ptr + 0)
                if heads > 1:
                    score_1 += tl.load(bias_ptr + 1)
                if heads > 2:
                    score_2 += tl.load(bias_ptr + 2)
                if heads > 3:
                    score_3 += tl.load(bias_ptr + 3)

            neg_large = -1.0e30
            max_abs = tl.where(full_mask, tl.abs(score_0), neg_large)
            if heads > 1:
                max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_1), neg_large))
            if heads > 2:
                max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_2), neg_large))
            if heads > 3:
                max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_3), neg_large))

            denom_heads = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            numer = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            p0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            p1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            p2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            p3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            if heads > 0:
                p0 = tl.where(full_mask, tl.exp(tl.abs(score_0) - max_abs), 0.0)
                denom_heads += p0
                numer += score_0 * p0
            if heads > 1:
                p1 = tl.where(full_mask, tl.exp(tl.abs(score_1) - max_abs), 0.0)
                denom_heads += p1
                numer += score_1 * p1
            if heads > 2:
                p2 = tl.where(full_mask, tl.exp(tl.abs(score_2) - max_abs), 0.0)
                denom_heads += p2
                numer += score_2 * p2
            if heads > 3:
                p3 = tl.where(full_mask, tl.exp(tl.abs(score_3) - max_abs), 0.0)
                denom_heads += p3
                numer += score_3 * p3

            safe_head_denom = tl.maximum(denom_heads, 1.0e-20)
            combined = numer / safe_head_denom
            probs = tl.where(full_mask, tl.exp(tl.abs(combined) - row_max[:, None]) / tl.maximum(row_denom[:, None], 1.0e-20), 0.0)
            signs = tl.where(combined > 0, 1.0, tl.where(combined < 0, -1.0, 0.0))
            grad_scores = tl.where(full_mask, signs * probs * (signs * grad_edges - edge_dot[:, None]), 0.0)

            sign0 = tl.where(score_0 > 0, 1.0, tl.where(score_0 < 0, -1.0, 0.0))
            sign1 = tl.where(score_1 > 0, 1.0, tl.where(score_1 < 0, -1.0, 0.0))
            sign2 = tl.where(score_2 > 0, 1.0, tl.where(score_2 < 0, -1.0, 0.0))
            sign3 = tl.where(score_3 > 0, 1.0, tl.where(score_3 < 0, -1.0, 0.0))
            g0 = tl.where(full_mask, (p0 / safe_head_denom) * (1.0 + sign0 * (score_0 - combined)), 0.0) if heads > 0 else tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            g1 = tl.where(full_mask, (p1 / safe_head_denom) * (1.0 + sign1 * (score_1 - combined)), 0.0) if heads > 1 else tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            g2 = tl.where(full_mask, (p2 / safe_head_denom) * (1.0 + sign2 * (score_2 - combined)), 0.0) if heads > 2 else tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            g3 = tl.where(full_mask, (p3 / safe_head_denom) * (1.0 + sign3 * (score_3 - combined)), 0.0) if heads > 3 else tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            if has_bias:
                if heads > 0:
                    bias0 += tl.sum(grad_scores * g0)
                if heads > 1:
                    bias1 += tl.sum(grad_scores * g1)
                if heads > 2:
                    bias2 += tl.sum(grad_scores * g2)
                if heads > 3:
                    bias3 += tl.sum(grad_scores * g3)
            for d_start in range(0, dim, BLOCK_D):
                offs_d = d_start + tl.arange(0, BLOCK_D)
                mask_d = offs_d < dim
                target_d = tl.load(
                    flat_ptr + pid_b * stride_flat_b + offs_m[:, None] * stride_flat_n + offs_d[None, :] * stride_flat_d,
                    mask=mask_m[:, None] & mask_d[None, :],
                    other=0.0,
                )
                source_d = tl.load(
                    flat_ptr + pid_b * stride_flat_b + offs_n[:, None] * stride_flat_n + offs_d[None, :] * stride_flat_d,
                    mask=mask_n[:, None] & mask_d[None, :],
                    other=0.0,
                )
                grad_layer_ptrs = grad_layer_ptr + pid_b * stride_grad_layer_b + offs_m[:, None] * stride_grad_layer_n + offs_d[None, :] * stride_grad_layer_d
                grad_layer_acc = tl.load(grad_layer_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
                if heads > 0:
                    source_head = tl.dot(tl.trans(grad_scores * g0), target_d, allow_tf32=False, out_dtype=tl.float32)
                    core0d = tl.load(core_ptr + 0 * stride_core_h + offs_d * stride_core_d, mask=mask_d, other=0.0)
                    grad_layer_acc += tl.dot(grad_scores * g0, source_d * core0d[None, :], allow_tf32=False, out_dtype=tl.float32)
                    tl.store(
                        grad_weights_partial_ptr + pid_b * stride_grad_weights_partial_b + pid_m * stride_grad_weights_partial_rb + source_block * stride_grad_weights_partial_sb + 0 * stride_grad_weights_partial_h + offs_d * stride_grad_weights_partial_d,
                        tl.sum(source_head * source_d, axis=0),
                        mask=mask_d,
                    )
                if heads > 1:
                    source_head = tl.dot(tl.trans(grad_scores * g1), target_d, allow_tf32=False, out_dtype=tl.float32)
                    core1d = tl.load(core_ptr + 1 * stride_core_h + offs_d * stride_core_d, mask=mask_d, other=0.0)
                    grad_layer_acc += tl.dot(grad_scores * g1, source_d * core1d[None, :], allow_tf32=False, out_dtype=tl.float32)
                    tl.store(
                        grad_weights_partial_ptr + pid_b * stride_grad_weights_partial_b + pid_m * stride_grad_weights_partial_rb + source_block * stride_grad_weights_partial_sb + 1 * stride_grad_weights_partial_h + offs_d * stride_grad_weights_partial_d,
                        tl.sum(source_head * source_d, axis=0),
                        mask=mask_d,
                    )
                if heads > 2:
                    source_head = tl.dot(tl.trans(grad_scores * g2), target_d, allow_tf32=False, out_dtype=tl.float32)
                    core2d = tl.load(core_ptr + 2 * stride_core_h + offs_d * stride_core_d, mask=mask_d, other=0.0)
                    grad_layer_acc += tl.dot(grad_scores * g2, source_d * core2d[None, :], allow_tf32=False, out_dtype=tl.float32)
                    tl.store(
                        grad_weights_partial_ptr + pid_b * stride_grad_weights_partial_b + pid_m * stride_grad_weights_partial_rb + source_block * stride_grad_weights_partial_sb + 2 * stride_grad_weights_partial_h + offs_d * stride_grad_weights_partial_d,
                        tl.sum(source_head * source_d, axis=0),
                        mask=mask_d,
                    )
                if heads > 3:
                    source_head = tl.dot(tl.trans(grad_scores * g3), target_d, allow_tf32=False, out_dtype=tl.float32)
                    core3d = tl.load(core_ptr + 3 * stride_core_h + offs_d * stride_core_d, mask=mask_d, other=0.0)
                    grad_layer_acc += tl.dot(grad_scores * g3, source_d * core3d[None, :], allow_tf32=False, out_dtype=tl.float32)
                    tl.store(
                        grad_weights_partial_ptr + pid_b * stride_grad_weights_partial_b + pid_m * stride_grad_weights_partial_rb + source_block * stride_grad_weights_partial_sb + 3 * stride_grad_weights_partial_h + offs_d * stride_grad_weights_partial_d,
                        tl.sum(source_head * source_d, axis=0),
                        mask=mask_d,
                    )
                tl.store(grad_layer_ptrs, grad_layer_acc, mask=mask_m[:, None] & mask_d[None, :])

        if has_bias:
            if heads > 0:
                tl.store(grad_bias_partial_ptr + pid_b * stride_grad_bias_partial_b + pid_m * stride_grad_bias_partial_rb + 0 * stride_grad_bias_partial_h, bias0)
            if heads > 1:
                tl.store(grad_bias_partial_ptr + pid_b * stride_grad_bias_partial_b + pid_m * stride_grad_bias_partial_rb + 1 * stride_grad_bias_partial_h, bias1)
            if heads > 2:
                tl.store(grad_bias_partial_ptr + pid_b * stride_grad_bias_partial_b + pid_m * stride_grad_bias_partial_rb + 2 * stride_grad_bias_partial_h, bias2)
            if heads > 3:
                tl.store(grad_bias_partial_ptr + pid_b * stride_grad_bias_partial_b + pid_m * stride_grad_bias_partial_rb + 3 * stride_grad_bias_partial_h, bias3)

    @triton.jit
    def _diagonal_signed_smoothmax_backward_source_owned_kernel(
        flat_ptr,
        core_ptr,
        bias_ptr,
        proj_state_ptr,
        proj_val_ptr,
        grad_state_ptr,
        grad_val_ptr,
        row_max_ptr,
        row_denom_ptr,
        edge_dot_ptr,
        grad_layer_ptr,
        stride_flat_b,
        stride_flat_n,
        stride_flat_d,
        stride_core_h,
        stride_core_d,
        stride_proj_state_b,
        stride_proj_state_n,
        stride_proj_val_b,
        stride_proj_val_n,
        stride_proj_val_d,
        stride_grad_state_b,
        stride_grad_state_n,
        stride_grad_val_b,
        stride_grad_val_n,
        stride_grad_val_d,
        stride_row_b,
        stride_row_m,
        stride_edge_dot_b,
        stride_edge_dot_m,
        stride_grad_layer_b,
        stride_grad_layer_n,
        stride_grad_layer_d,
        num_nodes,
        dim,
        has_bias: tl.constexpr,
        heads: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_b = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < num_nodes
        source_state = tl.load(
            proj_state_ptr + pid_b * stride_proj_state_b + offs_n * stride_proj_state_n,
            mask=mask_n,
            other=0.0,
        )

        for target_base in range(0, num_nodes, BLOCK_M):
            offs_m = target_base + tl.arange(0, BLOCK_M)
            mask_m = offs_m < num_nodes
            causal = offs_n[None, :] <= offs_m[:, None]
            full_mask = mask_m[:, None] & mask_n[None, :] & causal
            score_0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            score_1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            score_2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            score_3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            grad_edges = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            grad_state = tl.load(
                grad_state_ptr + pid_b * stride_grad_state_b + offs_m * stride_grad_state_n,
                mask=mask_m,
                other=0.0,
            )
            row_max = tl.load(row_max_ptr + pid_b * stride_row_b + offs_m * stride_row_m, mask=mask_m, other=-float("inf"))
            row_denom = tl.load(row_denom_ptr + pid_b * stride_row_b + offs_m * stride_row_m, mask=mask_m, other=1.0)
            edge_dot = tl.load(edge_dot_ptr + pid_b * stride_edge_dot_b + offs_m * stride_edge_dot_m, mask=mask_m, other=0.0)

            for k_start in range(0, dim, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)
                mask_k = offs_k < dim
                target_k = tl.load(
                    flat_ptr + pid_b * stride_flat_b + offs_m[:, None] * stride_flat_n + offs_k[None, :] * stride_flat_d,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )
                source_k = tl.load(
                    flat_ptr + pid_b * stride_flat_b + offs_n[:, None] * stride_flat_n + offs_k[None, :] * stride_flat_d,
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                grad_val_k = tl.load(
                    grad_val_ptr + pid_b * stride_grad_val_b + offs_m[:, None] * stride_grad_val_n + offs_k[None, :] * stride_grad_val_d,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )
                proj_val_k = tl.load(
                    proj_val_ptr + pid_b * stride_proj_val_b + offs_n[:, None] * stride_proj_val_n + offs_k[None, :] * stride_proj_val_d,
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                grad_edges += tl.dot(grad_val_k, tl.trans(proj_val_k), allow_tf32=False, out_dtype=tl.float32)
                if heads > 0:
                    core0 = tl.load(core_ptr + 0 * stride_core_h + offs_k * stride_core_d, mask=mask_k, other=0.0)
                    score_0 += tl.dot(target_k * core0[None, :], tl.trans(source_k), allow_tf32=False, out_dtype=tl.float32)
                if heads > 1:
                    core1 = tl.load(core_ptr + 1 * stride_core_h + offs_k * stride_core_d, mask=mask_k, other=0.0)
                    score_1 += tl.dot(target_k * core1[None, :], tl.trans(source_k), allow_tf32=False, out_dtype=tl.float32)
                if heads > 2:
                    core2 = tl.load(core_ptr + 2 * stride_core_h + offs_k * stride_core_d, mask=mask_k, other=0.0)
                    score_2 += tl.dot(target_k * core2[None, :], tl.trans(source_k), allow_tf32=False, out_dtype=tl.float32)
                if heads > 3:
                    core3 = tl.load(core_ptr + 3 * stride_core_h + offs_k * stride_core_d, mask=mask_k, other=0.0)
                    score_3 += tl.dot(target_k * core3[None, :], tl.trans(source_k), allow_tf32=False, out_dtype=tl.float32)

            grad_edges += grad_state[:, None] * source_state[None, :]
            if has_bias:
                if heads > 0:
                    score_0 += tl.load(bias_ptr + 0)
                if heads > 1:
                    score_1 += tl.load(bias_ptr + 1)
                if heads > 2:
                    score_2 += tl.load(bias_ptr + 2)
                if heads > 3:
                    score_3 += tl.load(bias_ptr + 3)

            neg_large = -1.0e30
            max_abs = tl.where(full_mask, tl.abs(score_0), neg_large)
            if heads > 1:
                max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_1), neg_large))
            if heads > 2:
                max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_2), neg_large))
            if heads > 3:
                max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_3), neg_large))
            denom_heads = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            numer = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            p0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            p1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            p2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            p3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            if heads > 0:
                p0 = tl.where(full_mask, tl.exp(tl.abs(score_0) - max_abs), 0.0)
                denom_heads += p0
                numer += score_0 * p0
            if heads > 1:
                p1 = tl.where(full_mask, tl.exp(tl.abs(score_1) - max_abs), 0.0)
                denom_heads += p1
                numer += score_1 * p1
            if heads > 2:
                p2 = tl.where(full_mask, tl.exp(tl.abs(score_2) - max_abs), 0.0)
                denom_heads += p2
                numer += score_2 * p2
            if heads > 3:
                p3 = tl.where(full_mask, tl.exp(tl.abs(score_3) - max_abs), 0.0)
                denom_heads += p3
                numer += score_3 * p3
            safe_head_denom = tl.maximum(denom_heads, 1.0e-20)
            combined = numer / safe_head_denom
            probs = tl.where(full_mask, tl.exp(tl.abs(combined) - row_max[:, None]) / tl.maximum(row_denom[:, None], 1.0e-20), 0.0)
            signs = tl.where(combined > 0, 1.0, tl.where(combined < 0, -1.0, 0.0))
            grad_scores = tl.where(full_mask, signs * probs * (signs * grad_edges - edge_dot[:, None]), 0.0)
            g0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            g1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            g2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            g3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            if heads > 0:
                sign0 = tl.where(score_0 > 0, 1.0, tl.where(score_0 < 0, -1.0, 0.0))
                g0 = tl.where(full_mask, (p0 / safe_head_denom) * (1.0 + sign0 * (score_0 - combined)), 0.0)
            if heads > 1:
                sign1 = tl.where(score_1 > 0, 1.0, tl.where(score_1 < 0, -1.0, 0.0))
                g1 = tl.where(full_mask, (p1 / safe_head_denom) * (1.0 + sign1 * (score_1 - combined)), 0.0)
            if heads > 2:
                sign2 = tl.where(score_2 > 0, 1.0, tl.where(score_2 < 0, -1.0, 0.0))
                g2 = tl.where(full_mask, (p2 / safe_head_denom) * (1.0 + sign2 * (score_2 - combined)), 0.0)
            if heads > 3:
                sign3 = tl.where(score_3 > 0, 1.0, tl.where(score_3 < 0, -1.0, 0.0))
                g3 = tl.where(full_mask, (p3 / safe_head_denom) * (1.0 + sign3 * (score_3 - combined)), 0.0)

            for d_start in range(0, dim, BLOCK_D):
                offs_d = d_start + tl.arange(0, BLOCK_D)
                mask_d = offs_d < dim
                target_d = tl.load(
                    flat_ptr + pid_b * stride_flat_b + offs_m[:, None] * stride_flat_n + offs_d[None, :] * stride_flat_d,
                    mask=mask_m[:, None] & mask_d[None, :],
                    other=0.0,
                )
                source_d = tl.load(
                    flat_ptr + pid_b * stride_flat_b + offs_n[:, None] * stride_flat_n + offs_d[None, :] * stride_flat_d,
                    mask=mask_n[:, None] & mask_d[None, :],
                    other=0.0,
                )
                source_acc = tl.load(
                    grad_layer_ptr + pid_b * stride_grad_layer_b + offs_n[:, None] * stride_grad_layer_n + offs_d[None, :] * stride_grad_layer_d,
                    mask=mask_n[:, None] & mask_d[None, :],
                    other=0.0,
                )
                if heads > 0:
                    source_head = tl.dot(tl.trans(grad_scores * g0), target_d, allow_tf32=False, out_dtype=tl.float32)
                    core0d = tl.load(core_ptr + 0 * stride_core_h + offs_d * stride_core_d, mask=mask_d, other=0.0)
                    source_acc += source_head * core0d[None, :]
                if heads > 1:
                    source_head = tl.dot(tl.trans(grad_scores * g1), target_d, allow_tf32=False, out_dtype=tl.float32)
                    core1d = tl.load(core_ptr + 1 * stride_core_h + offs_d * stride_core_d, mask=mask_d, other=0.0)
                    source_acc += source_head * core1d[None, :]
                if heads > 2:
                    source_head = tl.dot(tl.trans(grad_scores * g2), target_d, allow_tf32=False, out_dtype=tl.float32)
                    core2d = tl.load(core_ptr + 2 * stride_core_h + offs_d * stride_core_d, mask=mask_d, other=0.0)
                    source_acc += source_head * core2d[None, :]
                if heads > 3:
                    source_head = tl.dot(tl.trans(grad_scores * g3), target_d, allow_tf32=False, out_dtype=tl.float32)
                    core3d = tl.load(core_ptr + 3 * stride_core_h + offs_d * stride_core_d, mask=mask_d, other=0.0)
                    source_acc += source_head * core3d[None, :]
                tl.store(
                    grad_layer_ptr + pid_b * stride_grad_layer_b + offs_n[:, None] * stride_grad_layer_n + offs_d[None, :] * stride_grad_layer_d,
                    source_acc,
                    mask=mask_n[:, None] & mask_d[None, :],
                )

    @triton.jit
    def _lowrank_signed_smoothmax_backward_target_owned_kernel(
        target_ptr,
        source_proj_ptr,
        source_weighted_ptr,
        core_ptr,
        bias_ptr,
        proj_state_ptr,
        proj_val_ptr,
        grad_state_ptr,
        grad_val_ptr,
        row_max_ptr,
        row_denom_ptr,
        edge_dot_ptr,
        grad_target_ptr,
        grad_core_partial_ptr,
        grad_bias_partial_ptr,
        stride_target_h,
        stride_target_b,
        stride_target_n,
        stride_target_r,
        stride_source_proj_h,
        stride_source_proj_b,
        stride_source_proj_n,
        stride_source_proj_r,
        stride_source_weighted_h,
        stride_source_weighted_b,
        stride_source_weighted_n,
        stride_source_weighted_r,
        stride_core_h,
        stride_core_r,
        stride_proj_state_b,
        stride_proj_state_n,
        stride_proj_val_b,
        stride_proj_val_n,
        stride_proj_val_d,
        stride_grad_state_b,
        stride_grad_state_n,
        stride_grad_val_b,
        stride_grad_val_n,
        stride_grad_val_d,
        stride_row_b,
        stride_row_m,
        stride_edge_dot_b,
        stride_edge_dot_m,
        stride_grad_target_h,
        stride_grad_target_b,
        stride_grad_target_n,
        stride_grad_target_r,
        stride_grad_core_partial_b,
        stride_grad_core_partial_rb,
        stride_grad_core_partial_h,
        stride_grad_core_partial_r,
        stride_grad_bias_partial_b,
        stride_grad_bias_partial_rb,
        stride_grad_bias_partial_h,
        num_nodes,
        rank_dim,
        val_dim,
        has_bias: tl.constexpr,
        heads: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)
        owned_head = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < num_nodes
        grad_state = tl.load(grad_state_ptr + pid_b * stride_grad_state_b + offs_m * stride_grad_state_n, mask=mask_m, other=0.0)
        row_max = tl.load(row_max_ptr + pid_b * stride_row_b + offs_m * stride_row_m, mask=mask_m, other=-float("inf"))
        row_denom = tl.load(row_denom_ptr + pid_b * stride_row_b + offs_m * stride_row_m, mask=mask_m, other=1.0)
        edge_dot = tl.load(edge_dot_ptr + pid_b * stride_edge_dot_b + offs_m * stride_edge_dot_m, mask=mask_m, other=0.0)
        bias_partial = tl.zeros((), dtype=tl.float32)

        for source_base in range(0, num_nodes, BLOCK_N):
            offs_n = source_base + tl.arange(0, BLOCK_N)
            mask_n = offs_n < num_nodes
            causal = offs_n[None, :] <= offs_m[:, None]
            full_mask = mask_m[:, None] & mask_n[None, :] & causal
            score_0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            score_1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            score_2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            score_3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            grad_edges = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for v_start in range(0, val_dim, BLOCK_K):
                offs_v = v_start + tl.arange(0, BLOCK_K)
                mask_v = offs_v < val_dim
                grad_val_v = tl.load(
                    grad_val_ptr + pid_b * stride_grad_val_b + offs_m[:, None] * stride_grad_val_n + offs_v[None, :] * stride_grad_val_d,
                    mask=mask_m[:, None] & mask_v[None, :],
                    other=0.0,
                )
                proj_val_v = tl.load(
                    proj_val_ptr + pid_b * stride_proj_val_b + offs_n[:, None] * stride_proj_val_n + offs_v[None, :] * stride_proj_val_d,
                    mask=mask_n[:, None] & mask_v[None, :],
                    other=0.0,
                )
                source_state = tl.load(
                    proj_state_ptr + pid_b * stride_proj_state_b + offs_n * stride_proj_state_n,
                    mask=mask_n,
                    other=0.0,
                )
                grad_edges += tl.dot(grad_val_v, tl.trans(proj_val_v * source_state[:, None]), allow_tf32=False, out_dtype=tl.float32)

            grad_edges += grad_state[:, None] * tl.load(
                proj_state_ptr + pid_b * stride_proj_state_b + offs_n * stride_proj_state_n,
                mask=mask_n,
                other=0.0,
            )[None, :]

            for k_start in range(0, rank_dim, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)
                mask_k = offs_k < rank_dim
                if heads > 0:
                    a = tl.load(target_ptr + 0 * stride_target_h + pid_b * stride_target_b + offs_m[:, None] * stride_target_n + offs_k[None, :] * stride_target_r, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                    b = tl.load(source_weighted_ptr + 0 * stride_source_weighted_h + pid_b * stride_source_weighted_b + offs_n[:, None] * stride_source_weighted_n + offs_k[None, :] * stride_source_weighted_r, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
                    score_0 += tl.dot(a, tl.trans(b), allow_tf32=False, out_dtype=tl.float32)
                if heads > 1:
                    a = tl.load(target_ptr + 1 * stride_target_h + pid_b * stride_target_b + offs_m[:, None] * stride_target_n + offs_k[None, :] * stride_target_r, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                    b = tl.load(source_weighted_ptr + 1 * stride_source_weighted_h + pid_b * stride_source_weighted_b + offs_n[:, None] * stride_source_weighted_n + offs_k[None, :] * stride_source_weighted_r, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
                    score_1 += tl.dot(a, tl.trans(b), allow_tf32=False, out_dtype=tl.float32)
                if heads > 2:
                    a = tl.load(target_ptr + 2 * stride_target_h + pid_b * stride_target_b + offs_m[:, None] * stride_target_n + offs_k[None, :] * stride_target_r, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                    b = tl.load(source_weighted_ptr + 2 * stride_source_weighted_h + pid_b * stride_source_weighted_b + offs_n[:, None] * stride_source_weighted_n + offs_k[None, :] * stride_source_weighted_r, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
                    score_2 += tl.dot(a, tl.trans(b), allow_tf32=False, out_dtype=tl.float32)
                if heads > 3:
                    a = tl.load(target_ptr + 3 * stride_target_h + pid_b * stride_target_b + offs_m[:, None] * stride_target_n + offs_k[None, :] * stride_target_r, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                    b = tl.load(source_weighted_ptr + 3 * stride_source_weighted_h + pid_b * stride_source_weighted_b + offs_n[:, None] * stride_source_weighted_n + offs_k[None, :] * stride_source_weighted_r, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
                    score_3 += tl.dot(a, tl.trans(b), allow_tf32=False, out_dtype=tl.float32)

            if has_bias:
                if heads > 0:
                    score_0 += tl.load(bias_ptr + 0)
                if heads > 1:
                    score_1 += tl.load(bias_ptr + 1)
                if heads > 2:
                    score_2 += tl.load(bias_ptr + 2)
                if heads > 3:
                    score_3 += tl.load(bias_ptr + 3)

            neg_large = -1.0e30
            max_abs = tl.where(full_mask, tl.abs(score_0), neg_large)
            if heads > 1:
                max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_1), neg_large))
            if heads > 2:
                max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_2), neg_large))
            if heads > 3:
                max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_3), neg_large))
            denom_heads = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            numer = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            p0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            p1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            p2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            p3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            if heads > 0:
                p0 = tl.where(full_mask, tl.exp(tl.abs(score_0) - max_abs), 0.0)
                denom_heads += p0
                numer += score_0 * p0
            if heads > 1:
                p1 = tl.where(full_mask, tl.exp(tl.abs(score_1) - max_abs), 0.0)
                denom_heads += p1
                numer += score_1 * p1
            if heads > 2:
                p2 = tl.where(full_mask, tl.exp(tl.abs(score_2) - max_abs), 0.0)
                denom_heads += p2
                numer += score_2 * p2
            if heads > 3:
                p3 = tl.where(full_mask, tl.exp(tl.abs(score_3) - max_abs), 0.0)
                denom_heads += p3
                numer += score_3 * p3
            safe_head_denom = tl.maximum(denom_heads, 1.0e-20)
            combined = numer / safe_head_denom
            probs = tl.where(full_mask, tl.exp(tl.abs(combined) - row_max[:, None]) / tl.maximum(row_denom[:, None], 1.0e-20), 0.0)
            signs = tl.where(combined > 0, 1.0, tl.where(combined < 0, -1.0, 0.0))
            grad_scores = tl.where(full_mask, signs * probs * (signs * grad_edges - edge_dot[:, None]), 0.0)

            owned_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            if owned_head == 0:
                sign0 = tl.where(score_0 > 0, 1.0, tl.where(score_0 < 0, -1.0, 0.0))
                owned_g = tl.where(full_mask, (p0 / safe_head_denom) * (1.0 + sign0 * (score_0 - combined)), 0.0)
            elif owned_head == 1:
                sign1 = tl.where(score_1 > 0, 1.0, tl.where(score_1 < 0, -1.0, 0.0))
                owned_g = tl.where(full_mask, (p1 / safe_head_denom) * (1.0 + sign1 * (score_1 - combined)), 0.0)
            elif owned_head == 2:
                sign2 = tl.where(score_2 > 0, 1.0, tl.where(score_2 < 0, -1.0, 0.0))
                owned_g = tl.where(full_mask, (p2 / safe_head_denom) * (1.0 + sign2 * (score_2 - combined)), 0.0)
            elif owned_head == 3:
                sign3 = tl.where(score_3 > 0, 1.0, tl.where(score_3 < 0, -1.0, 0.0))
                owned_g = tl.where(full_mask, (p3 / safe_head_denom) * (1.0 + sign3 * (score_3 - combined)), 0.0)

            gs = grad_scores * owned_g
            if has_bias:
                bias_partial += tl.sum(gs)
            for r_start in range(0, rank_dim, BLOCK_R):
                offs_r = r_start + tl.arange(0, BLOCK_R)
                mask_r = offs_r < rank_dim
                target_r = tl.load(
                    target_ptr + owned_head * stride_target_h + pid_b * stride_target_b + offs_m[:, None] * stride_target_n + offs_r[None, :] * stride_target_r,
                    mask=mask_m[:, None] & mask_r[None, :],
                    other=0.0,
                )
                source_weighted_r = tl.load(
                    source_weighted_ptr + owned_head * stride_source_weighted_h + pid_b * stride_source_weighted_b + offs_n[:, None] * stride_source_weighted_n + offs_r[None, :] * stride_source_weighted_r,
                    mask=mask_n[:, None] & mask_r[None, :],
                    other=0.0,
                )
                grad_target_ptrs = (
                    grad_target_ptr
                    + owned_head * stride_grad_target_h
                    + pid_b * stride_grad_target_b
                    + offs_m[:, None] * stride_grad_target_n
                    + offs_r[None, :] * stride_grad_target_r
                )
                grad_target_acc = tl.load(grad_target_ptrs, mask=mask_m[:, None] & mask_r[None, :], other=0.0)
                grad_target_acc += tl.dot(gs, source_weighted_r, allow_tf32=False, out_dtype=tl.float32)
                tl.store(grad_target_ptrs, grad_target_acc, mask=mask_m[:, None] & mask_r[None, :])
                source_weighted_part = tl.dot(tl.trans(gs), target_r, allow_tf32=False, out_dtype=tl.float32)
                source_proj_r = tl.load(
                    source_proj_ptr + owned_head * stride_source_proj_h + pid_b * stride_source_proj_b + offs_n[:, None] * stride_source_proj_n + offs_r[None, :] * stride_source_proj_r,
                    mask=mask_n[:, None] & mask_r[None, :],
                    other=0.0,
                )
                core_partial = tl.load(
                    grad_core_partial_ptr + pid_b * stride_grad_core_partial_b + pid_m * stride_grad_core_partial_rb + owned_head * stride_grad_core_partial_h + offs_r * stride_grad_core_partial_r,
                    mask=mask_r,
                    other=0.0,
                )
                core_partial += tl.sum(source_weighted_part * source_proj_r, axis=0)
                tl.store(
                    grad_core_partial_ptr + pid_b * stride_grad_core_partial_b + pid_m * stride_grad_core_partial_rb + owned_head * stride_grad_core_partial_h + offs_r * stride_grad_core_partial_r,
                    core_partial,
                    mask=mask_r,
                )

        if has_bias:
            tl.store(grad_bias_partial_ptr + pid_b * stride_grad_bias_partial_b + pid_m * stride_grad_bias_partial_rb + owned_head * stride_grad_bias_partial_h, bias_partial)

    @triton.jit
    def _lowrank_signed_smoothmax_backward_source_owned_kernel(
        target_ptr,
        source_weighted_ptr,
        core_ptr,
        bias_ptr,
        proj_state_ptr,
        proj_val_ptr,
        grad_state_ptr,
        grad_val_ptr,
        row_max_ptr,
        row_denom_ptr,
        edge_dot_ptr,
        grad_source_ptr,
        stride_target_h,
        stride_target_b,
        stride_target_n,
        stride_target_r,
        stride_source_weighted_h,
        stride_source_weighted_b,
        stride_source_weighted_n,
        stride_source_weighted_r,
        stride_core_h,
        stride_core_r,
        stride_proj_state_b,
        stride_proj_state_n,
        stride_proj_val_b,
        stride_proj_val_n,
        stride_proj_val_d,
        stride_grad_state_b,
        stride_grad_state_n,
        stride_grad_val_b,
        stride_grad_val_n,
        stride_grad_val_d,
        stride_row_b,
        stride_row_m,
        stride_edge_dot_b,
        stride_edge_dot_m,
        stride_grad_source_h,
        stride_grad_source_b,
        stride_grad_source_n,
        stride_grad_source_r,
        num_nodes,
        rank_dim,
        val_dim,
        has_bias: tl.constexpr,
        heads: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_b = tl.program_id(1)
        owned_head = tl.program_id(2)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < num_nodes
        source_state = tl.load(proj_state_ptr + pid_b * stride_proj_state_b + offs_n * stride_proj_state_n, mask=mask_n, other=0.0)

        for target_base in range(0, num_nodes, BLOCK_M):
            offs_m = target_base + tl.arange(0, BLOCK_M)
            mask_m = offs_m < num_nodes
            causal = offs_n[None, :] <= offs_m[:, None]
            full_mask = mask_m[:, None] & mask_n[None, :] & causal
            score_0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            score_1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            score_2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            score_3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            grad_edges = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            grad_state = tl.load(grad_state_ptr + pid_b * stride_grad_state_b + offs_m * stride_grad_state_n, mask=mask_m, other=0.0)
            row_max = tl.load(row_max_ptr + pid_b * stride_row_b + offs_m * stride_row_m, mask=mask_m, other=-float("inf"))
            row_denom = tl.load(row_denom_ptr + pid_b * stride_row_b + offs_m * stride_row_m, mask=mask_m, other=1.0)
            edge_dot = tl.load(edge_dot_ptr + pid_b * stride_edge_dot_b + offs_m * stride_edge_dot_m, mask=mask_m, other=0.0)
            for v_start in range(0, val_dim, BLOCK_K):
                offs_v = v_start + tl.arange(0, BLOCK_K)
                mask_v = offs_v < val_dim
                grad_val_v = tl.load(
                    grad_val_ptr + pid_b * stride_grad_val_b + offs_m[:, None] * stride_grad_val_n + offs_v[None, :] * stride_grad_val_d,
                    mask=mask_m[:, None] & mask_v[None, :],
                    other=0.0,
                )
                proj_val_v = tl.load(
                    proj_val_ptr + pid_b * stride_proj_val_b + offs_n[:, None] * stride_proj_val_n + offs_v[None, :] * stride_proj_val_d,
                    mask=mask_n[:, None] & mask_v[None, :],
                    other=0.0,
                )
                grad_edges += tl.dot(grad_val_v, tl.trans(proj_val_v * source_state[:, None]), allow_tf32=False, out_dtype=tl.float32)
            grad_edges += grad_state[:, None] * source_state[None, :]

            for k_start in range(0, rank_dim, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)
                mask_k = offs_k < rank_dim
                if heads > 0:
                    a = tl.load(target_ptr + 0 * stride_target_h + pid_b * stride_target_b + offs_m[:, None] * stride_target_n + offs_k[None, :] * stride_target_r, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                    b = tl.load(source_weighted_ptr + 0 * stride_source_weighted_h + pid_b * stride_source_weighted_b + offs_n[:, None] * stride_source_weighted_n + offs_k[None, :] * stride_source_weighted_r, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
                    score_0 += tl.dot(a, tl.trans(b), allow_tf32=False, out_dtype=tl.float32)
                if heads > 1:
                    a = tl.load(target_ptr + 1 * stride_target_h + pid_b * stride_target_b + offs_m[:, None] * stride_target_n + offs_k[None, :] * stride_target_r, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                    b = tl.load(source_weighted_ptr + 1 * stride_source_weighted_h + pid_b * stride_source_weighted_b + offs_n[:, None] * stride_source_weighted_n + offs_k[None, :] * stride_source_weighted_r, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
                    score_1 += tl.dot(a, tl.trans(b), allow_tf32=False, out_dtype=tl.float32)
                if heads > 2:
                    a = tl.load(target_ptr + 2 * stride_target_h + pid_b * stride_target_b + offs_m[:, None] * stride_target_n + offs_k[None, :] * stride_target_r, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                    b = tl.load(source_weighted_ptr + 2 * stride_source_weighted_h + pid_b * stride_source_weighted_b + offs_n[:, None] * stride_source_weighted_n + offs_k[None, :] * stride_source_weighted_r, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
                    score_2 += tl.dot(a, tl.trans(b), allow_tf32=False, out_dtype=tl.float32)
                if heads > 3:
                    a = tl.load(target_ptr + 3 * stride_target_h + pid_b * stride_target_b + offs_m[:, None] * stride_target_n + offs_k[None, :] * stride_target_r, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                    b = tl.load(source_weighted_ptr + 3 * stride_source_weighted_h + pid_b * stride_source_weighted_b + offs_n[:, None] * stride_source_weighted_n + offs_k[None, :] * stride_source_weighted_r, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
                    score_3 += tl.dot(a, tl.trans(b), allow_tf32=False, out_dtype=tl.float32)

            if has_bias:
                if heads > 0:
                    score_0 += tl.load(bias_ptr + 0)
                if heads > 1:
                    score_1 += tl.load(bias_ptr + 1)
                if heads > 2:
                    score_2 += tl.load(bias_ptr + 2)
                if heads > 3:
                    score_3 += tl.load(bias_ptr + 3)
            neg_large = -1.0e30
            max_abs = tl.where(full_mask, tl.abs(score_0), neg_large)
            if heads > 1:
                max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_1), neg_large))
            if heads > 2:
                max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_2), neg_large))
            if heads > 3:
                max_abs = tl.maximum(max_abs, tl.where(full_mask, tl.abs(score_3), neg_large))
            denom_heads = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            numer = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            p0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            p1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            p2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            p3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            if heads > 0:
                p0 = tl.where(full_mask, tl.exp(tl.abs(score_0) - max_abs), 0.0)
                denom_heads += p0
                numer += score_0 * p0
            if heads > 1:
                p1 = tl.where(full_mask, tl.exp(tl.abs(score_1) - max_abs), 0.0)
                denom_heads += p1
                numer += score_1 * p1
            if heads > 2:
                p2 = tl.where(full_mask, tl.exp(tl.abs(score_2) - max_abs), 0.0)
                denom_heads += p2
                numer += score_2 * p2
            if heads > 3:
                p3 = tl.where(full_mask, tl.exp(tl.abs(score_3) - max_abs), 0.0)
                denom_heads += p3
                numer += score_3 * p3
            safe_head_denom = tl.maximum(denom_heads, 1.0e-20)
            combined = numer / safe_head_denom
            probs = tl.where(full_mask, tl.exp(tl.abs(combined) - row_max[:, None]) / tl.maximum(row_denom[:, None], 1.0e-20), 0.0)
            signs = tl.where(combined > 0, 1.0, tl.where(combined < 0, -1.0, 0.0))
            grad_scores = tl.where(full_mask, signs * probs * (signs * grad_edges - edge_dot[:, None]), 0.0)

            owned_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            if owned_head == 0:
                sign0 = tl.where(score_0 > 0, 1.0, tl.where(score_0 < 0, -1.0, 0.0))
                owned_g = tl.where(full_mask, (p0 / safe_head_denom) * (1.0 + sign0 * (score_0 - combined)), 0.0)
            elif owned_head == 1:
                sign1 = tl.where(score_1 > 0, 1.0, tl.where(score_1 < 0, -1.0, 0.0))
                owned_g = tl.where(full_mask, (p1 / safe_head_denom) * (1.0 + sign1 * (score_1 - combined)), 0.0)
            elif owned_head == 2:
                sign2 = tl.where(score_2 > 0, 1.0, tl.where(score_2 < 0, -1.0, 0.0))
                owned_g = tl.where(full_mask, (p2 / safe_head_denom) * (1.0 + sign2 * (score_2 - combined)), 0.0)
            elif owned_head == 3:
                sign3 = tl.where(score_3 > 0, 1.0, tl.where(score_3 < 0, -1.0, 0.0))
                owned_g = tl.where(full_mask, (p3 / safe_head_denom) * (1.0 + sign3 * (score_3 - combined)), 0.0)

            gs = grad_scores * owned_g
            for r_start in range(0, rank_dim, BLOCK_R):
                offs_r = r_start + tl.arange(0, BLOCK_R)
                mask_r = offs_r < rank_dim
                target_r = tl.load(
                    target_ptr + owned_head * stride_target_h + pid_b * stride_target_b + offs_m[:, None] * stride_target_n + offs_r[None, :] * stride_target_r,
                    mask=mask_m[:, None] & mask_r[None, :],
                    other=0.0,
                )
                source_weighted_part = tl.dot(tl.trans(gs), target_r, allow_tf32=False, out_dtype=tl.float32)
                core_owned = tl.load(core_ptr + owned_head * stride_core_h + offs_r * stride_core_r, mask=mask_r, other=0.0)
                grad_source_ptrs = (
                    grad_source_ptr
                    + owned_head * stride_grad_source_h
                    + pid_b * stride_grad_source_b
                    + offs_n[:, None] * stride_grad_source_n
                    + offs_r[None, :] * stride_grad_source_r
                )
                source_acc = tl.load(grad_source_ptrs, mask=mask_n[:, None] & mask_r[None, :], other=0.0)
                source_acc += source_weighted_part * core_owned[None, :]
                tl.store(grad_source_ptrs, source_acc, mask=mask_n[:, None] & mask_r[None, :])


def diagonal_signed_smoothmax_backward_owner(
    flat_val_bnd: torch.Tensor,
    core_hd: torch.Tensor,
    projected_state_bn: torch.Tensor,
    projected_val_bnd: torch.Tensor,
    grad_state_bn: torch.Tensor,
    grad_val_bnd: torch.Tensor,
    row_max_bm: torch.Tensor,
    row_denom_bm: torch.Tensor,
    edge_dot_bm: torch.Tensor,
    biases: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not triton_signed_smoothmax_available():
        raise RuntimeError("Triton is unavailable.")
    batch, nodes, dim = flat_val_bnd.shape
    heads = int(core_hd.shape[0])
    row_blocks = triton.cdiv(nodes, _diag_backward_block_config()[0])
    source_blocks = triton.cdiv(nodes, _diag_backward_block_config()[1])
    flat_val = flat_val_bnd.contiguous().to(dtype=torch.float32)
    core = core_hd.contiguous().to(dtype=torch.float32)
    projected_state = projected_state_bn.contiguous().to(dtype=torch.float32)
    projected_val = projected_val_bnd.contiguous().to(dtype=torch.float32)
    grad_state = grad_state_bn.contiguous().to(dtype=torch.float32)
    grad_val = grad_val_bnd.contiguous().to(dtype=torch.float32)
    row_max = row_max_bm.contiguous().to(dtype=torch.float32)
    row_denom = row_denom_bm.contiguous().to(dtype=torch.float32)
    edge_dot = edge_dot_bm.contiguous().to(dtype=torch.float32)
    bias = None if biases is None or biases.numel() == 0 else biases.contiguous().to(dtype=torch.float32, device=flat_val.device)
    grad_layer_target = torch.zeros_like(flat_val, dtype=torch.float32)
    grad_layer_source = torch.zeros_like(flat_val, dtype=torch.float32)
    grad_weights_partial = torch.zeros((batch, row_blocks, source_blocks, heads, dim), dtype=torch.float32, device=flat_val.device)
    grad_bias_partial = None if bias is None else torch.zeros((batch, row_blocks, heads), dtype=torch.float32, device=flat_val.device)
    block_m, block_n, block_k, block_d = _diag_backward_block_config()
    _diagonal_signed_smoothmax_backward_target_owned_kernel[(row_blocks, batch)](
        flat_val, core, bias if bias is not None else flat_val,
        projected_state, projected_val, grad_state, grad_val,
        row_max, row_denom, edge_dot,
        grad_layer_target, grad_weights_partial, grad_bias_partial if grad_bias_partial is not None else grad_layer_target,
        flat_val.stride(0), flat_val.stride(1), flat_val.stride(2),
        core.stride(0), core.stride(1),
        projected_state.stride(0), projected_state.stride(1),
        projected_val.stride(0), projected_val.stride(1), projected_val.stride(2),
        grad_state.stride(0), grad_state.stride(1),
        grad_val.stride(0), grad_val.stride(1), grad_val.stride(2),
        row_max.stride(0), row_max.stride(1),
        edge_dot.stride(0), edge_dot.stride(1),
        grad_layer_target.stride(0), grad_layer_target.stride(1), grad_layer_target.stride(2),
        grad_weights_partial.stride(0), grad_weights_partial.stride(1), grad_weights_partial.stride(2), grad_weights_partial.stride(3), grad_weights_partial.stride(4),
        grad_bias_partial.stride(0) if grad_bias_partial is not None else 0,
        grad_bias_partial.stride(1) if grad_bias_partial is not None else 0,
        grad_bias_partial.stride(2) if grad_bias_partial is not None else 0,
        0,
        num_nodes=nodes, dim=dim,
        has_bias=bias is not None, heads=heads,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_D=block_d,
        num_warps=4, num_stages=2,
    )
    _diagonal_signed_smoothmax_backward_source_owned_kernel[(triton.cdiv(nodes, block_n), batch)](
        flat_val, core, bias if bias is not None else flat_val,
        projected_state, projected_val, grad_state, grad_val,
        row_max, row_denom, edge_dot,
        grad_layer_source,
        flat_val.stride(0), flat_val.stride(1), flat_val.stride(2),
        core.stride(0), core.stride(1),
        projected_state.stride(0), projected_state.stride(1),
        projected_val.stride(0), projected_val.stride(1), projected_val.stride(2),
        grad_state.stride(0), grad_state.stride(1),
        grad_val.stride(0), grad_val.stride(1), grad_val.stride(2),
        row_max.stride(0), row_max.stride(1),
        edge_dot.stride(0), edge_dot.stride(1),
        grad_layer_source.stride(0), grad_layer_source.stride(1), grad_layer_source.stride(2),
        num_nodes=nodes, dim=dim,
        has_bias=bias is not None, heads=heads,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_D=block_d,
        num_warps=4, num_stages=2,
    )
    return grad_layer_target, grad_layer_source, grad_weights_partial, grad_bias_partial


def lowrank_signed_smoothmax_backward_owner(
    projected_target_hbnr: torch.Tensor,
    projected_source_hbnr: torch.Tensor,
    weighted_source_hbnr: torch.Tensor,
    core_hr: torch.Tensor,
    projected_state_bn: torch.Tensor,
    projected_val_bnd: torch.Tensor,
    grad_state_bn: torch.Tensor,
    grad_val_bnd: torch.Tensor,
    row_max_bm: torch.Tensor,
    row_denom_bm: torch.Tensor,
    edge_dot_bm: torch.Tensor,
    biases: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not triton_signed_smoothmax_available():
        raise RuntimeError("Triton is unavailable.")
    heads, batch, nodes, rank_dim = projected_target_hbnr.shape
    val_dim = int(projected_val_bnd.shape[-1])
    row_blocks = triton.cdiv(nodes, _lowrank_backward_block_config()[0])
    target = projected_target_hbnr.contiguous()
    source_proj = projected_source_hbnr.contiguous()
    source_weighted = weighted_source_hbnr.contiguous()
    core = core_hr.contiguous()
    projected_state = projected_state_bn.contiguous().to(dtype=torch.float32)
    projected_val = projected_val_bnd.contiguous().to(dtype=torch.float32)
    grad_state = grad_state_bn.contiguous().to(dtype=torch.float32)
    grad_val = grad_val_bnd.contiguous().to(dtype=torch.float32)
    row_max = row_max_bm.contiguous().to(dtype=torch.float32)
    row_denom = row_denom_bm.contiguous().to(dtype=torch.float32)
    edge_dot = edge_dot_bm.contiguous().to(dtype=torch.float32)
    bias = None if biases is None or biases.numel() == 0 else biases.contiguous().to(dtype=torch.float32, device=target.device)
    grad_target = torch.zeros_like(target, dtype=torch.float32)
    grad_source = torch.zeros_like(source_proj, dtype=torch.float32)
    grad_core_partial = torch.zeros((batch, row_blocks, heads, rank_dim), dtype=torch.float32, device=target.device)
    grad_bias_partial = None if bias is None else torch.zeros((batch, row_blocks, heads), dtype=torch.float32, device=target.device)
    block_m, block_n, block_k, block_r = _lowrank_backward_block_config()
    _lowrank_signed_smoothmax_backward_target_owned_kernel[(row_blocks, batch, heads)](
        target, source_proj, source_weighted, core, bias if bias is not None else target,
        projected_state, projected_val, grad_state, grad_val,
        row_max, row_denom, edge_dot,
        grad_target, grad_core_partial, grad_bias_partial if grad_bias_partial is not None else grad_target,
        target.stride(0), target.stride(1), target.stride(2), target.stride(3),
        source_proj.stride(0), source_proj.stride(1), source_proj.stride(2), source_proj.stride(3),
        source_weighted.stride(0), source_weighted.stride(1), source_weighted.stride(2), source_weighted.stride(3),
        core.stride(0), core.stride(1),
        projected_state.stride(0), projected_state.stride(1),
        projected_val.stride(0), projected_val.stride(1), projected_val.stride(2),
        grad_state.stride(0), grad_state.stride(1),
        grad_val.stride(0), grad_val.stride(1), grad_val.stride(2),
        row_max.stride(0), row_max.stride(1),
        edge_dot.stride(0), edge_dot.stride(1),
        grad_target.stride(0), grad_target.stride(1), grad_target.stride(2), grad_target.stride(3),
        grad_core_partial.stride(0), grad_core_partial.stride(1), grad_core_partial.stride(2), grad_core_partial.stride(3),
        grad_bias_partial.stride(0) if grad_bias_partial is not None else 0,
        grad_bias_partial.stride(1) if grad_bias_partial is not None else 0,
        grad_bias_partial.stride(2) if grad_bias_partial is not None else 0,
        num_nodes=nodes, rank_dim=rank_dim, val_dim=val_dim,
        has_bias=bias is not None, heads=heads,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_R=block_r,
        num_warps=4, num_stages=2,
    )
    _lowrank_signed_smoothmax_backward_source_owned_kernel[(triton.cdiv(nodes, block_n), batch, heads)](
        target, source_weighted, core, bias if bias is not None else target,
        projected_state, projected_val, grad_state, grad_val,
        row_max, row_denom, edge_dot,
        grad_source,
        target.stride(0), target.stride(1), target.stride(2), target.stride(3),
        source_weighted.stride(0), source_weighted.stride(1), source_weighted.stride(2), source_weighted.stride(3),
        core.stride(0), core.stride(1),
        projected_state.stride(0), projected_state.stride(1),
        projected_val.stride(0), projected_val.stride(1), projected_val.stride(2),
        grad_state.stride(0), grad_state.stride(1),
        grad_val.stride(0), grad_val.stride(1), grad_val.stride(2),
        row_max.stride(0), row_max.stride(1),
        edge_dot.stride(0), edge_dot.stride(1),
        grad_source.stride(0), grad_source.stride(1), grad_source.stride(2), grad_source.stride(3),
        num_nodes=nodes, rank_dim=rank_dim, val_dim=val_dim,
        has_bias=bias is not None, heads=heads,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_R=block_r,
        num_warps=4, num_stages=2,
    )
    return grad_target, grad_source, grad_core_partial, grad_bias_partial
