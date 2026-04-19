from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import gc
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from jakal_net import (
    DiagonalBilinearPairwise,
    describe_device,
    Layer,
    LayerDelta,
    LinearRoute,
    LowRankBilinearRoute,
    native_status,
    Propagation,
    resolve_device,
    ScalarAffine,
    SparsePropagation,
    SparseTransition,
    Transition,
)
from jakal_net.native_backend import (
    propagation_query_topk_native,
    transition_query_topk_native,
)


def _current_rss_bytes() -> int | None:
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.wintypes.DWORD),
                ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            ctypes.wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.wintypes.BOOL

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if not ok:
            return None
        return int(counters.WorkingSetSize)

    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as file:
            rss_pages = int(file.read().split()[1])
        return rss_pages * os.sysconf("SC_PAGE_SIZE")
    except (FileNotFoundError, OSError, ValueError):
        return None


class MemorySampler:
    def __init__(self, interval_s: float = 0.001) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline = _current_rss_bytes()
        self.peak = self.baseline

    def _sample(self) -> None:
        while not self._stop.is_set():
            current = _current_rss_bytes()
            if current is not None:
                if self.peak is None or current > self.peak:
                    self.peak = current
            time.sleep(self.interval_s)

    def __enter__(self) -> "MemorySampler":
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def peak_delta_bytes(self) -> int | None:
        if self.baseline is None or self.peak is None:
            return None
        return max(0, self.peak - self.baseline)


@dataclass
class BenchmarkResult:
    name: str
    implementation: str
    avg_ms: float
    peak_memory_mb: float | None
    logical_edges_per_sec: float
    max_abs_error: float


@dataclass
class BenchmarkFailure:
    name: str
    implementation: str
    error: str


def _maybe_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _clone_module(module: nn.Module) -> nn.Module:
    clone = type(module)(*module._init_args, **module._init_kwargs)  # type: ignore[attr-defined]
    clone.load_state_dict(module.state_dict())
    return clone


def _make_cloneable(module: nn.Module, *args, **kwargs) -> nn.Module:
    module._init_args = args  # type: ignore[attr-defined]
    module._init_kwargs = kwargs  # type: ignore[attr-defined]
    return module


def _measure(
    fn: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> tuple[float, float | None]:
    for _ in range(warmup):
        fn()
        _maybe_synchronize(device)

    gc.collect()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with MemorySampler() as sampler:
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        _maybe_synchronize(device)
        end = time.perf_counter()

    avg_ms = (end - start) * 1000.0 / iterations
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        peak_delta = sampler.peak_delta_bytes
        peak_memory_mb = None if peak_delta is None else peak_delta / (1024**2)
    return avg_ms, peak_memory_mb


def _max_abs_error(reference, candidate) -> float:
    if isinstance(reference, Layer) and isinstance(candidate, Layer):
        state_error = (reference.state - candidate.state).abs().max().item()
        val_error = (reference.val - candidate.val).abs().max().item()
        return max(state_error, val_error)
    if hasattr(reference, "delta_state") and hasattr(reference, "delta_val"):
        state_error = (reference.delta_state - candidate.delta_state).abs().max().item()
        val_error = (reference.delta_val - candidate.delta_val).abs().max().item()
        return max(state_error, val_error)
    raise TypeError("Unsupported output type for error calculation.")


def _batch_product(batch_shape: torch.Size | tuple[int, ...]) -> int:
    product = 1
    for value in batch_shape:
        product *= value
    return product


def _window_edge_count(num_nodes: int, window: int) -> int:
    return sum(min(index + 1, window + 1) for index in range(num_nodes))


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


def _query_propagation_reference(
    *,
    pairwise_fn: nn.Module,
    query_val: torch.Tensor,
    source_val: torch.Tensor,
    projected_state: torch.Tensor,
    projected_val: torch.Tensor,
    topk: int,
) -> LayerDelta:
    scores = pairwise_fn(query_val, source_val)
    selected = torch.topk(scores, k=min(topk, source_val.shape[-2]), dim=-1)
    edges = torch.nn.functional.softsign(selected.values)
    return LayerDelta(
        delta_state=(edges * _gather_state(projected_state, selected.indices)).sum(dim=-1),
        delta_val=(
            edges.unsqueeze(-1) * _gather_val(projected_val, selected.indices)
        ).sum(dim=-2),
    )


def _query_transition_reference(
    *,
    route_fn: nn.Module,
    sender_strength: torch.Tensor,
    src_val: torch.Tensor,
    query_val: torch.Tensor,
    projected_state: torch.Tensor,
    projected_val: torch.Tensor,
    topk: int,
) -> LayerDelta:
    logits = route_fn(src_val, query_val).transpose(-1, -2)
    selected = torch.topk(logits, k=min(topk, src_val.shape[-2]), dim=-1)
    routes = torch.softmax(selected.values, dim=-1)
    weighted_state = sender_strength * projected_state
    weighted_val = sender_strength.unsqueeze(-1) * projected_val
    return LayerDelta(
        delta_state=(routes * _gather_state(weighted_state, selected.indices)).sum(dim=-1),
        delta_val=(
            routes.unsqueeze(-1) * _gather_val(weighted_val, selected.indices)
        ).sum(dim=-2),
    )


def _format_result(result: BenchmarkResult) -> str:
    memory = "n/a" if result.peak_memory_mb is None else f"{result.peak_memory_mb:8.2f}"
    return (
        f"{result.name:28} {result.implementation:11} "
        f"{result.avg_ms:9.3f} ms  {memory:>8} MB  "
        f"{result.logical_edges_per_sec / 1e6:9.3f} Medges/s  "
        f"{result.max_abs_error:10.3e}"
    )


def _format_failure(failure: BenchmarkFailure) -> str:
    return f"{failure.name:28} {failure.implementation:11} ERROR      {failure.error}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = torch.float32
    print(f"using device: {describe_device(args.device)}")
    print(f"native backend: {native_status().backend_name or 'unavailable'}")

    batch_shape = (4,)
    batch_factor = _batch_product(batch_shape)

    prop_layer = Layer.zeros(dim=64, num_nodes=256, batch_shape=batch_shape, device=device, dtype=dtype)
    prop_layer = prop_layer.with_tensors(
        state=torch.randn_like(prop_layer.state),
        val=torch.randn_like(prop_layer.val),
    )
    trans_src = Layer.zeros(dim=64, num_nodes=256, batch_shape=batch_shape, device=device, dtype=dtype)
    trans_src = trans_src.with_tensors(
        state=torch.randn_like(trans_src.state),
        val=torch.randn_like(trans_src.val),
    )
    trans_dst = Layer.zeros(dim=64, num_nodes=128, batch_shape=batch_shape, device=device, dtype=dtype)
    query_val = torch.randn(*batch_shape, 16, 64, device=device, dtype=dtype)

    pairwise_template = _make_cloneable(DiagonalBilinearPairwise(dim=64), 64).to(device)
    state_proj_template = _make_cloneable(ScalarAffine(), 1.0, 0.0).to(device)
    route_template = _make_cloneable(LinearRoute(src_dim=64, dst_nodes=128), 64, 128).to(device)
    query_route_template = _make_cloneable(
        LowRankBilinearRoute(src_dim=64, dst_dim=64, rank=16),
        64,
        64,
        rank=16,
    ).to(device)
    val_proj_template = _make_cloneable(nn.Linear(64, 64), 64, 64).to(device)
    pairwise_template.requires_grad_(False)
    query_route_template.requires_grad_(False)
    state_proj_template.requires_grad_(False)
    val_proj_template.requires_grad_(False)
    query_projected_state = state_proj_template(trans_src.state)
    query_projected_val = val_proj_template(trans_src.val)
    query_sender_strength = torch.nn.functional.softplus(trans_src.state) + 0.1

    propagation_reference = Propagation(
        pairwise_fn=_clone_module(pairwise_template).to(device),
        state_proj_fn=_clone_module(state_proj_template).to(device),
        implementation="reference",
    )
    propagation_kernel = Propagation(
        pairwise_fn=_clone_module(pairwise_template).to(device),
        state_proj_fn=_clone_module(state_proj_template).to(device),
        implementation="kernel",
    )
    propagation_streaming = Propagation(
        pairwise_fn=_clone_module(pairwise_template).to(device),
        state_proj_fn=_clone_module(state_proj_template).to(device),
        implementation="streaming",
        target_block_size=64,
        source_block_size=64,
    )
    propagation_native = Propagation(
        pairwise_fn=_clone_module(pairwise_template).to(device),
        state_proj_fn=_clone_module(state_proj_template).to(device),
        implementation="native",
        target_block_size=64,
        source_block_size=64,
    )

    window_reference = SparsePropagation(
        pairwise_fn=_clone_module(pairwise_template).to(device),
        state_proj_fn=_clone_module(state_proj_template).to(device),
        sparse_type="window",
        window=8,
        implementation="reference",
    )
    window_kernel = SparsePropagation(
        pairwise_fn=_clone_module(pairwise_template).to(device),
        state_proj_fn=_clone_module(state_proj_template).to(device),
        sparse_type="window",
        window=8,
        implementation="kernel",
    )
    window_streaming = SparsePropagation(
        pairwise_fn=_clone_module(pairwise_template).to(device),
        state_proj_fn=_clone_module(state_proj_template).to(device),
        sparse_type="window",
        window=8,
        implementation="streaming",
        target_block_size=64,
        source_block_size=64,
    )
    window_native = SparsePropagation(
        pairwise_fn=_clone_module(pairwise_template).to(device),
        state_proj_fn=_clone_module(state_proj_template).to(device),
        sparse_type="window",
        window=8,
        implementation="native",
        target_block_size=64,
        source_block_size=64,
    )

    topk_propagation_reference = SparsePropagation(
        pairwise_fn=_clone_module(pairwise_template).to(device),
        state_proj_fn=_clone_module(state_proj_template).to(device),
        sparse_type="topk",
        topk=8,
        implementation="reference",
    )
    topk_propagation_kernel = SparsePropagation(
        pairwise_fn=_clone_module(pairwise_template).to(device),
        state_proj_fn=_clone_module(state_proj_template).to(device),
        sparse_type="topk",
        topk=8,
        implementation="kernel",
    )
    topk_propagation_streaming = SparsePropagation(
        pairwise_fn=_clone_module(pairwise_template).to(device),
        state_proj_fn=_clone_module(state_proj_template).to(device),
        sparse_type="topk",
        topk=8,
        implementation="streaming",
        target_block_size=64,
        source_block_size=64,
    )
    topk_propagation_native = SparsePropagation(
        pairwise_fn=_clone_module(pairwise_template).to(device),
        state_proj_fn=_clone_module(state_proj_template).to(device),
        sparse_type="topk",
        topk=8,
        implementation="native",
        target_block_size=64,
        source_block_size=64,
    )

    transition_reference = Transition(
        route_fn=_clone_module(route_template).to(device),
        state_proj_fn=_clone_module(state_proj_template).to(device),
        val_proj_fn=_clone_module(val_proj_template).to(device),
        implementation="reference",
    )
    transition_kernel = Transition(
        route_fn=_clone_module(route_template).to(device),
        state_proj_fn=_clone_module(state_proj_template).to(device),
        val_proj_fn=_clone_module(val_proj_template).to(device),
        implementation="kernel",
    )
    transition_streaming = Transition(
        route_fn=_clone_module(route_template).to(device),
        state_proj_fn=_clone_module(state_proj_template).to(device),
        val_proj_fn=_clone_module(val_proj_template).to(device),
        implementation="streaming",
        src_block_size=64,
    )
    transition_native = Transition(
        route_fn=_clone_module(route_template).to(device),
        state_proj_fn=_clone_module(state_proj_template).to(device),
        val_proj_fn=_clone_module(val_proj_template).to(device),
        implementation="native",
        src_block_size=64,
        dst_block_size=64,
    )

    sparse_transition_reference = SparseTransition(
        route_fn=_clone_module(route_template).to(device),
        topk=8,
        state_proj_fn=_clone_module(state_proj_template).to(device),
        val_proj_fn=_clone_module(val_proj_template).to(device),
        implementation="reference",
    )
    sparse_transition_kernel = SparseTransition(
        route_fn=_clone_module(route_template).to(device),
        topk=8,
        state_proj_fn=_clone_module(state_proj_template).to(device),
        val_proj_fn=_clone_module(val_proj_template).to(device),
        implementation="kernel",
    )
    sparse_transition_streaming = SparseTransition(
        route_fn=_clone_module(route_template).to(device),
        topk=8,
        state_proj_fn=_clone_module(state_proj_template).to(device),
        val_proj_fn=_clone_module(val_proj_template).to(device),
        implementation="streaming",
        src_block_size=64,
    )
    sparse_transition_native = SparseTransition(
        route_fn=_clone_module(route_template).to(device),
        topk=8,
        state_proj_fn=_clone_module(state_proj_template).to(device),
        val_proj_fn=_clone_module(val_proj_template).to(device),
        implementation="native",
        src_block_size=64,
        dst_block_size=64,
    )
    sparse_transition_pairwise_reference = SparseTransition(
        route_fn=_clone_module(query_route_template).to(device),
        topk=8,
        state_proj_fn=_clone_module(state_proj_template).to(device),
        val_proj_fn=_clone_module(val_proj_template).to(device),
        implementation="reference",
    )
    sparse_transition_pairwise_streaming = SparseTransition(
        route_fn=_clone_module(query_route_template).to(device),
        topk=8,
        state_proj_fn=_clone_module(state_proj_template).to(device),
        val_proj_fn=_clone_module(val_proj_template).to(device),
        implementation="streaming",
        src_block_size=64,
        dst_block_size=64,
    )
    sparse_transition_pairwise_kernel = SparseTransition(
        route_fn=_clone_module(query_route_template).to(device),
        topk=8,
        state_proj_fn=_clone_module(state_proj_template).to(device),
        val_proj_fn=_clone_module(val_proj_template).to(device),
        implementation="kernel",
        src_block_size=64,
        dst_block_size=64,
    )
    sparse_transition_pairwise_native = SparseTransition(
        route_fn=_clone_module(query_route_template).to(device),
        topk=8,
        state_proj_fn=_clone_module(state_proj_template).to(device),
        val_proj_fn=_clone_module(val_proj_template).to(device),
        implementation="native",
        src_block_size=64,
        dst_block_size=64,
    )

    benchmarks: list[tuple[str, list[tuple[str, Callable[[], object]]], int]] = [
        (
            "Propagation(dense)",
            [
                ("reference", lambda: propagation_reference.compute_delta(prop_layer)),
                ("kernel", lambda: propagation_kernel.compute_delta(prop_layer)),
                ("streaming", lambda: propagation_streaming.compute_delta(prop_layer)),
                ("native", lambda: propagation_native.compute_delta(prop_layer)),
            ],
            batch_factor * prop_layer.num_nodes * prop_layer.num_nodes,
        ),
        (
            "SparsePropagation(window)",
            [
                ("reference", lambda: window_reference.compute_delta(prop_layer)),
                ("kernel", lambda: window_kernel.compute_delta(prop_layer)),
                ("streaming", lambda: window_streaming.compute_delta(prop_layer)),
                ("native", lambda: window_native.compute_delta(prop_layer)),
            ],
            batch_factor * _window_edge_count(prop_layer.num_nodes, 8),
        ),
        (
            "SparsePropagation(topk)",
            [
                ("reference", lambda: topk_propagation_reference.compute_delta(prop_layer)),
                ("kernel", lambda: topk_propagation_kernel.compute_delta(prop_layer)),
                ("streaming", lambda: topk_propagation_streaming.compute_delta(prop_layer)),
                ("native", lambda: topk_propagation_native.compute_delta(prop_layer)),
            ],
            batch_factor * prop_layer.num_nodes * 8,
        ),
        (
            "Transition(dense)",
            [
                ("reference", lambda: transition_reference.compute_delta(trans_src, trans_dst)),
                ("kernel", lambda: transition_kernel.compute_delta(trans_src, trans_dst)),
                ("streaming", lambda: transition_streaming.compute_delta(trans_src, trans_dst)),
                ("native", lambda: transition_native.compute_delta(trans_src, trans_dst)),
            ],
            batch_factor * trans_src.num_nodes * trans_dst.num_nodes,
        ),
        (
            "SparseTransition(topk)",
            [
                ("reference", lambda: sparse_transition_reference.compute_delta(trans_src, trans_dst)),
                ("kernel", lambda: sparse_transition_kernel.compute_delta(trans_src, trans_dst)),
                ("streaming", lambda: sparse_transition_streaming.compute_delta(trans_src, trans_dst)),
                ("native", lambda: sparse_transition_native.compute_delta(trans_src, trans_dst)),
            ],
            batch_factor * trans_src.num_nodes * 8,
        ),
        (
            "SparseTransition(pairwise_topk)",
            [
                (
                    "reference",
                    lambda: sparse_transition_pairwise_reference.compute_delta(
                        trans_src, trans_dst
                    ),
                ),
                (
                    "streaming",
                    lambda: sparse_transition_pairwise_streaming.compute_delta(
                        trans_src, trans_dst
                    ),
                ),
                (
                    "kernel",
                    lambda: sparse_transition_pairwise_kernel.compute_delta(
                        trans_src, trans_dst
                    ),
                ),
                (
                    "native",
                    lambda: sparse_transition_pairwise_native.compute_delta(
                        trans_src, trans_dst
                    ),
                ),
            ],
            batch_factor * trans_src.num_nodes * 8,
        ),
        (
            "QueryPropagation(topk)",
            [
                (
                    "reference",
                    lambda: _query_propagation_reference(
                        pairwise_fn=pairwise_template,
                        query_val=query_val,
                        source_val=trans_src.val,
                        projected_state=query_projected_state,
                        projected_val=query_projected_val,
                        topk=32,
                    ),
                ),
                (
                    "native",
                    lambda: propagation_query_topk_native(
                        pairwise_fn=pairwise_template,
                        edge_compress_name="softsign",
                        query_val=query_val,
                        source_val=trans_src.val,
                        projected_state=query_projected_state,
                        projected_val=query_projected_val,
                        topk=32,
                        query_block_size=8,
                        source_block_size=0,
                    ),
                ),
            ],
            batch_factor * query_val.shape[-2] * 32,
        ),
        (
            "QueryTransition(topk)",
            [
                (
                    "reference",
                    lambda: _query_transition_reference(
                        route_fn=query_route_template,
                        sender_strength=query_sender_strength,
                        src_val=trans_src.val,
                        query_val=query_val,
                        projected_state=query_projected_state,
                        projected_val=query_projected_val,
                        topk=32,
                    ),
                ),
                (
                    "native",
                    lambda: transition_query_topk_native(
                        route_fn=query_route_template,
                        sender_strength=query_sender_strength,
                        src_val=trans_src.val,
                        query_val=query_val,
                        projected_state=query_projected_state,
                        projected_val=query_projected_val,
                        topk=32,
                        query_block_size=8,
                        source_block_size=0,
                    ),
                ),
            ],
            batch_factor * query_val.shape[-2] * 32,
        ),
    ]

    print("name                         impl           avg time    peak mem    throughput       max error")
    print("-" * 100)

    for name, implementations, logical_edges in benchmarks:
        outputs: dict[str, object] = {}
        failures: list[BenchmarkFailure] = []
        baseline_name: str | None = None

        for implementation_name, fn in implementations:
            try:
                outputs[implementation_name] = fn()
                if baseline_name is None:
                    baseline_name = implementation_name
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    BenchmarkFailure(
                        name=name,
                        implementation=implementation_name,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

        if baseline_name is None:
            for failure in failures:
                print(_format_failure(failure))
            continue

        reference_output = outputs[baseline_name]
        for implementation_name, fn in implementations:
            if implementation_name not in outputs:
                continue
            output = outputs[implementation_name]
            error = (
                0.0
                if implementation_name == baseline_name
                else _max_abs_error(reference_output, output)
            )
            try:
                avg_ms, peak_mem = _measure(
                    fn, device=device, warmup=args.warmup, iterations=args.iterations
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    BenchmarkFailure(
                        name=name,
                        implementation=implementation_name,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue

            result = BenchmarkResult(
                name=name,
                implementation=implementation_name,
                avg_ms=avg_ms,
                peak_memory_mb=peak_mem,
                logical_edges_per_sec=logical_edges / (avg_ms / 1000.0),
                max_abs_error=error,
            )
            print(_format_result(result))

        for failure in failures:
            print(_format_failure(failure))


if __name__ == "__main__":
    main()
