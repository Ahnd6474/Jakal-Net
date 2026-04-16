from __future__ import annotations

import argparse
import copy
import json
import time
from typing import Callable

import torch

from jakal_net import (
    BilinearPairwiseRoute,
    DiagonalBilinearRoute,
    DiagonalBilinearPairwise,
    HadamardMLPPairwise,
    Layer,
    LowRankBilinearRoute,
    MLPRoute,
    Propagation,
    SourceTargetHadamardMLPRoute,
    Transition,
    native_status,
)


def _clone_named_grads(module: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    result: dict[str, torch.Tensor | None] = {}
    for name, parameter in module.named_parameters():
        result[name] = None if parameter.grad is None else parameter.grad.detach().cpu().clone()
    return result


def _max_grad_diff(
    left: dict[str, torch.Tensor | None],
    right: dict[str, torch.Tensor | None],
) -> tuple[str | None, float]:
    worst_name: str | None = None
    worst_value = 0.0
    for name, left_grad in left.items():
        right_grad = right.get(name)
        if left_grad is None and right_grad is None:
            continue
        if left_grad is None or right_grad is None:
            return name, float("inf")
        diff = (left_grad - right_grad).abs().max().item()
        if diff > worst_value:
            worst_name = name
            worst_value = diff
    return worst_name, worst_value


def _benchmark_step(
    step_fn: Callable[[], None],
    *,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    for _ in range(warmup):
        step_fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(iters):
        step_fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    result = {
        "seconds_per_iter": elapsed / max(iters, 1),
    }
    if device.type == "cuda":
        result["peak_memory_bytes"] = float(torch.cuda.max_memory_allocated(device))
    return result


def _propagation_case(
    *,
    mode: str,
    device: torch.device,
    batch_size: int,
    num_nodes: int,
    dim: int,
    pairwise_kind: str,
    hidden_dim: int,
    seed: int,
) -> tuple[dict[str, float], torch.nn.Module, Layer]:
    torch.manual_seed(seed)
    layer = Layer(
        dim=dim,
        num_nodes=num_nodes,
        state=torch.randn(batch_size, num_nodes, device=device, requires_grad=True),
        val=torch.randn(batch_size, num_nodes, dim, device=device, requires_grad=True),
    )
    if pairwise_kind == "diagonal_bilinear":
        pairwise_fn = DiagonalBilinearPairwise(dim=dim).to(device)
    elif pairwise_kind == "hadamard_mlp":
        pairwise_fn = HadamardMLPPairwise(dim=dim, hidden_dim=hidden_dim).to(device)
    else:
        raise ValueError(f"Unsupported propagation pairwise kind: {pairwise_kind!r}")
    module = Propagation(
        pairwise_fn=pairwise_fn,
        implementation=mode,
    ).to(device)
    delta = module.compute_delta(layer)
    loss = delta.delta_state.square().mean() + delta.delta_val.square().mean()
    loss.backward()
    stats = {
        "loss": float(loss.detach().item()),
        "delta_state_norm": float(delta.delta_state.detach().norm().item()),
        "delta_val_norm": float(delta.delta_val.detach().norm().item()),
    }
    return stats, module, layer


def _transition_case(
    *,
    mode: str,
    device: torch.device,
    batch_size: int,
    src_nodes: int,
    dst_nodes: int,
    src_dim: int,
    dst_dim: int,
    hidden_dim: int,
    route_kind: str,
    seed: int,
) -> tuple[dict[str, float], torch.nn.Module, Layer]:
    torch.manual_seed(seed)
    src_layer = Layer(
        dim=src_dim,
        num_nodes=src_nodes,
        state=torch.randn(batch_size, src_nodes, device=device, requires_grad=True),
        val=torch.randn(batch_size, src_nodes, src_dim, device=device, requires_grad=True),
    )
    dst_layer = Layer.zeros(dim=dst_dim, num_nodes=dst_nodes, batch_shape=(batch_size,), device=device)
    if route_kind == "mlp":
        route_fn = MLPRoute(src_dim=src_dim, dst_nodes=dst_nodes, hidden_dim=hidden_dim).to(device)
    elif route_kind == "low_rank_bilinear":
        if src_dim != dst_dim:
            raise ValueError("low_rank_bilinear route requires matching src/dst dims.")
        route_fn = LowRankBilinearRoute(src_dim=src_dim, dst_dim=dst_dim, rank=hidden_dim).to(device)
    elif route_kind == "diagonal_bilinear":
        if src_dim != dst_dim:
            raise ValueError("diagonal_bilinear route requires matching src/dst dims.")
        route_fn = DiagonalBilinearRoute(src_dim=src_dim, dst_dim=dst_dim).to(device)
    elif route_kind == "bilinear":
        route_fn = BilinearPairwiseRoute(
            src_dim=src_dim,
            dst_dim=dst_dim,
            route_dim=hidden_dim,
        ).to(device)
    elif route_kind == "hadamard_mlp":
        route_fn = SourceTargetHadamardMLPRoute(
            src_dim=src_dim,
            dst_dim=dst_dim,
            route_dim=hidden_dim,
            hidden_dim=hidden_dim,
        ).to(device)
    else:
        raise ValueError(f"Unsupported transition route kind: {route_kind!r}")
    module = Transition(
        route_fn=route_fn,
        val_proj_fn=lambda val: val[..., :dst_dim],
        implementation=mode,
    ).to(device)
    delta = module.compute_delta(src_layer, dst_layer)
    loss = delta.delta_state.square().mean() + delta.delta_val.square().mean()
    loss.backward()
    stats = {
        "loss": float(loss.detach().item()),
        "delta_state_norm": float(delta.delta_state.detach().norm().item()),
        "delta_val_norm": float(delta.delta_val.detach().norm().item()),
    }
    return stats, module, src_layer


def _time_propagation(
    *,
    mode: str,
    device: torch.device,
    batch_size: int,
    num_nodes: int,
    dim: int,
    pairwise_kind: str,
    hidden_dim: int,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    if pairwise_kind == "diagonal_bilinear":
        pairwise_fn = DiagonalBilinearPairwise(dim=dim).to(device)
    elif pairwise_kind == "hadamard_mlp":
        pairwise_fn = HadamardMLPPairwise(dim=dim, hidden_dim=hidden_dim).to(device)
    else:
        raise ValueError(f"Unsupported propagation pairwise kind: {pairwise_kind!r}")
    module = Propagation(
        pairwise_fn=pairwise_fn,
        implementation=mode,
    ).to(device)

    def step() -> None:
        module.zero_grad(set_to_none=True)
        layer = Layer(
            dim=dim,
            num_nodes=num_nodes,
            state=torch.randn(batch_size, num_nodes, device=device, requires_grad=True),
            val=torch.randn(batch_size, num_nodes, dim, device=device, requires_grad=True),
        )
        delta = module.compute_delta(layer)
        loss = delta.delta_state.square().mean() + delta.delta_val.square().mean()
        loss.backward()

    return _benchmark_step(step, device=device, warmup=warmup, iters=iters)


def _time_transition(
    *,
    mode: str,
    device: torch.device,
    batch_size: int,
    src_nodes: int,
    dst_nodes: int,
    src_dim: int,
    dst_dim: int,
    hidden_dim: int,
    route_kind: str,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    if route_kind == "mlp":
        route_fn = MLPRoute(src_dim=src_dim, dst_nodes=dst_nodes, hidden_dim=hidden_dim).to(device)
    elif route_kind == "low_rank_bilinear":
        if src_dim != dst_dim:
            raise ValueError("low_rank_bilinear route requires matching src/dst dims.")
        route_fn = LowRankBilinearRoute(src_dim=src_dim, dst_dim=dst_dim, rank=hidden_dim).to(device)
    elif route_kind == "diagonal_bilinear":
        if src_dim != dst_dim:
            raise ValueError("diagonal_bilinear route requires matching src/dst dims.")
        route_fn = DiagonalBilinearRoute(src_dim=src_dim, dst_dim=dst_dim).to(device)
    elif route_kind == "bilinear":
        route_fn = BilinearPairwiseRoute(
            src_dim=src_dim,
            dst_dim=dst_dim,
            route_dim=hidden_dim,
        ).to(device)
    elif route_kind == "hadamard_mlp":
        route_fn = SourceTargetHadamardMLPRoute(
            src_dim=src_dim,
            dst_dim=dst_dim,
            route_dim=hidden_dim,
            hidden_dim=hidden_dim,
        ).to(device)
    else:
        raise ValueError(f"Unsupported transition route kind: {route_kind!r}")
    module = Transition(
        route_fn=route_fn,
        val_proj_fn=lambda val: val[..., :dst_dim],
        implementation=mode,
    ).to(device)
    dst_layer = Layer.zeros(dim=dst_dim, num_nodes=dst_nodes, batch_shape=(batch_size,), device=device)

    def step() -> None:
        module.zero_grad(set_to_none=True)
        src_layer = Layer(
            dim=src_dim,
            num_nodes=src_nodes,
            state=torch.randn(batch_size, src_nodes, device=device, requires_grad=True),
            val=torch.randn(batch_size, src_nodes, src_dim, device=device, requires_grad=True),
        )
        delta = module.compute_delta(src_layer, dst_layer)
        loss = delta.delta_state.square().mean() + delta.delta_val.square().mean()
        loss.backward()

    return _benchmark_step(step, device=device, warmup=warmup, iters=iters)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark dense native propagation and transition.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prop-nodes", type=int, default=256)
    parser.add_argument("--prop-dim", type=int, default=64)
    parser.add_argument("--prop-hidden-dim", type=int, default=96)
    parser.add_argument(
        "--prop-pairwise-kind",
        default="diagonal_bilinear",
        choices=["diagonal_bilinear", "hadamard_mlp"],
    )
    parser.add_argument("--trans-src-nodes", type=int, default=256)
    parser.add_argument("--trans-dst-nodes", type=int, default=256)
    parser.add_argument("--trans-src-dim", type=int, default=64)
    parser.add_argument("--trans-dst-dim", type=int, default=64)
    parser.add_argument("--trans-hidden-dim", type=int, default=96)
    parser.add_argument(
        "--transition-route-kind",
        default="mlp",
        choices=["mlp", "low_rank_bilinear", "diagonal_bilinear", "bilinear", "hadamard_mlp"],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["reference", "native"],
        choices=["reference", "kernel", "native"],
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    torch.manual_seed(1234)
    native = native_status(force_reload=True)

    report: dict[str, object] = {
        "device": str(device),
        "native_status": {
            "available": native.available,
            "backend_name": native.backend_name,
            "supported_ops": native.supported_ops,
            "supported_devices": native.supported_devices,
            "error": native.error,
        },
        "correctness": {},
        "benchmark": {},
    }

    propagation_runs: dict[str, tuple[dict[str, float], torch.nn.Module, Layer]] = {}
    for mode in args.modes:
        propagation_runs[mode] = _propagation_case(
            mode=mode,
            device=device,
            batch_size=args.batch_size,
            num_nodes=args.prop_nodes,
            dim=args.prop_dim,
            pairwise_kind=args.prop_pairwise_kind,
            hidden_dim=args.prop_hidden_dim,
            seed=101,
        )
        report["benchmark"] = report.get("benchmark", {})

    if "reference" in propagation_runs:
        ref_stats, ref_module, ref_layer = propagation_runs["reference"]
        report["correctness"] = report.get("correctness", {})
        report["correctness"]["propagation_reference"] = ref_stats
        ref_param_grads = _clone_named_grads(ref_module)
        ref_state_grad = ref_layer.state.grad.detach().cpu().clone()
        ref_val_grad = ref_layer.val.grad.detach().cpu().clone()
        ref_delta = ref_module.compute_delta(
            Layer(
                dim=args.prop_dim,
                num_nodes=args.prop_nodes,
                state=ref_layer.state.detach().clone(),
                val=ref_layer.val.detach().clone(),
            )
        )
        for mode, (stats, module, layer) in propagation_runs.items():
            if mode == "reference":
                continue
            param_name, param_diff = _max_grad_diff(ref_param_grads, _clone_named_grads(module))
            compare_delta = module.compute_delta(
                Layer(
                    dim=args.prop_dim,
                    num_nodes=args.prop_nodes,
                    state=layer.state.detach().clone(),
                    val=layer.val.detach().clone(),
                )
            )
            report["correctness"][f"propagation_vs_reference/{mode}"] = {
                "forward_state_max_diff": float(
                    (ref_delta.delta_state.detach() - compare_delta.delta_state.detach()).abs().max().item()
                ),
                "forward_val_max_diff": float(
                    (ref_delta.delta_val.detach() - compare_delta.delta_val.detach()).abs().max().item()
                ),
                "input_state_grad_max_diff": float(
                    (ref_state_grad - layer.state.grad.detach().cpu()).abs().max().item()
                ),
                "input_val_grad_max_diff": float(
                    (ref_val_grad - layer.val.grad.detach().cpu()).abs().max().item()
                ),
                "param_grad_name": param_name,
                "param_grad_max_diff": float(param_diff),
                "loss": stats["loss"],
            }

    transition_runs: dict[str, tuple[dict[str, float], torch.nn.Module, Layer]] = {}
    for mode in args.modes:
        transition_runs[mode] = _transition_case(
            mode=mode,
            device=device,
            batch_size=args.batch_size,
            src_nodes=args.trans_src_nodes,
            dst_nodes=args.trans_dst_nodes,
            src_dim=args.trans_src_dim,
            dst_dim=args.trans_dst_dim,
            hidden_dim=args.trans_hidden_dim,
            route_kind=args.transition_route_kind,
            seed=202,
        )

    if "reference" in transition_runs:
        ref_stats, ref_module, ref_layer = transition_runs["reference"]
        report["correctness"]["transition_reference"] = ref_stats
        ref_param_grads = _clone_named_grads(ref_module)
        ref_state_grad = ref_layer.state.grad.detach().cpu().clone()
        ref_val_grad = ref_layer.val.grad.detach().cpu().clone()
        ref_dst = Layer.zeros(
            dim=args.trans_dst_dim,
            num_nodes=args.trans_dst_nodes,
            batch_shape=(args.batch_size,),
            device=device,
        )
        ref_delta = ref_module.compute_delta(
            Layer(
                dim=args.trans_src_dim,
                num_nodes=args.trans_src_nodes,
                state=ref_layer.state.detach().clone(),
                val=ref_layer.val.detach().clone(),
            ),
            ref_dst,
        )
        for mode, (stats, module, layer) in transition_runs.items():
            if mode == "reference":
                continue
            param_name, param_diff = _max_grad_diff(ref_param_grads, _clone_named_grads(module))
            compare_delta = module.compute_delta(
                Layer(
                    dim=args.trans_src_dim,
                    num_nodes=args.trans_src_nodes,
                    state=layer.state.detach().clone(),
                    val=layer.val.detach().clone(),
                ),
                ref_dst,
            )
            report["correctness"][f"transition_vs_reference/{mode}"] = {
                "forward_state_max_diff": float(
                    (ref_delta.delta_state.detach() - compare_delta.delta_state.detach()).abs().max().item()
                ),
                "forward_val_max_diff": float(
                    (ref_delta.delta_val.detach() - compare_delta.delta_val.detach()).abs().max().item()
                ),
                "input_state_grad_max_diff": float(
                    (ref_state_grad - layer.state.grad.detach().cpu()).abs().max().item()
                ),
                "input_val_grad_max_diff": float(
                    (ref_val_grad - layer.val.grad.detach().cpu()).abs().max().item()
                ),
                "param_grad_name": param_name,
                "param_grad_max_diff": float(param_diff),
                "loss": stats["loss"],
            }

    benchmark_section = report["benchmark"]
    assert isinstance(benchmark_section, dict)
    benchmark_section["propagation"] = {
        mode: _time_propagation(
            mode=mode,
            device=device,
            batch_size=args.batch_size,
            num_nodes=args.prop_nodes,
            dim=args.prop_dim,
            pairwise_kind=args.prop_pairwise_kind,
            hidden_dim=args.prop_hidden_dim,
            warmup=args.warmup,
            iters=args.iters,
        )
        for mode in args.modes
    }
    benchmark_section["transition"] = {
        mode: _time_transition(
            mode=mode,
            device=device,
            batch_size=args.batch_size,
            src_nodes=args.trans_src_nodes,
            dst_nodes=args.trans_dst_nodes,
            src_dim=args.trans_src_dim,
            dst_dim=args.trans_dst_dim,
            hidden_dim=args.trans_hidden_dim,
            route_kind=args.transition_route_kind,
            warmup=args.warmup,
            iters=args.iters,
        )
        for mode in args.modes
    }

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
