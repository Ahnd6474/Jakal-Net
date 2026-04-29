from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from jakal_net._architectural_common import apply_delta  # noqa: E402
from jakal_net.causal_memory_lm import CausalHierarchicalMemoryLM  # noqa: E402
from jakal_net.core import LayerDelta  # noqa: E402
from train_causal_memory_lm import (  # noqa: E402
    FlatDocumentChunkBatcher,
    compute_masked_loss,
    load_flat_pretokenized_directory,
    move_batch_to_device,
    resolve_stage_bucket_weights,
    split_train_val_flat_documents_with_collection,
)


def _tensor_stats(prefix: str, tensor: torch.Tensor) -> dict[str, float]:
    data = tensor.detach().float()
    return {
        f"{prefix}_l2": float(torch.linalg.vector_norm(data).item()),
        f"{prefix}_mean_abs": float(data.abs().mean().item()),
        f"{prefix}_max_abs": float(data.abs().max().item()),
    }


def _grad_l2(tensor: torch.Tensor) -> float:
    grad = tensor.grad
    if grad is None:
        return 0.0
    return float(torch.linalg.vector_norm(grad.detach().float()).item())


def _edge_row_stats(edges: torch.Tensor) -> dict[str, float]:
    probs = edges.detach().float().abs()
    row_sums = probs.sum(dim=-1).clamp_min(1e-12)
    probs = probs / row_sums.unsqueeze(-1)
    row_max = probs.max(dim=-1).values
    entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
    return {
        "edge_row_max_mean": float(row_max.mean().item()),
        "edge_row_max_max": float(row_max.max().item()),
        "edge_entropy_mean": float(entropy.mean().item()),
        "edge_entropy_min": float(entropy.min().item()),
    }


def _build_model_kwargs(cfg: dict[str, Any], vocab_size: int) -> dict[str, Any]:
    return {
        "vocab_size": int(vocab_size),
        "dim": int(cfg["dim"]),
        "max_seq_len": int(cfg["seq_len"]),
        "s_layers": int(cfg["s_layers"]),
        "memory_slots": tuple(int(x) for x in cfg["memory_slots"]),
        "memory_update_intervals": tuple(int(x) for x in cfg["memory_update_intervals"]),
        "prediction_layers": int(cfg["prediction_layers"]),
        "s_window": int(cfg["s_window"]),
        "s_microbatch_size": None
        if int(cfg["s_microbatch_size"]) <= 0
        else int(cfg["s_microbatch_size"]),
        "prediction_window": int(cfg["prediction_window"]),
        "checkpoint_sequence_layers": bool(cfg["checkpoint_sequence_layers"]),
        "checkpoint_prediction_layers": bool(cfg["checkpoint_prediction_layers"]),
        "memory_topk": int(cfg["memory_topk"]),
        "memory_train_mode": str(cfg["memory_train_mode"]),
        "memory_eval_mode": str(cfg["memory_eval_mode"]),
        "eval_topk": int(cfg["eval_topk"]) if cfg["eval_topk"] is not None else None,
        "pairwise_kind": str(cfg["pairwise_kind"]),
        "route_kind": str(cfg["route_kind"]),
        "pairwise_rank": int(cfg["pairwise_rank"]),
        "route_rank": int(cfg["route_rank"]),
        "pairwise_heads": int(cfg["pairwise_heads"]),
        "route_heads": int(cfg["route_heads"]),
        "pairwise_frozen_heads": int(cfg["pairwise_frozen_heads"]),
        "route_frozen_heads": int(cfg["route_frozen_heads"]),
        "pairwise_anchor_heads": int(cfg["pairwise_anchor_heads"]),
        "route_anchor_heads": int(cfg["route_anchor_heads"]),
        "pairwise_anchor_kind": str(cfg["pairwise_anchor_kind"]),
        "route_anchor_kind": str(cfg["route_anchor_kind"]),
        "pairwise_head_aggregate": str(cfg["pairwise_head_aggregate"]),
        "sequence_anchor": not bool(cfg["disable_sequence_anchor"]),
        "scan_backend": str(cfg["scan_backend"]),
        "scan_checkpoint_chunk_size": None
        if int(cfg["scan_checkpoint_chunk_size"]) <= 0
        else int(cfg["scan_checkpoint_chunk_size"]),
        "implementation": str(cfg["implementation"]),
        "unit_norm_values": False,
        "feed_forward_layers": not bool(cfg["disable_feed_forward_layers"]),
        "memory_feed_forward_layers": not bool(cfg["disable_memory_feed_forward_layers"]),
        "disable_memory": bool(cfg["disable_memory"]),
        "disable_memory_read": bool(cfg["disable_memory_read"]),
        "disable_memory_propagation": bool(cfg["disable_memory_propagation"]),
        "feed_forward_hidden_mult": float(cfg["feed_forward_hidden_mult"]),
        "feed_forward_kind": str(cfg["feed_forward_kind"]),
        "feed_forward_residual_scale": float(cfg["feed_forward_residual_scale"]),
        "feed_forward_zero_init_output": not bool(cfg["feed_forward_random_output_init"]),
        "feed_forward_activation": str(cfg["feed_forward_activation"]),
        "knowledge_nodes": int(cfg["knowledge_nodes"]),
        "knowledge_route_topk": int(cfg["knowledge_route_topk"]),
        "knowledge_propagation_topk": int(cfg["knowledge_propagation_topk"]),
        "knowledge_propagation_layers": int(cfg["knowledge_propagation_layers"]),
    }


def _configure_unit_norm(model: CausalHierarchicalMemoryLM, enabled: bool) -> CausalHierarchicalMemoryLM:
    model.unit_norm_values = bool(enabled)
    model.s_module.unit_norm_values = bool(enabled)
    for propagation in model.prediction_layers:
        propagation.use_direction_only = bool(enabled)
    return model


def _sum_head_grad_norms(pairwise_fn: Any) -> tuple[float, float, float]:
    source_sq = 0.0
    target_sq = 0.0
    core_sq = 0.0
    for head in pairwise_fn.heads:
        if head.source_proj.weight.grad is not None:
            source_sq += float(torch.linalg.vector_norm(head.source_proj.weight.grad.detach().float()).item()) ** 2
        if head.target_proj.weight.grad is not None:
            target_sq += float(torch.linalg.vector_norm(head.target_proj.weight.grad.detach().float()).item()) ** 2
        if head.weight.grad is not None:
            core_sq += float(torch.linalg.vector_norm(head.weight.grad.detach().float()).item()) ** 2
    return source_sq**0.5, target_sq**0.5, core_sq**0.5


def _run_case(
    base_model: CausalHierarchicalMemoryLM,
    batch: Any,
    *,
    unit_norm: bool,
) -> dict[str, Any]:
    device = batch.context.device
    model = copy.deepcopy(base_model).to(device=device, dtype=torch.bfloat16)
    _configure_unit_norm(model, unit_norm)
    model.train()
    model.zero_grad(set_to_none=True)

    sequence_layer = model.s_module.encode(batch.context, state_projection=model.value_to_state)
    aligned_s = sequence_layer.val[:, 1:, :] if model.sequence_anchor else sequence_layer.val
    query_layer = model._memoryless_query_layer(aligned_s)
    input_query_val = query_layer.val
    input_query_val.retain_grad()

    propagation = model.prediction_layers[0]
    norm = model.prediction_norms[0]
    directional_val = propagation._directional_val(query_layer.state, query_layer.val)
    directional_val.retain_grad()

    head_scores = propagation.pairwise_fn.head_scores(directional_val, directional_val)
    head_scores.retain_grad()
    smoothmax_fp32 = torch.logsumexp(head_scores.float(), dim=-1) - math.log(
        float(propagation.pairwise_fn.num_heads)
    )
    scores = propagation.pairwise_fn(directional_val, directional_val)
    scores.retain_grad()
    edges = propagation.edge_compress_fn(scores)
    edges = propagation._weight_edges(edges, query_layer.state)
    edges.retain_grad()

    projected_state, projected_val = propagation._project_inputs(query_layer, directional_val=directional_val)
    if propagation.state_weight_edges:
        delta_state = edges.sum(dim=-1)
    else:
        delta_state = torch.einsum("...ij,...j->...i", edges, projected_state)
    delta_val = torch.einsum("...ij,...jd->...id", edges, projected_val)
    delta_state.retain_grad()
    delta_val.retain_grad()

    query_layer = apply_delta(
        query_layer,
        LayerDelta(delta_state=delta_state, delta_val=delta_val),
        residual=True,
        val_norm=norm,
        unit_norm_values=model.unit_norm_values,
    )

    for layer_index in range(1, len(model.prediction_layers)):
        next_prop = model.prediction_layers[layer_index]
        next_norm = model.prediction_norms[layer_index]
        delta = next_prop.compute_delta(query_layer)
        if model._can_use_dense_apply_fastpath(query_layer, next_prop):
            query_layer = model._apply_dense_delta_fastpath(
                query_layer, delta.delta_state, delta.delta_val, next_norm
            )
        else:
            query_layer = apply_delta(
                query_layer,
                delta,
                residual=True,
                val_norm=next_norm,
                unit_norm_values=model.unit_norm_values,
            )

        ffn = model.prediction_ffns[layer_index]
        if ffn.__class__.__name__ == "StateValueFeedForward":
            ffn_state, ffn_val = ffn(query_layer.state, query_layer.val)
            if model.unit_norm_values:
                query_layer = query_layer.with_tensors(state=ffn_state, val=ffn_val)
            else:
                query_layer = query_layer.with_tensors(state=ffn_state, val=ffn_val)
        else:
            query_layer = query_layer.with_tensors(val=ffn(query_layer.val))

    output_state_source = model.output_norm(query_layer.val)
    output_val = output_state_source
    logits = model.lm_head(output_val)
    loss = compute_masked_loss(logits, batch.target, batch.loss_mask)
    loss.backward()

    lowrank_source, lowrank_target, lowrank_core = _sum_head_grad_norms(propagation.pairwise_fn)
    return {
        "unit_norm": unit_norm,
        "loss": float(loss.detach().item()),
        **_tensor_stats("query_val", input_query_val),
        **_tensor_stats("directional_val", directional_val),
        **_tensor_stats("head_scores", head_scores),
        **_tensor_stats("smoothmax_fp32", smoothmax_fp32),
        **_tensor_stats("scores", scores),
        **_edge_row_stats(edges),
        **_tensor_stats("delta_state", delta_state),
        **_tensor_stats("delta_val", delta_val),
        "grad_query_val_l2": _grad_l2(input_query_val),
        "grad_directional_val_l2": _grad_l2(directional_val),
        "grad_head_scores_l2": _grad_l2(head_scores),
        "grad_scores_l2": _grad_l2(scores),
        "grad_edges_l2": _grad_l2(edges),
        "grad_delta_state_l2": _grad_l2(delta_state),
        "grad_delta_val_l2": _grad_l2(delta_val),
        "grad_lowrank_source_l2": lowrank_source,
        "grad_lowrank_target_l2": lowrank_target,
        "grad_lowrank_core_l2": lowrank_core,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = json.load(f)["args"]

    device = torch.device("cuda")
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    collection = load_flat_pretokenized_directory(
        REPO_ROOT / cfg["pretokenized_dir"],
        load_workers=1,
        max_loaded_shards=0,
        integrity_mode="meta",
        integrity_workers=1,
    )
    train_docs, _ = split_train_val_flat_documents_with_collection(
        collection, collection.document_refs, train_fraction=float(cfg["train_fraction"])
    )
    batcher = FlatDocumentChunkBatcher(
        collection,
        train_docs,
        batch_size=int(args.batch_size),
        device=device,
        active_shards_per_bucket=int(cfg["pretokenized_active_shards_per_bucket"]),
        shard_rotation_interval=int(cfg["pretokenized_shard_rotation_interval"]),
    )
    batcher.set_bucket_weights(resolve_stage_bucket_weights(stage_name="stage3"))
    batch = move_batch_to_device(batcher.next_batch(), device=device, non_blocking=False)

    model_kwargs = _build_model_kwargs(cfg, collection.vocab_size)
    base_model = CausalHierarchicalMemoryLM(**model_kwargs).to(device=device, dtype=torch.bfloat16)
    base_model.train()

    results = {
        "config": str(args.config),
        "batch_size": int(args.batch_size),
        "results": [
            _run_case(base_model, batch, unit_norm=False),
            _run_case(base_model, batch, unit_norm=True),
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
