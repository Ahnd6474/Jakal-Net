from __future__ import annotations

import argparse
import contextlib
import time

import torch
import torch.nn.functional as F

from jakal_net._architectural_common import apply_delta, unit_normalize_values
from jakal_net.causal_memory_lm import CausalMemoryLM
from jakal_net.core import Layer


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextlib.contextmanager
def timed(name: str):
    _sync()
    start = time.perf_counter()
    yield
    _sync()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(f"profile_section | {name} | {elapsed_ms:.3f} ms", flush=True)


def build_model(args: argparse.Namespace, device: torch.device) -> CausalMemoryLM:
    return CausalMemoryLM(
        vocab_size=args.vocab_size,
        dim=args.dim,
        max_seq_len=args.seq_len,
        s_layers=args.s_layers,
        memory_slots=tuple(args.memory_slots),
        memory_update_intervals=None
        if args.memory_update_intervals is None
        else tuple(args.memory_update_intervals),
        prediction_layers=args.prediction_layers,
        s_window=args.s_window,
        memory_topk=args.memory_topk,
        memory_train_mode=args.memory_train_mode,
        memory_eval_mode="topk",
        eval_topk=args.eval_topk,
        scan_backend=args.scan_backend,
        pairwise_kind=args.pairwise_kind,
        route_kind=args.route_kind,
        pairwise_rank=args.pairwise_rank,
        route_rank=args.route_rank,
        pairwise_heads=args.pairwise_heads,
        route_heads=args.route_heads,
        pairwise_anchor_heads=args.pairwise_anchor_heads,
        route_anchor_heads=args.route_anchor_heads,
        implementation="native",
        feed_forward_layers=not args.disable_feed_forward_layers,
        memory_feed_forward_layers=not (
            args.disable_feed_forward_layers or args.disable_memory_feed_forward_layers
        ),
        feed_forward_hidden_mult=args.feed_forward_hidden_mult,
        disable_memory=args.disable_memory,
        disable_memory_read=args.disable_memory_read,
        disable_memory_propagation=args.disable_memory_propagation,
    ).to(device)


def profile_s_encode_sections(
    model: CausalMemoryLM,
    input_ids: torch.Tensor,
) -> Layer:
    s_module = model.s_module
    batch_size, seq_len = input_ids.shape
    with timed("s_embed_norm"):
        token_val = s_module.token_embedding(input_ids)
        token_val = token_val + s_module.position_encoding(
            seq_len,
            device=token_val.device,
            dtype=token_val.dtype,
        ).unsqueeze(0)
        token_val = s_module.sequence_input_norm(token_val)
        token_state_source = token_val
        if s_module.unit_norm_values:
            token_val = unit_normalize_values(token_val)
    with timed("s_anchor_state"):
        anchor_val = s_module.anchor_val.expand(batch_size, 1, -1).to(
            device=token_val.device,
            dtype=token_val.dtype,
        )
        if s_module.unit_norm_values:
            anchor_val = unit_normalize_values(anchor_val)
        anchor_state = s_module.anchor_state.expand(batch_size, 1).to(
            device=token_val.device,
            dtype=token_val.dtype,
        )
        seq_val = torch.cat((anchor_val, token_val), dim=1)
        seq_state = torch.cat(
            (anchor_state, model.value_to_state(token_state_source).squeeze(-1)),
            dim=1,
        )
        layer = Layer(dim=s_module.dim, num_nodes=seq_len + 1, state=seq_state, val=seq_val)
    for layer_index, (propagation, norm) in enumerate(
        zip(s_module.sequence_layers, s_module.sequence_norms)
    ):
        with timed(f"s_layer_{layer_index}_compute_delta"):
            delta = propagation.compute_delta(layer)
        with timed(f"s_layer_{layer_index}_apply_delta_norm"):
            layer = apply_delta(
                layer,
                delta,
                residual=True,
                val_norm=norm,
                unit_norm_values=s_module.unit_norm_values,
            )
        with timed(f"s_layer_{layer_index}_ffn"):
            layer = s_module._apply_ffn(layer, s_module.sequence_ffns[layer_index])
    return layer


def profile_forward_sections(
    model: CausalMemoryLM,
    input_ids: torch.Tensor,
    *,
    autocast_dtype: torch.dtype | None,
    profile_s_internal: bool,
) -> torch.Tensor:
    autocast = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if input_ids.device.type == "cuda" and autocast_dtype is not None
        else contextlib.nullcontext()
    )
    with autocast:
        with timed("s_encode"):
            sequence_layer = (
                profile_s_encode_sections(model, input_ids)
                if profile_s_internal
                else model.s_module.encode(input_ids, state_projection=model.value_to_state)
            )
            aligned_s = sequence_layer.val[:, 1:, :]
        with timed("memory_init_reset"):
            memory_state = model.initialize_memory_state(
                aligned_s.shape[0],
                device=aligned_s.device,
                dtype=aligned_s.dtype,
            )
            current_memory = model.b_module.reset_state(
                memory_state,
                reset_mask=None,
                device=aligned_s.device,
                dtype=aligned_s.dtype,
            )
        if model.disable_memory:
            with timed("memoryless_query"):
                query_layer = model._memoryless_query_layer(aligned_s)
        else:
            with timed("b_scan"):
                scan_output = model._scan_memory(
                    aligned_s,
                    current_memory,
                    knowledge_state=None,
                    reset_mask=None,
                )
                query_layer = scan_output.query_layer
            if model.disable_memory_read:
                with timed("memoryless_query"):
                    query_layer = model._memoryless_query_layer(aligned_s)
        for layer_index, (propagation, norm) in enumerate(
            zip(model.prediction_layers, model.prediction_norms)
        ):
            with timed(f"prediction_propagation_{layer_index}"):
                query_layer = apply_delta(
                    query_layer,
                    propagation.compute_delta(query_layer),
                    residual=True,
                    val_norm=norm,
                    unit_norm_values=model.unit_norm_values,
                )
            with timed(f"prediction_ffn_{layer_index}"):
                ffn_val = model.prediction_ffns[layer_index](query_layer.val)
                if model.unit_norm_values:
                    ffn_val = unit_normalize_values(ffn_val)
                query_layer = query_layer.with_tensors(val=ffn_val)
        with timed("output_norm_head"):
            output_state_source = model.output_norm(query_layer.val)
            output_val = output_state_source
            if model.unit_norm_values:
                output_val = unit_normalize_values(output_val)
                query_layer = query_layer.with_tensors(
                    state=model.value_to_state(output_state_source).squeeze(-1),
                    val=output_val,
                )
            else:
                query_layer = query_layer.with_tensors(
                    state=model.value_to_state(output_state_source).squeeze(-1),
                    val=output_val,
                )
            _ = Layer(dim=model.dim, num_nodes=query_layer.num_nodes, state=query_layer.state, val=output_val)
            logits = model.lm_head(output_val)
    return logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--s-layers", type=int, default=6)
    parser.add_argument("--memory-slots", type=int, nargs="+", default=[384, 96, 24])
    parser.add_argument("--memory-update-intervals", type=int, nargs="+", default=None)
    parser.add_argument("--prediction-layers", type=int, default=3)
    parser.add_argument("--s-window", type=int, default=128)
    parser.add_argument("--memory-topk", type=int, default=16)
    parser.add_argument("--memory-train-mode", choices=("dense", "topk"), default="dense")
    parser.add_argument("--eval-topk", type=int, default=16)
    parser.add_argument(
        "--pairwise-kind",
        choices=("diagonal_bilinear", "low_rank_bilinear", "bilinear", "scaled_cosine"),
        default="low_rank_bilinear",
    )
    parser.add_argument(
        "--route-kind",
        choices=("diagonal_bilinear", "low_rank_bilinear", "bilinear", "fixed_projection", "query_normalized_dot"),
        default="low_rank_bilinear",
    )
    parser.add_argument("--pairwise-rank", type=int, default=128)
    parser.add_argument("--route-rank", type=int, default=128)
    parser.add_argument("--pairwise-heads", type=int, default=4)
    parser.add_argument("--route-heads", type=int, default=4)
    parser.add_argument("--pairwise-anchor-heads", type=int, default=0)
    parser.add_argument("--route-anchor-heads", type=int, default=0)
    parser.add_argument("--feed-forward-hidden-mult", type=float, default=2.0)
    parser.add_argument("--disable-feed-forward-layers", action="store_true")
    parser.add_argument("--disable-memory-feed-forward-layers", action="store_true")
    parser.add_argument("--disable-memory", action="store_true")
    parser.add_argument("--disable-memory-read", action="store_true")
    parser.add_argument("--disable-memory-propagation", action="store_true")
    parser.add_argument("--scan-backend", choices=("auto", "python", "native"), default="native")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--profile-s-internal", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = build_model(args, device)
    model.train()
    input_ids = torch.randint(args.vocab_size, (args.batch_size, args.seq_len), device=device)
    autocast_dtype = torch.bfloat16 if args.bf16 else None

    print(
        "profile_config | "
        f"batch={args.batch_size} | seq={args.seq_len} | dim={args.dim} | "
        f"s_layers={args.s_layers} | prediction_layers={args.prediction_layers} | "
        f"memory_slots={tuple(args.memory_slots)} | pairwise_kind={args.pairwise_kind} | "
        f"route_kind={args.route_kind} | memory_ffn={model.memory_feed_forward_layers} | "
        f"disable_memory={model.disable_memory} | disable_memory_read={model.disable_memory_read} | "
        f"disable_memory_propagation={model.disable_memory_propagation} | "
        f"memory_train_mode={args.memory_train_mode} | scan_backend={args.scan_backend} | "
        f"bf16={args.bf16} | backward={args.backward}",
        flush=True,
    )
    with timed("forward_total"):
        logits = profile_forward_sections(
            model,
            input_ids,
            autocast_dtype=autocast_dtype,
            profile_s_internal=args.profile_s_internal,
        )
    target = torch.randint(args.vocab_size, logits.shape[:-1], device=device)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1))
    print(f"profile_loss | {float(loss.detach()):.6f}", flush=True)
    if args.backward:
        with timed("backward_total"):
            loss.backward()
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024.0**3)
        print(f"profile_cuda | peak_allocated_gb={peak_gb:.3f}", flush=True)


if __name__ == "__main__":
    main()
