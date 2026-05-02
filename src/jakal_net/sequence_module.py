from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from jakal_net._architectural_common import (
    apply_delta,
    make_pairwise,
    signed_abs_softmax_edges,
    signed_softmax_state,
    softsign_state,
)
from jakal_net.core import Layer
from jakal_net.modules import LearnedPositionEncoding, make_residual_postmix
from jakal_net.native_backend import dense_apply_native, dense_apply_native_available
from jakal_net.propagation import SparsePropagation


class SModule(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        max_seq_len: int,
        s_layers: int,
        pairwise_kind: str,
        pairwise_rank: int,
        pairwise_heads: int = 1,
        pairwise_frozen_heads: int = 0,
        pairwise_anchor_heads: int = 0,
        pairwise_anchor_kind: str = "scaled_cosine",
        implementation: str,
        s_window: int | None = None,
        s_microbatch_size: int | None = None,
        checkpoint_sequence_layers: bool = False,
        direction_only_values: bool = False,
        feed_forward_layers: bool = True,
        feed_forward_kind: str = "ffn",
        feed_forward_hidden_mult: float = 2.0,
        feed_forward_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")
        if s_layers <= 0:
            raise ValueError("s_layers must be positive.")
        if s_microbatch_size is not None and s_microbatch_size <= 0:
            raise ValueError("s_microbatch_size must be positive when provided.")
        if feed_forward_hidden_mult <= 0.0:
            raise ValueError("feed_forward_hidden_mult must be positive.")
        if feed_forward_kind not in {"ffn", "linear"}:
            raise ValueError("feed_forward_kind must be 'ffn' or 'linear'.")
        if not 0.0 <= feed_forward_dropout < 1.0:
            raise ValueError("feed_forward_dropout must be in [0, 1).")

        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.s_microbatch_size = s_microbatch_size
        self.checkpoint_sequence_layers = checkpoint_sequence_layers
        self.direction_only_values = direction_only_values
        self.feed_forward_layers = bool(feed_forward_layers)
        self.feed_forward_kind = str(feed_forward_kind)
        self.feed_forward_hidden_mult = float(feed_forward_hidden_mult)
        self.feed_forward_dropout = float(feed_forward_dropout)

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_encoding = LearnedPositionEncoding(dim)
        self.anchor_state = nn.Parameter(torch.zeros(()))
        self.anchor_val = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.anchor_val, mean=0.0, std=0.02)

        if direction_only_values:
            self.sequence_input_norm = nn.Identity()
            self.sequence_norms = nn.ModuleList(nn.Identity() for _ in range(s_layers))
        else:
            self.sequence_input_norm = nn.LayerNorm(dim)
            self.sequence_norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(s_layers))
        self.sequence_layers = nn.ModuleList(
            SparsePropagation(
                pairwise_fn=make_pairwise(
                    pairwise_kind,
                    dim=dim,
                    rank=pairwise_rank,
                    heads=pairwise_heads,
                    frozen_heads=pairwise_frozen_heads,
                    anchor_heads=pairwise_anchor_heads,
                    anchor_kind=pairwise_anchor_kind,
                ),
                sparse_type="window",
                window=max_seq_len if s_window is None else max(1, s_window),
                edge_compress_fn=signed_abs_softmax_edges,
                state_weight_edges=False,
                implementation=implementation,
                residual=True,
                use_direction_only=direction_only_values,
            )
            for _ in range(s_layers)
        )
        self.sequence_ffns = nn.ModuleList(
            (
                make_residual_postmix(
                    self.feed_forward_kind,
                    dim,
                    hidden_mult=self.feed_forward_hidden_mult,
                    dropout=self.feed_forward_dropout,
                )
                if self.feed_forward_layers
                else nn.Identity()
            )
            for _ in range(s_layers)
        )
        self._dense_apply_native_calls = 0
        self._dense_apply_python_calls = 0

    def _can_use_dense_apply_fastpath(self, layer: Layer, propagation: SparsePropagation) -> bool:
        if propagation.sparse_type != "window":
            return False
        return int(propagation.window or 0) + 1 >= int(layer.num_nodes)

    @staticmethod
    def _norm_param_or_empty(module: nn.Module, name: str, reference: Tensor) -> Tensor:
        tensor = getattr(module, name, None)
        if isinstance(tensor, Tensor):
            return tensor.to(device=reference.device, dtype=reference.dtype)
        return reference.new_empty((0,))

    def reset_dense_apply_stats(self) -> None:
        self._dense_apply_native_calls = 0
        self._dense_apply_python_calls = 0

    def dense_apply_stats(self) -> dict[str, int]:
        return {
            "native_calls": int(self._dense_apply_native_calls),
            "python_calls": int(self._dense_apply_python_calls),
        }

    def _apply_dense_delta_fastpath(
        self,
        layer: Layer,
        delta_state: Tensor,
        delta_val: Tensor,
        norm: nn.Module,
        propagation: SparsePropagation,
    ) -> Layer:
        state_activation_name = "softsign" if self.direction_only_values else "signed_softmax"
        if propagation.implementation == "native" and dense_apply_native_available(layer.val.device.type):
            next_state, next_val = dense_apply_native(
                layer_state=layer.state,
                layer_val=layer.val,
                delta_state=delta_state,
                delta_val=delta_val,
                val_norm_weight=self._norm_param_or_empty(norm, "weight", layer.val),
                val_norm_bias=self._norm_param_or_empty(norm, "bias", layer.val),
                state_activation_name=state_activation_name,
            )
            self._dense_apply_native_calls += 1
            return layer.with_tensors(state=next_state, val=next_val)
        self._dense_apply_python_calls += 1
        updated_val = layer.val + delta_val
        return layer.with_tensors(
            state=softsign_state(layer.state + delta_state)
            if self.direction_only_values
            else signed_softmax_state(layer.state + delta_state),
            val=norm(updated_val),
        )

    def _apply_ffn(self, layer: Layer, ffn: nn.Module) -> Layer:
        val = ffn(layer.val)
        return layer.with_tensors(val=val)

    def encode(
        self,
        input_ids: Tensor,
        *,
        state_projection: nn.Module,
    ) -> Layer:
        if self.s_microbatch_size is None or input_ids.shape[0] <= self.s_microbatch_size:
            return self._encode_single(input_ids, state_projection=state_projection)

        chunks: list[Layer] = []
        for start in range(0, input_ids.shape[0], self.s_microbatch_size):
            end = min(start + self.s_microbatch_size, input_ids.shape[0])
            chunks.append(
                self._encode_single(input_ids[start:end], state_projection=state_projection)
            )
        return Layer(
            dim=self.dim,
            num_nodes=chunks[0].num_nodes,
            state=torch.cat([chunk.state for chunk in chunks], dim=0),
            val=torch.cat([chunk.val for chunk in chunks], dim=0),
        )

    def make_token_layer(
        self,
        token_val: Tensor,
        *,
        state_projection: nn.Module,
    ) -> Layer:
        token_state_source = token_val
        token_state = state_projection(token_state_source).squeeze(-1)
        if self.direction_only_values:
            token_state = softsign_state(token_state)
        return Layer(
            dim=self.dim,
            num_nodes=token_val.shape[-2],
            state=token_state,
            val=token_val,
        )

    def _encode_single(
        self,
        input_ids: Tensor,
        *,
        state_projection: nn.Module,
    ) -> Layer:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len].")
        batch_size, seq_len = input_ids.shape
        if seq_len <= 0:
            raise ValueError("seq_len must be positive.")
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds configured max_seq_len={self.max_seq_len}."
            )
        token_val = self.token_embedding(input_ids)
        token_val = token_val + self.position_encoding(
            seq_len,
            device=token_val.device,
            dtype=token_val.dtype,
        ).unsqueeze(0)
        token_val = self.sequence_input_norm(token_val)
        token_state_source = token_val

        anchor_val = self.anchor_val.expand(batch_size, 1, -1).to(
            device=token_val.device,
            dtype=token_val.dtype,
        )
        anchor_state = self.anchor_state.expand(batch_size, 1).to(
            device=token_val.device,
            dtype=token_val.dtype,
        )
        token_state = state_projection(token_state_source).squeeze(-1)
        if self.direction_only_values:
            anchor_state = softsign_state(anchor_state)
            token_state = softsign_state(token_state)
        seq_val = torch.cat((anchor_val, token_val), dim=1)
        seq_state = torch.cat((anchor_state, token_state), dim=1)
        layer = Layer(dim=self.dim, num_nodes=seq_len + 1, state=seq_state, val=seq_val)
        for layer_index, (propagation, norm) in enumerate(zip(self.sequence_layers, self.sequence_norms)):
            if self.checkpoint_sequence_layers and torch.is_grad_enabled():
                def _run_sequence_layer(state: Tensor, val: Tensor) -> tuple[Tensor, Tensor]:
                    current_layer = Layer(dim=self.dim, num_nodes=seq_len + 1, state=state, val=val)
                    next_layer = apply_delta(
                        current_layer,
                        propagation.compute_delta(current_layer),
                        residual=True,
                        val_norm=norm,
                        direction_only_values=self.direction_only_values,
                    )
                    return next_layer.state, next_layer.val

                next_state, next_val = torch_checkpoint(
                    _run_sequence_layer,
                    layer.state,
                    layer.val,
                    use_reentrant=False,
                )
                layer = Layer(dim=self.dim, num_nodes=seq_len + 1, state=next_state, val=next_val)
            else:
                delta = propagation.compute_delta(layer)
                if self._can_use_dense_apply_fastpath(layer, propagation):
                    layer = self._apply_dense_delta_fastpath(
                        layer,
                        delta.delta_state,
                        delta.delta_val,
                        norm,
                        propagation,
                    )
                else:
                    layer = apply_delta(
                        layer,
                        delta,
                        residual=True,
                        val_norm=norm,
                        direction_only_values=self.direction_only_values,
                    )
            layer = self._apply_ffn(layer, self.sequence_ffns[layer_index])
        return layer
