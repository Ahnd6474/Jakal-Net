from __future__ import annotations

import torch
from torch import Tensor, nn

from jakal_net._architectural_common import (
    apply_delta,
    layer_with_val_norm,
    make_pairwise,
    signed_abs_softmax_edges,
)
from jakal_net.core import Layer
from jakal_net.modules import LearnedPositionEncoding
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
        implementation: str,
        s_window: int | None = None,
        s_microbatch_size: int | None = None,
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

        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.s_microbatch_size = s_microbatch_size

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_encoding = LearnedPositionEncoding(dim)
        self.anchor_state = nn.Parameter(torch.zeros(()))
        self.anchor_val = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.anchor_val, mean=0.0, std=0.02)

        self.sequence_input_norm = nn.LayerNorm(dim)
        self.sequence_norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(s_layers))
        self.sequence_layers = nn.ModuleList(
            SparsePropagation(
                pairwise_fn=make_pairwise(pairwise_kind, dim=dim, rank=pairwise_rank),
                sparse_type="window",
                window=max_seq_len if s_window is None else max(1, s_window),
                edge_compress_fn=signed_abs_softmax_edges,
                state_weight_edges=True,
                implementation=implementation,
                residual=True,
            )
            for _ in range(s_layers)
        )

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
        return Layer(
            dim=self.dim,
            num_nodes=token_val.shape[-2],
            state=state_projection(token_val).squeeze(-1),
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

        anchor_val = self.anchor_val.expand(batch_size, 1, -1).to(
            device=token_val.device,
            dtype=token_val.dtype,
        )
        anchor_state = self.anchor_state.expand(batch_size, 1).to(
            device=token_val.device,
            dtype=token_val.dtype,
        )
        seq_val = torch.cat((anchor_val, token_val), dim=1)
        seq_state = torch.cat((anchor_state, state_projection(token_val).squeeze(-1)), dim=1)
        layer = Layer(dim=self.dim, num_nodes=seq_len + 1, state=seq_state, val=seq_val)
        for propagation, norm in zip(self.sequence_layers, self.sequence_norms):
            layer = apply_delta(
                layer,
                propagation.compute_delta(layer_with_val_norm(layer, norm)),
                residual=True,
                val_norm=norm,
            )
        return layer
