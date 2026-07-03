from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from jakal_net.core import Layer

PARAM_INIT_STD = 0.02


@dataclass(frozen=True, slots=True)
class ModelRecurrentState:
    memory_state: tuple[Tensor, ...]
    knowledge_state: Tensor | None = None


@dataclass(frozen=True, slots=True)
class MemoryScanOutput:
    logits: Tensor
    memory_state: tuple[Tensor, ...]
    knowledge_state: Tensor | None = None
    sequence_layer: Layer | None = None
    query_layer: Layer | None = None

    @property
    def recurrent_state(self) -> ModelRecurrentState:
        return ModelRecurrentState(
            memory_state=self.memory_state,
            knowledge_state=self.knowledge_state,
        )


class ValueNormStateProjection(nn.Module):
    def __init__(self, *, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, val: Tensor) -> Tensor:
        return torch.linalg.vector_norm(val, ord=2, dim=-1, keepdim=True).clamp_min(self.eps)


class _TokenEmbeddingStack(nn.Module):
    def __init__(self, *, vocab_size: int, dim: int, max_seq_len: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)

    def forward(self, input_ids: Tensor) -> Tensor:
        seq_len = int(input_ids.shape[1])
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        return self.token_embedding(input_ids) + self.position_embedding(positions)


class _KnowledgeTraceBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        memory_size: int,
        beta_dim: int,
        num_heads: int,
        num_layers: int,
        hops: int,
        relation_rank: int,
        activation: str,
    ) -> None:
        super().__init__()
        if memory_size <= 0:
            raise ValueError("memory_size must be positive.")
        if beta_dim <= 0:
            raise ValueError("beta_dim must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if hops <= 0:
            raise ValueError("hops must be positive.")
        if activation not in {"relu", "gelu", "softplus"}:
            raise ValueError(f"Unsupported knowledge activation: {activation!r}.")

        self.dim = int(dim)
        self.memory_size = int(memory_size)
        self.beta_dim = int(beta_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.hops = int(hops)
        self.relation_rank = int(relation_rank)
        self.activation_name = activation

        self.memory = nn.Parameter(torch.empty(self.memory_size, self.dim))
        self.relation = nn.Parameter(torch.empty(self.memory_size, self.memory_size))
        self.value = nn.Parameter(torch.empty(self.num_layers, self.memory_size, self.dim))
        self.share_logit = nn.Parameter(torch.full((self.num_layers, self.num_heads), -2.0))
        self.prior_logit = nn.Parameter(torch.full((self.num_layers, self.num_heads), -4.0))
        if self.hops > 1:
            self.hop_residual_logit = nn.Parameter(torch.full((self.num_layers, self.hops - 1), -2.0))
        else:
            self.register_parameter("hop_residual_logit", None)
        self.residual_gate_logit = nn.Parameter(torch.zeros(self.num_layers))

        self.track_stats = False
        self.last_stats: dict[str, float] = {}
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.memory, mean=0.0, std=PARAM_INIT_STD)
        nn.init.normal_(self.relation, mean=0.0, std=PARAM_INIT_STD)
        nn.init.normal_(self.value, mean=0.0, std=PARAM_INIT_STD)

    def lookup(self, hidden: Tensor) -> Tensor:
        return torch.matmul(hidden, self.memory.transpose(0, 1))

    def _activation(self, tensor: Tensor) -> Tensor:
        if self.activation_name == "relu":
            return F.relu(tensor)
        if self.activation_name == "gelu":
            return F.gelu(tensor)
        return F.softplus(tensor)

    def forward(
        self,
        activations: Tensor,
        *,
        attention_probs: Tensor,
        prior_trace: Tensor | None,
        layer_index: int,
    ) -> tuple[Tensor, Tensor, dict[str, float] | None]:
        if not 0 <= layer_index < self.num_layers:
            raise ValueError(f"layer_index must be in [0, {self.num_layers}), got {layer_index}.")
        batch_size, seq_len, _ = activations.shape
        if prior_trace is None:
            prior_trace = activations.new_zeros((batch_size, self.num_heads, self.memory_size))
        if attention_probs.shape != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                "attention_probs must have shape "
                f"({batch_size}, {self.num_heads}, {seq_len}, {seq_len}), got {tuple(attention_probs.shape)}."
            )

        shared = torch.einsum("bhts,bsk->bhtk", attention_probs, activations)
        share_mix = torch.sigmoid(self.share_logit[layer_index]).view(1, self.num_heads, 1, 1)
        traced_by_head = (
            (1.0 - share_mix) * activations.unsqueeze(1)
            + share_mix * shared
        )
        prior_mix = torch.sigmoid(self.prior_logit[layer_index]).view(1, self.num_heads, 1, 1)
        traced_by_head = traced_by_head + prior_mix * prior_trace.unsqueeze(2)
        traced = traced_by_head.mean(dim=1)
        relation_scores = self._activation(torch.matmul(traced, self.relation))
        if self.hops > 1:
            assert self.hop_residual_logit is not None
            propagated = relation_scores
            for hop_index in range(self.hops - 1):
                hop_update = self._activation(torch.matmul(propagated, self.relation))
                hop_gate = torch.sigmoid(self.hop_residual_logit[layer_index, hop_index]).to(dtype=hop_update.dtype)
                propagated = propagated + hop_gate * hop_update
            relation_scores = propagated
        next_trace = traced_by_head[:, :, -1, :]

        stats = None
        if self.track_stats:
            beta_probs = attention_probs.clamp_min(1.0e-12)
            beta_entropy = -(beta_probs * beta_probs.log()).sum(dim=-1).mean()
            stats = {
                "knowledge/share_mean": float(share_mix.detach().float().mean().item()),
                "knowledge/prior_mean": float(prior_mix.detach().float().mean().item()),
                "knowledge/beta_entropy": float(beta_entropy.detach().float().item()),
                "knowledge/trace_rms": float(next_trace.detach().float().square().mean().sqrt().item()),
                "knowledge/activation_mean": float(relation_scores.detach().float().mean().item()),
                "knowledge/relation_rms": float(self.relation.detach().float().square().mean().sqrt().item()),
            }
            if self.hop_residual_logit is not None:
                stats["knowledge/hop_gate_mean"] = float(
                    torch.sigmoid(self.hop_residual_logit[layer_index]).detach().float().mean().item()
                )

        return relation_scores, next_trace, stats


class _CausalTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        ff_dim: int,
        dropout: float,
        activation: str,
        enable_feed_forward: bool,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.enable_feed_forward = bool(enable_feed_forward)
        if self.enable_feed_forward:
            self.feed_forward_norm = nn.LayerNorm(dim)
            self.feed_forward_in = nn.Linear(dim, ff_dim)
            self.feed_forward_out = nn.Linear(ff_dim, dim)
            self.feed_forward_dropout = nn.Dropout(dropout)
            self.feed_forward_activation_name = activation
        else:
            self.feed_forward_norm = None
            self.feed_forward_in = None
            self.feed_forward_out = None
            self.feed_forward_dropout = None
            self.feed_forward_activation_name = activation

    def _activation(self, tensor: Tensor) -> Tensor:
        if self.feed_forward_activation_name == "gelu":
            return F.gelu(tensor)
        if self.feed_forward_activation_name == "relu":
            return F.relu(tensor)
        if self.feed_forward_activation_name == "silu":
            return F.silu(tensor)
        raise ValueError(f"Unsupported feed_forward_activation: {self.feed_forward_activation_name!r}.")

    def attend(
        self,
        hidden: Tensor,
        *,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        attention_input = self.attention_norm(hidden)
        query_weight = self.self_attention.in_proj_weight[: self.self_attention.embed_dim]
        query_bias = None
        if self.self_attention.in_proj_bias is not None:
            query_bias = self.self_attention.in_proj_bias[: self.self_attention.embed_dim]
        query_states = F.linear(attention_input, query_weight, query_bias)
        attention_output, _ = self.self_attention(
            attention_input,
            attention_input,
            attention_input,
            attn_mask=attention_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        hidden = hidden + self.attention_dropout(attention_output)
        return hidden, attention_input, _, query_states

    def feed_forward(self, hidden: Tensor) -> Tensor:
        if not self.enable_feed_forward:
            return hidden

        assert self.feed_forward_norm is not None
        assert self.feed_forward_in is not None
        assert self.feed_forward_out is not None
        assert self.feed_forward_dropout is not None
        feed_forward_hidden = self.feed_forward_norm(hidden)
        feed_forward_hidden = self.feed_forward_in(feed_forward_hidden)
        feed_forward_hidden = self._activation(feed_forward_hidden)
        feed_forward_hidden = self.feed_forward_dropout(feed_forward_hidden)
        feed_forward_hidden = self.feed_forward_out(feed_forward_hidden)
        return hidden + self.feed_forward_dropout(feed_forward_hidden)


class CausalMemoryLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int = 512,
        max_seq_len: int = 2048,
        transformer_layers: int = 6,
        scan_backend: str = "auto",
        scan_checkpoint_chunk_size: int | None = None,
        feed_forward_layers: bool = True,
        feed_forward_hidden_mult: float = 4.0,
        feed_forward_activation: str = "gelu",
        tie_embedding_head: bool = True,
        transformer_heads: int = 8,
        transformer_dropout: float = 0.0,
        knowledge_memory_size: int = 2048,
        knowledge_beta_dim: int = 64,
        knowledge_hops: int = 1,
        knowledge_relation_rank: int = 64,
        knowledge_activation: str = "relu",
        **legacy_kwargs: Any,
    ) -> None:
        super().__init__()
        del legacy_kwargs
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")
        if transformer_heads <= 0 or dim % transformer_heads != 0:
            raise ValueError("transformer_heads must be positive and divide dim.")
        if not 0.0 <= transformer_dropout < 1.0:
            raise ValueError("transformer_dropout must be in [0, 1).")
        if feed_forward_hidden_mult <= 0.0:
            raise ValueError("feed_forward_hidden_mult must be positive.")
        if scan_backend not in {"auto", "python", "native"}:
            raise ValueError(f"Unsupported scan_backend: {scan_backend!r}.")
        if scan_checkpoint_chunk_size is not None and scan_checkpoint_chunk_size <= 0:
            raise ValueError("scan_checkpoint_chunk_size must be positive when provided.")

        total_layers = int(transformer_layers)
        if total_layers <= 0:
            raise ValueError("transformer layer count must be positive.")

        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.max_seq_len = int(max_seq_len)
        self.transformer_layer_count = total_layers
        self.legacy_sequence_layers_count = total_layers
        self.legacy_prediction_layers_count = 0
        self.scan_backend = scan_backend
        self.scan_checkpoint_chunk_size = scan_checkpoint_chunk_size
        self.feed_forward_layers = bool(feed_forward_layers)
        self.feed_forward_hidden_mult = float(feed_forward_hidden_mult)
        self.feed_forward_activation = str(feed_forward_activation)
        self.transformer_heads = int(transformer_heads)
        self.transformer_dropout = float(transformer_dropout)
        self.knowledge_memory_size = int(knowledge_memory_size)
        self.knowledge_beta_dim = int(knowledge_beta_dim)
        self.knowledge_hops = int(knowledge_hops)
        self.knowledge_relation_rank = int(knowledge_relation_rank)
        self.knowledge_activation = str(knowledge_activation)
        self.value_to_state = ValueNormStateProjection()
        if self.feed_forward_activation == "gelu":
            encoder_activation: str | Any = "gelu"
        elif self.feed_forward_activation == "relu":
            encoder_activation = "relu"
        elif self.feed_forward_activation == "silu":
            encoder_activation = F.silu
        else:
            raise ValueError(f"Unsupported feed_forward_activation: {self.feed_forward_activation!r}.")

        self.s_module = _TokenEmbeddingStack(
            vocab_size=self.vocab_size,
            dim=self.dim,
            max_seq_len=self.max_seq_len,
        )
        self.knowledge_block = _KnowledgeTraceBlock(
            dim=self.dim,
            memory_size=self.knowledge_memory_size,
            beta_dim=self.knowledge_beta_dim,
            num_heads=self.transformer_heads,
            num_layers=self.transformer_layer_count,
            hops=self.knowledge_hops,
            relation_rank=self.knowledge_relation_rank,
            activation=self.knowledge_activation,
        )
        ff_dim = max(self.dim, int(round(self.dim * self.feed_forward_hidden_mult)))
        self.encoder_layers = nn.ModuleList(
            _CausalTransformerBlock(
                dim=self.dim,
                heads=self.transformer_heads,
                ff_dim=ff_dim,
                dropout=self.transformer_dropout,
                activation=self.feed_forward_activation,
                enable_feed_forward=self.feed_forward_layers,
            )
            for _ in range(self.transformer_layer_count)
        )
        self.output_norm = nn.LayerNorm(self.dim)
        self.lm_head = nn.Linear(self.dim, self.vocab_size, bias=False)
        if tie_embedding_head:
            self.lm_head.weight = self.s_module.token_embedding.weight
        self._reset_parameters(tie_embedding_head=tie_embedding_head)

    @property
    def s_layers(self) -> int:
        return self.legacy_sequence_layers_count

    @property
    def prediction_layers_count(self) -> int:
        return self.legacy_prediction_layers_count

    def _reset_parameters(self, *, tie_embedding_head: bool) -> None:
        nn.init.normal_(self.s_module.token_embedding.weight, mean=0.0, std=PARAM_INIT_STD)
        nn.init.normal_(self.s_module.position_embedding.weight, mean=0.0, std=PARAM_INIT_STD)
        if not tie_embedding_head:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=PARAM_INIT_STD)

    def set_track_stats(self, enabled: bool) -> None:
        self.knowledge_block.track_stats = bool(enabled)
        if not enabled:
            self.knowledge_block.last_stats = {}

    def collect_internal_stats(self) -> dict[str, float]:
        return dict(self.knowledge_block.last_stats)

    def initialize_memory_state(
        self,
        batch_size: int,
        *,
        device,
        dtype,
    ) -> tuple[Tensor, ...]:
        return (
            torch.zeros(
                int(batch_size),
                self.transformer_layer_count,
                self.transformer_heads,
                self.knowledge_memory_size,
                device=device,
                dtype=dtype,
            ),
        )

    def _coerce_memory_state(
        self,
        memory_state: Sequence[Tensor] | ModelRecurrentState | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        reset_mask: Tensor | None,
    ) -> Tensor:
        if isinstance(memory_state, ModelRecurrentState):
            memory_state = memory_state.memory_state
        if memory_state is None or len(tuple(memory_state)) == 0:
            trace = self.initialize_memory_state(batch_size, device=device, dtype=dtype)[0]
        else:
            trace = tuple(memory_state)[0]
            if trace.shape != (batch_size, self.transformer_layer_count, self.transformer_heads, self.knowledge_memory_size):
                raise ValueError(
                    "memory_state[0] must have shape "
                    f"({batch_size}, {self.transformer_layer_count}, {self.transformer_heads}, {self.knowledge_memory_size}), got {tuple(trace.shape)}."
                )
            trace = trace.to(device=device, dtype=dtype)
        if reset_mask is not None:
            if reset_mask.shape != (batch_size,):
                raise ValueError(
                    f"reset_mask must have shape ({batch_size},), got {tuple(reset_mask.shape)}."
                )
            trace = torch.where(
                reset_mask.to(device=device, dtype=torch.bool).view(batch_size, 1, 1, 1),
                torch.zeros_like(trace),
                trace,
            )
        return trace

    def forward(
        self,
        input_ids: Tensor,
        *,
        memory_state: Sequence[Tensor] | ModelRecurrentState | None = None,
        knowledge_state: Tensor | None = None,
        reset_mask: Tensor | None = None,
        return_memory_state: bool = False,
        return_layers: bool = False,
        return_logits: bool = True,
    ) -> Tensor | MemoryScanOutput:
        del knowledge_state
        batch_size, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
        if seq_len > self.max_seq_len:
            raise ValueError(f"input sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}.")

        embedded = self.s_module(input_ids)
        prior_trace = self._coerce_memory_state(
            memory_state,
            batch_size=batch_size,
            device=embedded.device,
            dtype=embedded.dtype,
            reset_mask=reset_mask,
        )
        causal_mask = torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool).triu(1)
        hidden = embedded
        next_trace_layers: list[Tensor] = []
        injected = embedded
        layer_stats: list[dict[str, float]] = []
        for layer_index, block in enumerate(self.encoder_layers):
            hidden_after_attention, attention_input, attention_probs, query_states = block.attend(
                hidden,
                attention_mask=causal_mask,
            )
            activations = self.knowledge_block.lookup(query_states)
            relation_scores, layer_trace, stats = self.knowledge_block(
                activations,
                attention_probs=attention_probs,
                prior_trace=prior_trace[:, layer_index],
                layer_index=layer_index,
            )
            delta = torch.matmul(relation_scores, self.knowledge_block.value[layer_index])
            gate = torch.sigmoid(self.knowledge_block.residual_gate_logit[layer_index]).to(dtype=delta.dtype)
            hidden = hidden_after_attention + gate * delta
            hidden = block.feed_forward(hidden)
            injected = hidden
            next_trace_layers.append(layer_trace)
            if stats is not None:
                stats["knowledge/gate_mean"] = float(gate.detach().float().mean().item())
                layer_stats.append(stats)
        if self.knowledge_block.track_stats and layer_stats:
            stat_names = layer_stats[0].keys()
            self.knowledge_block.last_stats = {
                name: float(sum(stat[name] for stat in layer_stats) / len(layer_stats))
                for name in stat_names
            }

        encoded = hidden
        output_val = self.output_norm(encoded)
        output_state = self.value_to_state(output_val).squeeze(-1)
        logits = self.lm_head(output_val) if return_logits else output_val.new_empty((0,))
        if not (return_memory_state or return_layers):
            return logits

        sequence_layer = None
        query_layer = None
        if return_layers:
            sequence_state = self.value_to_state(injected).squeeze(-1)
            sequence_layer = Layer(dim=self.dim, num_nodes=seq_len, state=sequence_state, val=injected)
            query_layer = Layer(dim=self.dim, num_nodes=seq_len, state=output_state, val=output_val)

        next_trace = torch.stack(next_trace_layers, dim=1)
        next_memory_state = (next_trace,)
        return MemoryScanOutput(
            logits=logits,
            memory_state=next_memory_state,
            knowledge_state=next_trace,
            sequence_layer=sequence_layer,
            query_layer=query_layer,
        )


CausalHierarchicalMemoryLM = CausalMemoryLM
