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
        if activation not in {"relu", "gelu", "softplus"}:
            raise ValueError(f"Unsupported knowledge activation: {activation!r}.")

        self.dim = int(dim)
        self.memory_size = int(memory_size)
        self.beta_dim = int(beta_dim)
        self.num_heads = int(num_heads)
        self.relation_rank = int(relation_rank)
        self.activation_name = activation

        self.memory = nn.Parameter(torch.empty(self.memory_size, self.dim))
        self.beta_query = nn.Parameter(torch.empty(self.num_heads, self.beta_dim, self.memory_size))
        self.beta_key = nn.Parameter(torch.empty(self.num_heads, self.beta_dim, self.memory_size))
        self.relation = nn.Parameter(torch.empty(self.memory_size, self.memory_size))
        self.distance_decay = nn.Parameter(torch.tensor(0.1))
        self.prior_logit = nn.Parameter(torch.tensor(0.0))
        self.residual_gate = nn.Linear(self.dim, self.dim)

        self.track_stats = False
        self.last_stats: dict[str, float] = {}
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.memory, mean=0.0, std=PARAM_INIT_STD)
        nn.init.xavier_uniform_(self.beta_query)
        nn.init.xavier_uniform_(self.beta_key)
        nn.init.normal_(self.relation, mean=0.0, std=PARAM_INIT_STD)
        nn.init.zeros_(self.residual_gate.weight)
        nn.init.zeros_(self.residual_gate.bias)

    def _activation(self, tensor: Tensor) -> Tensor:
        if self.activation_name == "relu":
            return F.relu(tensor)
        if self.activation_name == "gelu":
            return F.gelu(tensor)
        return F.softplus(tensor)

    def forward(
        self,
        hidden: Tensor,
        *,
        prior_trace: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        activations = torch.matmul(hidden, self.memory.transpose(0, 1))
        batch_size, seq_len, _ = activations.shape
        if prior_trace is None:
            prior_trace = activations.new_zeros((batch_size, self.num_heads, self.memory_size))
        history = torch.cat(
            (
                prior_trace.unsqueeze(2),
                activations.unsqueeze(1).expand(-1, self.num_heads, -1, -1),
            ),
            dim=2,
        )

        query = torch.einsum("btk,hdk->bthd", activations, self.beta_query)
        key = torch.einsum("bhsk,hdk->bhsd", history, self.beta_key)
        scores = torch.einsum("bthd,bhsd->bhts", query, key) / math.sqrt(float(self.beta_dim))

        device = hidden.device
        query_index = torch.arange(seq_len, device=device).view(1, seq_len, 1)
        key_index = torch.arange(seq_len + 1, device=device).view(1, 1, seq_len + 1)
        causal_mask = key_index > (query_index + 1)
        scores = scores.masked_fill(causal_mask.unsqueeze(1), torch.finfo(scores.dtype).min)

        positive_decay = F.softplus(self.distance_decay)
        distance = (query_index + 1 - key_index).clamp_min(0).to(dtype=scores.dtype)
        distance_bias = -positive_decay * distance
        distance_bias[..., 0] = self.prior_logit.to(dtype=scores.dtype)
        beta = torch.softmax(scores + distance_bias, dim=-1)

        traced_by_head = torch.einsum("bhts,bhsk->bhtk", beta, history)
        traced = traced_by_head.mean(dim=1)
        relation_scores = torch.matmul(traced, self.relation)
        relation_scores = self._activation(relation_scores)
        delta = torch.matmul(relation_scores, self.memory)
        gate = torch.sigmoid(self.residual_gate(hidden))
        updated = hidden + gate * delta
        next_trace = traced_by_head[:, :, -1, :]

        if self.track_stats:
            beta_probs = beta.clamp_min(1.0e-12)
            beta_entropy = -(beta_probs * beta_probs.log()).sum(dim=-1).mean()
            self.last_stats = {
                "knowledge/gate_mean": float(gate.detach().float().mean().item()),
                "knowledge/beta_entropy": float(beta_entropy.detach().float().item()),
                "knowledge/trace_rms": float(next_trace.detach().float().square().mean().sqrt().item()),
                "knowledge/activation_mean": float(relation_scores.detach().float().mean().item()),
                "knowledge/relation_rms": float(self.relation.detach().float().square().mean().sqrt().item()),
            }

        return updated, next_trace


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
            relation_rank=self.knowledge_relation_rank,
            activation=self.knowledge_activation,
        )
        ff_dim = max(self.dim, int(round(self.dim * self.feed_forward_hidden_mult)))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=self.transformer_heads,
            dim_feedforward=ff_dim,
            dropout=self.transformer_dropout,
            activation=encoder_activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.transformer_layer_count)
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
            if trace.shape != (batch_size, self.transformer_heads, self.knowledge_memory_size):
                raise ValueError(
                    "memory_state[0] must have shape "
                    f"({batch_size}, {self.transformer_heads}, {self.knowledge_memory_size}), got {tuple(trace.shape)}."
                )
            trace = trace.to(device=device, dtype=dtype)
        if reset_mask is not None:
            if reset_mask.shape != (batch_size,):
                raise ValueError(
                    f"reset_mask must have shape ({batch_size},), got {tuple(reset_mask.shape)}."
                )
            trace = torch.where(
                reset_mask.to(device=device, dtype=torch.bool).view(batch_size, 1, 1),
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
        injected, next_trace = self.knowledge_block(embedded, prior_trace=prior_trace)

        causal_mask = torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool).triu(1)
        encoded = self.encoder(injected, mask=causal_mask, is_causal=True)
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

        next_memory_state = (next_trace,)
        return MemoryScanOutput(
            logits=logits,
            memory_state=next_memory_state,
            knowledge_state=next_trace,
            sequence_layer=sequence_layer,
            query_layer=query_layer,
        )


CausalHierarchicalMemoryLM = CausalMemoryLM
