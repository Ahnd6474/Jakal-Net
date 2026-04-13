from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from jakal_net import (
    DiagonalBilinearPairwise,
    Layer,
    LayerDelta,
    LinearRoute,
    Propagation,
    SparsePropagation,
    SparseTransition,
    Transition,
)
from jakal_net.kernel_common import apply_slot_mask_to_state, apply_slot_mask_to_val


def _scale_delta(delta: LayerDelta, scale: float) -> LayerDelta:
    if scale == 1.0:
        return delta
    return LayerDelta(
        delta_state=delta.delta_state * scale,
        delta_val=delta.delta_val * scale,
    )


def _apply_scaled_delta(layer: Layer, delta: LayerDelta, scale: float) -> Layer:
    if scale == 0.0:
        return layer
    return layer.apply_delta(_scale_delta(delta, scale))


def _stabilize_layer(layer: Layer, val_norm: nn.LayerNorm) -> Layer:
    return layer.with_tensors(
        state=torch.tanh(layer.state),
        val=val_norm(layer.val),
    )


def _apply_layer_slot_mask(layer: Layer, slot_mask: Tensor) -> Layer:
    return layer.with_tensors(
        state=apply_slot_mask_to_state(layer.state, slot_mask),
        val=apply_slot_mask_to_val(layer.val, slot_mask),
    )


def _make_sparse_or_dense_propagation(
    *,
    dim: int,
    sparse_type: str,
    implementation: str,
    window: int | None = None,
    topk: int | None = None,
    pairwise_fn: nn.Module | None = None,
) -> Propagation | SparsePropagation:
    pairwise = DiagonalBilinearPairwise(dim=dim) if pairwise_fn is None else pairwise_fn
    if sparse_type == "window":
        if window is None:
            raise ValueError("window propagation requires window.")
        return SparsePropagation(
            pairwise_fn=pairwise,
            sparse_type="window",
            window=window,
            implementation=implementation,
        )
    if sparse_type == "topk":
        if topk is None:
            raise ValueError("topk propagation requires topk.")
        return SparsePropagation(
            pairwise_fn=pairwise,
            sparse_type="topk",
            topk=topk,
            implementation=implementation,
        )
    raise ValueError(f"Unsupported sparse_type: {sparse_type!r}.")


def _make_transition(
    *,
    dim: int,
    dst_nodes: int,
    route_topk: int,
    implementation: str,
    merge_mode: str = "add",
) -> Transition | SparseTransition:
    if route_topk >= dst_nodes:
        return Transition(
            route_fn=LinearRoute(src_dim=dim, dst_nodes=dst_nodes),
            implementation=implementation,
            merge_mode=merge_mode,
        )
    return SparseTransition(
        route_fn=LinearRoute(src_dim=dim, dst_nodes=dst_nodes),
        topk=route_topk,
        implementation=implementation,
        merge_mode=merge_mode,
    )


@dataclass(frozen=True, slots=True)
class ProgressiveBStageSpec:
    num_layers: int
    expanded_nodes: int
    compressed_nodes: int
    alpha_b: float
    beta_s_to_b: float
    beta_b_to_s: float

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if self.expanded_nodes <= 0:
            raise ValueError("expanded_nodes must be positive.")
        if self.compressed_nodes <= 0:
            raise ValueError("compressed_nodes must be positive.")
        if self.compressed_nodes > self.expanded_nodes:
            raise ValueError("compressed_nodes must be less than or equal to expanded_nodes.")
        for name, value in (
            ("alpha_b", self.alpha_b),
            ("beta_s_to_b", self.beta_s_to_b),
            ("beta_b_to_s", self.beta_b_to_s),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")


def build_progressive_b_stage_specs(
    seq_nodes: int,
    *,
    lite_layers: int = 2,
    mid_layers: int = 2,
    full_layers: int = 1,
) -> list[ProgressiveBStageSpec]:
    if seq_nodes <= 0:
        raise ValueError("seq_nodes must be positive.")
    return [
        ProgressiveBStageSpec(
            num_layers=lite_layers,
            expanded_nodes=max(1, math.ceil(seq_nodes * 1.05)),
            compressed_nodes=max(1, math.ceil(seq_nodes * 0.90)),
            alpha_b=0.3,
            beta_s_to_b=0.25,
            beta_b_to_s=0.15,
        ),
        ProgressiveBStageSpec(
            num_layers=mid_layers,
            expanded_nodes=max(1, math.ceil(seq_nodes * 1.10)),
            compressed_nodes=max(1, math.ceil(seq_nodes * 0.80)),
            alpha_b=0.65,
            beta_s_to_b=0.55,
            beta_b_to_s=0.35,
        ),
        ProgressiveBStageSpec(
            num_layers=full_layers,
            expanded_nodes=max(1, math.ceil(seq_nodes * 1.20)),
            compressed_nodes=max(1, math.ceil(seq_nodes * 0.70)),
            alpha_b=1.0,
            beta_s_to_b=0.9,
            beta_b_to_s=0.8,
        ),
    ]


class ProgressiveBJointBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        seq_nodes: int,
        expanded_nodes: int,
        compressed_nodes: int,
        alpha_b: float,
        beta_s_to_b: float,
        beta_b_to_s: float,
        s_window: int = 4,
        route_topk: int = 4,
        expanded_topk: int = 4,
        compressed_topk: int = 4,
        expanded_sparse_type: str = "topk",
        compressed_sparse_type: str = "topk",
        expanded_window: int | None = None,
        compressed_window: int | None = None,
        implementation: str = "streaming",
        s_delta_scale: float = 0.25,
        b_delta_scale: float = 0.20,
        cross_delta_scale: float = 0.15,
        s_pairwise_fn: nn.Module | None = None,
        expanded_pairwise_fn: nn.Module | None = None,
        compressed_pairwise_fn: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.seq_nodes = seq_nodes
        self.expanded_nodes = expanded_nodes
        self.compressed_nodes = compressed_nodes
        self.alpha_b = alpha_b
        self.beta_s_to_b = beta_s_to_b
        self.beta_b_to_s = beta_b_to_s
        self.s_delta_scale = s_delta_scale
        self.b_delta_scale = b_delta_scale
        self.cross_delta_scale = cross_delta_scale
        self.s_val_norm = nn.LayerNorm(dim)
        self.expanded_val_norm = nn.LayerNorm(dim)
        self.compressed_val_norm = nn.LayerNorm(dim)

        self.s_propagation = _make_sparse_or_dense_propagation(
            dim=dim,
            sparse_type="window",
            window=s_window,
            implementation=implementation,
            pairwise_fn=s_pairwise_fn,
        )
        self.expand_transition = _make_transition(
            dim=dim,
            dst_nodes=expanded_nodes,
            route_topk=min(route_topk, expanded_nodes),
            implementation=implementation,
        )
        self.expanded_propagation = _make_sparse_or_dense_propagation(
            dim=dim,
            sparse_type=expanded_sparse_type,
            window=expanded_window,
            topk=min(expanded_topk, expanded_nodes),
            implementation=implementation,
            pairwise_fn=expanded_pairwise_fn,
        )
        self.b_to_s = _make_transition(
            dim=dim,
            dst_nodes=seq_nodes,
            route_topk=min(route_topk, seq_nodes),
            implementation=implementation,
        )
        self.compress_transition = _make_transition(
            dim=dim,
            dst_nodes=compressed_nodes,
            route_topk=min(route_topk, compressed_nodes),
            implementation=implementation,
        )
        self.s_to_b = _make_transition(
            dim=dim,
            dst_nodes=compressed_nodes,
            route_topk=min(route_topk, compressed_nodes),
            implementation=implementation,
        )
        self.compressed_propagation = _make_sparse_or_dense_propagation(
            dim=dim,
            sparse_type=compressed_sparse_type,
            window=compressed_window,
            topk=min(compressed_topk, compressed_nodes),
            implementation=implementation,
            pairwise_fn=compressed_pairwise_fn,
        )
        self.compressed_adapter = Transition(
            route_fn=LinearRoute(src_dim=dim, dst_nodes=compressed_nodes),
            merge_mode="replace",
            implementation=implementation,
        )

    def _make_layer_like(self, reference: Layer, num_nodes: int) -> Layer:
        return Layer.zeros(
            dim=self.dim,
            num_nodes=num_nodes,
            batch_shape=tuple(reference.batch_shape),
            device=reference.state.device,
            dtype=reference.state.dtype,
        )

    def _prepare_compressed_layer(
        self, s_layer: Layer, compressed_b: Layer | None
    ) -> Layer:
        if compressed_b is None:
            return self._make_layer_like(s_layer, self.compressed_nodes)
        if compressed_b.num_nodes == self.compressed_nodes:
            return compressed_b
        adapted = self._make_layer_like(s_layer, self.compressed_nodes)
        return self.compressed_adapter(compressed_b, adapted)

    def forward_sequence_only(self, s_layer: Layer) -> Layer:
        s_layer = _apply_scaled_delta(
            s_layer,
            self.s_propagation.compute_delta(s_layer),
            self.s_delta_scale,
        )
        return _stabilize_layer(s_layer, self.s_val_norm)

    def forward(self, s_layer: Layer, compressed_b: Layer | None = None) -> tuple[Layer, Layer]:
        s_layer = _apply_scaled_delta(
            s_layer,
            self.s_propagation.compute_delta(s_layer),
            self.s_delta_scale,
        )
        s_layer = _stabilize_layer(s_layer, self.s_val_norm)
        compressed_b = self._prepare_compressed_layer(s_layer, compressed_b)
        compressed_b = _stabilize_layer(compressed_b, self.compressed_val_norm)

        expanded_b = self._make_layer_like(s_layer, self.expanded_nodes)
        expanded_b = _apply_scaled_delta(
            expanded_b,
            self.expand_transition.compute_delta(compressed_b, expanded_b),
            self.alpha_b * self.b_delta_scale,
        )
        expanded_b = _apply_scaled_delta(
            expanded_b,
            self.expanded_propagation.compute_delta(expanded_b),
            self.alpha_b * self.b_delta_scale,
        )
        expanded_b = _stabilize_layer(expanded_b, self.expanded_val_norm)

        s_layer = _apply_scaled_delta(
            s_layer,
            self.b_to_s.compute_delta(expanded_b, s_layer),
            self.alpha_b * self.beta_b_to_s * self.cross_delta_scale,
        )
        s_layer = _stabilize_layer(s_layer, self.s_val_norm)

        next_compressed = compressed_b.clone()
        next_compressed = _apply_scaled_delta(
            next_compressed,
            self.compress_transition.compute_delta(expanded_b, next_compressed),
            self.alpha_b * self.b_delta_scale,
        )
        next_compressed = _apply_scaled_delta(
            next_compressed,
            self.s_to_b.compute_delta(s_layer, next_compressed),
            self.alpha_b * self.beta_s_to_b * self.cross_delta_scale,
        )
        next_compressed = _apply_scaled_delta(
            next_compressed,
            self.compressed_propagation.compute_delta(next_compressed),
            self.alpha_b * self.b_delta_scale,
        )
        next_compressed = _stabilize_layer(next_compressed, self.compressed_val_norm)
        return s_layer, next_compressed


class ProgressiveBExampleLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        seq_nodes: int,
        warmup_layers: int = 2,
        stage_specs: Sequence[ProgressiveBStageSpec] | None = None,
        final_refine_layers: int = 2,
        s_window: int = 4,
        route_topk: int = 4,
        expanded_topk: int = 4,
        compressed_topk: int = 4,
        expanded_sparse_type: str = "topk",
        compressed_sparse_type: str = "topk",
        expanded_window: int | None = None,
        compressed_window: int | None = None,
        implementation: str = "streaming",
        prediction_slot_index: int = -1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.seq_nodes = seq_nodes
        self.prediction_slot_index = prediction_slot_index
        self.stage_specs = tuple(stage_specs or build_progressive_b_stage_specs(seq_nodes))

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Parameter(torch.zeros(seq_nodes, dim))
        self.state_init = nn.Linear(dim, 1)
        nn.init.zeros_(self.state_init.weight)
        nn.init.zeros_(self.state_init.bias)
        self.sequence_val_norm = nn.LayerNorm(dim)
        shared_s_pairwise = DiagonalBilinearPairwise(dim=dim)
        shared_expanded_pairwise = DiagonalBilinearPairwise(dim=dim)
        shared_compressed_pairwise = DiagonalBilinearPairwise(dim=dim)
        self.s_warmup = nn.ModuleList(
            [
                _make_sparse_or_dense_propagation(
                    dim=dim,
                    sparse_type="window",
                    window=s_window,
                    implementation=implementation,
                    pairwise_fn=shared_s_pairwise,
                )
                for _ in range(warmup_layers)
            ]
        )
        self.joint_blocks = nn.ModuleList(
            [
                ProgressiveBJointBlock(
                    dim=dim,
                    seq_nodes=seq_nodes,
                    expanded_nodes=stage.expanded_nodes,
                    compressed_nodes=stage.compressed_nodes,
                    alpha_b=stage.alpha_b,
                    beta_s_to_b=stage.beta_s_to_b,
                    beta_b_to_s=stage.beta_b_to_s,
                    s_window=s_window,
                    route_topk=route_topk,
                    expanded_topk=expanded_topk,
                    compressed_topk=compressed_topk,
                    expanded_sparse_type=expanded_sparse_type,
                    compressed_sparse_type=compressed_sparse_type,
                    expanded_window=expanded_window,
                    compressed_window=compressed_window,
                    implementation=implementation,
                    s_pairwise_fn=shared_s_pairwise,
                    expanded_pairwise_fn=shared_expanded_pairwise,
                    compressed_pairwise_fn=shared_compressed_pairwise,
                )
                for stage in self.stage_specs
                for _ in range(stage.num_layers)
            ]
        )
        self.s_refine = nn.ModuleList(
            [
                _make_sparse_or_dense_propagation(
                    dim=dim,
                    sparse_type="window",
                    window=s_window,
                    implementation=implementation,
                    pairwise_fn=shared_s_pairwise,
                )
                for _ in range(final_refine_layers)
            ]
        )
        self.readout_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def initialize_sequence_layer(
        self,
        token_ids: Tensor,
        *,
        slot_mask: Tensor | None = None,
    ) -> Layer:
        token_val = self.token_embedding(token_ids)
        position_val = self.position_embedding.unsqueeze(0)
        combined = token_val + position_val
        state = self.state_init(combined).squeeze(-1)
        layer = Layer(
            dim=self.dim,
            num_nodes=self.seq_nodes,
            state=state,
            val=combined,
        )
        layer = _stabilize_layer(layer, self.sequence_val_norm)
        if slot_mask is not None:
            layer = _apply_layer_slot_mask(layer, slot_mask)
        return layer

    def read_prediction_slot(self, s_layer: Layer) -> Tensor:
        index = self.prediction_slot_index
        if index < 0:
            index = s_layer.num_nodes + index
        return s_layer.val[..., index, :]

    def read_prediction_slots(self, s_layer: Layer, indices: Tensor) -> Tensor:
        if tuple(indices.shape) != tuple(s_layer.batch_shape):
            raise ValueError(
                "indices must match s_layer batch shape, "
                f"expected {tuple(s_layer.batch_shape)}, got {tuple(indices.shape)}."
            )
        flat_val = s_layer.val.reshape(-1, s_layer.num_nodes, self.dim)
        flat_indices = indices.reshape(-1)
        batch_index = torch.arange(flat_val.shape[0], device=flat_val.device)
        gathered = flat_val[batch_index, flat_indices]
        return gathered.reshape(*indices.shape, self.dim)

    def _apply_sequence_mask(self, s_layer: Layer, slot_mask: Tensor) -> Layer:
        return _apply_layer_slot_mask(s_layer, slot_mask)

    def forward_full_sequence_causal(
        self, token_ids: Tensor, *, return_layers: bool = False
    ) -> Tensor | tuple[Tensor, Layer, None]:
        if token_ids.shape[-1] != self.seq_nodes:
            raise ValueError(
                "full-sequence causal expects token_ids.shape[-1] to equal seq_nodes, "
                f"got {token_ids.shape[-1]} and {self.seq_nodes}."
            )

        # This path keeps strict causality by using the causal S operators only.
        s_layer = self.initialize_sequence_layer(token_ids)
        for op in self.s_warmup:
            s_layer = _apply_scaled_delta(s_layer, op.compute_delta(s_layer), 0.25)
            s_layer = _stabilize_layer(s_layer, self.sequence_val_norm)
        for block in self.joint_blocks:
            s_layer = block.forward_sequence_only(s_layer)
        for op in self.s_refine:
            s_layer = _apply_scaled_delta(s_layer, op.compute_delta(s_layer), 0.25)
            s_layer = _stabilize_layer(s_layer, self.sequence_val_norm)

        logits = self.lm_head(self.readout_norm(s_layer.val))
        if return_layers:
            return logits, s_layer, None
        return logits

    def forward_teacher_forcing(
        self, token_ids: Tensor, *, return_layers: bool = False
    ) -> Tensor | tuple[Tensor, Layer, Layer | None]:
        if token_ids.shape[-1] != self.seq_nodes:
            raise ValueError(
                "teacher forcing expects token_ids.shape[-1] to equal seq_nodes, "
                f"got {token_ids.shape[-1]} and {self.seq_nodes}."
            )

        base_layer = self.initialize_sequence_layer(token_ids)
        batch_shape = tuple(base_layer.batch_shape)
        flat_batch = math.prod(batch_shape) if batch_shape else 1
        device = token_ids.device

        visible_upto = torch.arange(self.seq_nodes, device=device)
        base_mask = visible_upto.view(self.seq_nodes, 1) >= torch.arange(
            self.seq_nodes, device=device
        ).view(1, self.seq_nodes)
        slot_mask = base_mask.unsqueeze(0).expand(flat_batch, -1, -1)
        prediction_indices = visible_upto.view(1, self.seq_nodes).expand(flat_batch, -1)

        expanded_state = (
            base_layer.state.reshape(flat_batch, self.seq_nodes)
            .unsqueeze(1)
            .expand(flat_batch, self.seq_nodes, self.seq_nodes)
            .reshape(flat_batch * self.seq_nodes, self.seq_nodes)
        )
        expanded_val = (
            base_layer.val.reshape(flat_batch, self.seq_nodes, self.dim)
            .unsqueeze(1)
            .expand(flat_batch, self.seq_nodes, self.seq_nodes, self.dim)
            .reshape(flat_batch * self.seq_nodes, self.seq_nodes, self.dim)
        )
        slot_mask_flat = slot_mask.reshape(flat_batch * self.seq_nodes, self.seq_nodes)
        prediction_indices_flat = prediction_indices.reshape(flat_batch * self.seq_nodes)

        s_layer = Layer(
            dim=self.dim,
            num_nodes=self.seq_nodes,
            state=expanded_state,
            val=expanded_val,
        )
        s_layer = self._apply_sequence_mask(s_layer, slot_mask_flat)
        compressed_b: Layer | None = None

        for op in self.s_warmup:
            s_layer = _apply_scaled_delta(s_layer, op.compute_delta(s_layer), 0.25)
            s_layer = _stabilize_layer(s_layer, self.sequence_val_norm)
            s_layer = self._apply_sequence_mask(s_layer, slot_mask_flat)
        for block in self.joint_blocks:
            s_layer, compressed_b = block(s_layer, compressed_b)
            s_layer = self._apply_sequence_mask(s_layer, slot_mask_flat)
        for op in self.s_refine:
            s_layer = _apply_scaled_delta(s_layer, op.compute_delta(s_layer), 0.25)
            s_layer = _stabilize_layer(s_layer, self.sequence_val_norm)
            s_layer = self._apply_sequence_mask(s_layer, slot_mask_flat)

        prediction_slot = self.readout_norm(
            self.read_prediction_slots(s_layer, prediction_indices_flat)
        )
        logits = self.lm_head(prediction_slot).reshape(*batch_shape, self.seq_nodes, self.vocab_size)
        if return_layers:
            return logits, s_layer, compressed_b
        return logits

    def forward(
        self,
        token_ids: Tensor,
        *,
        return_layers: bool = False,
        teacher_forcing: bool = False,
        full_sequence_causal: bool = False,
    ) -> Tensor | tuple[Tensor, Layer, Layer | None]:
        if teacher_forcing and full_sequence_causal:
            raise ValueError("teacher_forcing and full_sequence_causal cannot both be enabled.")
        if full_sequence_causal:
            return self.forward_full_sequence_causal(token_ids, return_layers=return_layers)
        if teacher_forcing:
            return self.forward_teacher_forcing(token_ids, return_layers=return_layers)
        s_layer = self.initialize_sequence_layer(token_ids)
        compressed_b: Layer | None = None

        for op in self.s_warmup:
            s_layer = _apply_scaled_delta(s_layer, op.compute_delta(s_layer), 0.25)
            s_layer = _stabilize_layer(s_layer, self.sequence_val_norm)
        for block in self.joint_blocks:
            s_layer, compressed_b = block(s_layer, compressed_b)
        for op in self.s_refine:
            s_layer = _apply_scaled_delta(s_layer, op.compute_delta(s_layer), 0.25)
            s_layer = _stabilize_layer(s_layer, self.sequence_val_norm)

        prediction_slot = self.readout_norm(self.read_prediction_slot(s_layer))
        logits = self.lm_head(prediction_slot)
        if return_layers:
            return logits, s_layer, compressed_b
        return logits


@dataclass(frozen=True, slots=True)
class CharVocab:
    stoi: dict[str, int]
    itos: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> Tensor:
        return torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(self.itos[idx] for idx in token_ids)


@dataclass(frozen=True, slots=True)
class NextTokenBatch:
    context: Tensor
    target: Tensor


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    train_losses: tuple[float, ...]
    val_losses: tuple[float, ...]


def build_char_vocab(text: str) -> CharVocab:
    symbols = tuple(sorted(set(text)))
    return CharVocab(stoi={ch: idx for idx, ch in enumerate(symbols)}, itos=symbols)


def split_train_val(tokens: Tensor, *, train_fraction: float = 0.9) -> tuple[Tensor, Tensor]:
    split_index = int(tokens.numel() * train_fraction)
    split_index = max(2, min(split_index, tokens.numel() - 2))
    return tokens[:split_index], tokens[split_index:]


def sample_next_token_batch(
    tokens: Tensor,
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device | str,
    teacher_forcing: bool = False,
    full_sequence_causal: bool = False,
) -> NextTokenBatch:
    if teacher_forcing and full_sequence_causal:
        raise ValueError("teacher_forcing and full_sequence_causal cannot both be enabled.")
    max_start = tokens.numel() - seq_len - 1
    starts = torch.randint(0, max_start + 1, (batch_size,))
    context = torch.stack([tokens[start : start + seq_len] for start in starts], dim=0)
    if teacher_forcing or full_sequence_causal:
        target = torch.stack([tokens[start + 1 : start + seq_len + 1] for start in starts], dim=0)
    else:
        target = torch.stack([tokens[start + seq_len] for start in starts], dim=0)
    return NextTokenBatch(context=context.to(device), target=target.to(device))


def compute_next_token_loss(
    model: nn.Module,
    batch: NextTokenBatch,
    *,
    teacher_forcing: bool = False,
    full_sequence_causal: bool = False,
) -> tuple[Tensor, Tensor]:
    if teacher_forcing and full_sequence_causal:
        raise ValueError("teacher_forcing and full_sequence_causal cannot both be enabled.")
    logits = model(
        batch.context,
        teacher_forcing=teacher_forcing,
        full_sequence_causal=full_sequence_causal,
    )
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        batch.target.reshape(-1),
    )
    return loss, logits


@torch.no_grad()
def estimate_next_token_loss(
    model: nn.Module,
    tokens: Tensor,
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device | str,
    eval_steps: int,
    teacher_forcing: bool = False,
    full_sequence_causal: bool = False,
) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for _ in range(eval_steps):
        batch = sample_next_token_batch(
            tokens,
            seq_len=seq_len,
            batch_size=batch_size,
            device=device,
            teacher_forcing=teacher_forcing,
            full_sequence_causal=full_sequence_causal,
        )
        loss, _ = compute_next_token_loss(
            model,
            batch,
            teacher_forcing=teacher_forcing,
            full_sequence_causal=full_sequence_causal,
        )
        losses.append(float(loss.item()))
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def train_next_token_model(
    model: nn.Module,
    train_tokens: Tensor,
    val_tokens: Tensor,
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device | str,
    steps: int,
    eval_interval: int,
    eval_steps: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    grad_clip: float | None = 1.0,
    teacher_forcing: bool = False,
    full_sequence_causal: bool = False,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> TrainingHistory:
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    train_losses: list[float] = []
    val_losses: list[float] = []

    for step in range(1, steps + 1):
        batch = sample_next_token_batch(
            train_tokens,
            seq_len=seq_len,
            batch_size=batch_size,
            device=device,
            teacher_forcing=teacher_forcing,
            full_sequence_causal=full_sequence_causal,
        )
        optimizer.zero_grad(set_to_none=True)
        loss, _ = compute_next_token_loss(
            model,
            batch,
            teacher_forcing=teacher_forcing,
            full_sequence_causal=full_sequence_causal,
        )
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        if progress_callback is not None:
            progress_callback(step, steps, float(loss.item()))

        if step % eval_interval == 0 or step == 1 or step == steps:
            train_losses.append(
                estimate_next_token_loss(
                    model,
                    train_tokens,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    device=device,
                    eval_steps=eval_steps,
                    teacher_forcing=teacher_forcing,
                    full_sequence_causal=full_sequence_causal,
                )
            )
            val_losses.append(
                estimate_next_token_loss(
                    model,
                    val_tokens,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    device=device,
                    eval_steps=eval_steps,
                    teacher_forcing=teacher_forcing,
                    full_sequence_causal=full_sequence_causal,
                )
            )

    return TrainingHistory(
        train_losses=tuple(train_losses),
        val_losses=tuple(val_losses),
    )


@torch.no_grad()
def generate_next_tokens(
    model: nn.Module,
    prompt: Tensor,
    *,
    max_new_tokens: int,
    seq_len: int,
    device: torch.device | str,
) -> Tensor:
    was_training = model.training
    model.eval()
    generated = prompt.to(device).clone()
    for _ in range(max_new_tokens):
        context = generated[-seq_len:].unsqueeze(0)
        logits = model(context)
        next_token = torch.argmax(logits, dim=-1)
        generated = torch.cat((generated, next_token), dim=0)
    if was_training:
        model.train()
    return generated


def perplexity_from_loss(loss: float) -> float:
    return float(math.exp(loss))
