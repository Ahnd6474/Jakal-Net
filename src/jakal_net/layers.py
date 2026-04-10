from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CollectionEdgeScorer(nn.Module):
    def __init__(self, node_dim: int, edge_hidden_dim: int) -> None:
        super().__init__()
        self.source_norm = nn.LayerNorm(node_dim)
        self.target_norm = nn.LayerNorm(node_dim)
        self.source_up = nn.Linear(node_dim, edge_hidden_dim)
        self.target_proj = nn.Linear(node_dim, edge_hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(edge_hidden_dim * 3, edge_hidden_dim),
            nn.GELU(),
            nn.Linear(edge_hidden_dim, 1),
        )

    def forward(
        self,
        source_values: Tensor,
        target_values: Tensor,
        source_mask: Tensor | None = None,
    ) -> Tensor:
        source = self.source_up(self.source_norm(source_values))
        target = self.target_proj(self.target_norm(target_values))

        num_targets = target.size(1)
        num_sources = source.size(1)
        pair_features = torch.cat(
            [
                source.unsqueeze(2).expand(-1, -1, num_targets, -1),
                target.unsqueeze(1).expand(-1, num_sources, -1, -1),
                source.unsqueeze(2) * target.unsqueeze(1),
            ],
            dim=-1,
        )
        edges = F.softsign(self.mlp(pair_features).squeeze(-1))

        if source_mask is not None:
            edges = edges * source_mask.to(edges.dtype).unsqueeze(-1)

        return edges


class FirstCollectionLayer(nn.Module):
    def __init__(self, node_dim: int, edge_hidden_dim: int) -> None:
        super().__init__()
        self.text_edge_scorer = CollectionEdgeScorer(node_dim, edge_hidden_dim)
        self.image_edge_scorer = CollectionEdgeScorer(node_dim, edge_hidden_dim)
        self.text_state_norm = nn.LayerNorm(node_dim)
        self.image_state_norm = nn.LayerNorm(node_dim)
        self.text_state_proj = nn.Linear(node_dim, node_dim)
        self.image_state_proj = nn.Linear(node_dim, node_dim)
        self.state_norm = nn.LayerNorm(node_dim)
        self.value_norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        *,
        text_values: Tensor,
        image_values: Tensor,
        latent_state_init: Tensor,
        latent_value_init: Tensor,
        text_mask: Tensor | None = None,
        image_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        text_to_latent = self.text_edge_scorer(text_values, latent_value_init, text_mask)
        image_to_latent = self.image_edge_scorer(image_values, latent_value_init, image_mask)

        text_state_features = self.text_state_proj(self.text_state_norm(text_values))
        image_state_features = self.image_state_proj(self.image_state_norm(image_values))

        latent_state_raw = torch.einsum(
            "bsk,bsd->bkd",
            text_to_latent,
            text_state_features,
        )
        latent_state_raw = latent_state_raw + torch.einsum(
            "bsk,bsd->bkd",
            image_to_latent,
            image_state_features,
        )

        latent_value_raw = torch.einsum(
            "bsk,bsd->bkd",
            text_to_latent,
            text_values,
        )
        latent_value_raw = latent_value_raw + torch.einsum(
            "bsk,bsd->bkd",
            image_to_latent,
            image_values,
        )

        latent_state = self.state_norm(latent_state_raw + latent_state_init)
        latent_value = self.value_norm(latent_value_raw + latent_value_init)

        aux = {
            "text_to_latent": text_to_latent,
            "image_to_latent": image_to_latent,
        }
        return latent_state, latent_value, aux


class LatentEdgeScorer(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_hidden_dim: int,
        *,
        allow_self_edges: bool,
    ) -> None:
        super().__init__()
        self.allow_self_edges = allow_self_edges
        self.source_norm = nn.LayerNorm(node_dim)
        self.target_norm = nn.LayerNorm(node_dim)
        self.source_proj = nn.Linear(node_dim, edge_hidden_dim)
        self.target_proj = nn.Linear(node_dim, edge_hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(edge_hidden_dim * 3, edge_hidden_dim),
            nn.GELU(),
            nn.Linear(edge_hidden_dim, 1),
        )

    def forward(self, latent_values: Tensor) -> Tensor:
        source = self.source_proj(self.source_norm(latent_values))
        target = self.target_proj(self.target_norm(latent_values))
        num_latents = latent_values.size(1)

        pair_features = torch.cat(
            [
                source.unsqueeze(2).expand(-1, -1, num_latents, -1),
                target.unsqueeze(1).expand(-1, num_latents, -1, -1),
                source.unsqueeze(2) * target.unsqueeze(1),
            ],
            dim=-1,
        )
        edges = F.softsign(self.mlp(pair_features).squeeze(-1))

        if not self.allow_self_edges:
            diag = torch.eye(num_latents, device=latent_values.device, dtype=torch.bool)
            edges = edges.masked_fill(diag.unsqueeze(0), 0.0)

        return edges


class LatentPropagationLayer(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_hidden_dim: int,
        *,
        allow_self_edges: bool,
    ) -> None:
        super().__init__()
        self.edge_scorer = LatentEdgeScorer(
            node_dim,
            edge_hidden_dim,
            allow_self_edges=allow_self_edges,
        )

    def forward(self, state: Tensor, value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        edges = self.edge_scorer(value)
        propagated_state = torch.einsum("bpq,bpd->bqd", edges, state)
        propagated_value = torch.einsum("bpq,bpd->bqd", edges, value)

        gate = F.softplus(propagated_state)
        next_state = gate
        next_value = gate * propagated_value
        return next_state, next_value, edges


class ResidualReadout(nn.Module):
    def __init__(self, node_dim: int, dropout: float) -> None:
        super().__init__()
        self.summary_norm = nn.LayerNorm(node_dim)
        self.text_proj = nn.Linear(node_dim, node_dim)
        self.image_proj = nn.Linear(node_dim, node_dim)
        self.text_norm = nn.LayerNorm(node_dim)
        self.image_norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

    def pool_latents(self, latent_state: Tensor, latent_value: Tensor) -> Tensor:
        weights = torch.softmax(latent_state.mean(dim=-1), dim=-1)
        pooled = torch.einsum("bk,bkd->bd", weights, latent_value)
        return self.summary_norm(pooled)

    def forward(
        self,
        text_values: Tensor,
        image_values: Tensor,
        latent_state: Tensor,
        latent_value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        summary = self.pool_latents(latent_state, latent_value)
        text_delta = self.dropout(self.text_proj(summary)).unsqueeze(1)
        image_delta = self.dropout(self.image_proj(summary)).unsqueeze(1)
        refined_text = self.text_norm(text_values + text_delta)
        refined_image = self.image_norm(image_values + image_delta)
        return refined_text, refined_image, summary
