from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .encoders import (
    LightImageEncoder,
    LightTextEncoder,
    PretrainedImageEncoder,
    PretrainedTextEncoder,
    TorchvisionResNetImageEncoder,
    masked_mean_pool,
)
from .layers import FirstCollectionLayer, LatentPropagationLayer, ResidualReadout


@dataclass(slots=True)
class JakalNetConfig:
    vocab_size: int = 32_000
    pad_token_id: int = 0
    max_text_tokens: int = 64
    image_size: int = 128
    patch_size: int = 16
    image_channels: int = 3
    node_dim: int = 256
    edge_hidden_dim: int = 384
    ff_multiplier: int = 4
    encoder_dropout: float = 0.1
    readout_dropout: float = 0.0
    text_encoder_layers: int = 2
    image_encoder_layers: int = 2
    num_heads: int = 4
    num_latents: int = 24
    num_propagation_layers: int = 2
    allow_self_edges: bool = True
    contrastive_temperature: float = 0.07
    use_pretrained_encoders: bool = False
    text_encoder_name: str = "distilbert-base-uncased"
    image_encoder_name: str = "resnet18"
    image_encoder_backend: str = "torchvision"
    train_text_encoder: bool = False
    train_image_encoder: bool = False
    local_files_only: bool = False
    drop_image_cls_token: bool = True

    def __post_init__(self) -> None:
        if self.node_dim % self.num_heads != 0:
            raise ValueError("node_dim must be divisible by num_heads")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.num_propagation_layers < 1:
            raise ValueError("num_propagation_layers must be at least 1")

    @classmethod
    def coco_pretrained(cls) -> "JakalNetConfig":
        return cls(
            max_text_tokens=48,
            node_dim=256,
            edge_hidden_dim=384,
            num_heads=4,
            num_latents=24,
            num_propagation_layers=2,
            use_pretrained_encoders=True,
            text_encoder_name="distilbert-base-uncased",
            image_encoder_name="resnet18",
            image_encoder_backend="torchvision",
            train_text_encoder=False,
            train_image_encoder=False,
            drop_image_cls_token=True,
        )


@dataclass(slots=True)
class JakalNetOutput:
    text_tokens: Tensor
    image_tokens: Tensor
    text_mask: Tensor
    image_mask: Tensor
    latent_state: Tensor
    latent_value: Tensor
    latent_summary: Tensor
    text_pooled: Tensor
    image_pooled: Tensor
    match_logit: Tensor
    similarity_map: Tensor
    alignment_logits: Tensor
    collection_edges: dict[str, Tensor]
    propagation_edges: list[Tensor]


class MatchingHead(nn.Module):
    def __init__(self, node_dim: int) -> None:
        super().__init__()
        pair_dim = node_dim * 4
        self.mlp = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, node_dim * 2),
            nn.GELU(),
            nn.Linear(node_dim * 2, 1),
        )

    def forward(self, text_embeddings: Tensor, image_embeddings: Tensor) -> Tensor:
        pair = torch.cat(
            [
                text_embeddings,
                image_embeddings,
                torch.abs(text_embeddings - image_embeddings),
                text_embeddings * image_embeddings,
            ],
            dim=-1,
        )
        return self.mlp(pair).squeeze(-1)


class JakalNetModel(nn.Module):
    def __init__(self, config: JakalNetConfig) -> None:
        super().__init__()
        self.config = config
        if config.use_pretrained_encoders:
            self.text_encoder = PretrainedTextEncoder(
                model_name=config.text_encoder_name,
                output_dim=config.node_dim,
                trainable=config.train_text_encoder,
                local_files_only=config.local_files_only,
            )
            if config.image_encoder_backend == "torchvision":
                self.image_encoder = TorchvisionResNetImageEncoder(
                    model_name=config.image_encoder_name,
                    output_dim=config.node_dim,
                    trainable=config.train_image_encoder,
                )
            elif config.image_encoder_backend == "transformers":
                self.image_encoder = PretrainedImageEncoder(
                    model_name=config.image_encoder_name,
                    output_dim=config.node_dim,
                    trainable=config.train_image_encoder,
                    local_files_only=config.local_files_only,
                    drop_cls_token=config.drop_image_cls_token,
                )
            else:
                raise ValueError(
                    f"unsupported image_encoder_backend `{config.image_encoder_backend}`"
                )
        else:
            self.text_encoder = LightTextEncoder(
                vocab_size=config.vocab_size,
                hidden_dim=config.node_dim,
                max_length=config.max_text_tokens,
                num_layers=config.text_encoder_layers,
                num_heads=config.num_heads,
                ff_multiplier=config.ff_multiplier,
                dropout=config.encoder_dropout,
                pad_token_id=config.pad_token_id,
            )
            self.image_encoder = LightImageEncoder(
                image_size=config.image_size,
                patch_size=config.patch_size,
                in_channels=config.image_channels,
                hidden_dim=config.node_dim,
                num_layers=config.image_encoder_layers,
                num_heads=config.num_heads,
                ff_multiplier=config.ff_multiplier,
                dropout=config.encoder_dropout,
            )
        self.text_write_norm = nn.LayerNorm(config.node_dim)
        self.image_write_norm = nn.LayerNorm(config.node_dim)
        self.latent_state_init = nn.Parameter(torch.zeros(config.num_latents, config.node_dim))
        self.latent_value_init = nn.Parameter(
            torch.randn(config.num_latents, config.node_dim) * 0.02
        )
        self.collection_layer = FirstCollectionLayer(
            node_dim=config.node_dim,
            edge_hidden_dim=config.edge_hidden_dim,
        )
        self.propagation_layers = nn.ModuleList(
            [
                LatentPropagationLayer(
                    node_dim=config.node_dim,
                    edge_hidden_dim=config.edge_hidden_dim,
                    allow_self_edges=config.allow_self_edges,
                )
                for _ in range(config.num_propagation_layers)
            ]
        )
        self.readout = ResidualReadout(
            node_dim=config.node_dim,
            dropout=config.readout_dropout,
        )
        self.matching_head = MatchingHead(config.node_dim)

    def compute_matching_logits(
        self,
        text_embeddings: Tensor,
        image_embeddings: Tensor,
    ) -> Tensor:
        return self.matching_head(text_embeddings, image_embeddings)

    def forward(
        self,
        input_ids: Tensor,
        images: Tensor,
        attention_mask: Tensor | None = None,
    ) -> JakalNetOutput:
        text_encoded, text_mask = self.text_encoder(input_ids, attention_mask)
        image_encoded, image_mask = self.image_encoder(images)
        text_values = self.text_write_norm(text_encoded)
        image_values = self.image_write_norm(image_encoded)

        batch_size = text_values.size(0)
        latent_state_init = self.latent_state_init.unsqueeze(0).expand(batch_size, -1, -1)
        latent_value_init = self.latent_value_init.unsqueeze(0).expand(batch_size, -1, -1)

        latent_state, latent_value, collection_edges = self.collection_layer(
            text_values=text_values,
            image_values=image_values,
            latent_state_init=latent_state_init,
            latent_value_init=latent_value_init,
            text_mask=text_mask,
            image_mask=image_mask,
        )

        propagation_edges: list[Tensor] = []
        for layer in self.propagation_layers:
            latent_state, latent_value, edges = layer(latent_state, latent_value)
            propagation_edges.append(edges)

        refined_text, refined_image, latent_summary = self.readout(
            text_values,
            image_values,
            latent_state,
            latent_value,
        )

        text_pooled = masked_mean_pool(refined_text, text_mask)
        image_pooled = masked_mean_pool(refined_image, image_mask)
        normalized_text = torch.nn.functional.normalize(refined_text, dim=-1)
        normalized_image = torch.nn.functional.normalize(refined_image, dim=-1)
        similarity_map = torch.einsum("btd,bpd->btp", normalized_text, normalized_image)
        alignment_logits = torch.einsum("btd,bpd->btp", refined_text, refined_image)
        alignment_logits = alignment_logits / math.sqrt(self.config.node_dim)
        match_logit = self.matching_head(text_pooled, image_pooled)

        return JakalNetOutput(
            text_tokens=refined_text,
            image_tokens=refined_image,
            text_mask=text_mask,
            image_mask=image_mask,
            latent_state=latent_state,
            latent_value=latent_value,
            latent_summary=latent_summary,
            text_pooled=text_pooled,
            image_pooled=image_pooled,
            match_logit=match_logit,
            similarity_map=similarity_map,
            alignment_logits=alignment_logits,
            collection_edges=collection_edges,
            propagation_edges=propagation_edges,
        )
