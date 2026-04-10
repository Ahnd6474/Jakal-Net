from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn

if TYPE_CHECKING:
    from .model import JakalNetOutput


@dataclass(slots=True)
class JakalLossBundle:
    contrastive: Tensor
    matching: Tensor
    grounding: Tensor | None
    total: Tensor


def bidirectional_contrastive_loss(
    text_embeddings: Tensor,
    image_embeddings: Tensor,
    *,
    temperature: float,
) -> Tensor:
    if text_embeddings.shape != image_embeddings.shape:
        raise ValueError("text and image embeddings must have the same shape")

    normalized_text = F.normalize(text_embeddings, dim=-1)
    normalized_image = F.normalize(image_embeddings, dim=-1)
    logits = normalized_text @ normalized_image.transpose(0, 1)
    logits = logits / temperature
    targets = torch.arange(logits.size(0), device=logits.device)

    loss_t2i = F.cross_entropy(logits, targets)
    loss_i2t = F.cross_entropy(logits.transpose(0, 1), targets)
    return 0.5 * (loss_t2i + loss_i2t)


def binary_matching_loss(
    matching_head: nn.Module,
    text_embeddings: Tensor,
    image_embeddings: Tensor,
) -> Tensor:
    if text_embeddings.shape != image_embeddings.shape:
        raise ValueError("text and image embeddings must have the same shape")
    if text_embeddings.size(0) < 2:
        raise ValueError("matching loss needs batch_size >= 2 to build negatives")

    positive_logits = matching_head(text_embeddings, image_embeddings)
    negative_images = torch.roll(image_embeddings, shifts=1, dims=0)
    negative_logits = matching_head(text_embeddings, negative_images)

    logits = torch.cat([positive_logits, negative_logits], dim=0)
    labels = torch.cat(
        [
            torch.ones_like(positive_logits),
            torch.zeros_like(negative_logits),
        ],
        dim=0,
    )
    return F.binary_cross_entropy_with_logits(logits, labels)


def grounding_bce_loss(
    alignment_logits: Tensor,
    targets: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    if alignment_logits.shape != targets.shape:
        raise ValueError("alignment logits and grounding targets must have the same shape")

    loss = F.binary_cross_entropy_with_logits(alignment_logits, targets, reduction="none")
    if valid_mask is None:
        return loss.mean()

    weights = valid_mask.to(loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def compute_jakal_losses(
    output: "JakalNetOutput",
    matching_head: nn.Module,
    *,
    temperature: float,
    contrastive_weight: float = 1.0,
    matching_weight: float = 1.0,
    grounding_weight: float = 1.0,
    grounding_targets: Tensor | None = None,
    grounding_mask: Tensor | None = None,
) -> JakalLossBundle:
    contrastive = bidirectional_contrastive_loss(
        output.text_pooled,
        output.image_pooled,
        temperature=temperature,
    )
    matching = binary_matching_loss(
        matching_head,
        output.text_pooled,
        output.image_pooled,
    )
    total = contrastive_weight * contrastive + matching_weight * matching

    grounding = None
    if grounding_targets is not None:
        grounding = grounding_bce_loss(
            output.alignment_logits,
            grounding_targets,
            grounding_mask,
        )
        total = total + grounding_weight * grounding

    return JakalLossBundle(
        contrastive=contrastive,
        matching=matching,
        grounding=grounding,
        total=total,
    )
