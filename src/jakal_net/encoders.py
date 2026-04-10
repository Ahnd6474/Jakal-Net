from __future__ import annotations

import torch
from torch import Tensor, nn


def masked_mean_pool(sequence: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(sequence.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (sequence * weights).sum(dim=1) / denom


def build_2d_sincos_position_embedding(
    height: int,
    width: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if dim % 4 != 0:
        raise ValueError("image hidden dimension must be divisible by 4")

    base_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
    y = torch.arange(height, device=device, dtype=base_dtype)
    x = torch.arange(width, device=device, dtype=base_dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

    quarter_dim = dim // 4
    omega = torch.arange(quarter_dim, device=device, dtype=base_dtype)
    denom = max(quarter_dim - 1, 1)
    omega = 1.0 / (10000 ** (omega / denom))

    y_embed = grid_y.reshape(-1, 1) * omega.reshape(1, -1)
    x_embed = grid_x.reshape(-1, 1) * omega.reshape(1, -1)

    embedding = torch.cat(
        [y_embed.sin(), y_embed.cos(), x_embed.sin(), x_embed.cos()],
        dim=-1,
    )
    return embedding.unsqueeze(0).to(dtype=dtype)


def _freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def _require_transformers() -> tuple[object, object]:
    try:
        from transformers import AutoModel, AutoConfig
    except ImportError as exc:
        raise ImportError(
            "transformers is required for pretrained encoders. "
            "Install it with `pip install transformers`."
        ) from exc
    return AutoModel, AutoConfig


def _require_torchvision_models() -> tuple[object, object]:
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for pretrained image encoders. "
            "Install it with `pip install torchvision`."
        ) from exc
    return ResNet18_Weights, resnet18


class LightTextEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_dim: int,
        max_length: int,
        num_layers: int,
        num_heads: int,
        ff_multiplier: int,
        dropout: float,
        pad_token_id: int,
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_dim,
            padding_idx=pad_token_id,
        )
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ff_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("text input must have shape [batch, tokens]")

        batch_size, seq_len = input_ids.shape
        if seq_len > self.position_embedding.num_embeddings:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_length "
                f"{self.position_embedding.num_embeddings}"
            )

        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id)
        attention_mask = attention_mask.bool()

        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(positions).unsqueeze(0)
        x = self.dropout(x)
        x = self.encoder(x, src_key_padding_mask=~attention_mask)
        x = self.norm(x)

        return x, attention_mask.expand(batch_size, -1)


class LightImageEncoder(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ff_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ff_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        if images.ndim != 4:
            raise ValueError("image input must have shape [batch, channels, height, width]")

        patches = self.patch_embedding(images)
        batch_size, hidden_dim, grid_h, grid_w = patches.shape
        x = patches.flatten(2).transpose(1, 2)
        pos = build_2d_sincos_position_embedding(
            grid_h,
            grid_w,
            hidden_dim,
            device=images.device,
            dtype=x.dtype,
        )
        x = self.dropout(x + pos)
        x = self.encoder(x)
        x = self.norm(x)

        mask = torch.ones(batch_size, x.size(1), device=images.device, dtype=torch.bool)
        return x, mask


class PretrainedTextEncoder(nn.Module):
    def __init__(
        self,
        *,
        model_name: str,
        output_dim: int,
        trainable: bool,
        local_files_only: bool,
    ) -> None:
        super().__init__()
        AutoModel, _ = _require_transformers()
        self.backbone = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        hidden_dim = getattr(self.backbone.config, "hidden_size", None)
        if hidden_dim is None:
            raise ValueError(f"text model `{model_name}` does not expose hidden_size")

        if not trainable:
            _freeze_module(self.backbone)

        self.proj = nn.Identity() if hidden_dim == output_dim else nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        tokens = outputs.last_hidden_state
        tokens = self.norm(self.proj(tokens))

        if attention_mask is None:
            attention_mask = torch.ones(
                tokens.size(0),
                tokens.size(1),
                device=tokens.device,
                dtype=torch.bool,
            )

        return tokens, attention_mask.bool()


class PretrainedImageEncoder(nn.Module):
    MODEL_TYPES_WITH_CLS = {
        "beit",
        "deit",
        "dinov2",
        "vit",
    }

    def __init__(
        self,
        *,
        model_name: str,
        output_dim: int,
        trainable: bool,
        local_files_only: bool,
        drop_cls_token: bool,
    ) -> None:
        super().__init__()
        AutoModel, _ = _require_transformers()
        self.backbone = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        hidden_dim = getattr(self.backbone.config, "hidden_size", None)
        if hidden_dim is None:
            raise ValueError(f"image model `{model_name}` does not expose hidden_size")

        if not trainable:
            _freeze_module(self.backbone)

        self.drop_cls_token = drop_cls_token
        self.model_type = getattr(self.backbone.config, "model_type", "")
        self.proj = nn.Identity() if hidden_dim == output_dim else nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        outputs = self.backbone(pixel_values=images)
        tokens = outputs.last_hidden_state

        if (
            self.drop_cls_token
            and self.model_type in self.MODEL_TYPES_WITH_CLS
            and tokens.size(1) > 1
        ):
            tokens = tokens[:, 1:, :]

        tokens = self.norm(self.proj(tokens))
        mask = torch.ones(tokens.size(0), tokens.size(1), device=tokens.device, dtype=torch.bool)
        return tokens, mask


class TorchvisionResNetImageEncoder(nn.Module):
    def __init__(
        self,
        *,
        model_name: str,
        output_dim: int,
        trainable: bool,
    ) -> None:
        super().__init__()
        ResNet18_Weights, resnet18 = _require_torchvision_models()
        if model_name != "resnet18":
            raise ValueError(
                f"unsupported torchvision image encoder `{model_name}`; "
                "currently only `resnet18` is supported"
            )

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        if not trainable:
            _freeze_module(backbone)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layers = nn.Sequential(
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        hidden_dim = backbone.fc.in_features
        self.proj = nn.Identity() if hidden_dim == output_dim else nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        x = self.stem(images)
        x = self.layers(x)
        batch_size, channels, height, width = x.shape
        tokens = x.reshape(batch_size, channels, height * width).transpose(1, 2)
        tokens = self.norm(self.proj(tokens))
        mask = torch.ones(batch_size, tokens.size(1), device=tokens.device, dtype=torch.bool)
        return tokens, mask
