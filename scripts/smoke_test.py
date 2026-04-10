from __future__ import annotations

import torch

from jakal_net import JakalNetConfig, JakalNetModel, compute_jakal_losses


def main() -> None:
    torch.manual_seed(7)

    config = JakalNetConfig(
        vocab_size=2_048,
        max_text_tokens=32,
        image_size=64,
        patch_size=8,
        node_dim=128,
        edge_hidden_dim=192,
        text_encoder_layers=2,
        image_encoder_layers=2,
        num_heads=4,
        num_latents=16,
        num_propagation_layers=2,
    )
    model = JakalNetModel(config)

    batch_size = 4
    seq_len = 24
    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))
    input_ids[:, -4:] = config.pad_token_id
    attention_mask = input_ids.ne(config.pad_token_id)
    images = torch.randn(batch_size, config.image_channels, config.image_size, config.image_size)

    output = model(input_ids, images, attention_mask)
    losses = compute_jakal_losses(
        output,
        model.matching_head,
        temperature=config.contrastive_temperature,
    )

    print("text tokens:", tuple(output.text_tokens.shape))
    print("image tokens:", tuple(output.image_tokens.shape))
    print("latent state:", tuple(output.latent_state.shape))
    print("latent value:", tuple(output.latent_value.shape))
    print("similarity map:", tuple(output.similarity_map.shape))
    print("contrastive loss:", float(losses.contrastive.detach()))
    print("matching loss:", float(losses.matching.detach()))
    print("total loss:", float(losses.total.detach()))


if __name__ == "__main__":
    main()
