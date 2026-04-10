from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from jakal_net import CocoBatchCollator, CocoCaptionsDataset, JakalNetConfig, JakalNetModel
from jakal_net.losses import compute_jakal_losses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Jakal-Net on COCO captions.")
    parser.add_argument("--train-images-dir", required=True, type=Path)
    parser.add_argument("--train-annotations", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--text-model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--image-model", type=str, default="resnet18")
    parser.add_argument("--image-encoder-backend", type=str, default="torchvision")
    parser.add_argument("--max-text-tokens", type=int, default=48)
    parser.add_argument("--node-dim", type=int, default=256)
    parser.add_argument("--edge-hidden-dim", type=int, default=384)
    parser.add_argument("--num-latents", type=int, default=24)
    parser.add_argument("--num-propagation-layers", type=int, default=2)
    parser.add_argument("--train-text-encoder", action="store_true")
    parser.add_argument("--train-image-encoder", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--save-path", type=Path, default=Path("checkpoints/jakal_coco.pt"))
    return parser.parse_args()


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    dataset = CocoCaptionsDataset(
        images_dir=args.train_images_dir,
        annotations_file=args.train_annotations,
        max_samples=args.max_samples,
    )
    collator = CocoBatchCollator(
        text_model_name=args.text_model,
        image_model_name=args.image_model,
        image_encoder_backend=args.image_encoder_backend,
        max_text_tokens=args.max_text_tokens,
        local_files_only=args.local_files_only,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    config = JakalNetConfig(
        max_text_tokens=args.max_text_tokens,
        node_dim=args.node_dim,
        edge_hidden_dim=args.edge_hidden_dim,
        num_latents=args.num_latents,
        num_propagation_layers=args.num_propagation_layers,
        use_pretrained_encoders=True,
        text_encoder_name=args.text_model,
        image_encoder_name=args.image_model,
        image_encoder_backend=args.image_encoder_backend,
        train_text_encoder=args.train_text_encoder,
        train_image_encoder=args.train_image_encoder,
        local_files_only=args.local_files_only,
    )

    model = JakalNetModel(config).to(device)
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        for batch in dataloader:
            global_step += 1
            batch = move_batch_to_device(batch, device)

            output = model(
                batch["input_ids"],
                batch["images"],
                batch["attention_mask"],
            )
            losses = compute_jakal_losses(
                output,
                model.matching_head,
                temperature=config.contrastive_temperature,
            )

            optimizer.zero_grad(set_to_none=True)
            losses.total.backward()
            optimizer.step()

            print(
                f"epoch={epoch + 1} step={global_step} "
                f"total={losses.total.detach().item():.4f} "
                f"contrastive={losses.contrastive.detach().item():.4f} "
                f"matching={losses.matching.detach().item():.4f}"
            )

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
        },
        args.save_path,
    )
    print(f"saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
