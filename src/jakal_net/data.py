from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset


def _require_transformers_tokenizer() -> object:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for COCO text preprocessing. "
            "Install it with `pip install transformers`."
        ) from exc
    return AutoTokenizer


def _require_torchvision_transforms() -> object:
    try:
        from torchvision.models import ResNet18_Weights
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for COCO image preprocessing. "
            "Install it with `pip install torchvision`."
        ) from exc
    return ResNet18_Weights


@dataclass(slots=True)
class CocoCaptionRecord:
    annotation_id: int
    image_id: int
    image_path: str
    caption: str


class CocoCaptionsDataset(Dataset[CocoCaptionRecord]):
    def __init__(
        self,
        *,
        images_dir: str | Path,
        annotations_file: str | Path,
        max_samples: int | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.annotations_file = Path(annotations_file)
        if not self.images_dir.exists():
            raise FileNotFoundError(f"COCO image directory not found: {self.images_dir}")
        if not self.annotations_file.exists():
            raise FileNotFoundError(
                f"COCO annotations file not found: {self.annotations_file}"
            )

        with self.annotations_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        image_map = {
            int(image["id"]): self.images_dir / str(image["file_name"])
            for image in payload["images"]
        }

        records: list[CocoCaptionRecord] = []
        for annotation in payload["annotations"]:
            caption = str(annotation["caption"]).strip()
            if not caption:
                continue

            image_id = int(annotation["image_id"])
            image_path = image_map.get(image_id)
            if image_path is None or not image_path.exists():
                continue

            records.append(
                CocoCaptionRecord(
                    annotation_id=int(annotation["id"]),
                    image_id=image_id,
                    image_path=str(image_path),
                    caption=caption,
                )
            )
            if max_samples is not None and len(records) >= max_samples:
                break

        if not records:
            raise ValueError("no valid COCO caption records were loaded")

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> CocoCaptionRecord:
        return self.records[index]


class CocoBatchCollator:
    def __init__(
        self,
        *,
        text_model_name: str,
        image_model_name: str,
        image_encoder_backend: str,
        max_text_tokens: int,
        local_files_only: bool,
    ) -> None:
        AutoTokenizer = _require_transformers_tokenizer()
        self.tokenizer = AutoTokenizer.from_pretrained(
            text_model_name,
            use_fast=False,
            local_files_only=local_files_only,
        )
        self.max_text_tokens = max_text_tokens
        self.image_encoder_backend = image_encoder_backend

        if image_encoder_backend == "torchvision":
            ResNet18_Weights = _require_torchvision_transforms()
            if image_model_name != "resnet18":
                raise ValueError(
                    f"unsupported torchvision image encoder `{image_model_name}`; "
                    "currently only `resnet18` is supported"
                )
            self.image_transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
        else:
            raise ValueError(
                f"unsupported image_encoder_backend `{image_encoder_backend}` in collator"
            )

    def __call__(self, batch: list[CocoCaptionRecord]) -> dict[str, Any]:
        captions = [record.caption for record in batch]
        pil_images = []
        for record in batch:
            with Image.open(record.image_path) as image:
                pil_images.append(image.convert("RGB"))

        tokenized = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=self.max_text_tokens,
            return_tensors="pt",
        )
        image_tensors = torch.stack([self.image_transform(image) for image in pil_images], dim=0)

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"].bool(),
            "images": image_tensors,
            "captions": captions,
            "image_ids": torch.tensor([record.image_id for record in batch], dtype=torch.long),
            "annotation_ids": torch.tensor(
                [record.annotation_id for record in batch],
                dtype=torch.long,
            ),
            "image_paths": [record.image_path for record in batch],
        }
