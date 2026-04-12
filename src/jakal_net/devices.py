from __future__ import annotations

import torch


def resolve_device(device_name: str) -> torch.device:
    normalized = device_name.strip().lower()
    if normalized in {"cpu", "cuda"}:
        return torch.device(normalized)
    if normalized in {"directml", "dml"}:
        try:
            import torch_directml
        except ImportError as exc:
            raise RuntimeError(
                "DirectML device requested but torch-directml is not installed."
            ) from exc
        return torch_directml.device()
    return torch.device(device_name)


def describe_device(device_name: str) -> str:
    normalized = device_name.strip().lower()
    if normalized in {"directml", "dml"}:
        try:
            import torch_directml
        except ImportError:
            return "directml (unavailable)"
        return f"directml:{torch_directml.default_device()} ({torch_directml.device_name(0)})"
    return str(resolve_device(device_name))
