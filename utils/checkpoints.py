from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def _state_dict_or_none(component: Any) -> dict[str, Any] | None:
    if component is None or not hasattr(component, "state_dict"):
        return None
    return component.state_dict()


def save_checkpoint(
    model,
    optimizer=None,
    scheduler=None,
    epoch=None,
    step=None,
    output_dir=None,
    filename="checkpoint.pt",
):
    """Save a crash-safe checkpoint using an atomic replace."""
    if output_dir is None:
        raise ValueError("output_dir is required for save_checkpoint")

    checkpoint_dir = Path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / filename
    temp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": _state_dict_or_none(optimizer),
        "scheduler_state_dict": _state_dict_or_none(scheduler),
        "epoch": epoch,
        "step": step,
    }
    torch.save(payload, temp_path)
    os.replace(temp_path, checkpoint_path)
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load available checkpoint state and return restored objects plus metadata."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    metadata = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict"}
    }
    return model, optimizer, scheduler, metadata


def get_latest_checkpoint(output_dir):
    """Return the newest checkpoint file below output_dir, or None if absent."""
    checkpoint_dir = Path(output_dir)
    if not checkpoint_dir.exists():
        return None

    candidates = [
        path
        for path in checkpoint_dir.rglob("*.pt")
        if path.is_file() and not path.name.endswith(".tmp")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)
