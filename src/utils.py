from pathlib import Path
import torch


def make_safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "-").replace("=", "-")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

import os

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_f1):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_f1": best_f1,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["best_f1"]
