from pathlib import Path


def make_safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "-").replace("=", "-")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
