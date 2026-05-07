from __future__ import annotations

from .factory import get_model


MODEL_KEY = "mbert"
MODEL_NAME = "bert-base-multilingual-cased"


def create_model(config: dict, num_labels: int):
    """Lightweight mBERT wrapper that delegates to the shared model factory."""
    merged_config = {**config, "model": MODEL_KEY}
    return get_model(merged_config, num_labels)
