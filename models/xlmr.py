from __future__ import annotations

from .factory import get_model


MODEL_KEY = "xlmr"
MODEL_NAME = "xlm-roberta-base"


def create_model(config: dict, num_labels: int):
    """Lightweight XLM-R wrapper that delegates to the shared model factory."""
    merged_config = {**config, "model": MODEL_KEY}
    return get_model(merged_config, num_labels)
