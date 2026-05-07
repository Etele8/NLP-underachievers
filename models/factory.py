from __future__ import annotations

from typing import Any

from transformers import AutoModelForTokenClassification


MODEL_NAME_MAP = {
    "mbert": "bert-base-multilingual-cased",
    "xlmr": "xlm-roberta-base",
}


def _apply_lora_if_requested(model: Any, config: dict[str, Any]) -> Any:
    if not bool(config.get("lora", False)):
        return model

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError:
        print("Warning: peft is not installed; continuing without LoRA.")
        return model

    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=int(config.get("lora_r", 8)),
        lora_alpha=int(config.get("lora_alpha", 16)),
        lora_dropout=float(config.get("lora_dropout", 0.1)),
    )
    return get_peft_model(model, lora_config)


def get_model(config: dict[str, Any], num_labels: int):
    """Create a Hugging Face token classification model for an experiment.

    This factory intentionally contains no training logic. It preserves the
    standard Hugging Face forward pass and only adds optional PEFT LoRA wrapping
    when requested and available.
    """
    model_key = str(config.get("model", "")).lower()
    model_name = MODEL_NAME_MAP.get(model_key)
    if model_name is None:
        supported = ", ".join(sorted(MODEL_NAME_MAP))
        raise ValueError(f"Unknown model {model_key!r}. Supported models: {supported}")

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    return _apply_lora_if_requested(model, config)
