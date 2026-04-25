from __future__ import annotations

from typing import Iterable

from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer, PreTrainedTokenizerBase


MODEL_NAME_FALLBACKS = {
    "FacebookAI/xlm-roberta-base": ["FacebookAI/xlm-roberta-base", "xlm-roberta-base"],
    "xlm-roberta-base": ["xlm-roberta-base", "FacebookAI/xlm-roberta-base"],
}


def iter_model_name_candidates(model_name: str) -> Iterable[str]:
    yielded: set[str] = set()
    for candidate in [model_name, *MODEL_NAME_FALLBACKS.get(model_name, [])]:
        if candidate not in yielded:
            yielded.add(candidate)
            yield candidate


def load_tokenizer(model_name: str) -> tuple[PreTrainedTokenizerBase, str]:
    last_error: Exception | None = None
    for candidate in iter_model_name_candidates(model_name):
        try:
            return AutoTokenizer.from_pretrained(candidate), candidate
        except Exception as exc:  # pragma: no cover - exercised by runtime environment
            last_error = exc
    raise RuntimeError(f"Unable to load tokenizer for {model_name}") from last_error


def create_token_classifier(
    model_name: str,
    num_labels: int,
    label2id: dict[str, int],
    id2label: dict[int, str],
    dropout: float | None = None,
):
    last_error: Exception | None = None
    for candidate in iter_model_name_candidates(model_name):
        try:
            config = AutoConfig.from_pretrained(candidate)
            config.num_labels = num_labels
            config.label2id = label2id
            config.id2label = id2label
            if dropout is not None:
                if hasattr(config, "hidden_dropout_prob"):
                    config.hidden_dropout_prob = dropout
                if hasattr(config, "attention_probs_dropout_prob"):
                    config.attention_probs_dropout_prob = dropout

            model = AutoModelForTokenClassification.from_pretrained(candidate, config=config)
            return model, candidate
        except Exception as exc:  # pragma: no cover - exercised by runtime environment
            last_error = exc
    raise RuntimeError(f"Unable to load model for {model_name}") from last_error
