from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedTokenizerBase
from transformers.modeling_outputs import TokenClassifierOutput


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


def _entity_type_order(label2id: dict[str, int]) -> tuple[list[str], list[int]]:
    entity_types = ["O"]
    for label, _ in sorted(label2id.items(), key=lambda item: item[1]):
        if label == "O":
            continue
        entity_type = label.split("-", 1)[1]
        if entity_type not in entity_types:
            entity_types.append(entity_type)
    entity_type_to_index = {entity_type: idx for idx, entity_type in enumerate(entity_types)}
    label_entity_type_ids = []
    for label, _ in sorted(label2id.items(), key=lambda item: item[1]):
        if label == "O":
            label_entity_type_ids.append(entity_type_to_index["O"])
        else:
            label_entity_type_ids.append(entity_type_to_index[label.split("-", 1)[1]])
    return entity_types, label_entity_type_ids


class LanguageAwareTokenClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        config,
        num_labels: int,
        label2id: dict[str, int],
        id2label: dict[int, str],
        lid2id: dict[str, int] | None = None,
        use_language_bias: bool = False,
        use_lid_feature: bool = False,
        language_embedding_dim: int = 32,
        language_gate_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.num_labels = num_labels
        self.label2id = label2id
        self.id2label = id2label
        self.lid2id = lid2id or {"<pad>": 0}
        self.use_language_bias = bool(use_language_bias)
        self.use_lid_feature = bool(use_lid_feature)
        self.lid_pad_id = self.lid2id.get("<pad>", 0)

        classifier_dropout = getattr(config, "classifier_dropout", None)
        hidden_dropout = getattr(config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(classifier_dropout if classifier_dropout is not None else hidden_dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        entity_types, label_entity_type_ids = _entity_type_order(label2id)
        self.entity_types = entity_types
        self.register_buffer(
            "label_entity_type_ids",
            torch.tensor(label_entity_type_ids, dtype=torch.long),
            persistent=False,
        )

        if self.use_language_bias:
            self.language_embedding = nn.Embedding(
                num_embeddings=len(self.lid2id),
                embedding_dim=language_embedding_dim,
                padding_idx=self.lid_pad_id,
            )
            gate_input_dim = config.hidden_size + language_embedding_dim if self.use_lid_feature else language_embedding_dim
            self.language_gate = nn.Sequential(
                nn.Linear(gate_input_dim, language_gate_hidden_dim),
                nn.Tanh(),
                nn.Linear(language_gate_hidden_dim, len(entity_types)),
                nn.Sigmoid(),
            )
            self.entity_type_language_bias = nn.Parameter(torch.zeros(len(entity_types), len(self.lid2id)))
        else:
            self.language_embedding = None
            self.language_gate = None
            self.entity_type_language_bias = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        lid_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> TokenClassifierOutput:
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        sequence_output = self.dropout(encoder_outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        if self.use_language_bias:
            if lid_ids is None:
                raise ValueError("lid_ids are required when use_language_bias is enabled.")

            language_embeddings = self.language_embedding(lid_ids)
            gate_input = (
                torch.cat([sequence_output, language_embeddings], dim=-1)
                if self.use_lid_feature
                else language_embeddings
            )
            gate = self.language_gate(gate_input)
            bias_lookup = self.entity_type_language_bias.t()[lid_ids]
            label_adjustment = (gate * bias_lookup)[..., self.label_entity_type_ids]
            logits = logits + label_adjustment

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


def create_token_classifier(
    model_name: str,
    num_labels: int,
    label2id: dict[str, int],
    id2label: dict[int, str],
    lid2id: dict[str, int] | None = None,
    dropout: float | None = None,
    use_language_bias: bool = False,
    use_lid_feature: bool = False,
    language_embedding_dim: int = 32,
    language_gate_hidden_dim: int = 128,
):
    last_error: Exception | None = None
    for candidate in iter_model_name_candidates(model_name):
        try:
            config = AutoConfig.from_pretrained(candidate)
            config.num_labels = num_labels
            config.label2id = label2id
            config.id2label = id2label
            config.use_language_bias = use_language_bias
            config.use_lid_feature = use_lid_feature
            config.language_embedding_dim = language_embedding_dim
            config.language_gate_hidden_dim = language_gate_hidden_dim
            if dropout is not None:
                if hasattr(config, "hidden_dropout_prob"):
                    config.hidden_dropout_prob = dropout
                if hasattr(config, "attention_probs_dropout_prob"):
                    config.attention_probs_dropout_prob = dropout

            encoder = AutoModel.from_pretrained(candidate, config=config)
            model = LanguageAwareTokenClassifier(
                encoder=encoder,
                config=config,
                num_labels=num_labels,
                label2id=label2id,
                id2label=id2label,
                lid2id=lid2id,
                use_language_bias=use_language_bias,
                use_lid_feature=use_lid_feature,
                language_embedding_dim=language_embedding_dim,
                language_gate_hidden_dim=language_gate_hidden_dim,
            )
            return model, candidate
        except Exception as exc:  # pragma: no cover - exercised by runtime environment
            last_error = exc
    raise RuntimeError(f"Unable to load model for {model_name}") from last_error
