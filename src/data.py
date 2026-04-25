from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


QUOTED_TOKEN_PATTERN = re.compile(r"'((?:\\.|[^'\\])*)'|\"((?:\\.|[^\"\\])*)\"", re.DOTALL)
BIO_TAG_PATTERN = re.compile(r"^(?P<prefix>[BI])-(?P<entity>[A-Za-z0-9_]+)$")


def _clean_quoted_token(token: str) -> str:
    return token.replace("\\'", "'").replace('\\"', '"').replace("\\\\", "\\")


def parse_list_cell(cell: str) -> list[str]:
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        raise ValueError("Cannot parse null list cell.")

    text = str(cell).strip()
    if not text:
        return []

    matches = QUOTED_TOKEN_PATTERN.findall(text)
    values = [_clean_quoted_token(single if single else double) for single, double in matches]

    if text and not values:
        raise ValueError(f"Failed to parse quoted list cell: {text[:200]}")

    return values


def validate_bio_tags(tags: list[str]) -> None:
    previous_tag: str | None = None

    for index, tag in enumerate(tags):
        if tag == "O":
            previous_tag = tag
            continue

        match = BIO_TAG_PATTERN.match(tag)
        if match is None:
            raise ValueError(
                f"Malformed BIO tag at index {index}: previous_tag={previous_tag!r}, current_tag={tag!r}"
            )

        prefix = match.group("prefix")
        entity = match.group("entity")

        if prefix == "B":
            previous_tag = tag
            continue

        if previous_tag is None or previous_tag == "O":
            raise ValueError(
                f"Invalid BIO transition at index {index}: previous_tag={previous_tag!r}, current_tag={tag!r}"
            )

        previous_match = BIO_TAG_PATTERN.match(previous_tag)
        if previous_match is None or previous_match.group("entity") != entity:
            raise ValueError(
                f"Invalid BIO transition at index {index}: previous_tag={previous_tag!r}, current_tag={tag!r}"
            )

        previous_tag = tag


def _has_usable_labels(tags: list[str]) -> bool:
    return bool(tags) and any(tag.strip() for tag in tags)


def load_split_csv(path: str | Path, split_name: str) -> list[dict[str, Any]]:
    csv_path = Path(path)
    frame = pd.read_csv(csv_path)

    missing_columns = {"words", "ner"} - set(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    examples: list[dict[str, Any]] = []
    for row_id, row in frame.iterrows():
        try:
            tokens = parse_list_cell(row["words"])
        except ValueError as exc:
            raise ValueError(
                f"Failed parsing words for split={split_name}, row_id={row_id}: {exc}"
            ) from exc

        try:
            ner_tags = parse_list_cell(row["ner"])
        except ValueError as exc:
            raise ValueError(
                f"Failed parsing ner for split={split_name}, row_id={row_id}: {exc}"
            ) from exc

        if not tokens:
            raise ValueError(f"Empty token list for split={split_name}, row_id={row_id}")
        if len(tokens) != len(ner_tags):
            raise ValueError(
                f"Length mismatch for split={split_name}, row_id={row_id}: "
                f"{len(tokens)} tokens vs {len(ner_tags)} labels"
            )

        has_labels = _has_usable_labels(ner_tags)
        if not has_labels and split_name != "test":
            raise ValueError(f"Encountered empty labels for labeled split={split_name}, row_id={row_id}")
        if has_labels and any(not tag for tag in ner_tags):
            raise ValueError(f"Encountered blank label token for split={split_name}, row_id={row_id}")

        if has_labels:
            try:
                validate_bio_tags(ner_tags)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid BIO tags for split={split_name}, row_id={row_id}: {exc}"
                ) from exc

        examples.append(
            {
                "id": int(row_id),
                "split": split_name,
                "tokens": tokens,
                "ner_tags": ner_tags,
                "has_labels": has_labels,
            }
        )

    return examples


def build_label_maps(
    train_examples: list[dict[str, Any]],
    val_examples: list[dict[str, Any]],
    test_examples: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, int], dict[int, str]]:
    label_set: set[str] = set()
    for example in [*train_examples, *val_examples, *(test_examples or [])]:
        if not example.get("has_labels", True):
            continue
        label_set.update(example["ner_tags"])

    label_set.discard("")
    ordered_labels = ["O"] + sorted(label for label in label_set if label != "O")
    label2id = {label: idx for idx, label in enumerate(ordered_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def save_label_maps(label2id: dict[str, int], id2label: dict[int, str], output_dir: str | Path) -> Path:
    output_path = Path(output_dir) / "label_maps.json"
    payload = {
        "label2id": label2id,
        "id2label": {str(idx): label for idx, label in id2label.items()},
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


class TokenizedNERDataset(Dataset):
    def __init__(
        self,
        examples: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        label2id: dict[str, int],
        max_length: int,
        label_all_tokens: bool = False,
    ) -> None:
        if not tokenizer.is_fast:
            raise ValueError("Fast tokenizer is required for word_ids-based alignment.")

        self.examples = examples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.label_all_tokens = label_all_tokens
        self.features: list[dict[str, Any]] = []

        for example in examples:
            encoding = tokenizer(
                example["tokens"],
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=False,
            )
            word_ids = encoding.word_ids()
            labels: list[int] = []
            previous_word_id: int | None = None

            for word_id in word_ids:
                if word_id is None:
                    labels.append(-100)
                elif not example.get("has_labels", True):
                    labels.append(-100)
                else:
                    gold_label = example["ner_tags"][word_id]
                    if word_id != previous_word_id or label_all_tokens:
                        labels.append(label2id[gold_label])
                    else:
                        labels.append(-100)
                previous_word_id = word_id

            self.features.append(
                {
                    "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "tokens": example["tokens"],
                    "gold_labels": example["ner_tags"],
                    "word_ids": list(word_ids),
                    "example_id": example["id"],
                    "split": example["split"],
                    "has_labels": example.get("has_labels", True),
                }
            )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.features[index]


def create_data_collator(tokenizer: PreTrainedTokenizerBase) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
        model_features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        batch = tokenizer.pad(model_features, padding=True, return_tensors="pt")
        max_length = batch["input_ids"].shape[1]

        padded_labels = []
        padded_word_ids = []
        for feature in features:
            label_row = feature["labels"].tolist()
            word_ids = list(feature["word_ids"])
            pad_size = max_length - len(label_row)
            padded_labels.append(label_row + ([-100] * pad_size))
            padded_word_ids.append(word_ids + ([None] * pad_size))

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        batch["tokens"] = [feature["tokens"] for feature in features]
        batch["gold_labels"] = [feature["gold_labels"] for feature in features]
        batch["word_ids"] = padded_word_ids
        batch["example_ids"] = [feature["example_id"] for feature in features]
        batch["splits"] = [feature["split"] for feature in features]
        batch["has_labels"] = [feature["has_labels"] for feature in features]
        return batch

    return collate_fn


def decode_predictions(
    logits: torch.Tensor | list[list[list[float]]],
    labels: torch.Tensor | list[list[int]],
    id2label: dict[int, str],
) -> tuple[list[list[str]], list[list[str]]]:
    logits_tensor = torch.as_tensor(logits)
    labels_tensor = torch.as_tensor(labels)
    pred_ids = torch.argmax(logits_tensor, dim=-1)

    all_predictions: list[list[str]] = []
    all_labels: list[list[str]] = []
    for pred_row, label_row in zip(pred_ids.tolist(), labels_tensor.tolist()):
        row_predictions: list[str] = []
        row_labels: list[str] = []
        for pred_id, label_id in zip(pred_row, label_row):
            if label_id == -100:
                continue
            row_predictions.append(id2label[int(pred_id)])
            row_labels.append(id2label[int(label_id)])
        all_predictions.append(row_predictions)
        all_labels.append(row_labels)
    return all_predictions, all_labels


def summarize_examples(examples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    examples = list(examples)
    label_counter: Counter[str] = Counter()
    entity_counter: Counter[str] = Counter()
    lengths: list[int] = []
    labeled_examples = 0

    for example in examples:
        lengths.append(len(example["tokens"]))
        if example.get("has_labels", True):
            labeled_examples += 1
            label_counter.update(example["ner_tags"])
            entity_counter.update(tag[2:] for tag in example["ner_tags"] if tag.startswith("B-"))

    mean_length = sum(lengths) / len(lengths) if lengths else 0.0
    max_length = max(lengths) if lengths else 0
    return {
        "num_examples": len(examples),
        "num_labeled_examples": labeled_examples,
        "mean_tokens_per_example": round(mean_length, 2),
        "max_tokens_per_example": max_length,
        "label_distribution": dict(sorted(label_counter.items())),
        "entity_type_counts": dict(sorted(entity_counter.items())),
    }
