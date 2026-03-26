from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_iob2_file(path: Path) -> List[Dict[str, List[str]]]:
    """Parse train/dev IOB2 file into sentence-level tokens/labels."""
    examples: List[Dict[str, List[str]]] = []
    tokens: List[str] = []
    labels: List[str] = []

    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            if not line.strip():
                if tokens:
                    examples.append({"tokens": tokens, "labels": labels})
                    tokens = []
                    labels = []
                continue

            if line.startswith("#"):
                continue

            parts = re.split(r"\t|\s+", line.strip())
            if len(parts) < 2:
                raise ValueError(
                    f"Malformed IOB2 line at {path}:{lineno}: '{line}' (needs token and label)."
                )

            if len(parts) == 2:
                token = parts[0]
                label = parts[1]
            else:
                if parts[0].isdigit():
                    token = parts[1]
                    label = parts[2]
                else:
                    token = parts[0]
                    label = parts[1]

            tokens.append(token)
            labels.append(label)

    if tokens:
        examples.append({"tokens": tokens, "labels": labels})

    return examples


def parse_iob2_test_file(path: Path) -> Tuple[List[Dict[str, List[str]]], List[Dict[str, Any]]]:
    """Parse test IOB2 file and preserve lines for exact reconstruction."""
    examples: List[Dict[str, List[str]]] = []
    line_items: List[Dict[str, Any]] = []
    tokens: List[str] = []

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                line_items.append({"type": "blank", "text": line})
                if tokens:
                    examples.append({"tokens": tokens})
                    tokens = []
                continue

            if line.startswith("#"):
                line_items.append({"type": "comment", "text": line})
                continue

            parts = re.split(r"\t|\s+", line.strip())

            if len(parts) == 1:
                token = parts[0]
            elif len(parts) == 2:
                token = parts[0]
            else:
                token = parts[1] if parts[0].isdigit() else parts[0]

            tokens.append(token)
            line_items.append(
                {
                    "type": "token",
                    "parts": parts,
                    "token": token,
                }
            )

    if tokens:
        examples.append({"tokens": tokens})

    return examples, line_items


def build_label_vocab(train_examples: List[Dict[str, List[str]]], dev_examples: List[Dict[str, List[str]]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build label2id and id2label from train+dev labels."""
    label_set: List[str] = []
    seen = set()

    for example in train_examples + dev_examples:
        for lab in example["labels"]:
            if lab not in seen:
                seen.add(lab)
                label_set.append(lab)

    if "O" not in seen:
        label_set.insert(0, "O")

    label2id = {lab: idx for idx, lab in enumerate(label_set)}
    id2label = {idx: lab for lab, idx in label2id.items()}
    return label2id, id2label
