from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TokenClassificationDataset(Dataset):
    def __init__(self, encodings: Dict[str, List[Any]]):
        self.encodings = encodings

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.encodings["labels"][idx], dtype=torch.long),
        }


def align_labels_with_tokens(
    examples: List[Dict[str, List[str]]],
    tokenizer: AutoTokenizer,
    label2id: Dict[str, int],
    max_length: int = 128,
) -> Dict[str, List[Any]]:
    """Tokenize and align word-level labels to tokenizer subwords."""
    tokenized = tokenizer(
        [ex["tokens"] for ex in examples],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_attention_mask=True,
        return_offsets_mapping=False,
    )

    all_labels = []
    all_word_ids = []
    for i, ex in enumerate(examples):
        labels = ex.get("labels")
        word_ids = tokenized.word_ids(batch_index=i)
        aligned_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif labels is None:
                aligned_labels.append(-100)
            else:
                label = labels[word_idx]
                if word_idx != previous_word_idx:
                    aligned_labels.append(label2id[label])
                else:
                    aligned_labels.append(-100)
                previous_word_idx = word_idx

        all_labels.append(aligned_labels)
        all_word_ids.append(word_ids)

    tokenized["labels"] = all_labels
    tokenized["word_ids"] = all_word_ids
    return tokenized


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


from seqeval.metrics import precision_score, recall_score, f1_score, classification_report


def compute_seqeval_metrics(predictions: List[List[str]], references: List[List[str]]) -> Dict[str, Any]:
    metrics = {
        "precision": precision_score(references, predictions),
        "recall": recall_score(references, predictions),
        "f1": f1_score(references, predictions),
        "report": classification_report(references, predictions, digits=4),
    }
    return metrics
