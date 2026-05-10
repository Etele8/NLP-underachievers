from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from seqeval.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


def compute_metrics(predictions: list[list[str]], golds: list[list[str]]) -> dict[str, Any]:
    return {
        "precision": precision_score(golds, predictions, zero_division=0),
        "recall": recall_score(golds, predictions, zero_division=0),
        "f1": f1_score(golds, predictions, zero_division=0),
        "accuracy": accuracy_score(golds, predictions),
        "classification_report": classification_report(golds, predictions, digits=4, zero_division=0),
    }


def save_predictions_table(
    examples: list[dict[str, Any]],
    gold_labels: list[list[str]],
    pred_labels: list[list[str]],
    path: str | Path,
) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for example, gold, pred in zip(examples, gold_labels, pred_labels):
        records.append(
            {
                "split": example["split"],
                "row_id": example["id"],
                "tokens": example["tokens"],
                "lid_tags": example.get("lid_tags", []),
                "gold_labels": gold,
                "pred_labels": pred,
            }
        )

    if path_obj.suffix.lower() == ".jsonl":
        with path_obj.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return

    with path_obj.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "row_id", "tokens", "lid_tags", "gold_labels", "pred_labels"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "split": record["split"],
                    "row_id": record["row_id"],
                    "tokens": json.dumps(record["tokens"], ensure_ascii=False),
                    "lid_tags": json.dumps(record["lid_tags"], ensure_ascii=False),
                    "gold_labels": json.dumps(record["gold_labels"], ensure_ascii=False),
                    "pred_labels": json.dumps(record["pred_labels"], ensure_ascii=False),
                }
            )
