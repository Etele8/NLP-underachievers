from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def extract_spans(tokens: list[str], labels: list[str]) -> list[dict[str, Any]]:
    """Convert a BIO-tagged sequence into a list of entity span dicts."""
    spans: list[dict[str, Any]] = []
    i = 0
    while i < len(labels):
        if labels[i].startswith("B-"):
            entity_type = labels[i][2:]
            start = i
            j = i + 1
            while j < len(labels) and labels[j] == f"I-{entity_type}":
                j += 1
            spans.append(
                {
                    "start": start,
                    "end": j,
                    "entity_type": entity_type,
                    "text": " ".join(tokens[start:j]) if tokens else "",
                }
            )
            i = j
        else:
            i += 1
    return spans


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(0, min(a_end, b_end) - max(a_start, b_start)) > 0


def _categorize_fn(gold_span: dict[str, Any], fp_spans: list[dict[str, Any]]) -> tuple[str, str | None]:
    """Classify a missed gold span; return (error_type, pred_entity_type)."""
    for p in fp_spans:
        if _overlaps(gold_span["start"], gold_span["end"], p["start"], p["end"]):
            if p["entity_type"] != gold_span["entity_type"]:
                return "label_confusion", p["entity_type"]
            return "boundary_error", p["entity_type"]
    return "missed_span", None


def _categorize_fp(pred_span: dict[str, Any], fn_spans: list[dict[str, Any]], gold_spans: list[dict[str, Any]]) -> tuple[str, str | None]:
    """Classify a spurious predicted span; return (error_type, gold_entity_type)."""
    for g in fn_spans:
        if _overlaps(pred_span["start"], pred_span["end"], g["start"], g["end"]):
            if g["entity_type"] != pred_span["entity_type"]:
                return "label_confusion", g["entity_type"]
            return "boundary_error", g["entity_type"]
    # Check if it overlaps any gold span that was a TP (boundary error case)
    for g in gold_spans:
        if _overlaps(pred_span["start"], pred_span["end"], g["start"], g["end"]):
            return "boundary_error", g["entity_type"]
    return "spurious_span", None


def analyze_sentence(
    row_id: int,
    tokens: list[str],
    gold_labels: list[str],
    pred_labels: list[str],
    lid_tags: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compare gold vs predicted labels for one sentence.
    Returns counts (tp, fp, fn) and a list of per-error dicts for analysis.
    """
    gold_spans = extract_spans(tokens, gold_labels)
    pred_spans = extract_spans(tokens, pred_labels)

    gold_set = {(s["start"], s["end"], s["entity_type"]) for s in gold_spans}
    pred_set = {(s["start"], s["end"], s["entity_type"]) for s in pred_spans}
    tp_keys = gold_set & pred_set

    fn_spans = [s for s in gold_spans if (s["start"], s["end"], s["entity_type"]) not in tp_keys]
    fp_spans = [s for s in pred_spans if (s["start"], s["end"], s["entity_type"]) not in tp_keys]

    errors: list[dict[str, Any]] = []
    for fn in fn_spans:
        error_type, pred_entity = _categorize_fn(fn, fp_spans)
        errors.append(
            {
                "row_id": row_id,
                "error_type": error_type,
                "side": "fn",
                "gold_entity_type": fn["entity_type"],
                "pred_entity_type": pred_entity,
                "span_text": fn["text"],
                "context_tokens": tokens,
                "gold_labels": gold_labels,
                "pred_labels": pred_labels,
                "lid_tags": lid_tags or [],
            }
        )

    for fp in fp_spans:
        error_type, gold_entity = _categorize_fp(fp, fn_spans, gold_spans)
        errors.append(
            {
                "row_id": row_id,
                "error_type": error_type,
                "side": "fp",
                "gold_entity_type": gold_entity,
                "pred_entity_type": fp["entity_type"],
                "span_text": fp["text"],
                "context_tokens": tokens,
                "gold_labels": gold_labels,
                "pred_labels": pred_labels,
                "lid_tags": lid_tags or [],
            }
        )

    return {"tp": len(tp_keys), "fp": len(fp_spans), "fn": len(fn_spans), "errors": errors}


def compute_entity_confusion_matrix(
    gold_seqs: list[list[str]],
    pred_seqs: list[list[str]],
    entity_types: list[str],
) -> dict[str, dict[str, int]]:
    """
    Entity-level confusion matrix (span boundaries must match exactly).
    Rows = gold entity types (+ 'O' for spurious FP).
    Cols = predicted entity types (+ 'O' for missed FN).
    """
    labels = entity_types + ["O"]
    matrix: dict[str, dict[str, int]] = {g: {p: 0 for p in labels} for g in labels}

    for gold_seq, pred_seq in zip(gold_seqs, pred_seqs):
        dummy = [str(i) for i in range(len(gold_seq))]
        gold_spans = extract_spans(dummy, gold_seq)
        pred_spans = extract_spans(dummy, pred_seq)

        gold_set = {(s["start"], s["end"], s["entity_type"]) for s in gold_spans}
        pred_set = {(s["start"], s["end"], s["entity_type"]) for s in pred_spans}

        matched_gold: set[tuple] = set()
        matched_pred: set[tuple] = set()

        for g_s, g_e, g_t in gold_set:
            for p_s, p_e, p_t in pred_set:
                if g_s == p_s and g_e == p_e:
                    matrix[g_t][p_t] += 1
                    matched_gold.add((g_s, g_e, g_t))
                    matched_pred.add((p_s, p_e, p_t))

        for _, _, g_t in gold_set - matched_gold:
            matrix[g_t]["O"] += 1

        for _, _, p_t in pred_set - matched_pred:
            matrix["O"][p_t] += 1

    return matrix


def load_predictions(jsonl_path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def sample_errors(errors: list[dict[str, Any]], n: int = 100, seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    return rng.sample(errors, min(n, len(errors)))
