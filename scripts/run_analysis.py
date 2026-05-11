"""
Full evaluation & analysis pipeline for outputs/real_ner/ runs.

Produces in outputs/analysis/:
  - model_comparison.csv          overall precision/recall/F1 across all runs
  - per_class_f1.csv              per-entity F1 for every run
  - per_class_f1.png              grouped bar chart
  - training_curves.png           val-F1 & train-loss vs epoch for every run
  - confusion_matrix_<run>.csv    entity-level confusion matrix per run
  - confusion_matrix_<run>.png    heatmap per run
  - error_categories.csv          FN/FP breakdown by error type per run
  - error_categories.png          stacked bar chart
  - error_samples_<run>.json      up to 100 sampled errors per run

Run from project root:
    python scripts/run_analysis.py [--runs-dir outputs/real_ner] [--output-dir outputs/analysis]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.error_analysis import (
    analyze_sentence,
    compute_entity_confusion_matrix,
    load_predictions,
    sample_errors,
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

ENTITY_TYPES = ["EVENT", "GROUP", "LOC", "ORG", "OTHER", "PER", "PROD", "TIME", "TITLE"]

# ── helpers ──────────────────────────────────────────────────────────────────

def _run_label(name: str) -> str:
    """Strip timestamp suffix and prettify run directory name."""
    parts = name.rsplit("_", 2)
    base = "_".join(parts[:-2]) if len(parts) == 3 and parts[-1].isdigit() and parts[-2].isdigit() else name
    return base.replace("mbert", "mBERT").replace("xlmr", "XLM-R").replace("_language_bias", "+LangBias").replace("_full_ft", " FullFT")


def _load_run(run_dir: Path) -> dict:
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))

    epoch_metrics: list[dict] = []
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    epoch_metrics.append(json.loads(line))

    best_metrics = json.loads((run_dir / "best_validation_metrics.json").read_text(encoding="utf-8"))
    run_summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    predictions = load_predictions(run_dir / "validation_predictions_best.jsonl")

    return {
        "name": run_dir.name,
        "label": _run_label(run_dir.name),
        "config": config,
        "epoch_metrics": epoch_metrics,
        "best_metrics": best_metrics,
        "run_summary": run_summary,
        "predictions": predictions,
    }


def _compute_per_class_f1(predictions: list[dict]) -> dict[str, dict[str, float]]:
    """Re-derive per-class metrics from prediction records using seqeval."""
    gold_seqs = [r["gold_labels"] for r in predictions if r["gold_labels"]]
    pred_seqs = [r["pred_labels"] for r in predictions if r["gold_labels"]]

    report_str = classification_report(gold_seqs, pred_seqs, digits=4, zero_division=0)
    per_class: dict[str, dict[str, float]] = {}

    for line in report_str.splitlines():
        line = line.strip()
        if not line or line.startswith("micro") or line.startswith("macro") or line.startswith("weighted"):
            continue
        parts = line.split()
        # Format: entity  prec  recall  f1  support
        if len(parts) == 5:
            entity = parts[0]
            try:
                per_class[entity] = {
                    "precision": float(parts[1]),
                    "recall": float(parts[2]),
                    "f1": float(parts[3]),
                    "support": int(parts[4]),
                }
            except ValueError:
                continue

    return per_class


def _run_error_analysis(predictions: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Run span-level error analysis over all prediction records."""
    all_errors: list[dict] = []
    category_counts: dict[str, int] = defaultdict(int)

    for rec in predictions:
        if not rec["gold_labels"]:
            continue
        result = analyze_sentence(
            row_id=rec["row_id"],
            tokens=rec["tokens"],
            gold_labels=rec["gold_labels"],
            pred_labels=rec["pred_labels"],
            lid_tags=rec.get("lid_tags"),
        )
        for err in result["errors"]:
            category_counts[err["error_type"]] += 1
        all_errors.extend(result["errors"])

    return all_errors, dict(category_counts)


# ── plots ─────────────────────────────────────────────────────────────────────

def _plot_per_class_f1(per_class_by_run: dict[str, dict[str, dict]], out_path: Path) -> None:
    labels = ENTITY_TYPES
    run_names = list(per_class_by_run.keys())
    x = np.arange(len(labels))
    width = 0.8 / len(run_names)
    offsets = np.linspace(-(len(run_names) - 1) / 2, (len(run_names) - 1) / 2, len(run_names)) * width

    fig, ax = plt.subplots(figsize=(14, 5))
    for offset, run_name in zip(offsets, run_names):
        f1_vals = [per_class_by_run[run_name].get(et, {}).get("f1", 0.0) for et in labels]
        ax.bar(x + offset, f1_vals, width, label=run_name)

    ax.set_xlabel("Entity Type")
    ax.set_ylabel("F1")
    ax.set_title("Per-Class F1 by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_training_curves(runs: list[dict], out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for run in runs:
        epochs_data = run["epoch_metrics"]
        if not epochs_data:
            continue
        epochs = [d["epoch"] for d in epochs_data]
        f1 = [d["val_f1"] for d in epochs_data]
        loss = [d["train_loss"] for d in epochs_data]
        label = run["label"]
        ax1.plot(epochs, f1, marker="o", label=label)
        ax2.plot(epochs, loss, marker="o", label=label)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation F1")
    ax1.set_title("Validation F1 per Epoch")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Train Loss")
    ax2.set_title("Training Loss per Epoch")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_confusion_matrix(matrix: dict[str, dict[str, int]], label: str, out_path: Path) -> None:
    row_labels = ENTITY_TYPES
    col_labels = ENTITY_TYPES + ["O"]

    data = np.array([[matrix.get(r, {}).get(c, 0) for c in col_labels] for r in row_labels], dtype=float)

    # Row-normalise (recall orientation: of each gold type, what was predicted)
    row_sums = data.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm = data / row_sums

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.03)

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(f"Entity Confusion Matrix — {label}\n(row-normalised; 'O' col = missed spans)")

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            count = int(data[i, j])
            if count > 0:
                color = "white" if norm[i, j] > 0.6 else "black"
                ax.text(j, i, str(count), ha="center", va="center", fontsize=7, color=color)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_error_categories(category_by_run: dict[str, dict[str, int]], out_path: Path) -> None:
    error_types = ["missed_span", "spurious_span", "boundary_error", "label_confusion"]
    run_names = list(category_by_run.keys())
    x = np.arange(len(run_names))
    width = 0.6
    bottoms = np.zeros(len(run_names))

    colors = ["#e74c3c", "#f39c12", "#3498db", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(10, 5))

    for et, color in zip(error_types, colors):
        vals = [category_by_run[rn].get(et, 0) for rn in run_names]
        ax.bar(x, vals, width, bottom=bottoms, label=et.replace("_", " ").title(), color=color)
        bottoms += np.array(vals, dtype=float)

    ax.set_xticks(x)
    ax.set_xticklabels(run_names, rotation=15, ha="right")
    ax.set_ylabel("Error Count")
    ax.set_title("Error Categories per Model")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run full evaluation analysis on real_ner outputs.")
    parser.add_argument("--runs-dir", default="outputs/real_ner", help="Directory containing run subdirs.")
    parser.add_argument("--output-dir", default="outputs/analysis", help="Where to write analysis artifacts.")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir())
    if not run_dirs:
        print(f"No run directories found in {runs_dir}")
        return

    print(f"Found {len(run_dirs)} runs: {[d.name for d in run_dirs]}")

    runs = [_load_run(d) for d in run_dirs]

    # ── 1. overall comparison table ──────────────────────────────────────────
    comparison_rows = []
    for run in runs:
        bm = run["best_metrics"]
        rs = run["run_summary"]
        comparison_rows.append(
            {
                "Run": run["label"],
                "Best Epoch": rs.get("last_epoch_metrics", {}).get("epoch", "?"),
                "Precision": round(bm["precision"], 4),
                "Recall": round(bm["recall"], 4),
                "F1": round(bm["f1"], 4),
                "Accuracy": round(bm["accuracy"], 4),
            }
        )
    df_comparison = pd.DataFrame(comparison_rows)
    df_comparison.to_csv(out_dir / "model_comparison.csv", index=False)
    print("\n── Model Comparison ──")
    print(df_comparison.to_string(index=False))

    # ── 2. per-class F1 table ─────────────────────────────────────────────────
    per_class_by_run: dict[str, dict[str, dict]] = {}
    for run in runs:
        per_class_by_run[run["label"]] = _compute_per_class_f1(run["predictions"])

    rows = []
    for et in ENTITY_TYPES:
        row: dict = {"Entity": et}
        for run in runs:
            pc = per_class_by_run[run["label"]].get(et, {})
            row[f"{run['label']} P"] = round(pc.get("precision", 0.0), 4)
            row[f"{run['label']} R"] = round(pc.get("recall", 0.0), 4)
            row[f"{run['label']} F1"] = round(pc.get("f1", 0.0), 4)
    rows.append(row)

    # Simpler: one F1 column per run
    f1_rows = []
    for et in ENTITY_TYPES:
        row = {"Entity": et}
        for run in runs:
            row[run["label"]] = round(per_class_by_run[run["label"]].get(et, {}).get("f1", 0.0), 4)
        f1_rows.append(row)

    df_per_class = pd.DataFrame(f1_rows)
    df_per_class.to_csv(out_dir / "per_class_f1.csv", index=False)
    print("\n── Per-Class F1 ──")
    print(df_per_class.to_string(index=False))

    # Full P/R/F1 per entity per model
    detail_rows = []
    for et in ENTITY_TYPES:
        for run in runs:
            pc = per_class_by_run[run["label"]].get(et, {})
            detail_rows.append(
                {
                    "Entity": et,
                    "Run": run["label"],
                    "Precision": round(pc.get("precision", 0.0), 4),
                    "Recall": round(pc.get("recall", 0.0), 4),
                    "F1": round(pc.get("f1", 0.0), 4),
                    "Support": pc.get("support", 0),
                }
            )
    pd.DataFrame(detail_rows).to_csv(out_dir / "per_class_detail.csv", index=False)

    # ── 3. training curves ────────────────────────────────────────────────────
    _plot_training_curves(runs, out_dir / "training_curves.png")
    print("Saved: training_curves.png")

    # Save training curve data as CSV
    curve_rows = []
    for run in runs:
        for em in run["epoch_metrics"]:
            curve_rows.append({"run": run["label"], **em})
    pd.DataFrame(curve_rows).to_csv(out_dir / "training_curves.csv", index=False)

    # ── 4. per-class F1 plot ──────────────────────────────────────────────────
    _plot_per_class_f1(per_class_by_run, out_dir / "per_class_f1.png")
    print("Saved: per_class_f1.png")

    # ── 5. error analysis ─────────────────────────────────────────────────────
    category_by_run: dict[str, dict[str, int]] = {}
    for run in runs:
        print(f"\nRunning error analysis for {run['label']} ...")
        all_errors, cat_counts = _run_error_analysis(run["predictions"])
        category_by_run[run["label"]] = cat_counts
        print(f"  Total errors: {len(all_errors)}  |  Categories: {cat_counts}")

        # Confusion matrix
        gold_seqs = [r["gold_labels"] for r in run["predictions"] if r["gold_labels"]]
        pred_seqs = [r["pred_labels"] for r in run["predictions"] if r["gold_labels"]]
        matrix = compute_entity_confusion_matrix(gold_seqs, pred_seqs, ENTITY_TYPES)

        run_slug = run["name"]
        df_matrix = pd.DataFrame(
            {row_key: {col_key: matrix[row_key][col_key] for col_key in ENTITY_TYPES + ["O"]} for row_key in ENTITY_TYPES}
        ).T
        df_matrix.to_csv(out_dir / f"confusion_matrix_{run_slug}.csv")
        _plot_confusion_matrix(matrix, run["label"], out_dir / f"confusion_matrix_{run_slug}.png")
        print(f"  Saved confusion matrix for {run['label']}")

        # Sample 100 errors
        sampled = sample_errors(all_errors, n=100)
        with open(out_dir / f"error_samples_{run_slug}.json", "w", encoding="utf-8") as fh:
            json.dump(sampled, fh, ensure_ascii=False, indent=2)
        print(f"  Saved {len(sampled)} error samples")

    # Error category summary CSV
    cat_rows = []
    for run_label, cats in category_by_run.items():
        for cat, count in cats.items():
            cat_rows.append({"Run": run_label, "ErrorType": cat, "Count": count})
    df_cats = pd.DataFrame(cat_rows)
    df_cats.to_csv(out_dir / "error_categories.csv", index=False)

    # Pivot for display
    df_cats_pivot = df_cats.pivot(index="Run", columns="ErrorType", values="Count").fillna(0).astype(int)
    print("\n── Error Categories ──")
    print(df_cats_pivot.to_string())

    _plot_error_categories(category_by_run, out_dir / "error_categories.png")
    print("Saved: error_categories.png")

    print(f"\nAll analysis artifacts saved to {out_dir}/")


if __name__ == "__main__":
    main()
