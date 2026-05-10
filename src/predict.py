from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .data import TokenizedNERDataset, create_data_collator, load_split_csv
from .evaluate import compute_metrics, save_predictions_table
from .model_factory import create_token_classifier, load_tokenizer
from .train import evaluate
from .utils import ensure_dir, get_device, load_checkpoint, load_yaml_config, save_json


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def _select_split(config: dict[str, Any], split: str) -> tuple[str, list[dict[str, Any]]]:
    if split == "train":
        return config["train_path"], load_split_csv(config["train_path"], "train")
    if split == "validation":
        return config["validation_path"], load_split_csv(config["validation_path"], "validation")
    if split == "test":
        return config["test_path"], load_split_csv(config["test_path"], "test")
    raise ValueError(f"Unsupported split={split!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prediction for the real-data NER pipeline.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved checkpoint.")
    parser.add_argument("--split", required=True, choices=["train", "validation", "test"])
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    run_output_dir = ensure_dir(checkpoint_path.parent)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    effective_config = dict(config)
    effective_config.update(checkpoint.get("config", {}))
    _, examples = _select_split(effective_config, args.split)

    tokenizer, resolved_tokenizer_name = load_tokenizer(effective_config["model_name"])
    label2id = checkpoint["label2id"]
    id2label = {int(idx): label for idx, label in checkpoint["id2label"].items()}
    lid2id = checkpoint.get("lid2id") or {"<pad>": 0}

    model, resolved_model_name = create_token_classifier(
        effective_config["model_name"],
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        lid2id=lid2id,
        dropout=effective_config.get("dropout"),
        use_language_bias=bool(effective_config.get("use_language_bias", False)),
        use_lid_feature=bool(effective_config.get("use_lid_feature", False)),
        language_embedding_dim=int(effective_config.get("language_embedding_dim", 32)),
        language_gate_hidden_dim=int(effective_config.get("language_gate_hidden_dim", 128)),
    )
    load_checkpoint(checkpoint_path, model)

    dataset = TokenizedNERDataset(
        examples,
        tokenizer,
        label2id=label2id,
        lid2id=lid2id,
        max_length=int(effective_config["max_length"]),
        label_all_tokens=bool(effective_config["label_all_tokens"]),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(effective_config["eval_batch_size"]),
        shuffle=False,
        num_workers=int(effective_config.get("num_workers", 0)),
        collate_fn=create_data_collator(tokenizer),
        pin_memory=torch.cuda.is_available(),
    )

    device = get_device()
    model.to(device)
    loss, metrics, prediction_rows = evaluate(model, dataloader, device, id2label)

    predictions_jsonl = run_output_dir / f"predictions_{args.split}.jsonl"
    predictions_csv = run_output_dir / f"predictions_{args.split}.csv"
    save_predictions_table(
        prediction_rows,
        [row["gold_labels"] for row in prediction_rows],
        [row["pred_labels"] for row in prediction_rows],
        predictions_jsonl,
    )
    save_predictions_table(
        prediction_rows,
        [row["gold_labels"] for row in prediction_rows],
        [row["pred_labels"] for row in prediction_rows],
        predictions_csv,
    )

    labeled_predictions = [row["pred_labels"] for row in prediction_rows if row["gold_labels"]]
    labeled_golds = [row["gold_labels"] for row in prediction_rows if row["gold_labels"]]
    if labeled_golds:
        metrics = compute_metrics(labeled_predictions, labeled_golds)
        save_json({"loss": loss, **metrics}, run_output_dir / f"metrics_{args.split}.json")
        (run_output_dir / f"classification_report_{args.split}.txt").write_text(
            metrics["classification_report"],
            encoding="utf-8",
        )
        LOGGER.info("Saved prediction metrics for split=%s", args.split)
    else:
        save_json(
            {
                "loss": loss,
                "classification_report": "No gold labels available for this split.",
                "resolved_model_name": resolved_model_name,
                "resolved_tokenizer_name": resolved_tokenizer_name,
            },
            run_output_dir / f"metrics_{args.split}.json",
        )
        LOGGER.info("Split=%s has no usable gold labels; saved predictions without seqeval metrics", args.split)


if __name__ == "__main__":
    main()
