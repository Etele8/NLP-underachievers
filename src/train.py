from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from .data import (
    TokenizedNERDataset,
    build_label_maps,
    build_lid_maps,
    create_data_collator,
    load_split_csv,
    save_label_maps,
    save_lid_maps,
    summarize_entity_language_bias,
    summarize_examples,
)
from .evaluate import compute_metrics, save_predictions_table
from .model_factory import create_token_classifier, load_tokenizer
from .utils import (
    count_trainable_parameters,
    ensure_dir,
    get_device,
    load_checkpoint,
    load_yaml_config,
    save_checkpoint,
    save_config_copy,
    save_json,
    set_seed,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def _log_dataset_stats(split_name: str, examples: list[dict[str, Any]]) -> None:
    stats = summarize_examples(examples)
    LOGGER.info(
        "%s stats: examples=%s labeled=%s mean_tokens=%.2f max_tokens=%s labels=%s entities=%s",
        split_name,
        stats["num_examples"],
        stats["num_labeled_examples"],
        stats["mean_tokens_per_example"],
        stats["max_tokens_per_example"],
        stats["label_distribution"],
        stats["entity_type_counts"],
    )


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _extract_word_level_predictions(
    logits: torch.Tensor,
    batch: dict[str, Any],
    id2label: dict[int, str],
) -> list[dict[str, Any]]:
    pred_ids = torch.argmax(logits.detach().cpu(), dim=-1).tolist()
    rows: list[dict[str, Any]] = []

    for example_index, row_pred_ids in enumerate(pred_ids):
        word_ids = batch["word_ids"][example_index]
        tokens = batch["tokens"][example_index]
        gold_labels = batch["gold_labels"][example_index]
        lid_tags = batch["lid_tags"][example_index]
        has_labels = batch["has_labels"][example_index]
        seen_word_ids: list[int] = []
        seen_word_id_set: set[int] = set()
        row_predictions: list[str] = []

        for pred_id, word_id in zip(row_pred_ids, word_ids):
            if word_id is None or word_id in seen_word_id_set or word_id >= len(tokens):
                continue
            row_predictions.append(id2label[int(pred_id)])
            seen_word_ids.append(word_id)
            seen_word_id_set.add(word_id)

        observed_tokens = [tokens[word_id] for word_id in seen_word_ids]
        observed_golds = [gold_labels[word_id] for word_id in seen_word_ids] if has_labels else []
        observed_lids = [lid_tags[word_id] for word_id in seen_word_ids]
        rows.append(
            {
                "id": batch["example_ids"][example_index],
                "split": batch["splits"][example_index],
                "tokens": observed_tokens,
                "lid_tags": observed_lids,
                "pred_labels": row_predictions,
                "gold_labels": observed_golds,
                "truncated": len(observed_tokens) < len(tokens),
                "original_token_count": len(tokens),
            }
        )

    return rows


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    id2label: dict[int, str],
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    losses: list[float] = []
    prediction_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in dataloader:
            device_batch = _move_batch_to_device(batch, device)
            has_supervision = bool((device_batch["labels"] != -100).any().item())
            if has_supervision:
                outputs = model(
                    input_ids=device_batch["input_ids"],
                    attention_mask=device_batch["attention_mask"],
                    lid_ids=device_batch["lid_ids"],
                    labels=device_batch["labels"],
                )
                losses.append(float(outputs.loss.detach().cpu().item()))
            else:
                outputs = model(
                    input_ids=device_batch["input_ids"],
                    attention_mask=device_batch["attention_mask"],
                    lid_ids=device_batch["lid_ids"],
                )

            prediction_rows.extend(_extract_word_level_predictions(outputs.logits, batch, id2label))

    usable_predictions = [row["pred_labels"] for row in prediction_rows if row["gold_labels"]]
    usable_golds = [row["gold_labels"] for row in prediction_rows if row["gold_labels"]]

    metrics = (
        compute_metrics(usable_predictions, usable_golds)
        if usable_golds
        else {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "classification_report": "No gold labels available for this split.",
        }
    )
    average_loss = sum(losses) / len(losses) if losses else 0.0
    return average_loss, metrics, prediction_rows


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    gradient_clip_norm: float | None,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    running_loss = 0.0

    progress = tqdm(dataloader, desc=f"train {epoch}/{total_epochs}", leave=False)
    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        device_batch = _move_batch_to_device(batch, device)
        outputs = model(
            input_ids=device_batch["input_ids"],
            attention_mask=device_batch["attention_mask"],
            lid_ids=device_batch["lid_ids"],
            labels=device_batch["labels"],
        )
        loss = outputs.loss
        loss.backward()

        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += float(loss.detach().cpu().item())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(1, len(dataloader))


def _write_metrics_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _prepare_run_dir(config: dict[str, Any]) -> Path:
    if "run_dir" in config:
        return ensure_dir(config["run_dir"])

    base_dir = ensure_dir(config["output_dir"])
    run_name = config.get("run_name", "real_ner_run")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(base_dir / f"{run_name}_{timestamp}")


def train(config: dict[str, Any], smoke_test: bool = False) -> dict[str, Any]:
    set_seed(int(config["seed"]))

    LOGGER.info("Loading dataset splits")
    train_examples = load_split_csv(config["train_path"], "train")
    validation_examples = load_split_csv(config["validation_path"], "validation")
    test_examples = load_split_csv(config["test_path"], "test")

    if smoke_test:
        LOGGER.info("Smoke test enabled: limiting train/validation to first 32 examples and one epoch")
        train_examples = train_examples[:32]
        validation_examples = validation_examples[:32]
        config = dict(config)
        config["epochs"] = 1

    output_dir = _prepare_run_dir(config)
    save_config_copy(config, output_dir)

    _log_dataset_stats("train", train_examples)
    _log_dataset_stats("validation", validation_examples)
    _log_dataset_stats("test", test_examples)

    label2id, id2label = build_label_maps(train_examples, validation_examples, test_examples)
    lid2id, id2lid = build_lid_maps(train_examples, validation_examples, test_examples)
    LOGGER.info("Label vocabulary: %s", label2id)
    LOGGER.info("Number of labels: %s", len(label2id))
    LOGGER.info("Language ID vocabulary: %s", lid2id)

    save_label_maps(label2id, id2label, output_dir)
    save_lid_maps(lid2id, id2lid, output_dir)
    save_json(summarize_entity_language_bias(train_examples), output_dir / "train_entity_language_bias.json")
    save_json(summarize_entity_language_bias(validation_examples), output_dir / "validation_entity_language_bias.json")

    tokenizer, resolved_tokenizer_name = load_tokenizer(config["model_name"])
    LOGGER.info("Loaded tokenizer: requested=%s resolved=%s", config["model_name"], resolved_tokenizer_name)

    train_dataset = TokenizedNERDataset(
        train_examples,
        tokenizer,
        label2id,
        lid2id,
        max_length=int(config["max_length"]),
        label_all_tokens=bool(config["label_all_tokens"]),
    )
    validation_dataset = TokenizedNERDataset(
        validation_examples,
        tokenizer,
        label2id,
        lid2id,
        max_length=int(config["max_length"]),
        label_all_tokens=bool(config["label_all_tokens"]),
    )

    collator = create_data_collator(tokenizer)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=collator,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["eval_batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=collator,
        pin_memory=pin_memory,
    )

    model, resolved_model_name = create_token_classifier(
        config["model_name"],
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        lid2id=lid2id,
        dropout=config.get("dropout"),
        use_language_bias=bool(config.get("use_language_bias", False)),
        use_lid_feature=bool(config.get("use_lid_feature", False)),
        language_embedding_dim=int(config.get("language_embedding_dim", 32)),
        language_gate_hidden_dim=int(config.get("language_gate_hidden_dim", 128)),
    )
    LOGGER.info("Loaded model: requested=%s resolved=%s", config["model_name"], resolved_model_name)
    LOGGER.info("Trainable parameters: %s", count_trainable_parameters(model))

    device = get_device()
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    total_steps = len(train_loader) * int(config["epochs"])
    warmup_steps = int(total_steps * float(config["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    metrics_path = output_dir / "metrics.jsonl"
    best_metric_name = str(config.get("save_best_metric", "f1"))
    if best_metric_name != "f1":
        raise ValueError(f"Unsupported save_best_metric={best_metric_name!r}; only 'f1' is supported.")

    best_metric = float("-inf")
    epochs = int(config["epochs"])
    latest_summary: dict[str, Any] | None = None
    start_epoch = 1
    resume_checkpoint = output_dir / "last.pt"
    if bool(config.get("resume_from_checkpoint", True)) and resume_checkpoint.exists():
        checkpoint = load_checkpoint(resume_checkpoint, model, optimizer, scheduler)
        _move_optimizer_state_to_device(optimizer, device)
        completed_epoch = int(checkpoint["epoch"])
        best_metric = float(checkpoint["best_metric"])
        start_epoch = completed_epoch + 1
        LOGGER.info("Resuming from %s after epoch %s", resume_checkpoint, completed_epoch)

    if start_epoch > epochs:
        LOGGER.info("Checkpoint already completed %s/%s epochs; nothing to train", epochs, epochs)
        run_summary_path = output_dir / "run_summary.json"
        if run_summary_path.exists():
            return json.loads(run_summary_path.read_text(encoding="utf-8"))
        return {
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric,
            "last_epoch_metrics": {},
            "resolved_model_name": resolved_model_name,
            "resolved_tokenizer_name": resolved_tokenizer_name,
            "output_dir": str(output_dir),
            "best_checkpoint": str(output_dir / "best.pt"),
            "last_checkpoint": str(output_dir / "last.pt"),
        }

    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            gradient_clip_norm=config.get("gradient_clip_norm"),
            epoch=epoch,
            total_epochs=epochs,
        )

        val_loss, val_metrics, prediction_rows = evaluate(
            model,
            validation_loader,
            device,
            id2label,
        )
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_accuracy": val_metrics["accuracy"],
            "learning_rate": current_lr,
        }
        _write_metrics_jsonl(metrics_path, epoch_metrics)
        LOGGER.info("Epoch metrics: %s", epoch_metrics)
        LOGGER.info("Validation classification report:\n%s", val_metrics["classification_report"])

        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            best_metric=max(best_metric, val_metrics["f1"]) if best_metric != float("-inf") else val_metrics["f1"],
            config=config,
            label2id=label2id,
            id2label=id2label,
            lid2id=lid2id,
            id2lid=id2lid,
        )
        save_predictions_table(
            prediction_rows,
            [row["gold_labels"] for row in prediction_rows],
            [row["pred_labels"] for row in prediction_rows],
            output_dir / "validation_predictions_last.jsonl",
        )
        save_predictions_table(
            prediction_rows,
            [row["gold_labels"] for row in prediction_rows],
            [row["pred_labels"] for row in prediction_rows],
            output_dir / "validation_predictions_last.csv",
        )

        if val_metrics["f1"] > best_metric:
            best_metric = val_metrics["f1"]
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch=epoch,
                best_metric=best_metric,
                config=config,
                label2id=label2id,
                id2label=id2label,
                lid2id=lid2id,
                id2lid=id2lid,
            )
            save_predictions_table(
                prediction_rows,
                [row["gold_labels"] for row in prediction_rows],
                [row["pred_labels"] for row in prediction_rows],
                output_dir / "validation_predictions_best.jsonl",
            )
            save_predictions_table(
                prediction_rows,
                [row["gold_labels"] for row in prediction_rows],
                [row["pred_labels"] for row in prediction_rows],
                output_dir / "validation_predictions_best.csv",
            )
            save_json(val_metrics, output_dir / "best_validation_metrics.json")

        latest_summary = epoch_metrics

    if latest_summary is not None:
        run_summary = {
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric,
            "last_epoch_metrics": latest_summary,
            "resolved_model_name": resolved_model_name,
            "resolved_tokenizer_name": resolved_tokenizer_name,
            "output_dir": str(output_dir),
            "best_checkpoint": str(output_dir / "best.pt"),
            "last_checkpoint": str(output_dir / "last.pt"),
        }
        save_json(run_summary, output_dir / "run_summary.json")
    else:
        run_summary = {
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric,
            "last_epoch_metrics": {},
            "resolved_model_name": resolved_model_name,
            "resolved_tokenizer_name": resolved_tokenizer_name,
            "output_dir": str(output_dir),
            "best_checkpoint": "",
            "last_checkpoint": "",
        }

    LOGGER.info("Training finished. Output directory: %s", output_dir)
    return run_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train real-data token classification model.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--smoke-test", action="store_true", help="Run on a tiny subset for one epoch.")
    args = parser.parse_args()

    train(load_yaml_config(args.config), smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
