from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from data import build_label_vocab, parse_iob2_file
from modeling import TokenClassificationDataset, align_labels_with_tokens, compute_seqeval_metrics, get_device
from utils import ensure_dir, make_safe_model_name, save_checkpoint, load_checkpoint



logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def evaluate(
    model: AutoModelForTokenClassification,
    dataloader: DataLoader,
    id2label: Dict[int, str],
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            for pred_row, label_row in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                sent_pred = []
                sent_label = []
                for p, l in zip(pred_row, label_row):
                    if l == -100:
                        continue
                    sent_pred.append(id2label[p])
                    sent_label.append(id2label[l])
                all_preds.append(sent_pred)
                all_labels.append(sent_label)

    metrics = compute_seqeval_metrics(all_preds, all_labels)
    return {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "report": metrics["report"],
        "predictions": all_preds,
        "references": all_labels,
    }


def save_dev_predictions(
    sentences: List[Dict[str, Any]],
    predictions: List[List[str]],
    output_file: Path,
) -> None:
    with output_file.open("w", encoding="utf-8") as f:
        for ex, pred in zip(sentences, predictions):
            for idx, (token, label) in enumerate(zip(ex["tokens"], pred), start=1):
                f.write(f"{idx}\t{token}\t{label}\n")
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NER token classification baseline")
    parser.add_argument("--model_name", type=str, required=True, choices=[
        "google-bert/bert-base-multilingual-cased",
        "FacebookAI/xlm-roberta-base",
    ])
    parser.add_argument("--train_file", type=str, default="data-baseline/en_ewt-ud-train.iob2")
    parser.add_argument("--dev_file", type=str, default="data-baseline/en_ewt-ud-dev.iob2")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    train_path = Path(args.train_file)
    dev_path = Path(args.dev_file)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    logging.info("Loading train/dev data")
    train_examples = parse_iob2_file(train_path)
    dev_examples = parse_iob2_file(dev_path)

    logging.info("Building label vocabulary")
    label2id, id2label = build_label_vocab(train_examples, dev_examples)

    logging.info("Loading tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    logging.info("Preparing datasets")
    train_encodings = align_labels_with_tokens(train_examples, tokenizer, label2id, max_length=args.max_length)
    dev_encodings = align_labels_with_tokens(dev_examples, tokenizer, label2id, max_length=args.max_length)

    train_dataset = TokenClassificationDataset(train_encodings)
    dev_dataset = TokenClassificationDataset(dev_encodings)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    device = get_device()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    start_epoch = 1

    if args.resume:
        ckpt_path = None

        if args.resume == "auto" and args.checkpoint_dir:
            candidate = Path(args.checkpoint_dir) / "last.pt"
            if candidate.exists():
                ckpt_path = candidate
        else:
            ckpt_path = Path(args.resume)

        if ckpt_path and ckpt_path.exists():
            logging.info(f"Resuming from {ckpt_path}")
            start_epoch, best_f1 = load_checkpoint(
                ckpt_path, model, optimizer, scheduler
            )
            start_epoch += 1

    best_f1 = 0.0
    best_dir = output_dir / f"best_{make_safe_model_name(args.model_name)}"

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))
        logging.info(f"Epoch {epoch}/{args.epochs} train loss: {avg_loss:.4f}")

        eval_results = evaluate(model, dev_loader, id2label, device)
        logging.info(f"Dev precision: {eval_results['precision']:.4f} recall: {eval_results['recall']:.4f} f1: {eval_results['f1']:.4f}")
        logging.info("Dev classification report:\n%s", eval_results["report"])

        if eval_results["f1"] > best_f1:
            best_f1 = eval_results["f1"]
            ensure_dir(best_dir)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            logging.info(f"Saved best model to {best_dir} (F1={best_f1:.4f})")

        dev_pred_path = output_dir / f"dev_predictions_{make_safe_model_name(args.model_name)}.iob2"
        save_dev_predictions(dev_examples, eval_results["predictions"], dev_pred_path)

        if args.checkpoint_dir:
                ckpt_dir = Path(args.checkpoint_dir)
                ensure_dir(ckpt_dir)

                save_checkpoint(
                    ckpt_dir / "last.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_f1,
                )

    logging.info("Training finished. Best dev F1: %.4f", best_f1)
    logging.info("Best model dir: %s", best_dir)


if __name__ == "__main__":
    main()
