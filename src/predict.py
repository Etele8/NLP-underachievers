from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

from data import parse_iob2_test_file
from modeling import align_labels_with_tokens, get_device
from utils import make_safe_model_name, ensure_dir


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")




def reconstruct_test_output(line_items: List[Dict[str, Any]], pred_labels: List[str], output_file: Path) -> None:
    idx = 0
    with output_file.open("w", encoding="utf-8") as f:
        for item in line_items:
            if item["type"] == "blank":
                f.write("\n")
            elif item["type"] == "comment":
                f.write(item["text"] + "\n")
            elif item["type"] == "token":
                if idx >= len(pred_labels):
                    raise RuntimeError("Prediction count mismatch during reconstruction")

                parts = item["parts"].copy()
                if item["has_label"]:
                    parts[-1] = pred_labels[idx]
                else:
                    parts.append(pred_labels[idx])

                f.write("\t".join(parts) + "\n")
                idx += 1

    if idx != len(pred_labels):
        raise RuntimeError("Not all predictions were consumed during reconstruction")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict test file with NER token classification model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_file", type=str, default="data-baseline/en_ewt-ud-test-masked.iob2")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--id2label_file", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    model = AutoModelForTokenClassification.from_pretrained(str(checkpoint))
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))

    # id2label is stored in model.config
    id2label = {int(k): v for k, v in model.config.id2label.items()} if model.config.id2label else {}
    if not id2label:
        raise ValueError("id2label mapping missing from model config")

    test_path = Path(args.test_file)
    test_examples, line_items = parse_iob2_test_file(test_path)

    device = get_device()
    model.to(device)

    tokenized_test = align_labels_with_tokens(test_examples, tokenizer, label2id={}, max_length=args.max_length)

    input_ids = torch.tensor(tokenized_test["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(tokenized_test["attention_mask"], dtype=torch.long)
    word_ids_list = tokenized_test.get("word_ids", [])

    dataset = torch.utils.data.TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    raw_preds: List[str] = []
    model.eval()
    with torch.no_grad():
        batch_start = 0
        for batch in dataloader:
            batch_input_ids = batch[0].to(device)
            batch_attention_mask = batch[1].to(device)
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()

            for local_idx, pred_row in enumerate(preds):
                global_idx = batch_start + local_idx
                if global_idx >= len(word_ids_list):
                    break
                word_ids = word_ids_list[global_idx]
                prev = None
                for p, word_idx in zip(pred_row, word_ids):
                    if word_idx is None:
                        continue
                    if word_idx != prev:
                        raw_preds.append(id2label[p])
                        prev = word_idx
            batch_start += len(preds)

    if args.model_name is None:
        safe_model_name = make_safe_model_name(checkpoint.name)
    else:
        safe_model_name = make_safe_model_name(args.model_name)

    output_file = output_dir / f"predictions_{safe_model_name}.iob2"
    reconstruct_test_output(line_items, raw_preds, output_file)

    logging.info("Test predictions written to %s", output_file)


if __name__ == "__main__":
    main()
