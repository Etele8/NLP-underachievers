from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOG_FIELDS = ["timestamp", "event", "experiment", "metrics_json"]


def _ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_config(config: dict[str, Any], output_dir: str | Path) -> Path:
    """Save the experiment config as JSON without touching existing logs."""
    output_path = _ensure_output_dir(output_dir) / "config.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def log_metrics(metrics_dict: dict[str, Any], output_dir: str | Path) -> Path:
    """Append metrics to logs.csv without rewriting or truncating old rows."""
    output_path = _ensure_output_dir(output_dir) / "logs.csv"
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": metrics_dict.get("event", ""),
        "experiment": metrics_dict.get("experiment", ""),
        "metrics_json": json.dumps(metrics_dict, sort_keys=True),
    }

    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return output_path


def log_error(error_message: str, output_dir: str | Path) -> Path:
    """Append an error message or stack trace to error.log."""
    output_path = _ensure_output_dir(output_dir) / "error.log"
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(error_message.rstrip())
        handle.write("\n")
    return output_path
