from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

from experiments.configs import experiments
from src.train import train as train_model
from src.utils import load_yaml_config, save_config_copy
from utils.logger import log_error, log_metrics, save_config


PROJECT_ROOT = Path(__file__).resolve().parent

BASELINE_CONFIG = {
    "model": "mbert",
    "lr": 3e-5,
    "batch_size": 8,
    "epochs": 5,
    "weight_decay": 0.01,
    "max_seq_length": 128,
}

OUTPUT_ROOT = PROJECT_ROOT / "outputs"
CONFIG_ROOT = PROJECT_ROOT / "configs"
MODEL_OUTPUT_DIRS = {
    "mbert": "mber",
    "xlmr": "xlmr",
}
MODEL_CONFIG_FILES = {
    "mbert": CONFIG_ROOT / "mbert.yaml",
    "xlmr": CONFIG_ROOT / "xlmr.yaml",
}

# These keys define experiment identity. Keeping model and LoRA here prevents
# mBERT/XLM-R or LoRA/full-finetune runs from writing into the same folder.
TRACKED_PARAMETERS = (
    "model",
    "lr",
    "batch_size",
    "epochs",
    "weight_decay",
    "max_seq_length",
    "lora",
)

PARAMETER_GROUPS = {
    "model": "model",
    "lr": "learning_rate",
    "batch_size": "batch_size",
    "epochs": "epochs",
    "weight_decay": "weight_decay",
    "max_seq_length": "max_seq_length",
    "lora": "lora",
}

PARAMETER_PREFIXES = {
    "model": "model",
    "lr": "lr",
    "batch_size": "bs",
    "epochs": "ep",
    "weight_decay": "wd",
    "max_seq_length": "seq",
    "lora": "lora",
}


def _format_value(value: Any) -> str:
    """Format values for stable, readable folder names."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        formatted = f"{value:g}"
        return formatted.replace("e-0", "e-").replace("e+0", "e+")
    return str(value).replace("/", "-").replace(" ", "_")


def _parameter_token(key: str, value: Any, *, compact: bool = False) -> str:
    """Create the path token for one changed parameter."""
    if key == "lora":
        return "lora" if bool(value) else "full"
    if compact and key in {"batch_size", "epochs", "weight_decay", "max_seq_length"}:
        return f"{PARAMETER_PREFIXES[key]}{_format_value(value)}"
    return f"{PARAMETER_PREFIXES[key]}_{_format_value(value)}"


def detect_changed_hyperparameters(
    config: dict[str, Any],
    baseline_config: dict[str, Any],
) -> tuple[list[str], str]:
    """Return changed parameter keys and the folder name for this config.

    Folder names intentionally include only values that differ from the
    baseline. Single-parameter changes are grouped under that parameter; mixed
    changes are grouped under outputs/mixed/.
    """
    changed = [
        key
        for key in TRACKED_PARAMETERS
        if config.get(key, False if key == "lora" else None)
        != baseline_config.get(key, False if key == "lora" else None)
    ]

    if not changed:
        return [], "baseline"

    compact = len(changed) > 1
    folder_name = "_".join(
        _parameter_token(key, config.get(key, True), compact=compact) for key in changed
    )
    return changed, folder_name


def _model_output_dir(config: dict[str, Any]) -> Path:
    model_key = str(config.get("model", "")).lower()
    folder_name = MODEL_OUTPUT_DIRS.get(model_key)
    if folder_name is None:
        supported = ", ".join(sorted(MODEL_OUTPUT_DIRS))
        raise ValueError(f"Unknown model {model_key!r}. Supported models: {supported}")
    return OUTPUT_ROOT / folder_name


def _baseline_for_config_model(
    config: dict[str, Any],
    baseline_config: dict[str, Any],
) -> dict[str, Any]:
    model_baseline = dict(baseline_config)
    model_baseline["model"] = config.get("model", baseline_config.get("model"))
    return model_baseline


def build_output_dir(config: dict[str, Any], baseline_config: dict[str, Any]) -> Path:
    """Map an experiment config to a collision-resistant output directory."""
    output_root = _model_output_dir(config)
    model_baseline = _baseline_for_config_model(config, baseline_config)
    changed, folder_name = detect_changed_hyperparameters(config, model_baseline)
    if not changed:
        return output_root / "baseline" / folder_name
    if len(changed) == 1:
        return output_root / PARAMETER_GROUPS[changed[0]] / folder_name
    return output_root / "mixed" / folder_name


def _json_default(value: Any) -> str:
    """Keep logging resilient when training returns non-JSON objects."""
    return str(value)


def _write_metrics_json(metrics: dict[str, Any], output_dir: Path) -> Path:
    """Write the latest experiment-level metrics snapshot."""
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, default=_json_default, indent=2, sort_keys=True)
        handle.write("\n")
    return metrics_path


def _prepare_experiment_dir(output_dir: Path) -> None:
    """Create the required output layout safely."""
    output_dir.mkdir(parents=True, exist_ok=True)


def _load_model_training_template(model_key: str) -> dict[str, Any]:
    config_path = MODEL_CONFIG_FILES.get(model_key)
    if config_path is None:
        supported = ", ".join(sorted(MODEL_CONFIG_FILES))
        raise ValueError(f"Unknown model {model_key!r}. Supported models: {supported}")
    return load_yaml_config(config_path)


def _resolve_project_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return PROJECT_ROOT / path_obj


def _build_training_config(
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Build the src.train YAML-style config for one sweep item."""
    model_key = str(config["model"]).lower()
    training_config = _load_model_training_template(model_key)
    experiment_name = output_dir.name

    training_config.update(
        {
            "run_name": experiment_name,
            "run_dir": str(output_dir),
            "output_dir": str(output_dir),
            "learning_rate": config["lr"],
            "batch_size": config["batch_size"],
            "epochs": config["epochs"],
            "weight_decay": config["weight_decay"],
            "max_length": config["max_seq_length"],
            "experiment": config,
        }
    )
    for key in ("train_path", "validation_path", "test_path"):
        training_config[key] = str(_resolve_project_path(training_config[key]))
    return training_config


def _required_data_paths() -> list[Path]:
    """Return all dataset files needed by the configured model templates."""
    paths: set[Path] = set()
    for model_key in MODEL_CONFIG_FILES:
        training_config = _load_model_training_template(model_key)
        for key in ("train_path", "validation_path", "test_path"):
            paths.add(_resolve_project_path(training_config[key]))
    return sorted(paths)


def _validate_required_data() -> None:
    missing_paths = [path for path in _required_data_paths() if not path.exists()]
    if not missing_paths:
        return

    missing = "\n".join(f"- {path}" for path in missing_paths)
    raise FileNotFoundError(
        "Missing required dataset files:\n"
        f"{missing}\n"
        "Download them with: .venv/bin/python scripts/download_data.py"
    )


def run_single_experiment(
    config: dict[str, Any],
    baseline_config: dict[str, Any],
) -> None:
    output_dir = build_output_dir(config, baseline_config)
    _prepare_experiment_dir(output_dir)

    print("\n=== Running experiment ===")
    print(f"Output directory: {output_dir}")
    print(f"Config: {json.dumps(config, sort_keys=True)}")

    training_config = _build_training_config(config, output_dir)

    save_config(training_config, output_dir)
    save_config_copy(training_config, output_dir)
    log_metrics({"event": "started", "output_dir": str(output_dir)}, output_dir)

    result = train_model(training_config)
    checkpoint_path = result.get("best_checkpoint") or result.get("last_checkpoint") or ""

    metrics = {
        "status": "success",
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
        "result": result,
    }
    _write_metrics_json(metrics, output_dir)
    log_metrics(
        {
            "event": "completed",
            "status": "success",
            "output_dir": str(output_dir),
            "checkpoint": str(checkpoint_path),
        },
        output_dir,
    )
    print(f"Success: {output_dir}")


def run_all_experiments() -> None:
    _validate_required_data()

    for index, config in enumerate(experiments, start=1):
        output_dir = build_output_dir(config, BASELINE_CONFIG)
        print(f"\n[{index}/{len(experiments)}] Starting: {output_dir}")

        try:
            run_single_experiment(config, BASELINE_CONFIG)
        except Exception:
            output_dir.mkdir(parents=True, exist_ok=True)
            stack_trace = traceback.format_exc()
            log_error(stack_trace, output_dir)
            log_metrics({"event": "failed", "output_dir": str(output_dir)}, output_dir)
            _write_metrics_json(
                {
                    "status": "failed",
                    "output_dir": str(output_dir),
                    "error": stack_trace,
                },
                output_dir,
            )
            print(f"Failure: {output_dir}")
            print(stack_trace)
            print("Continuing to next experiment.")


def main() -> None:
    run_all_experiments()


if __name__ == "__main__":
    main()
