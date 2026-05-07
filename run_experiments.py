from __future__ import annotations

import importlib
import inspect
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from experiments.configs import experiments
from models.factory import get_model
from utils.checkpoints import get_latest_checkpoint, load_checkpoint, save_checkpoint
from utils.logger import log_error, log_metrics, save_config


BASELINE_CONFIG = {
    "model": "mbert",
    "lr": 3e-5,
    "batch_size": 8,
    "epochs": 5,
    "weight_decay": 0.01,
    "max_seq_length": 128,
}

DEFAULT_NUM_LABELS = 2
OUTPUT_ROOT = Path("outputs")
TRAIN_MODULE_CANDIDATES = ("train", "trainer", "training", "src.train")
MODEL_OUTPUT_DIRS = {
    "mbert": "mber",
    "xlmr": "xlmr",
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


def _find_train_function() -> Callable[..., Any]:
    """Find an existing train(...) function without importing dataset code directly."""
    import_errors: list[str] = []
    for module_name in TRAIN_MODULE_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            import_errors.append(f"{module_name}: {exc}")
            continue

        train_fn = getattr(module, "train", None)
        if callable(train_fn):
            return train_fn

    details = "\n".join(import_errors) if import_errors else "No import errors were raised."
    raise RuntimeError(
        "Could not find a callable train function. Expected train(...) in one of: "
        f"{', '.join(TRAIN_MODULE_CANDIDATES)}.\nImport details:\n{details}"
    )


def _call_training_flexibly(
    train_fn: Callable[..., Any],
    model,
    config: dict[str, Any],
    output_dir: Path,
) -> Any:
    """Try common training signatures used in collaborative ML repos."""
    attempts = (
        (
            "train(model=model, config=config, output_dir=...)",
            (),
            {"model": model, "config": config, "output_dir": str(output_dir)},
        ),
        ("train(model, config)", (model, config), {}),
        ("train(config)", (config,), {}),
    )
    errors: list[str] = []

    for description, args, kwargs in attempts:
        try:
            # Prefer signature binding so incompatible call shapes are skipped
            # before any training side effects begin.
            inspect.signature(train_fn).bind(*args, **kwargs)
            return train_fn(*args, **kwargs)
        except TypeError as exc:
            errors.append(f"{description}: {exc}")

    raise RuntimeError(
        "Found train function, but none of the supported call styles worked. "
        f"Errors: {json.dumps(errors, indent=2)}"
    )


def _infer_num_labels(config: dict[str, Any]) -> int:
    """Infer label count when config provides it; otherwise use a safe placeholder."""
    if "num_labels" in config:
        return int(config["num_labels"])
    label2id = config.get("label2id")
    if isinstance(label2id, dict):
        return len(label2id)
    return DEFAULT_NUM_LABELS


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


def _unique_checkpoint_name() -> str:
    """Create a checkpoint filename that will not overwrite older savepoints."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"checkpoint_{timestamp}.pt"


def _prepare_experiment_dir(output_dir: Path) -> Path:
    """Create the required output layout safely."""
    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def _build_training_config(
    config: dict[str, Any],
    output_dir: Path,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    """Add non-invasive runtime hints commonly expected by training code."""
    training_config = dict(config)
    training_config.setdefault("learning_rate", config.get("lr"))
    training_config.setdefault("max_length", config.get("max_seq_length"))
    training_config.setdefault("output_dir", str(output_dir))
    training_config.setdefault("checkpoint_dir", str(checkpoint_dir))
    return training_config


def _load_existing_checkpoint_if_present(model, checkpoint_dir: Path):
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint is None:
        return model, {}

    print(f"Resuming from checkpoint ... {latest_checkpoint}")
    model, _, _, metadata = load_checkpoint(latest_checkpoint, model)
    return model, metadata


def _save_completion_checkpoint(model, checkpoint_dir: Path) -> Path:
    """Save an end-of-experiment checkpoint without overwriting older checkpoints."""
    return save_checkpoint(
        model,
        output_dir=checkpoint_dir,
        filename=_unique_checkpoint_name(),
    )


def run_single_experiment(
    config: dict[str, Any],
    baseline_config: dict[str, Any],
) -> None:
    output_dir = build_output_dir(config, baseline_config)
    checkpoint_dir = _prepare_experiment_dir(output_dir)

    print("\n=== Running experiment ===")
    print(f"Output directory: {output_dir}")
    print(f"Config: {json.dumps(config, sort_keys=True)}")

    training_config = _build_training_config(config, output_dir, checkpoint_dir)

    save_config(training_config, output_dir)
    log_metrics({"event": "started", "output_dir": str(output_dir)}, output_dir)

    train_fn = _find_train_function()
    model = get_model(training_config, num_labels=_infer_num_labels(training_config))
    model, checkpoint_metadata = _load_existing_checkpoint_if_present(model, checkpoint_dir)
    if checkpoint_metadata:
        log_metrics(
            {
                "event": "resumed",
                "output_dir": str(output_dir),
                "checkpoint_metadata": checkpoint_metadata,
            },
            output_dir,
        )

    result = _call_training_flexibly(train_fn, model, training_config, output_dir)
    checkpoint_path = _save_completion_checkpoint(model, checkpoint_dir)

    metrics = {
        "status": "success",
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
        "result": result if isinstance(result, dict) else {},
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
