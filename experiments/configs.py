from __future__ import annotations

from typing import Iterable


LEARNING_RATES = [1e-5, 3e-5, 5e-5]
EPOCHS = [3, 5, 10]
BATCH_SIZES = [4, 8, 16]
WEIGHT_DECAYS = [0.0, 0.01, 0.1]
MAX_SEQ_LENGTHS = [64, 128, 256]
MODELS = ["mbert", "xlmr"]

BASELINE_HYPERPARAMETERS = {
    "lr": 3e-5,
    "epochs": 5,
    "batch_size": 8,
    "weight_decay": 0.01,
    "max_seq_length": 128,
    "language_bias": False,
    "use_lid_feature": False,
}

SWEEP_VALUES = {
    "lr": LEARNING_RATES,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZES,
    "weight_decay": WEIGHT_DECAYS,
    "max_seq_length": MAX_SEQ_LENGTHS,
}


def generate_lr_sweep(models: Iterable[str], lrs: Iterable[float]) -> list[dict]:
    """Generate model/lr experiment combinations."""
    return [{"model": model, "lr": lr} for model in models for lr in lrs]


def _format_float(value: float) -> str:
    """Create compact, stable float strings for experiment names."""
    formatted = f"{value:g}"
    return formatted.replace("e-0", "e-").replace("e+0", "e+")


def generate_experiment_name(config: dict) -> str:
    """Generate a readable name from the experiment hyperparameters."""
    name = (
        f"{config['model']}"
        f"_lr{_format_float(float(config['lr']))}"
        f"_ep{config['epochs']}"
        f"_bs{config['batch_size']}"
        f"_wd{_format_float(float(config['weight_decay']))}"
        f"_seq{config['max_seq_length']}"
    )
    if bool(config.get("lora", False)):
        name += "_lora"
    if bool(config.get("language_bias", False)):
        name += "_langbias"
    if bool(config.get("use_lid_feature", False)):
        name += "_lidfeat"
    return name


def _baseline_for_model(model: str) -> dict:
    """Return the baseline hyperparameters for one model family."""
    return {"model": model, **BASELINE_HYPERPARAMETERS}


def _single_parameter_variants(baseline: dict) -> list[dict]:
    """Generate configs that differ from baseline by exactly one hyperparameter."""
    variants = []
    for parameter, values in SWEEP_VALUES.items():
        baseline_value = baseline[parameter]
        for value in values:
            if value == baseline_value:
                continue

            experiment = dict(baseline)
            experiment[parameter] = value
            variants.append(experiment)
    return variants


def _language_bias_variant(baseline: dict) -> dict:
    experiment = dict(baseline)
    experiment["language_bias"] = True
    experiment["use_lid_feature"] = True
    return experiment


def generate_full_experiment_sweep(
    max_experiments: int | None = None,
) -> list[dict]:
    """Generate multilingual token classification experiments.

    Each model gets its own baseline and one-hyperparameter-at-a-time variants.
    A variant keeps all baseline values fixed except the parameter being swept.
    """
    sweep = []
    for model in MODELS:
        baseline = _baseline_for_model(model)
        sweep.append(baseline)
        sweep.append(_language_bias_variant(baseline))
        sweep.extend(_single_parameter_variants(baseline))

    if max_experiments is None:
        return sweep
    if max_experiments < 1:
        raise ValueError("max_experiments must be None or a positive integer")
    return sweep[:max_experiments]


experiments = generate_full_experiment_sweep()
