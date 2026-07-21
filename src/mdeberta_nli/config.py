"""Training configuration for mDeBERTa NLI fine-tuning.

Defines the :class:`TrainConfig` dataclass and :func:`load_config`, which
builds a config from dataclass defaults, an optional YAML file, and explicit
keyword overrides. No training logic lives here.
"""

from dataclasses import dataclass, field, fields

import yaml


@dataclass
class TrainConfig:
    model_name: str = "microsoft/mdeberta-v3-base"
    use_xnli: bool = True
    use_26lang: bool = True
    max_length: int = 128
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    learning_rate: float = 2e-5
    num_train_epochs: float = 2.0
    max_steps: int = -1                 # -1 means "use epochs"
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    output_dir: str = "outputs"
    seed: int = 42
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500
    smoke_test: bool = False
    max_train_samples: int | None = None       # cap total training rows (None = all)
    smoke_languages: list[str] = field(default_factory=lambda: ["en", "de", "zh"])
    smoke_rows_per_lang: int = 100


def load_config(config_path: str | None = None, **overrides) -> TrainConfig:
    """Build a :class:`TrainConfig`.

    Precedence: dataclass defaults < YAML file (if given) < non-None overrides.
    Unknown YAML keys raise a ``ValueError``. Returns a :class:`TrainConfig`.
    """
    valid_keys = {f.name for f in fields(TrainConfig)}

    values: dict = {}

    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Config file {config_path!r} must contain a YAML mapping, "
                f"got {type(loaded).__name__}."
            )
        unknown = sorted(set(loaded) - valid_keys)
        if unknown:
            raise ValueError(
                "Unknown config key(s) in "
                f"{config_path!r}: {', '.join(unknown)}. "
                f"Valid keys are: {', '.join(sorted(valid_keys))}."
            )
        values.update(loaded)

    for key, value in overrides.items():
        if value is not None:
            values[key] = value

    return TrainConfig(**values)
