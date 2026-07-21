#!/usr/bin/env python
"""Entrypoint A: fine-tune mDeBERTa-v3 on multilingual NLI with the HF Trainer API.

Usage:
    uv run scripts/train_trainer.py --config configs/smoke.yaml --smoke-test
    uv run scripts/train_trainer.py --config configs/full.yaml
    uv run scripts/train_trainer.py --model-name microsoft/mdeberta-v3-base --max-steps 2
"""

from __future__ import annotations

import argparse
import os

# Allow unsupported ops to fall back to CPU on Apple Silicon (MPS).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from mdeberta_nli.config import load_config
from mdeberta_nli.data import prepare_datasets
from mdeberta_nli.device import get_device, get_precision
from mdeberta_nli.metrics import compute_metrics, per_language_accuracy
from mdeberta_nli.model import load_model, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train mDeBERTa-v3 on multilingual NLI using the HF Trainer API."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file (optional).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Force smoke-test mode (tiny subset, few steps).",
    )
    # Passthrough overrides. Defaults are None so we only override keys the user set.
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Maps to per_device_train_batch_size.",
    )
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> dict:
    """Collect only the config keys the user explicitly passed on the CLI."""
    overrides: dict = {}
    if args.model_name is not None:
        overrides["model_name"] = args.model_name
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.max_train_samples is not None:
        overrides["max_train_samples"] = args.max_train_samples
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.num_train_epochs is not None:
        overrides["num_train_epochs"] = args.num_train_epochs
    if args.batch_size is not None:
        overrides["per_device_train_batch_size"] = args.batch_size
    return overrides


def main() -> None:
    args = parse_args()

    overrides = build_overrides(args)
    config = load_config(args.config, **overrides)
    if args.smoke_test:
        config.smoke_test = True

    set_seed(config.seed)

    device = get_device()
    prec = get_precision(device)
    print(f"[train_trainer] device={device} precision={prec}")

    tokenizer = load_tokenizer(config)
    model = load_model(config)

    train_ds, eval_ds = prepare_datasets(config, tokenizer)
    print(
        f"[train_trainer] train_rows={len(train_ds)} eval_rows={len(eval_ds)}"
    )

    collator = DataCollatorWithPadding(tokenizer)

    logging_dir = os.path.join(config.output_dir, "runs")

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps if config.max_steps and config.max_steps > 0 else -1,
        seed=config.seed,
        fp16=prec["fp16"],
        bf16=prec["bf16"],
        logging_dir=logging_dir,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # Per-language accuracy on the eval set.
    print("[train_trainer] running per-language accuracy evaluation...")
    prediction_output = trainer.predict(eval_ds)
    preds = prediction_output.predictions
    labels = prediction_output.label_ids
    languages = eval_ds["language"]

    per_lang = per_language_accuracy(preds, labels, languages)
    print("[train_trainer] per-language accuracy:")
    for lang, acc in sorted(per_lang.items()):
        print(f"  {lang}: {acc:.4f}")

    # Log per-language accuracy to TensorBoard.
    trainer.log({f"eval_per_lang_accuracy/{lang}": acc for lang, acc in per_lang.items()})


if __name__ == "__main__":
    main()
