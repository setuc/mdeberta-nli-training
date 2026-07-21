#!/usr/bin/env python
"""Entrypoint B: custom PyTorch training loop with HuggingFace ``accelerate``.

This is the *educational* counterpart to ``scripts/train_trainer.py``. Where the
Trainer entrypoint hides the training loop behind a single ``trainer.train()``
call, this file spells the loop out explicitly: we build the optimizer and
scheduler ourselves, iterate over batches, run forward/backward/step by hand,
and drive evaluation + TensorBoard logging + checkpoint saving manually.

``accelerate`` is used only for the boring-but-important plumbing: it picks the
right mixed-precision dtype, moves model/optimizer/dataloaders onto the device,
scales the backward pass, and gathers predictions across processes at eval time.
The design stays single-process here but is distributed-ready (a config flag away).

Usage mirrors the Trainer entrypoint::

    uv run scripts/train_accelerate.py --config configs/smoke.yaml --smoke-test
    uv run scripts/train_accelerate.py --config configs/full.yaml --max-steps 50
"""

from __future__ import annotations

import argparse
import math
import os

# Let unsupported MPS ops silently fall back to CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import DataCollatorWithPadding, get_scheduler

from mdeberta_nli.config import TrainConfig, load_config
from mdeberta_nli.data import prepare_datasets
from mdeberta_nli.device import get_device, get_precision
from mdeberta_nli.metrics import compute_metrics, per_language_accuracy
from mdeberta_nli.model import load_model, load_tokenizer


# ---------------------------------------------------------------------------
# CLI / config handling — mirrors scripts/train_trainer.py exactly.
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse CLI args.

    Every TrainConfig-backed flag defaults to ``None`` so that
    ``load_config`` can apply the precedence rule:
        dataclass defaults < YAML file < non-None CLI overrides.
    Only flags the user actually passes end up overriding the YAML.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune mDeBERTa-v3 on multilingual NLI (custom accelerate loop)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file. CLI flags override its values.",
    )

    # --- model / data selection ---
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--use-xnli", dest="use_xnli", action="store_true", default=None)
    parser.add_argument("--no-xnli", dest="use_xnli", action="store_false", default=None)
    parser.add_argument("--use-26lang", dest="use_26lang", action="store_true", default=None)
    parser.add_argument("--no-26lang", dest="use_26lang", action="store_false", default=None)
    parser.add_argument("--max-length", type=int, default=None)

    # --- optimization hyperparameters ---
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)

    # --- bookkeeping ---
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)

    # --- data-shaping / verification ---
    parser.add_argument("--smoke-test", dest="smoke_test", action="store_true", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    """Turn parsed CLI args into a validated ``TrainConfig``.

    We forward every non-``--config`` flag as a keyword override. ``load_config``
    ignores ``None`` values, so unset flags never clobber the YAML/defaults.
    """
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    return load_config(args.config, **overrides)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    accelerator: Accelerator,
    model,
    eval_dataloader: DataLoader,
    eval_languages: list[str],
) -> tuple[dict, dict]:
    """Run a full pass over the eval set and return (overall, per-language) metrics.

    ``eval_languages`` is the list of language codes aligned by *dataset row
    order*. Because the eval DataLoader uses ``shuffle=False``, the i-th example
    the model sees is ``eval_languages[i]`` — so we can reconstruct the language
    of each gathered prediction by its position in the (unshuffled) stream.

    We use ``accelerator.gather_for_metrics`` so this stays correct under
    distributed / multi-process setups: it gathers tensors across processes and
    automatically drops the duplicate samples the DataLoader pads the last batch
    with. To keep the language alignment intact we also gather a per-example
    *index* tensor and use those indices to slice ``eval_languages``.
    """
    model.eval()

    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []

    # A monotonically increasing global index for every example, so that after
    # gathering (which may reorder/pad across processes) we can recover which
    # language each prediction belongs to.
    seen = 0
    for batch in eval_dataloader:
        labels = batch["labels"]
        # Per-example index for this batch, on the same device for gathering.
        batch_size = labels.shape[0]
        indices = torch.arange(seen, seen + batch_size, device=labels.device)
        seen += batch_size

        outputs = model(**batch)
        logits = outputs.logits

        # gather_for_metrics de-duplicates the tail-padding of the last batch.
        logits, labels, indices = accelerator.gather_for_metrics((logits, labels, indices))

        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        all_indices.append(indices.detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    indices = np.concatenate(all_indices, axis=0)

    # Map each gathered prediction back to its language via its original index.
    languages = [eval_languages[i] for i in indices]

    overall = compute_metrics((logits, labels))
    per_lang = per_language_accuracy(logits, labels, languages)

    model.train()
    return overall, per_lang


def log_eval(writer: SummaryWriter, step: int, overall: dict, per_lang: dict) -> None:
    """Write eval metrics to TensorBoard and print a readable summary."""
    accuracy = overall["accuracy"]
    writer.add_scalar("eval/accuracy", accuracy, step)
    for lang, acc in sorted(per_lang.items()):
        writer.add_scalar(f"eval/accuracy_{lang}", acc, step)

    print(f"\n[eval @ step {step}] overall accuracy = {accuracy:.4f}")
    per_lang_str = "  ".join(f"{lang}={acc:.3f}" for lang, acc in sorted(per_lang.items()))
    print(f"[eval @ step {step}] per-language: {per_lang_str}\n")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    config = build_config(args)

    # --- device & precision --------------------------------------------------
    # get_device() picks cuda > mps > cpu; get_precision() turns that into the
    # bf16/fp16 flags accelerate understands. We collapse those into the single
    # string ("bf16" | "fp16" | "no") that Accelerator's mixed_precision wants.
    device = get_device()
    precision = get_precision(device)
    if precision.get("bf16"):
        mixed_precision = "bf16"
    elif precision.get("fp16"):
        mixed_precision = "fp16"
    else:
        mixed_precision = "no"

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    accelerator.print(
        f"Device: {device} | mixed_precision: {mixed_precision} | "
        f"grad_accum: {config.gradient_accumulation_steps}"
    )

    # Make the whole run reproducible (seeds python/numpy/torch across processes).
    set_seed(config.seed)

    # --- tokenizer / model / data -------------------------------------------
    tokenizer = load_tokenizer(config)
    model = load_model(config)
    train_dataset, eval_dataset = prepare_datasets(config, tokenizer)

    # The eval dataset carries a string "language" column that the model can't
    # consume. Pull it out (aligned by row index, since eval is never shuffled)
    # before handing a language-free dataset to the collator.
    eval_languages = list(eval_dataset["language"])
    eval_dataset = eval_dataset.remove_columns("language")

    # --- dataloaders ---------------------------------------------------------
    # DataCollatorWithPadding pads each batch to its own longest sequence
    # (dynamic padding) rather than a fixed max_length — faster and less memory.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.per_device_train_batch_size,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=False,  # MUST stay False so eval_languages stays index-aligned.
        batch_size=config.per_device_eval_batch_size,
        collate_fn=data_collator,
    )

    # --- optimizer -----------------------------------------------------------
    # Standard "no weight decay on bias / LayerNorm" grouping.
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    # --- work out the training-step budget -----------------------------------
    # One "optimizer step" happens every gradient_accumulation_steps micro-batches.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps
    )
    if config.max_steps and config.max_steps > 0:
        max_steps = config.max_steps
        # Enough epochs to cover the requested steps (last one likely partial).
        num_train_epochs = math.ceil(max_steps / num_update_steps_per_epoch)
    else:
        num_train_epochs = math.ceil(config.num_train_epochs)
        max_steps = int(config.num_train_epochs * num_update_steps_per_epoch)

    # --- scheduler -----------------------------------------------------------
    num_warmup_steps = int(config.warmup_ratio * max_steps)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_steps,
    )

    # --- hand everything to accelerate --------------------------------------
    # After prepare(), model/optimizer/dataloaders are device-placed and
    # (in distributed setups) wrapped appropriately.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # TensorBoard writer lives under output_dir/runs. Only the main process
    # writes, to avoid clobbering under multi-process runs.
    writer = None
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(config.output_dir, "runs"))

    accelerator.print(
        f"Training: {max_steps} optimizer steps "
        f"(~{num_train_epochs} epochs, {num_warmup_steps} warmup steps)\n"
        f"Train examples: {len(train_dataset)} | Eval examples: {len(eval_dataset)}"
    )

    # --- the training loop ---------------------------------------------------
    global_step = 0          # counts optimizer steps (post-accumulation)
    running_loss = 0.0       # accumulates loss for logging_steps averaging
    running_count = 0
    model.train()

    progress_done = False
    for epoch in range(num_train_epochs):
        for batch in train_dataloader:
            # accumulate() handles gradient accumulation bookkeeping: it only
            # syncs grads / lets us step on the boundary micro-batch.
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Track the *un-scaled* loss value for logging.
            running_loss += loss.detach().float().item()
            running_count += 1

            # An optimizer step actually landed on this micro-batch boundary.
            if accelerator.sync_gradients:
                global_step += 1

                # ---- periodic train-loss logging ----
                if writer is not None and global_step % config.logging_steps == 0:
                    avg_loss = running_loss / max(running_count, 1)
                    writer.add_scalar("train/loss", avg_loss, global_step)
                    writer.add_scalar(
                        "train/learning_rate", lr_scheduler.get_last_lr()[0], global_step
                    )
                    accelerator.print(
                        f"step {global_step}/{max_steps} | loss {avg_loss:.4f}"
                    )
                    running_loss = 0.0
                    running_count = 0

                # ---- periodic evaluation ----
                if global_step % config.eval_steps == 0:
                    overall, per_lang = evaluate(
                        accelerator, model, eval_dataloader, eval_languages
                    )
                    if writer is not None:
                        log_eval(writer, global_step, overall, per_lang)

                # ---- stop once the step budget is exhausted ----
                if global_step >= max_steps:
                    progress_done = True
                    break

        if progress_done:
            break

    # --- final evaluation ----------------------------------------------------
    overall, per_lang = evaluate(accelerator, model, eval_dataloader, eval_languages)
    if writer is not None:
        log_eval(writer, global_step, overall, per_lang)

    if writer is not None:
        writer.flush()
        writer.close()

    # --- save ---------------------------------------------------------------
    # Barrier so every process has finished before we touch disk, then unwrap
    # the (possibly DDP-wrapped) model back to the plain HF model to save it.
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        config.output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(config.output_dir)
        accelerator.print(f"Saved model + tokenizer to {config.output_dir}")


if __name__ == "__main__":
    main()
