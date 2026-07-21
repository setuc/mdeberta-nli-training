"""Model and tokenizer loading for mDeBERTa-v3 multilingual NLI.

Thin wrappers around the HF ``Auto*`` classes. The NLI head is a 3-way classifier
using the standard label convention shared across XNLI and the 26lang corpus:
``0=entailment, 1=neutral, 2=contradiction``.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from mdeberta_nli.config import TrainConfig

# Canonical label mapping for the NLI task. Kept here so the model's config
# carries human-readable labels into any saved checkpoint / inference pipeline.
ID2LABEL = {0: "entailment", 1: "neutral", 2: "contradiction"}
LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}


def load_tokenizer(config: TrainConfig):
    """Load the fast tokenizer for ``config.model_name``.

    mDeBERTa-v3 uses a SentencePiece tokenizer, so ``sentencepiece`` and
    ``protobuf`` must be installed (they are declared in pyproject).
    """
    return AutoTokenizer.from_pretrained(config.model_name, use_fast=True)


def load_model(config: TrainConfig):
    """Load ``AutoModelForSequenceClassification`` with a 3-way NLI head.

    We force ``dtype=torch.float32`` for the master weights. This matters: the
    mDeBERTa-v3 checkpoint on the Hub is stored in float16, and since
    transformers v5 ``from_pretrained`` loads weights in the checkpoint's native
    dtype rather than upcasting to float32. float16 master weights break AdamW —
    its ``eps=1e-8`` underflows to 0 in float16, so the very first optimizer step
    divides by ~0 and produces NaN across the model (independent of learning
    rate). Mixed precision (fp16/bf16 *compute*) is handled separately by the
    training args / accelerate; the master weights must stay float32.
    """
    return AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        dtype=torch.float32,
    )
