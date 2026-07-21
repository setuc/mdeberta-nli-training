"""Model and tokenizer loading for mDeBERTa-v3 multilingual NLI.

Thin wrappers around the HF ``Auto*`` classes. The NLI head is a 3-way classifier
using the standard label convention shared across XNLI and the 26lang corpus:
``0=entailment, 1=neutral, 2=contradiction``.
"""

from __future__ import annotations

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
    """Load ``AutoModelForSequenceClassification`` with a 3-way NLI head."""
    return AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
