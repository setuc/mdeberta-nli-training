"""Tests for mdeberta_nli.model.

The dtype test is a regression guard: the mDeBERTa checkpoint is stored in
float16, and transformers v5 loads weights in the checkpoint's native dtype.
float16 master weights make AdamW produce NaN on the first optimizer step
(eps=1e-8 underflows in fp16). load_model must force float32.

These tests download the model, so they are skipped when the Hub is
unreachable (offline / CI without network).
"""

import pytest
import torch

from mdeberta_nli.config import TrainConfig
from mdeberta_nli.model import ID2LABEL, LABEL2ID, load_model


def _requires_hub(fn):
    """Skip a test if the model cannot be fetched (no network / no cache)."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - network/hub errors are the skip reason
        pytest.skip(f"model unavailable (offline?): {exc}")


def test_label_maps_are_consistent():
    assert ID2LABEL == {0: "entailment", 1: "neutral", 2: "contradiction"}
    assert LABEL2ID == {"entailment": 0, "neutral": 1, "contradiction": 2}


@pytest.mark.slow
def test_load_model_is_float32_and_three_class():
    """Master weights must be float32 (fp16 breaks AdamW) with a 3-way head."""
    config = TrainConfig()
    model = _requires_hub(lambda: load_model(config))

    dtypes = {p.dtype for p in model.parameters()}
    assert dtypes == {torch.float32}, f"expected all float32 master weights, got {dtypes}"
    assert model.config.num_labels == 3
    assert model.config.id2label[0] == "entailment"
