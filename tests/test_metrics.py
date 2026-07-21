"""Tests for mdeberta_nli.metrics.

Hand-built logits/labels with known accuracy. No model, no training.
Label convention: 0=entailment, 1=neutral, 2=contradiction.
"""

import numpy as np
import pytest

from mdeberta_nli.metrics import compute_metrics, per_language_accuracy


def _one_hot_logits(class_ids):
    """Build 2D logits whose argmax equals each given class id (3 classes)."""
    logits = np.full((len(class_ids), 3), -1.0, dtype=np.float32)
    for i, c in enumerate(class_ids):
        logits[i, c] = 5.0
    return logits


# --- compute_metrics -----------------------------------------------------


def test_compute_metrics_all_correct():
    labels = np.array([0, 1, 2, 0])
    logits = _one_hot_logits([0, 1, 2, 0])
    out = compute_metrics((logits, labels))
    assert out["accuracy"] == pytest.approx(1.0)


def test_compute_metrics_partial():
    # 2 of 4 predictions correct -> 0.5
    labels = np.array([0, 1, 2, 0])
    logits = _one_hot_logits([0, 1, 0, 2])  # idx 0,1 correct; 2,3 wrong
    out = compute_metrics((logits, labels))
    assert out["accuracy"] == pytest.approx(0.5)


def test_compute_metrics_none_correct():
    labels = np.array([0, 0, 0])
    logits = _one_hot_logits([1, 2, 1])
    out = compute_metrics((logits, labels))
    assert out["accuracy"] == pytest.approx(0.0)


# --- per_language_accuracy -----------------------------------------------


def test_per_language_accuracy_groups_with_2d_logits():
    # en: both correct -> 1.0 ; de: one of two -> 0.5
    languages = ["en", "en", "de", "de"]
    labels = np.array([0, 1, 2, 0])
    logits = _one_hot_logits([0, 1, 2, 1])  # de second wrong
    out = per_language_accuracy(logits, labels, languages)
    assert set(out) == {"en", "de"}
    assert out["en"] == pytest.approx(1.0)
    assert out["de"] == pytest.approx(0.5)


def test_per_language_accuracy_accepts_1d_predictions():
    # Predictions already reduced to class ids (1D).
    languages = ["fr", "fr", "zh"]
    labels = np.array([1, 2, 0])
    preds = np.array([1, 2, 2])  # fr both right, zh wrong
    out = per_language_accuracy(preds, labels, languages)
    assert out["fr"] == pytest.approx(1.0)
    assert out["zh"] == pytest.approx(0.0)


def test_per_language_accuracy_single_language():
    languages = ["en", "en", "en", "en"]
    labels = np.array([0, 1, 2, 0])
    preds = np.array([0, 1, 2, 1])  # 3/4
    out = per_language_accuracy(preds, labels, languages)
    assert list(out) == ["en"]
    assert out["en"] == pytest.approx(0.75)
