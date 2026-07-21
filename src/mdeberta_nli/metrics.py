"""Metrics for mDeBERTa NLI training and evaluation."""

from __future__ import annotations

import numpy as np


def _extract_logits_and_labels(eval_pred):
    """Return (predictions, labels) from a tuple or an object with attributes."""
    if hasattr(eval_pred, "predictions") and hasattr(eval_pred, "label_ids"):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        predictions, labels = eval_pred
    return predictions, labels


def _to_class_ids(predictions):
    """Convert predictions to 1D class ids, taking argmax over 2D logits."""
    predictions = np.asarray(predictions)
    if predictions.ndim >= 2:
        predictions = np.argmax(predictions, axis=-1)
    return predictions.reshape(-1)


def compute_metrics(eval_pred) -> dict:
    """Compute accuracy from an eval_pred (logits, labels) tuple or object."""
    predictions, labels = _extract_logits_and_labels(eval_pred)
    preds = _to_class_ids(predictions)
    labels = np.asarray(labels).reshape(-1)
    accuracy = float((preds == labels).mean()) if labels.size else 0.0
    return {"accuracy": accuracy}


def per_language_accuracy(predictions, labels, languages) -> dict:
    """Compute per-language accuracy.

    predictions may be 2D logits (argmax applied) or 1D class ids. languages is a
    list parallel to predictions/labels. Returns {lang: accuracy_float}.
    """
    preds = _to_class_ids(predictions)
    labels = np.asarray(labels).reshape(-1)
    languages = np.asarray(list(languages))

    result: dict = {}
    for lang in np.unique(languages):
        mask = languages == lang
        lang_preds = preds[mask]
        lang_labels = labels[mask]
        acc = float((lang_preds == lang_labels).mean()) if lang_labels.size else 0.0
        result[str(lang)] = acc
    return result
