"""Tests for mdeberta_nli.data — strictly offline.

We never hit the network. Two strategies, in priority order:

1. If ``data.py`` exposes a small pure normalization helper (something that maps
   a raw source row/batch into the common ``{premise, hypothesis, label,
   language}`` schema), we test that directly -- it's the smallest pure unit.
2. Otherwise we drive ``prepare_datasets`` end to end with:
     * ``monkeypatch`` on ``mdeberta_nli.data.load_dataset`` returning a tiny
       in-memory ``datasets.Dataset`` / ``DatasetDict``, and
     * a lightweight fake tokenizer (no model download),
   and assert the output contract: a ``(train, eval)`` pair where ``eval`` has a
   ``language`` column, the label column is named ``labels``, and tokenized
   fields exist.

Label convention: 0=entailment, 1=neutral, 2=contradiction.
"""

import inspect

import pytest
from datasets import Dataset, DatasetDict

import mdeberta_nli.data as data_mod
from mdeberta_nli.config import TrainConfig


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeTokenizer:
    """Minimal stand-in for a HF fast tokenizer used for PAIR encoding.

    Supports being called with either scalars (``str, str``) or batches
    (``list[str], list[str]``), mirroring how ``datasets.Dataset.map`` invokes a
    tokenizer with or without ``batched=True``. Produces ``input_ids`` and
    ``attention_mask`` only (no ``token_type_ids``; the contract treats that as
    optional).
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __call__(self, premise, hypothesis=None, truncation=False, max_length=None, **kwargs):
        batched = isinstance(premise, (list, tuple))
        premises = list(premise) if batched else [premise]
        if hypothesis is None:
            hypotheses = [None] * len(premises)
        else:
            hypotheses = list(hypothesis) if batched else [hypothesis]

        input_ids, attention_mask = [], []
        for p, h in zip(premises, hypotheses):
            text = p if h is None else f"{p} {h}"
            ids = [ord(c) % 100 + 1 for c in text.replace(" ", "")]
            if truncation and max_length is not None:
                ids = ids[:max_length]
            input_ids.append(ids)
            attention_mask.append([1] * len(ids))

        if batched:
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        return {"input_ids": input_ids[0], "attention_mask": attention_mask[0]}


def _common_schema_split(langs, rows_per_lang=3):
    """A datasets.Dataset already in the common {premise, hypothesis, label,
    language} schema. Used as the fake return value for load_dataset so
    prepare_datasets' normalization has valid, predictable inputs regardless of
    which raw source it thinks it is reading."""
    premise, hypothesis, label, language = [], [], [], []
    for lang in langs:
        for i in range(rows_per_lang):
            premise.append(f"premise {lang} {i}")
            hypothesis.append(f"hypothesis {lang} {i}")
            label.append(i % 3)  # cycles 0,1,2 -> entailment/neutral/contradiction
            language.append(lang)
    return Dataset.from_dict(
        {
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label,
            "language": language,
        }
    )


def _fake_load_dataset(langs=("en", "de", "zh")):
    """Return a load_dataset replacement that ignores its args and yields a
    DatasetDict with train/validation/test splits in the common schema."""

    def _loader(*args, **kwargs):
        return DatasetDict(
            {
                "train": _common_schema_split(langs),
                "validation": _common_schema_split(langs),
                "test": _common_schema_split(langs),
            }
        )

    return _loader


# --------------------------------------------------------------------------
# Strategy 1: the pure normalize helper (smallest testable unit)
# --------------------------------------------------------------------------


def _find_normalize_helper():
    """Return (name, fn) of the exposed normalize helper, or None.

    data.py's helper operates on a whole datasets.Dataset plus a language tag;
    it is the smallest pure unit for the normalization / label-mapping logic.
    """
    for name in dir(data_mod):
        if "normalize" not in name.lower():
            continue
        obj = getattr(data_mod, name)
        if callable(obj):
            return name, obj
    return None


def _call_normalize(fn, dataset, language):
    """Call a normalize helper whether it takes (dataset) or (dataset, language)."""
    sig = inspect.signature(fn)
    if len(sig.parameters) >= 2:
        return fn(dataset, language)
    return fn(dataset)


def test_normalize_helper_stamps_language_and_maps_schema():
    """XNLI-style source (no per-row language) -> common 4-column schema, with
    the passed language stamped on every row and int labels preserved."""
    found = _find_normalize_helper()
    if found is None:
        pytest.skip("data.py exposes no dedicated normalize helper; covered by prepare_datasets test")
    _name, fn = found

    raw = Dataset.from_dict(
        {
            "premise": ["p0", "p1", "p2"],
            "hypothesis": ["h0", "h1", "h2"],
            "label": [0, 1, 2],  # entailment / neutral / contradiction
            # extra source-specific column that must be dropped:
            "genre": ["a", "b", "c"],
        }
    )
    out = _call_normalize(fn, raw, "en")

    assert isinstance(out, Dataset)
    assert set(out.column_names) == {"premise", "hypothesis", "label", "language"}
    assert out["language"] == ["en", "en", "en"]
    assert out["label"] == [0, 1, 2]
    assert all(isinstance(v, int) for v in out["label"])
    assert out["premise"] == ["p0", "p1", "p2"]


def test_normalize_helper_prefers_existing_per_row_language():
    """When the source already carries a per-row language field, it is kept
    instead of the stamped tag."""
    found = _find_normalize_helper()
    if found is None:
        pytest.skip("data.py exposes no dedicated normalize helper; covered by prepare_datasets test")
    _name, fn = found

    raw = Dataset.from_dict(
        {
            "premise": ["p0", "p1"],
            "hypothesis": ["h0", "h1"],
            "label": [2, 0],
            "language": ["de", "fr"],
        }
    )
    out = _call_normalize(fn, raw, "mixed")
    assert set(out.column_names) == {"premise", "hypothesis", "label", "language"}
    # Per-row language wins over the stamped "mixed".
    assert out["language"] == ["de", "fr"]


# --------------------------------------------------------------------------
# Strategy 2: prepare_datasets end-to-end (offline, fake load_dataset)
# --------------------------------------------------------------------------


def test_prepare_datasets_offline_contract(monkeypatch):
    monkeypatch.setattr(data_mod, "load_dataset", _fake_load_dataset(), raising=True)

    cfg = TrainConfig(
        max_length=16,
        seed=7,
        # Keep both sources on; the fake ignores the name and returns the same
        # small schema for whichever source data.py requests.
    )
    tokenizer = FakeTokenizer()

    result = data_mod.prepare_datasets(cfg, tokenizer)

    # Returns a (train, eval) pair.
    assert isinstance(result, tuple)
    assert len(result) == 2
    train_ds, eval_ds = result
    assert isinstance(train_ds, Dataset)
    assert isinstance(eval_ds, Dataset)

    # Label column is named "labels" and is integer-typed.
    assert "labels" in train_ds.column_names
    assert "labels" in eval_ds.column_names

    # Tokenized fields exist on both splits.
    for ds in (train_ds, eval_ds):
        assert "input_ids" in ds.column_names
        assert "attention_mask" in ds.column_names

    # eval must additionally keep the string "language" column.
    assert "language" in eval_ds.column_names
    assert len(eval_ds) > 0
    assert isinstance(eval_ds[0]["language"], str)

    # Labels stay within the 3-class range.
    assert all(0 <= lbl <= 2 for lbl in train_ds["labels"])

    # Tokenization respected max_length truncation.
    assert all(len(ids) <= cfg.max_length for ids in train_ds["input_ids"])


def test_prepare_datasets_respects_max_train_samples(monkeypatch):
    monkeypatch.setattr(data_mod, "load_dataset", _fake_load_dataset(), raising=True)

    cfg = TrainConfig(max_length=16, seed=7, max_train_samples=4)
    train_ds, _eval_ds = data_mod.prepare_datasets(cfg, FakeTokenizer())
    assert len(train_ds) <= 4
