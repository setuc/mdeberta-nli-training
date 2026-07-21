"""Dataset loading, normalization, and tokenization for mDeBERTa NLI training.

Normalization contract
-----------------------
Every raw source is normalized to exactly four columns BEFORE tokenizing:

    premise   : str   -- the premise sentence
    hypothesis: str   -- the hypothesis sentence
    label     : int   -- 0=entailment, 1=neutral, 2=contradiction
    language  : str   -- ISO-ish language tag (e.g. "en"), or "mixed"

Both supported sources (``facebook/xnli`` per-language configs and
``MoritzLaurer/multilingual-NLI-26lang-2mil7``) already use the label
convention 0=entailment / 1=neutral / 2=contradiction, so labels are copied
through unchanged.

After normalization the datasets are tokenized as a PAIR
(``tokenizer(premise, hypothesis, truncation=True, max_length=...)``) and the
integer label column is renamed to ``labels`` (int64). No padding is applied
here -- callers are expected to use ``DataCollatorWithPadding`` for dynamic
padding at batch time.

Splits
------
- TRAIN: concatenation of the requested languages' XNLI ``train`` splits and,
  when enabled, the 26lang ``train`` corpus. ``config.max_train_samples`` caps
  the concatenated training set (shuffle(seed) then select).
- EVAL: XNLI ``validation`` splits, one per language, each row tagged with its
  language string. The eval dataset intentionally RETAINS the ``language``
  column so callers can compute per-language accuracy. HF ``Trainer`` ignores
  columns the model does not consume, so callers MUST NOT drop ``language``
  before per-language scoring.

Smoke mode (``config.smoke_test``)
----------------------------------
- Only ``config.smoke_languages`` are used, and each XNLI split is sliced to
  ``config.smoke_rows_per_lang`` rows via ``.select(range(...))``.
- The 26lang corpus is skipped entirely (it is far too large for a smoke run).
"""

from __future__ import annotations

from datasets import Dataset, concatenate_datasets, load_dataset

# XNLI lives at the namespaced repo id "facebook/xnli". The bare legacy name
# "xnli" is rejected by current datasets/huggingface_hub (repo ids must be
# "namespace/name").
XNLI_DATASET = "facebook/xnli"

# XNLI per-language configs available on the Hub. XNLI ships one config per
# language, each with train/validation/test and columns premise/hypothesis/label.
XNLI_LANGUAGES = [
    "ar", "bg", "de", "el", "en", "es", "fr", "hi",
    "ru", "sw", "th", "tr", "ur", "vi", "zh",
]

# The common schema produced by every normalization step, before tokenizing.
NORMALIZED_COLUMNS = ["premise", "hypothesis", "label", "language"]

_26LANG_DATASET = "MoritzLaurer/multilingual-NLI-26lang-2mil7"


def _select_xnli_languages(config) -> list[str]:
    """Resolve which XNLI language configs to load for this run."""
    if config.smoke_test:
        langs = list(config.smoke_languages)
    else:
        langs = list(XNLI_LANGUAGES)
    # Defensive: keep only languages XNLI actually provides a config for, so a
    # stray smoke_languages entry does not blow up the whole load.
    unknown = [lang for lang in langs if lang not in XNLI_LANGUAGES]
    if unknown:
        raise ValueError(
            f"Requested XNLI language(s) {unknown} have no XNLI config. "
            f"Valid options: {XNLI_LANGUAGES}"
        )
    return langs


def _normalize_columns(dataset: Dataset, language: str | None) -> Dataset:
    """Reduce an arbitrary NLI dataset to the normalized 4-column schema.

    ``language`` is the tag applied to every row when the source has no usable
    per-row language field. XNLI configs are single-language, so we pass the
    config name. For 26lang we pass "mixed" unless a language field is present.
    """
    columns = dataset.column_names

    # Assumption: both supported sources expose "premise", "hypothesis" and
    # "label" columns directly. XNLI does; the 26lang corpus does too. We assert
    # this loudly rather than silently producing a malformed dataset.
    for required in ("premise", "hypothesis", "label"):
        if required not in columns:
            raise ValueError(
                f"Source dataset is missing required column '{required}'. "
                f"Available columns: {columns}"
            )

    # Prefer an existing per-row language field if one exists (26lang variants
    # sometimes carry "lang" or "language"); otherwise stamp the given tag.
    lang_field = None
    for candidate in ("language", "lang"):
        if candidate in columns:
            lang_field = candidate
            break

    def _to_schema(batch):
        n = len(batch["premise"])
        if lang_field is not None:
            languages = [str(v) for v in batch[lang_field]]
        else:
            languages = [language] * n
        return {
            "premise": [str(p) for p in batch["premise"]],
            "hypothesis": [str(h) for h in batch["hypothesis"]],
            "label": [int(v) for v in batch["label"]],
            "language": languages,
        }

    return dataset.map(
        _to_schema,
        batched=True,
        remove_columns=columns,
        desc="normalize",
    )


def _load_xnli(config) -> tuple[list[Dataset], list[Dataset]]:
    """Load and normalize XNLI train + validation splits per language.

    Returns (train_parts, eval_parts) as lists of normalized Datasets.
    """
    langs = _select_xnli_languages(config)
    train_parts: list[Dataset] = []
    eval_parts: list[Dataset] = []

    for lang in langs:
        ds = load_dataset(XNLI_DATASET, lang)

        train_split = ds["train"]
        val_split = ds["validation"]

        if config.smoke_test:
            n = config.smoke_rows_per_lang
            train_split = train_split.select(range(min(n, len(train_split))))
            val_split = val_split.select(range(min(n, len(val_split))))

        train_parts.append(_normalize_columns(train_split, language=lang))
        eval_parts.append(_normalize_columns(val_split, language=lang))

    return train_parts, eval_parts


def _load_26lang(config) -> list[Dataset]:
    """Load and normalize the 26lang train corpus (train-only).

    Inspects the returned DatasetDict's splits at runtime rather than
    hardcoding a split name. Returns normalized Datasets tagged "mixed"
    (unless the source carries a per-row language field).
    """
    dsd = load_dataset(_26LANG_DATASET)

    # Inspect available splits at runtime. This corpus is train-only, but its
    # split may be named "train" or something else; concatenate whatever it has.
    parts: list[Dataset] = []
    for split_name in dsd.keys():
        parts.append(_normalize_columns(dsd[split_name], language="mixed"))
    return parts


def prepare_datasets(config, tokenizer) -> tuple:
    """Build tokenized (train_dataset, eval_dataset) per the data contract.

    Both returned objects are ``datasets.Dataset`` with columns:
    input_ids, attention_mask, token_type_ids (if the tokenizer produces them)
    and labels (int64). The eval dataset additionally keeps the string column
    ``language`` for per-language accuracy -- callers must not drop it.
    """
    train_parts: list[Dataset] = []
    eval_parts: list[Dataset] = []

    if config.use_xnli:
        xnli_train, xnli_eval = _load_xnli(config)
        train_parts.extend(xnli_train)
        eval_parts.extend(xnli_eval)

    # Skip the 26lang corpus when disabled OR during smoke tests (too large to
    # download/process for a quick local check).
    if config.use_26lang and not config.smoke_test:
        train_parts.extend(_load_26lang(config))

    if not train_parts:
        raise ValueError(
            "No training data selected: enable config.use_xnli and/or "
            "config.use_26lang (26lang is skipped in smoke_test mode)."
        )
    if not eval_parts:
        raise ValueError(
            "No evaluation data available. Enable config.use_xnli to obtain "
            "XNLI validation splits for evaluation."
        )

    train_dataset = concatenate_datasets(train_parts)
    eval_dataset = concatenate_datasets(eval_parts)

    # Cap total training rows if requested: shuffle deterministically first so
    # the cap is a representative sample across all concatenated sources.
    if config.max_train_samples is not None:
        n = min(config.max_train_samples, len(train_dataset))
        train_dataset = train_dataset.shuffle(seed=config.seed).select(range(n))

    def _tokenize(batch):
        return tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            max_length=config.max_length,
        )

    # Tokenize TRAIN: drop the raw text + language columns (the model never sees
    # them); keep "label" so it can be renamed to "labels".
    train_dataset = train_dataset.map(
        _tokenize,
        batched=True,
        remove_columns=["premise", "hypothesis", "language"],
        desc="tokenize-train",
    )
    train_dataset = train_dataset.rename_column("label", "labels")

    # Tokenize EVAL: drop only the raw text columns; KEEP "language" for
    # per-language accuracy. "label" is renamed to "labels" as well.
    eval_dataset = eval_dataset.map(
        _tokenize,
        batched=True,
        remove_columns=["premise", "hypothesis"],
        desc="tokenize-eval",
    )
    eval_dataset = eval_dataset.rename_column("label", "labels")

    return train_dataset, eval_dataset
