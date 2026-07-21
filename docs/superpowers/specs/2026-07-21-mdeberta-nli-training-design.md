# mDeBERTa-v3 Multilingual NLI Training — Design

**Date:** 2026-07-21
**Status:** Approved

## Goal

Build a training pipeline that fine-tunes `microsoft/mdeberta-v3-base` on multilingual
Natural Language Inference (NLI) data, replicating the approach behind
`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` style models. The pipeline must:

- Be **smoke-testable locally** on Apple Silicon (M4 Pro, MPS backend).
- Run the **full training on a single NVIDIA GPU VM** (CUDA).
- Be managed with **`uv`** for reproducible environments.
- Live in a **public GitHub repo `mdeberta-nli-training`** under account `setuc`.

## Scope

- **Datasets:** `facebook/xnli` (all_languages, 15 langs) **+**
  `MoritzLaurer/multilingual-NLI-26lang-2mil7` (~2.7M pairs, 27 langs).
- **Task:** 3-class sequence classification — `0=entailment, 1=neutral, 2=contradiction`.
- **Two training entrypoints** (to test and learn both approaches):
  1. HF `Trainer` API (production-standard, least code).
  2. Custom PyTorch loop with HF `accelerate` (educational, more control).
- **Tracking:** TensorBoard (local files, works offline on the VM).
- **Verification:** both a `--smoke-test` tiny-subset mode *and* a "full pipeline,
  few steps" mode (via `max_steps` override), plus a CPU-only `pytest` suite.

Explicitly out of scope: multi-GPU/distributed (design stays accelerate-compatible so
it's a config flag away, but not implemented/tested now), W&B, hyperparameter search.

## Architecture

Shared core modules consumed by two thin training entrypoints.

```
mdeberta-nli-training/
├── pyproject.toml            # uv-managed deps
├── README.md                 # setup + run instructions (local + GPU VM)
├── .python-version           # 3.11
├── .gitignore
├── configs/
│   ├── smoke.yaml            # tiny subset, MPS/CPU, max_steps=2
│   └── full.yaml             # XNLI + 26lang, single GPU, real hyperparams
├── src/mdeberta_nli/
│   ├── __init__.py
│   ├── config.py             # TrainConfig dataclass, YAML load + CLI override
│   ├── data.py               # load + normalize + concat datasets, tokenize
│   ├── model.py              # AutoModelForSequenceClassification (num_labels=3)
│   ├── metrics.py            # accuracy + per-language accuracy
│   └── device.py             # cuda/mps/cpu detection + fp16/bf16 gating
├── scripts/
│   ├── train_trainer.py      # entrypoint A: HF Trainer
│   ├── train_accelerate.py   # entrypoint B: custom accelerate loop
│   └── setup_env.sh          # uv sync + optional dataset/model pre-download
└── tests/
    ├── test_config.py        # YAML load + override precedence
    ├── test_data.py          # normalization schema, label mapping (mocked/tiny)
    ├── test_device.py        # device + precision selection logic
    └── test_metrics.py       # accuracy + per-language grouping
```

## Component Responsibilities

### `config.py`
`TrainConfig` dataclass holding: `model_name`, dataset selection flags, `max_length`,
`per_device_train_batch_size`, `learning_rate`, `num_train_epochs`, `max_steps`,
`output_dir`, `smoke_test`, `seed`, `logging_steps`, `eval_steps`, `save_steps`.
Loaded from a YAML file; CLI flags override YAML; `--smoke-test` forces the smoke subset.

### `data.py`
- Loads `facebook/xnli` and `MoritzLaurer/multilingual-NLI-26lang-2mil7`.
- Normalizes both to a common schema `{premise, hypothesis, label, language}`.
- Concatenates into one `datasets.Dataset`; XNLI validation/test kept separate,
  tagged per-language for per-language accuracy.
- Tokenizes premise+hypothesis pairs (truncation to `max_length`); uses
  `DataCollatorWithPadding` for dynamic padding.
- Smoke mode: slices ~300 rows from 3 languages, no full download of the 2.7M set
  (uses streaming/`split="train[:N]"` where possible).

### `model.py`
`load_model(config)` → `AutoModelForSequenceClassification` with `num_labels=3` and
`id2label`/`label2id` set. `load_tokenizer(config)` → fast tokenizer.

### `metrics.py`
`compute_metrics` for overall accuracy; `per_language_accuracy(preds, labels, langs)`
for the multilingual signal. Both usable by Trainer's `compute_metrics` and the
accelerate loop's eval step.

### `device.py`
`get_device()` → `cuda` > `mps` > `cpu`. `get_precision(device)` → `bf16` on CUDA
(if supported), disables fp16 on MPS, fp32 on CPU. Returns flags the entrypoints pass
to `TrainingArguments`/`accelerate`.

### `scripts/train_trainer.py`
Wires config → data → model → `Trainer` with `TrainingArguments` (TensorBoard
`report_to="tensorboard"`), runs `train()` + `evaluate()`, saves model.

### `scripts/train_accelerate.py`
Same wiring, but explicit training loop: `accelerate` prepares model/optimizer/
dataloaders, manual forward/backward/step, manual eval + TensorBoard `SummaryWriter`,
manual checkpoint save.

### `scripts/setup_env.sh`
`uv sync` to build the env; optional `--download` flag to pre-fetch model + datasets
into the HF cache so the GPU VM can train without re-downloading / offline.

## Data Flow

```
YAML config + CLI → TrainConfig
        ↓
data.load_datasets() → normalize → concat → tokenize → (train, xnli_val)
        ↓
model.load_model() + tokenizer
        ↓
device.get_device()/get_precision()
        ↓
[train_trainer.py]  Trainer(train, eval, compute_metrics) → train() → save
   or
[train_accelerate.py]  accelerate loop → per-step log → eval → save
        ↓
TensorBoard logs in output_dir/runs + saved model in output_dir
```

## Error Handling

- Missing/invalid config keys → fail fast with a clear message in `config.py`.
- Dataset download failure → surface HF error; `setup_env.sh --download` lets the VM
  pre-cache so training itself doesn't hit the network.
- MPS unsupported-op fallback → set `PYTORCH_ENABLE_MPS_FALLBACK=1` in scripts and warn.
- OOM guidance documented in README (reduce batch size / max_length / grad accumulation).

## Testing

CPU-only `pytest` suite (no real training, tiny/mocked data):
- `test_config.py` — YAML load, CLI override precedence, smoke-test forcing.
- `test_data.py` — normalization maps both sources to the common schema; label values
  and ordering (entailment/neutral/contradiction) correct.
- `test_device.py` — device priority and precision gating logic.
- `test_metrics.py` — overall accuracy and per-language grouping correctness.

End-to-end verification via `--smoke-test` (a few hundred rows, `max_steps=2`) on MPS,
and a "full pipeline, few steps" run using `max_steps` override.

## Workflow

**Local (M4 Pro):**
```
uv sync
uv run scripts/train_trainer.py --config configs/smoke.yaml --smoke-test
uv run pytest
```
**GPU VM:**
```
git clone https://github.com/setuc/mdeberta-nli-training
cd mdeberta-nli-training
bash scripts/setup_env.sh --download
uv run scripts/train_trainer.py --config configs/full.yaml
```

## Dependencies (uv / pyproject.toml)

`torch`, `transformers`, `datasets`, `accelerate`, `evaluate`, `scikit-learn`,
`sentencepiece` (mDeBERTa tokenizer), `protobuf`, `tensorboard`, `pyyaml`.
Dev: `pytest`.
