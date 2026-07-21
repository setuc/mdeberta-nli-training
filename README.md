# mDeBERTa-v3 Multilingual NLI Training

Fine-tune [`microsoft/mdeberta-v3-base`](https://huggingface.co/microsoft/mdeberta-v3-base)
for multilingual Natural Language Inference (NLI) on **XNLI** and the
**multilingual-NLI-26lang-2mil7** corpus. The resulting 3-way classifier
(`entailment` / `neutral` / `contradiction`) is suitable for zero-shot
classification across ~100 languages.

The pipeline is designed to **smoke-test locally on Apple Silicon (MPS)** and
run the **full job on a single NVIDIA GPU**, with the same code path. Two
training entrypoints are provided so you can compare approaches:

| Script | Approach | Use for |
| --- | --- | --- |
| `scripts/train_trainer.py` | HF `Trainer` API | Production-standard, least code |
| `scripts/train_accelerate.py` | Custom loop + `accelerate` | Learning / full control |

## Layout

```
src/mdeberta_nli/     shared core (config, data, model, metrics, device)
scripts/              training entrypoints + setup_env.sh
configs/              smoke.yaml (tiny/fast) and full.yaml (real run)
tests/                CPU-only pytest suite (no downloads)
```

The core modules are shared by both entrypoints, so data normalization,
tokenization, metrics, and device/precision selection behave identically
regardless of which loop you run.

## Requirements

- Python 3.11 (pinned in `.python-version`)
- [`uv`](https://docs.astral.sh/uv/) for environment management
- Local smoke test: any machine (CPU / Apple Silicon MPS)
- Full training: one CUDA GPU (A100 / L4 / 3090-class or better)

## Setup

```bash
# Sync the environment (creates .venv from pyproject.toml)
bash scripts/setup_env.sh

# On the GPU VM, warm the HF cache up-front (model + a dataset probe):
bash scripts/setup_env.sh --download
```

`uv sync` alone also works. `PYTORCH_ENABLE_MPS_FALLBACK=1` is exported by the
scripts so unsupported ops fall back to CPU on Apple Silicon.

## Local smoke test (verify the pipeline)

Runs a few hundred rows from 3 languages for 2 optimizer steps — proves the
full data → model → train → eval → per-language-accuracy path end to end.

```bash
uv run python scripts/train_trainer.py    --config configs/smoke.yaml --smoke-test
uv run python scripts/train_accelerate.py --config configs/smoke.yaml --smoke-test
```

> **Smoke-test note:** over only 2 steps the metrics are meaningless (accuracy
> ~0.33 = 3-class chance); the smoke test only proves the pipeline runs end to
> end. Use the "real bounded local run" below to confirm the model actually
> learns.

Run the test suite (CPU-only, no network):

```bash
uv run pytest              # full suite
uv run pytest -m "not slow"  # skip the model-download regression test
```

### Real bounded local run (proves it actually learns)

`configs/local.yaml` trains a genuine epoch over a bounded slice of XNLI
(3 languages × 4000 rows) on the Mac GPU — no `max_steps` cap. This is the
right way to confirm the pipeline *learns*, not just wires up:

```bash
uv run python scripts/train_trainer.py --config configs/local.yaml
```

Reference result on an M4 Pro (~23 min, one epoch): eval accuracy climbs from
0.333 (3-class chance) to **~0.71** (en 0.74 / de 0.70 / zh 0.68) — cross-lingual
transfer from English data lifting German and Chinese.

> **Note on precision:** the model is loaded in **float32** on purpose (see
> `model.py`). The mDeBERTa checkpoint is stored in float16, and transformers v5
> keeps the checkpoint dtype — but float16 master weights make AdamW's `eps`
> underflow and produce NaN on the first step. Mixed-precision *compute* (bf16 on
> CUDA) is configured separately and is unaffected.

## Full training (GPU VM)

```bash
git clone https://github.com/setuc/mdeberta-nli-training
cd mdeberta-nli-training
bash scripts/setup_env.sh --download
uv run python scripts/train_trainer.py --config configs/full.yaml
```

`full.yaml` trains on all XNLI languages + the 26lang corpus. The 26lang set is
~2.7M pairs — for a cheaper first run, cap it:

```bash
uv run python scripts/train_trainer.py --config configs/full.yaml --max-train-samples 500000
```

### CLI overrides

Any of these override the YAML (only keys you pass take effect):

```
--model-name --output-dir --max-steps --max-train-samples
--learning-rate --num-train-epochs --batch-size --smoke-test
```

## Monitoring

Both entrypoints log to TensorBoard under `<output_dir>/runs`:

```bash
uv run tensorboard --logdir outputs/full/runs
```

Overall accuracy and **per-language accuracy** on the XNLI validation splits
are logged and printed at the end of training.

## Configuration

`configs/*.yaml` keys map 1:1 to `TrainConfig` fields in
`src/mdeberta_nli/config.py`. Precedence is: dataclass defaults < YAML file <
CLI overrides. Unknown YAML keys fail fast with a clear error.

## Troubleshooting

- **Out of memory (CUDA):** lower `per_device_train_batch_size`, reduce
  `max_length` (default 128), or raise `gradient_accumulation_steps`.
- **Dataset download fails:** run `setup_env.sh --download` on a machine with
  network access to warm the HF cache before training.
- **`nan` loss:** if you see this after modifying `model.py`, check the model is
  loaded in float32 (`dtype=torch.float32`). float16 master weights make AdamW
  produce NaN on the first optimizer step regardless of learning rate or device.
