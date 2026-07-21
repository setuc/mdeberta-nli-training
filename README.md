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

> **MPS note:** over only 2 steps with fp32 you may see a `nan` loss or a large
> `eval_loss`. That is an artifact of MPS + a randomly-initialized classifier
> head over almost no steps — **not** a pipeline bug. Accuracy sits at ~0.33
> (chance for 3 classes), which is expected before real training. On CUDA with
> bf16 and real step counts this stabilizes.

Run the test suite (CPU-only, no network):

```bash
uv run pytest
```

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
- **`nan` loss locally on MPS:** expected over tiny step counts — see the smoke
  test note above. Verify on CUDA before treating it as a real problem.
