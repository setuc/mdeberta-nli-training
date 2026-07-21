#!/usr/bin/env bash
# Environment setup for the mdeberta_nli project.
#
# Usage:
#   ./scripts/setup_env.sh              # sync deps only
#   ./scripts/setup_env.sh --download   # sync deps + warm the HF cache
#
# Make executable once with:  chmod +x scripts/setup_env.sh
set -euo pipefail

# Let unsupported ops fall back to CPU on Apple Silicon (MPS) instead of erroring.
export PYTORCH_ENABLE_MPS_FALLBACK=1

DOWNLOAD=0
for arg in "$@"; do
  case "$arg" in
    --download)
      DOWNLOAD=1
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: $0 [--download]" >&2
      exit 2
      ;;
  esac
done

echo ">> Syncing dependencies with uv..."
uv sync

MODEL_NAME="microsoft/mdeberta-v3-base"

if [[ "$DOWNLOAD" -eq 1 ]]; then
  echo ">> Pre-fetching model + tiny dataset probe to warm the HF cache..."
  # With --download, a failure here is fatal so the user knows the warmup did not happen.
  uv run python -c '
import sys
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset

model_name = "microsoft/mdeberta-v3-base"
print(f"  - fetching model files for {model_name} ...")
snapshot_download(repo_id=model_name)
AutoConfig.from_pretrained(model_name)
AutoTokenizer.from_pretrained(model_name, use_fast=True)

print("  - probing datasets (tiny slice) ...")
load_dataset("facebook/xnli", "en", split="validation[:2]")
print("  - HF cache warmed.")
'
else
  echo ">> Skipping model/dataset download (pass --download to warm the HF cache)."
  # Best-effort, non-fatal probe: warn clearly but never hard-fail without --download.
  if ! uv run python -c '
from transformers import AutoConfig
AutoConfig.from_pretrained("microsoft/mdeberta-v3-base")
' 2>/dev/null; then
    echo "WARNING: could not verify access to microsoft/mdeberta-v3-base." >&2
    echo "WARNING: re-run with --download once you have network/HF access." >&2
  fi
fi

echo ">> Done."
