"""Device and mixed-precision selection helpers.

This module centralizes the logic for picking a compute device and the
matching mixed-precision configuration so the rest of the training code does
not have to sprinkle backend checks everywhere.

Precision policy rationale
---------------------------
* On CUDA we prefer bf16 whenever the hardware supports it
  (``torch.cuda.is_bf16_supported()``). bf16 keeps the same exponent range as
  fp32, so it avoids the overflow/underflow and loss-scaling headaches that
  fp16 requires. When bf16 is unavailable we fall back to fp16, which is still
  a large speed/memory win on CUDA GPUs and is well supported there.
* On Apple's MPS (Metal) backend fp16 is deliberately DISABLED. Half-precision
  training on MPS has historically been unreliable -- it can silently produce
  NaNs/Infs, and autocast/loss-scaling support is far less mature than on
  CUDA. Running fp32 on MPS is slower but numerically trustworthy, so we do not
  enable any mixed precision there.
* On CPU neither bf16 nor fp16 autocast training makes sense for this workload,
  so both are left off.

MPS availability is probed defensively via ``getattr`` because
``torch.backends.mps`` does not exist on older PyTorch builds or on platforms
without the Metal backend, and importing this module must never fail there.
"""

import torch


def get_device() -> str:
    """Return the best available device string: "cuda", "mps", or "cpu"."""
    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and getattr(mps_backend, "is_available", None) is not None:
        if mps_backend.is_available():
            return "mps"

    return "cpu"


def get_precision(device: str) -> dict:
    """Return the mixed-precision flags for the given device.

    * cuda: bf16 if ``torch.cuda.is_bf16_supported()`` else fp16.
    * mps or cpu: neither (fp16 is unreliable on MPS; see module docstring).
    """
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return {"bf16": True, "fp16": False}
        return {"bf16": False, "fp16": True}

    return {"bf16": False, "fp16": False}
