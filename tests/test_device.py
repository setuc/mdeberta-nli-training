"""Tests for mdeberta_nli.device.

We never require a real GPU. All backend probes are monkeypatched so the
priority logic (cuda > mps > cpu) and the precision mapping can be exercised
deterministically on any machine, including CPU-only CI.
"""

import torch

from mdeberta_nli.device import get_device, get_precision


def _patch_backends(monkeypatch, *, cuda, mps, bf16=False):
    """Force the three backend probes used by device.py."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: bf16)

    # device.py reads torch.backends.mps via getattr, then calls .is_available().
    # Ensure the attribute exists and is patched regardless of platform.
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        import types

        mps_backend = types.SimpleNamespace()
        monkeypatch.setattr(torch.backends, "mps", mps_backend, raising=False)
    monkeypatch.setattr(mps_backend, "is_available", lambda: mps, raising=False)


# --- get_device priority ------------------------------------------------


def test_get_device_cuda_wins(monkeypatch):
    _patch_backends(monkeypatch, cuda=True, mps=True)
    assert get_device() == "cuda"


def test_get_device_mps_when_no_cuda(monkeypatch):
    _patch_backends(monkeypatch, cuda=False, mps=True)
    assert get_device() == "mps"


def test_get_device_cpu_when_nothing(monkeypatch):
    _patch_backends(monkeypatch, cuda=False, mps=False)
    assert get_device() == "cpu"


def test_get_device_cuda_wins_even_without_mps(monkeypatch):
    _patch_backends(monkeypatch, cuda=True, mps=False)
    assert get_device() == "cuda"


# --- get_precision mapping ----------------------------------------------


def test_precision_cuda_bf16_supported(monkeypatch):
    _patch_backends(monkeypatch, cuda=True, mps=False, bf16=True)
    assert get_precision("cuda") == {"bf16": True, "fp16": False}


def test_precision_cuda_bf16_unsupported_falls_back_to_fp16(monkeypatch):
    _patch_backends(monkeypatch, cuda=True, mps=False, bf16=False)
    assert get_precision("cuda") == {"bf16": False, "fp16": True}


def test_precision_mps_disables_all():
    assert get_precision("mps") == {"bf16": False, "fp16": False}


def test_precision_cpu_disables_all():
    assert get_precision("cpu") == {"bf16": False, "fp16": False}
