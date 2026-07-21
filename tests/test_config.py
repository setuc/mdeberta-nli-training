"""Tests for mdeberta_nli.config.

Covers the precedence contract of ``load_config``:
    dataclass defaults < YAML file < non-None keyword overrides
plus fail-fast behaviour on unknown YAML keys.

No network / no training. YAML files are written to pytest's ``tmp_path``.
"""

import textwrap

import pytest

from mdeberta_nli.config import TrainConfig, load_config


def _write_yaml(tmp_path, body: str):
    path = tmp_path / "cfg.yaml"
    path.write_text(textwrap.dedent(body))
    return str(path)


def test_defaults_load_without_yaml():
    cfg = load_config()
    assert isinstance(cfg, TrainConfig)
    # Spot-check a representative sample of the documented defaults.
    assert cfg.model_name == "microsoft/mdeberta-v3-base"
    assert cfg.use_xnli is True
    assert cfg.use_26lang is True
    assert cfg.max_length == 128
    assert cfg.per_device_train_batch_size == 16
    assert cfg.per_device_eval_batch_size == 32
    assert cfg.learning_rate == pytest.approx(2e-5)
    assert cfg.num_train_epochs == pytest.approx(2.0)
    assert cfg.max_steps == -1
    assert cfg.seed == 42
    assert cfg.smoke_test is False
    assert cfg.max_train_samples is None
    # default_factory list must be a fresh list, not a shared mutable default.
    assert cfg.smoke_languages == ["en", "de", "zh"]
    assert cfg.smoke_rows_per_lang == 100


def test_default_factory_list_is_not_shared():
    a = load_config()
    b = load_config()
    a.smoke_languages.append("fr")
    assert b.smoke_languages == ["en", "de", "zh"]


def test_yaml_overrides_defaults(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
        learning_rate: 5.0e-5
        max_length: 256
        num_train_epochs: 3
        smoke_languages:
          - en
          - fr
        """,
    )
    cfg = load_config(path)
    assert cfg.learning_rate == pytest.approx(5e-5)
    assert cfg.max_length == 256
    assert cfg.num_train_epochs == pytest.approx(3.0)
    assert cfg.smoke_languages == ["en", "fr"]
    # Untouched keys keep their defaults.
    assert cfg.model_name == "microsoft/mdeberta-v3-base"
    assert cfg.seed == 42


def test_keyword_overrides_beat_yaml(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
        learning_rate: 5.0e-5
        max_length: 256
        """,
    )
    cfg = load_config(path, learning_rate=1e-3, max_length=64)
    assert cfg.learning_rate == pytest.approx(1e-3)
    assert cfg.max_length == 64


def test_none_overrides_do_not_win(tmp_path):
    """Only non-None overrides take effect (per the contract)."""
    path = _write_yaml(
        tmp_path,
        """
        learning_rate: 5.0e-5
        """,
    )
    cfg = load_config(path, learning_rate=None)
    # None must NOT clobber the YAML value.
    assert cfg.learning_rate == pytest.approx(5e-5)


def test_overrides_win_without_yaml():
    cfg = load_config(None, seed=123, smoke_test=True)
    assert cfg.seed == 123
    assert cfg.smoke_test is True


def test_unknown_yaml_key_raises_value_error(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
        learning_rate: 5.0e-5
        not_a_real_key: 7
        """,
    )
    with pytest.raises(ValueError) as excinfo:
        load_config(path)
    # Message should name the offending key to be actionable.
    assert "not_a_real_key" in str(excinfo.value)
