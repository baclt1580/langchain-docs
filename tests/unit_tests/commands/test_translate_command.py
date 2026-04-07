"""Tests for translate command environment configuration."""

import pytest

from pipeline.commands.translate import _load_settings_from_env


def test_load_settings_from_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use default runtime options when optional env vars are absent."""
    monkeypatch.setenv("DOCS_AI_OPENAI_BASEURL", "https://api.openai.com/v1")
    monkeypatch.setenv("DOCS_AI_OPENAI_APIKEY", "key")
    monkeypatch.setenv("DOCS_AI_OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("DOCS_TARGET_LANGUAGES", "zh-CN,ja")
    monkeypatch.delenv("DOCS_TRANSLATE_CONCURRENCY", raising=False)
    monkeypatch.delenv("DOCS_TRANSLATE_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("DOCS_TRANSLATE_MAX_RETRIES", raising=False)

    settings = _load_settings_from_env()

    assert settings.concurrency == 3
    assert settings.timeout_seconds == 120.0
    assert settings.max_retries == 3


def test_load_settings_from_env_custom_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load concurrency/timeout/retries from env."""
    monkeypatch.setenv("DOCS_AI_OPENAI_BASEURL", "https://api.openai.com/v1")
    monkeypatch.setenv("DOCS_AI_OPENAI_APIKEY", "key")
    monkeypatch.setenv("DOCS_AI_OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("DOCS_TARGET_LANGUAGES", "zh-CN")
    monkeypatch.setenv("DOCS_TRANSLATE_CONCURRENCY", "8")
    monkeypatch.setenv("DOCS_TRANSLATE_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("DOCS_TRANSLATE_MAX_RETRIES", "5")

    settings = _load_settings_from_env()

    assert settings.concurrency == 8
    assert settings.timeout_seconds == 30.0
    assert settings.max_retries == 5


def test_load_settings_from_env_invalid_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject invalid concurrency values."""
    monkeypatch.setenv("DOCS_AI_OPENAI_BASEURL", "https://api.openai.com/v1")
    monkeypatch.setenv("DOCS_AI_OPENAI_APIKEY", "key")
    monkeypatch.setenv("DOCS_AI_OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("DOCS_TARGET_LANGUAGES", "zh-CN")
    monkeypatch.setenv("DOCS_TRANSLATE_CONCURRENCY", "0")
    monkeypatch.delenv("DOCS_TRANSLATE_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("DOCS_TRANSLATE_MAX_RETRIES", raising=False)

    with pytest.raises(ValueError, match="DOCS_TRANSLATE_CONCURRENCY must be >= 1"):
        _load_settings_from_env()
