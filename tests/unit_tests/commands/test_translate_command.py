"""Tests for translate command environment configuration."""

from pathlib import Path
from threading import Lock

import pytest

from pipeline.commands.translate import (
    TranslationRunContext,
    _load_settings_from_env,
    _prepare_translation_work_items,
    _translate_language_sequential,
)
from pipeline.tools.translation import (
    LanguageTarget,
    TranslationManifest,
    compute_sha256,
)


def test_load_settings_from_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use default runtime options when optional env vars are absent."""
    monkeypatch.setenv("DOCS_AI_OPENAI_BASEURL", "https://api.openai.com/v1")
    monkeypatch.setenv("DOCS_AI_OPENAI_APIKEY", "key")
    monkeypatch.setenv("DOCS_AI_OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("DOCS_TARGET_LANGUAGES", "zh-CN,ja")
    monkeypatch.delenv("DOCS_TRANSLATE_CONCURRENCY", raising=False)
    monkeypatch.delenv("DOCS_TRANSLATE_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("DOCS_TRANSLATE_MAX_RETRIES", raising=False)
    monkeypatch.delenv("DOCS_TRANSLATE_MAX_PROMPT_TOKENS", raising=False)

    settings = _load_settings_from_env()

    assert settings.concurrency == 3
    assert settings.timeout_seconds == 120.0
    assert settings.max_retries == 3
    assert settings.max_prompt_tokens == 6000


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
    monkeypatch.setenv("DOCS_TRANSLATE_MAX_PROMPT_TOKENS", "9000")

    settings = _load_settings_from_env()

    assert settings.concurrency == 8
    assert settings.timeout_seconds == 30.0
    assert settings.max_retries == 5
    assert settings.max_prompt_tokens == 9000


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
    monkeypatch.delenv("DOCS_TRANSLATE_MAX_PROMPT_TOKENS", raising=False)

    with pytest.raises(ValueError, match="DOCS_TRANSLATE_CONCURRENCY must be >= 1"):
        _load_settings_from_env()


def test_load_settings_from_env_invalid_max_prompt_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject prompt budgets that are too small for the translation prompt."""
    monkeypatch.setenv("DOCS_AI_OPENAI_BASEURL", "https://api.openai.com/v1")
    monkeypatch.setenv("DOCS_AI_OPENAI_APIKEY", "key")
    monkeypatch.setenv("DOCS_AI_OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("DOCS_TARGET_LANGUAGES", "zh-CN")
    monkeypatch.delenv("DOCS_TRANSLATE_CONCURRENCY", raising=False)
    monkeypatch.delenv("DOCS_TRANSLATE_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("DOCS_TRANSLATE_MAX_RETRIES", raising=False)
    monkeypatch.setenv("DOCS_TRANSLATE_MAX_PROMPT_TOKENS", "128")

    with pytest.raises(
        ValueError,
        match="DOCS_TRANSLATE_MAX_PROMPT_TOKENS must be >= 512",
    ):
        _load_settings_from_env()


def test_prepare_translation_work_items_skips_up_to_date_files(
    tmp_path: Path,
) -> None:
    """Skip files that already have a matching translated output."""
    src_root = tmp_path / "src"
    docs_dir = src_root / "oss"
    docs_dir.mkdir(parents=True)

    completed_file = docs_dir / "completed.mdx"
    completed_file.write_text("completed content\n", encoding="utf-8")

    pending_file = docs_dir / "pending.mdx"
    pending_file.write_text("pending content\n", encoding="utf-8")

    i18n_root = src_root / "i18n"
    manifest = TranslationManifest(i18n_root / "translation-hashes.json")
    target_language = LanguageTarget("zh-CN", "Chinese (Simplified)")
    completed_relative_path = "oss/completed.mdx"
    completed_hash = compute_sha256(completed_file.read_text(encoding="utf-8"))
    manifest.set_hash(target_language.code, completed_relative_path, completed_hash)

    completed_output = i18n_root / target_language.code / completed_relative_path
    completed_output.parent.mkdir(parents=True, exist_ok=True)
    completed_output.write_text("已完成\n", encoding="utf-8")

    context = TranslationRunContext(
        src_root=src_root,
        i18n_root=i18n_root,
        manifest=manifest,
        translator=None,
        manifest_lock=Lock(),
        force=False,
        dry_run=False,
        concurrency=1,
    )

    pending_items, skipped_count = _prepare_translation_work_items(
        target_language=target_language,
        source_files=[completed_file, pending_file],
        context=context,
    )

    assert skipped_count == 1
    assert [item.relative_path for item in pending_items] == ["oss/pending.mdx"]
    assert pending_items[0].output_path == (
        i18n_root / target_language.code / "oss/pending.mdx"
    )


def test_translate_language_sequential_starts_progress_at_completed_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Initialize file progress with the number of already completed files."""
    src_root = tmp_path / "src"
    docs_dir = src_root / "langsmith"
    docs_dir.mkdir(parents=True)

    completed_file = docs_dir / "completed.mdx"
    completed_file.write_text("completed content\n", encoding="utf-8")

    pending_file = docs_dir / "pending.mdx"
    pending_file.write_text("pending content\n", encoding="utf-8")

    i18n_root = src_root / "i18n"
    manifest = TranslationManifest(i18n_root / "translation-hashes.json")
    target_language = LanguageTarget("zh-CN", "Chinese (Simplified)")
    completed_relative_path = "langsmith/completed.mdx"
    completed_hash = compute_sha256(completed_file.read_text(encoding="utf-8"))
    manifest.set_hash(target_language.code, completed_relative_path, completed_hash)

    completed_output = i18n_root / target_language.code / completed_relative_path
    completed_output.parent.mkdir(parents=True, exist_ok=True)
    completed_output.write_text("已完成\n", encoding="utf-8")

    context = TranslationRunContext(
        src_root=src_root,
        i18n_root=i18n_root,
        manifest=manifest,
        translator=None,
        manifest_lock=Lock(),
        force=False,
        dry_run=True,
        concurrency=1,
    )
    pending_items, skipped_count = _prepare_translation_work_items(
        target_language=target_language,
        source_files=[completed_file, pending_file],
        context=context,
    )

    progress_instances: list[object] = []

    class FakeTqdm:
        """Capture progress-bar initialization for assertions."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args
            self.total = kwargs["total"]
            self.initial = kwargs.get("initial", 0)
            self.updates: list[int] = []
            progress_instances.append(self)

        def __enter__(self) -> "FakeTqdm":
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def update(self, amount: int) -> None:
            self.updates.append(amount)

    monkeypatch.setattr("pipeline.commands.translate.tqdm", FakeTqdm)

    translated_count, total_skipped = _translate_language_sequential(
        target_language=target_language,
        total_files=2,
        pending_items=pending_items,
        initial_skipped=skipped_count,
        context=context,
    )

    assert translated_count == 1
    assert total_skipped == 1
    assert len(progress_instances) == 1
    assert progress_instances[0].total == 2
    assert progress_instances[0].initial == 1
    assert progress_instances[0].updates == [1]
