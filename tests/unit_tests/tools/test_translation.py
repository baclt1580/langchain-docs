"""Tests for translation utilities."""

import os
from pathlib import Path

import pytest

from pipeline.tools.translation import (
    TranslationManifest,
    compute_sha256,
    load_dotenv_file,
    parse_target_languages,
)


def test_parse_target_languages_with_mixed_formats() -> None:
    """Parse target language strings with default and custom labels."""
    targets = parse_target_languages("zh-CN,ja:Japanese,fr|French,zh-CN")
    assert [target.code for target in targets] == ["zh-CN", "ja", "fr"]
    assert [target.label for target in targets] == [
        "Chinese (Simplified)",
        "Japanese",
        "French",
    ]


def test_parse_target_languages_invalid_code() -> None:
    """Raise when language code is invalid."""
    with pytest.raises(ValueError, match="Invalid language code"):
        parse_target_languages("invalid-code-123456789")


def test_translation_manifest_round_trip(tmp_path: Path) -> None:
    """Persist and load hash values by language and source file."""
    manifest_path = tmp_path / "translation-hashes.json"
    manifest = TranslationManifest(manifest_path)

    source_hash = compute_sha256("# hello")
    manifest.set_hash("zh-CN", "oss/langchain/overview.mdx", source_hash)
    manifest.save()

    loaded = TranslationManifest(manifest_path)
    loaded.load()
    assert (
        loaded.get_hash("zh-CN", "oss/langchain/overview.mdx")
        == source_hash
    )


def test_load_dotenv_file_parses_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load dotenv values including comments and quoted strings."""
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        (
            "# comment\n"
            "DOCS_AI_OPENAI_BASEURL=https://api.example.com/v1\n"
            'DOCS_AI_OPENAI_MODEL="gpt-4o-mini"\n'
            "DOCS_TARGET_LANGUAGES=zh-CN,ja # trailing comment\n"
            "EMPTY_VALUE=\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("DOCS_AI_OPENAI_BASEURL", raising=False)
    monkeypatch.delenv("DOCS_AI_OPENAI_MODEL", raising=False)
    monkeypatch.delenv("DOCS_TARGET_LANGUAGES", raising=False)
    monkeypatch.delenv("EMPTY_VALUE", raising=False)

    loaded_count = load_dotenv_file(dotenv_path)

    assert loaded_count == 4
    assert os.getenv("DOCS_AI_OPENAI_BASEURL") == "https://api.example.com/v1"
    assert os.getenv("DOCS_AI_OPENAI_MODEL") == "gpt-4o-mini"
    assert os.getenv("DOCS_TARGET_LANGUAGES") == "zh-CN,ja"
    assert os.getenv("EMPTY_VALUE") == ""


def test_load_dotenv_file_does_not_override_existing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep existing env vars unless override_existing=True."""
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("DOCS_AI_OPENAI_MODEL=gpt-from-file\n", encoding="utf-8")

    monkeypatch.setenv("DOCS_AI_OPENAI_MODEL", "gpt-existing")
    loaded_count = load_dotenv_file(dotenv_path, override_existing=False)

    assert loaded_count == 0
    assert os.getenv("DOCS_AI_OPENAI_MODEL") == "gpt-existing"
