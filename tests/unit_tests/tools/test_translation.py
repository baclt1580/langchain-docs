"""Tests for translation utilities."""

import os
from pathlib import Path

import pytest

from pipeline.tools.translation import (
    _TRANSLATION_SYSTEM_PROMPT,
    OpenAICompatibleTranslator,
    TranslationManifest,
    _chunk_markdown_for_translation,
    compute_sha256,
    load_dotenv_file,
    parse_target_languages,
)


class _FakeEncoding:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


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


def test_chunk_markdown_prefers_heading_boundaries() -> None:
    """Split large documents at heading boundaries before finer block splits."""
    content = (
        "---\n"
        "title: Sample\n"
        "---\n\n"
        "# Alpha\n\n"
        "Alpha paragraph.\n\n"
        "## Beta\n\n"
        "Beta paragraph.\n\n"
        "## Gamma\n\n"
        "Gamma paragraph.\n"
    )

    chunks = _chunk_markdown_for_translation(
        content,
        max_content_tokens=55,
        count_tokens=len,
    )

    assert chunks == [
        "---\ntitle: Sample\n---\n\n# Alpha\n\nAlpha paragraph.\n\n",
        "## Beta\n\nBeta paragraph.\n\n## Gamma\n\nGamma paragraph.\n",
    ]
    assert "".join(chunks) == content


def test_chunk_markdown_falls_back_to_block_boundaries_without_splitting_fences(
) -> None:
    """Keep fenced blocks intact when a section must be split by block."""
    content = (
        "# Heading\n\n"
        "First paragraph is long enough to force a split.\n\n"
        "```python\n"
        "print('hello')\n"
        "```\n\n"
        "Second paragraph is also long enough to stay on its own.\n"
    )

    chunks = _chunk_markdown_for_translation(
        content,
        max_content_tokens=70,
        count_tokens=len,
    )

    assert chunks == [
        "# Heading\n\nFirst paragraph is long enough to force a split.\n\n",
        "```python\nprint('hello')\n```\n\n",
        "Second paragraph is also long enough to stay on its own.\n",
    ]
    assert "".join(chunks) == content


def test_chunk_markdown_raises_for_oversized_unsplittable_block() -> None:
    """Raise a clear error when a protected block still exceeds the budget."""
    content = "```python\n" + ("print('x')\n" * 20) + "```\n"

    with pytest.raises(ValueError, match="DOCS_TRANSLATE_MAX_PROMPT_TOKENS"):
        _chunk_markdown_for_translation(
            content,
            max_content_tokens=40,
            count_tokens=len,
        )


def test_translate_markdown_translates_each_chunk_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translate each computed chunk independently and concatenate results."""
    monkeypatch.setattr(
        "pipeline.tools.translation._resolve_token_encoding",
        lambda _model: _FakeEncoding(),
    )
    translator = OpenAICompatibleTranslator(
        base_url="https://api.example.com/v1",
        api_key="test-key",
        model="gpt-4o-mini",
        max_prompt_tokens=40,
    )
    content = (
        "# Alpha\n\n"
        "Alpha paragraph.\n\n"
        "## Beta\n\n"
        "Beta paragraph.\n"
    )
    seen_chunks: list[str] = []

    monkeypatch.setattr(translator, "_count_tokens", len)
    monkeypatch.setattr(translator, "_estimate_prompt_overhead", lambda _lang: 0)

    def fake_translate_chunk(chunk: str, target_language: str) -> str:
        seen_chunks.append(chunk)
        assert target_language == "Chinese (Simplified)"
        return chunk.upper()

    monkeypatch.setattr(translator, "_translate_chunk", fake_translate_chunk)

    translated = translator.translate_markdown(
        content,
        target_language="Chinese (Simplified)",
    )

    assert seen_chunks == [
        "# Alpha\n\nAlpha paragraph.\n\n",
        "## Beta\n\nBeta paragraph.\n",
    ]
    assert translated == content.upper()


def test_translation_prompts_preserve_mdx_structure_rules() -> None:
    """Keep strict MDX structure-preservation rules in the translation prompts."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "pipeline.tools.translation._resolve_token_encoding",
        lambda _model: _FakeEncoding(),
    )
    translator = OpenAICompatibleTranslator(
        base_url="https://api.example.com/v1",
        api_key="test-key",
        model="gpt-4o-mini",
    )

    user_prompt = translator._build_user_prompt(
        "```typescript\nconst x = 1;\n```\n",
        "Chinese (Simplified)",
    )

    assert "Preserve the exact document structure" in _TRANSLATION_SYSTEM_PROMPT
    assert "Never add, remove, unwrap, or rebalance Markdown or MDX wrappers" in (
        _TRANSLATION_SYSTEM_PROMPT
    )
    assert "Keep frontmatter keys, delimiters, and field order unchanged." in (
        user_prompt
    )
    assert "Preserve all Markdown fences exactly" in user_prompt
    assert "Do not remove or rewrite import/export lines" in user_prompt
    assert "Do not translate code, inline code, identifiers" in user_prompt
    assert "translation must remain a fenced code block" in user_prompt
    monkeypatch.undo()
