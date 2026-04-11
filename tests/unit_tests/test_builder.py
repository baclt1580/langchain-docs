"""Tests for the DocumentationBuilder class.

This module contains comprehensive tests for the DocumentationBuilder class,
covering all methods and edge cases including file extension handling,
directory structure preservation, and error conditions.
"""

from pathlib import Path

import pytest

from pipeline.core.builder import DocumentationBuilder
from tests.unit_tests.utils import File, file_system


def test_builder_initialization() -> None:
    """Test DocumentationBuilder initialization.

    Verifies that the builder is correctly initialized with the provided
    source and build directories, and that the copy_extensions set contains
    the expected file extensions.
    """
    with file_system([]) as fs:
        builder = DocumentationBuilder(fs.src_dir, fs.build_dir)
        assert builder.src_dir == fs.src_dir
        assert builder.build_dir == fs.build_dir
        assert builder.copy_extensions == {
            ".mdx",
            ".md",
            ".json",
            ".svg",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".yml",
            ".yaml",
            ".css",
            ".js",
            ".jsx",
            ".tsx",
            ".txt",
            ".woff2",
            ".woff",
            ".ttf",
            ".html",
        }


def test_build_all_empty_directory() -> None:
    """Test building from an empty directory.

    Verifies that the builder handles empty source directories correctly.
    """
    with file_system([]) as fs:
        builder = DocumentationBuilder(fs.src_dir, fs.build_dir)
        builder.build_all()
        assert not fs.list_build_files()


def test_build_all_supported_files() -> None:
    """Test building all supported file types.

    Verifies that the builder correctly copies all supported file types
    while maintaining directory structure.
    """
    files = [
        # LangGraph (oss) files - both Python and JavaScript versions
        File(path="oss/index.mdx", content="# Welcome"),
        File(path="oss/config.json", content='{"name": "test"}'),
        File(path="oss/guides/setup.md", content="# Setup Guide"),
        # LangSmith files
        File(path="langsmith/home.mdx", content="# LangSmith"),
        # Shared files
        File(path="images/logo.png", bytes=b"PNG_DATA"),
        File(path="docs.json", content='{"name": "test"}'),
    ]

    with file_system(files) as fs:
        builder = DocumentationBuilder(fs.src_dir, fs.build_dir)
        builder.build_all()

        # Verify all files were copied with correct structure
        build_files = set(fs.list_build_files())

        # Python version of LangGraph files
        assert Path("oss/python/index.mdx") in build_files
        assert Path("oss/python/config.json") in build_files
        assert Path("oss/python/guides/setup.mdx") in build_files

        # JavaScript version of LangGraph files
        assert Path("oss/javascript/index.mdx") in build_files
        assert Path("oss/javascript/config.json") in build_files
        assert Path("oss/javascript/guides/setup.mdx") in build_files

        # LangSmith files
        assert Path("langsmith/home.mdx") in build_files

        # Shared files
        assert Path("images/logo.png") in build_files
        assert Path("docs.json") in build_files

        # Total number of files should be:
        # - 3 files * 2 versions (Python/JavaScript) for LangGraph
        # - 1 file for LangSmith
        # - 2 shared files
        assert len(build_files) == 9


def test_build_all_unsupported_files() -> None:
    """Test building with unsupported file types.

    Verifies that the builder skips unsupported file types.
    """
    files = [
        # LangGraph files with supported and unsupported types
        File(
            path="oss/index.mdx",
            content="# Welcome",
        ),
        File(
            path="oss/ignored.csv",
            content="This should be ignored",
        ),
        File(
            path="oss/data.csv",
            content="col1,col2\n1,2",
        ),
        # LangSmith files with supported and unsupported types
        File(
            path="langsmith/home.mdx",
            content="# Guide",
        ),
        File(
            path="langsmith/data.csv",
            content="This should be ignored",
        ),
        # Shared files with supported and unsupported types
        File(
            path="images/logo.png",
            bytes=b"PNG_DATA",
        ),
        File(
            path="notes.csv",
            content="col1,col2\n1,2",
        ),
        File(
            path="docs.json",
            content='{"name": "test"}',
        ),
    ]

    with file_system(files) as fs:
        builder = DocumentationBuilder(fs.src_dir, fs.build_dir)
        builder.build_all()

        # Verify only supported files were copied
        build_files = set(fs.list_build_files())

        # Python version of LangGraph files (only .mdx)
        assert Path("oss/python/index.mdx") in build_files
        assert Path("oss/python/ignored.csv") not in build_files
        assert Path("oss/python/data.csv") not in build_files

        # JavaScript version of LangGraph files (only .mdx)
        assert Path("oss/javascript/index.mdx") in build_files
        assert Path("oss/javascript/ignored.csv") not in build_files
        assert Path("oss/javascript/data.csv") not in build_files

        # LangSmith files (only .mdx)
        assert Path("langsmith/home.mdx") in build_files
        assert Path("langsmith/data.csv") not in build_files

        # Shared files (only .png)
        assert Path("images/logo.png") in build_files
        assert Path("notes.csv") not in build_files
        assert Path("docs.json") in build_files

        # Total number of files should be:
        # - 1 file * 2 versions (Python/JavaScript) for LangGraph
        # - 1 file for LangSmith
        # - 2 shared files
        assert len(build_files) == 5


def test_build_single_file() -> None:
    """Test building a single file.

    Verifies that the builder correctly copies a single file
    when requested.
    """
    files = [
        File(
            path="index.mdx",
            content="# Welcome",
        ),
        File(
            path="config.json",
            content='{"name": "test"}',
        ),
    ]

    with file_system(files) as fs:
        builder = DocumentationBuilder(fs.src_dir, fs.build_dir)
        builder.build_file(fs.src_dir / "index.mdx")

        # Verify only the requested file was copied
        build_files = fs.list_build_files()
        assert len(build_files) == 1
        assert Path("index.mdx") in build_files
        assert not fs.build_file_exists("config.json")


def test_build_multiple_files() -> None:
    """Test building multiple specific files.

    Verifies that the builder correctly copies multiple specified files
    while maintaining directory structure.
    """
    files = [
        File(
            path="index.mdx",
            content="# Welcome",
        ),
        File(
            path="config.json",
            content='{"name": "test"}',
        ),
        File(
            path="guides/setup.md",
            content="# Setup Guide",
        ),
    ]

    with file_system(files) as fs:
        builder = DocumentationBuilder(fs.src_dir, fs.build_dir)
        builder.build_files(
            [
                fs.src_dir / "index.mdx",
                fs.src_dir / "guides/setup.md",
            ],
        )

        # Verify only specified files were copied
        build_files = fs.list_build_files()
        assert len(build_files) == 2
        assert Path("index.mdx") in build_files
        assert Path("guides/setup.mdx") in build_files
        assert not fs.build_file_exists("config.json")


def test_build_nonexistent_file() -> None:
    """Test building a nonexistent file.

    Verifies that the builder handles attempts to build
    nonexistent files gracefully.
    """
    with file_system([]) as fs:
        builder = DocumentationBuilder(fs.src_dir, fs.build_dir)
        with pytest.raises(AssertionError):
            builder.build_file(fs.src_dir / "nonexistent.md")


def test_build_all_prefers_configured_i18n_overlay() -> None:
    """Test that the configured site translation overlays live build output."""
    files = [
        File(path="docs.json", content='{"name": "test"}'),
        File(path="index.mdx", content="# English home"),
        File(path="oss/overview.mdx", content="# English overview"),
        File(path="i18n/config.json", content='{"siteLanguage": "zh-CN"}'),
        File(path="i18n/zh-CN/index.mdx", content="# 中文首页"),
        File(path="i18n/zh-CN/oss/overview.mdx", content="# 中文总览"),
    ]

    with file_system(files) as fs:
        builder = DocumentationBuilder(fs.src_dir, fs.build_dir)
        builder.build_all()

        home_output = (fs.build_dir / "index.mdx").read_text(encoding="utf-8")
        overview_output = (fs.build_dir / "oss" / "python" / "overview.mdx").read_text(
            encoding="utf-8"
        )

        assert "中文首页" in home_output
        assert "中文总览" in overview_output
        assert not fs.build_file_exists("i18n/zh-CN/index.mdx")


def test_build_translated_source_file_to_live_output() -> None:
    """Test that building a translated file writes to the live doc path."""
    files = [
        File(path="i18n/config.json", content='{"siteLanguage": "zh-CN"}'),
        File(path="langsmith/home.mdx", content="# English home"),
        File(path="i18n/zh-CN/langsmith/home.mdx", content="# 中文主页"),
    ]

    with file_system(files) as fs:
        builder = DocumentationBuilder(fs.src_dir, fs.build_dir)
        translated_path = fs.src_dir / "i18n" / "zh-CN" / "langsmith" / "home.mdx"

        builder.build_file(translated_path)

        output = (fs.build_dir / "langsmith" / "home.mdx").read_text(encoding="utf-8")
        assert "中文主页" in output
        assert not fs.build_file_exists("i18n/zh-CN/langsmith/home.mdx")


def test_deleted_translation_falls_back_to_source_content() -> None:
    """Test that deleting a translated page rebuilds the source fallback."""
    files = [
        File(path="i18n/config.json", content='{"siteLanguage": "zh-CN"}'),
        File(path="langsmith/home.mdx", content="# English home"),
        File(path="i18n/zh-CN/langsmith/home.mdx", content="# 中文主页"),
    ]

    with file_system(files) as fs:
        builder = DocumentationBuilder(fs.src_dir, fs.build_dir)
        translated_path = fs.src_dir / "i18n" / "zh-CN" / "langsmith" / "home.mdx"

        builder.build_all()
        translated_path.unlink()
        builder.handle_deleted_file(translated_path)

        output = (fs.build_dir / "langsmith" / "home.mdx").read_text(encoding="utf-8")
        assert "English home" in output


def test_build_all_rewrites_chat_embed_to_hosted_url() -> None:
    """Test that the copied chat embed avoids the local port 4100 dependency."""
    files = [
        File(path="docs.json", content='{"name": "test"}'),
    ]

    with file_system(files) as fs:
        pkg_dist = (
            fs.src_dir.parent
            / "node_modules"
            / "@langchain"
            / "docs-sandbox"
            / "dist"
        )
        pkg_dist.mkdir(parents=True, exist_ok=True)
        (pkg_dist / "ChatLangChainEmbed.js").write_text(
            'const local = "http://localhost:4100";\n'
            'const prod = "https://ui-patterns.langchain.com/react";\n',
            encoding="utf-8",
        )

        builder = DocumentationBuilder(fs.src_dir, fs.build_dir)
        builder.build_all()

        output = (fs.build_dir / "ChatLangChainEmbed.js").read_text(encoding="utf-8")
        assert "http://localhost:4100" not in output
        assert "https://ui-patterns.langchain.com/react" in output


def test_build_all_restores_missing_code_fence_for_translated_snippet() -> None:
    """Test that translated snippets recover the source code fence when missing."""
    files = [
        File(path="docs.json", content='{"name": "test"}'),
        File(path="i18n/config.json", content='{"siteLanguage": "zh-CN"}'),
        File(
            path="snippets/backend-composite-js.mdx",
            content=(
                "```typescript\n"
                'import { createDeepAgent } from "deepagents";\n'
                "```\n"
            ),
        ),
        File(
            path="i18n/zh-CN/snippets/backend-composite-js.mdx",
            content='import { createDeepAgent } from "deepagents";\n',
        ),
    ]

    with file_system(files) as fs:
        builder = DocumentationBuilder(fs.src_dir, fs.build_dir)
        builder.build_all()

        output = (fs.build_dir / "snippets" / "backend-composite-js.mdx").read_text(
            encoding="utf-8"
        )
        assert output.startswith("```typescript\n")
        assert output.rstrip().endswith("```")


def test_build_all_logs_translated_snippet_missing_code_fence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that missing snippet fences are reported during the build."""
    files = [
        File(path="docs.json", content='{"name": "test"}'),
        File(path="i18n/config.json", content='{"siteLanguage": "zh-CN"}'),
        File(
            path="snippets/backend-state-js.mdx",
            content="```typescript\nconst agent = createDeepAgent();\n```\n",
        ),
        File(
            path="i18n/zh-CN/snippets/backend-state-js.mdx",
            content="const agent = createDeepAgent();\n",
        ),
    ]

    with file_system(files) as fs:
        builder = DocumentationBuilder(fs.src_dir, fs.build_dir)

        with caplog.at_level("WARNING"):
            builder.build_all()

    assert "Snippet missing code fence" in caplog.text
    assert "i18n" in caplog.text
    assert "backend-state-js.mdx" in caplog.text
