"""Tests for the file watcher functionality."""

import asyncio
from pathlib import Path
from unittest.mock import Mock

from pipeline.core.builder import DocumentationBuilder
from pipeline.core.watcher import DocsFileHandler
from tests.unit_tests.utils import file_system


def test_should_ignore_backup_files() -> None:
    """Test that backup files with ~ suffix are properly ignored."""
    with file_system([]) as fs:
        builder = DocumentationBuilder(fs.src_dir, fs.build_dir)
        event_queue = asyncio.Queue()
        handler = DocsFileHandler(builder, event_queue, Mock())

        backup_files = [
            Path("langchain-models.mdx~"),
            Path("src/oss/langchain/models.mdx~"),
            Path("documentation.md~"),
            Path("config.json~"),
        ]

        for file_path in backup_files:
            assert handler._should_ignore_file(file_path), (
                f"Should ignore backup file: {file_path}"
            )


def test_should_ignore_temporary_files() -> None:
    """Test that various temporary files are properly ignored."""
    with file_system([]) as fs:
        builder = DocumentationBuilder(fs.src_dir, fs.build_dir)
        event_queue = asyncio.Queue()
        handler = DocsFileHandler(builder, event_queue, Mock())

        temp_files = [
            Path("file.bak"),
            Path("file.orig"),
            Path(".file.tmp"),
            Path(".file.swp"),
            Path("document.bak"),
            Path("backup.orig"),
        ]

        for file_path in temp_files:
            assert handler._should_ignore_file(file_path), (
                f"Should ignore temporary file: {file_path}"
            )


def test_should_not_ignore_valid_files() -> None:
    """Test that valid documentation files are NOT ignored."""
    with file_system([]) as fs:
        builder = DocumentationBuilder(fs.src_dir, fs.build_dir)
        event_queue = asyncio.Queue()
        handler = DocsFileHandler(builder, event_queue, Mock())

        valid_files = [
            Path("langchain-models.mdx"),
            Path("documentation.md"),
            Path("config.json"),
            Path("image.png"),
            Path("script.js"),
            Path("styles.css"),
            Path("data.yml"),
            Path("info.yaml"),
            Path("icon.svg"),
            Path("photo.jpg"),
            Path("picture.jpeg"),
            Path("animation.gif"),
        ]

        for file_path in valid_files:
            assert not handler._should_ignore_file(file_path), (
                f"Should NOT ignore valid file: {file_path}"
            )


def test_edge_cases() -> None:
    """Test edge cases for file filtering."""
    with file_system([]) as fs:
        builder = DocumentationBuilder(fs.src_dir, fs.build_dir)
        event_queue = asyncio.Queue()
        handler = DocsFileHandler(builder, event_queue, Mock())

        edge_cases = [
            (Path("file~name.mdx"), False),
            (Path("test~123.md"), False),
            (Path("file~"), True),
            (Path("name.ext~"), True),
            (Path(".gitignore"), False),
            (Path(".config.json"), False),
            (Path("file.backup.bak"), True),
            (Path("file.old.orig"), True),
        ]

        for file_path, should_ignore in edge_cases:
            result = handler._should_ignore_file(file_path)
            assert result == should_ignore, (
                f"File {file_path}: expected ignore={should_ignore}, got {result}"
            )
