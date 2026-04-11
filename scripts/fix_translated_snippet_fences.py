"""Restore missing code fences in translated snippet MDX files.

This script compares translated snippet files under `src/i18n/<lang>/snippets/`
against the source snippet files under `src/snippets/`. When a translated file
contains only the inner code content but the source file is fenced as a markdown
code block, the script restores the missing fence in the translated file.

Usage:
    python scripts/fix_translated_snippet_fences.py
    python scripts/fix_translated_snippet_fences.py --lang zh-CN --write
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Restore missing code fences in translated snippet MDX files."
    )
    parser.add_argument(
        "--lang",
        default="zh-CN",
        help="Translated language directory under src/i18n/ (default: zh-CN)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write changes back to disk. Without this flag, runs in preview mode.",
    )
    return parser.parse_args()


def get_source_fence(source_content: str) -> tuple[str, str] | None:
    """Return the opening fence marker and suffix from a fenced source snippet."""
    stripped = source_content.lstrip()
    first_line = stripped.splitlines()[0] if stripped else ""
    match = re.match(r"^(```|~~~)([^\r\n]*)$", first_line)
    if match is None:
        return None
    return match.group(1), match.group(2)


def restore_fence(source_content: str, translated_content: str) -> str | None:
    """Restore the source code fence onto a translated snippet when missing."""
    if translated_content.lstrip().startswith(("```", "~~~")):
        return None

    source_fence = get_source_fence(source_content)
    if source_fence is None:
        return None

    normalized_content = translated_content.strip("\n")
    if not normalized_content:
        return None

    fence, suffix = source_fence
    return f"{fence}{suffix}\n{normalized_content}\n{fence}\n"


def main() -> int:
    """Find and optionally fix translated snippet files missing code fences."""
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    source_snippets_dir = repo_root / "src" / "snippets"
    translated_snippets_dir = repo_root / "src" / "i18n" / args.lang / "snippets"

    if not translated_snippets_dir.is_dir():
        print(f"Translated snippets directory not found: {translated_snippets_dir}")
        return 1

    changed_files: list[Path] = []

    for translated_file in sorted(translated_snippets_dir.rglob("*.mdx")):
        relative_path = translated_file.relative_to(translated_snippets_dir)
        source_file = source_snippets_dir / relative_path
        if not source_file.is_file():
            continue

        source_content = source_file.read_text(encoding="utf-8")
        translated_content = translated_file.read_text(encoding="utf-8")

        restored_content = restore_fence(source_content, translated_content)
        if restored_content is None or restored_content == translated_content:
            continue

        changed_files.append(translated_file)
        print(f"Fixing missing code fence: {translated_file.relative_to(repo_root)}")
        if args.write:
            translated_file.write_text(restored_content, encoding="utf-8")

    if not changed_files:
        print("No translated snippet fences needed fixing.")
        return 0

    mode = "Updated" if args.write else "Would update"
    print(f"{mode} {len(changed_files)} translated snippet file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
