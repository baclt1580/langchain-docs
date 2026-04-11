"""Documentation builder implementation."""

import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import ClassVar

import yaml
from tqdm import tqdm

from pipeline.preprocessors import preprocess_markdown

_IS_CI = os.environ.get("CI", "").lower() in ("true", "1")
_I18N_TRANSLATED_PATH_PARTS = 3
_CHAT_LANGCHAIN_LOCAL_EMBED_URL = "http://localhost:4100"
_CHAT_LANGCHAIN_PROD_EMBED_URL = "https://ui-patterns.langchain.com/react"

logger = logging.getLogger(__name__)


class DocumentationBuilder:
    """Builds documentation from source files to build directory.

    This class handles the process of copying supported documentation files
    from a source directory to a build directory, maintaining the directory
    structure and preserving file metadata.

    Attributes:
        src_dir: Path to the source directory containing documentation files.
        build_dir: Path to the build directory where files will be copied.
        copy_extensions: Set of file extensions that are supported for copying.
    """

    def __init__(self, src_dir: Path, build_dir: Path) -> None:
        """Initialize the DocumentationBuilder.

        Args:
            src_dir: Path to the source directory containing documentation files.
            build_dir: Path to the build directory where files will be copied.
        """
        self.src_dir = src_dir
        self.build_dir = build_dir

        # File extensions to copy directly
        self.copy_extensions: set[str] = {
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

        # Mapping of language codes to full names for URLs
        self.language_url_names = {
            "python": "python",
            "js": "javascript",
        }
        self.i18n_dir = self.src_dir / "i18n"
        self.i18n_config_path = self.i18n_dir / "config.json"
        self.site_language: str | None = None
        self.site_translation_root: Path | None = None
        self._refresh_site_language()

    def _refresh_site_language(self) -> None:
        """Load the configured site language and cache its translation root."""
        self.site_language = self._load_site_language()
        self.site_translation_root = (
            self.i18n_dir / self.site_language if self.site_language else None
        )

    def _load_site_language(self) -> str | None:
        """Load the language code that should be published at the site root."""
        if not self.i18n_config_path.is_file():
            return None

        try:
            raw_config = json.loads(self.i18n_config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            logger.warning(
                "Failed to load i18n site config from %s",
                self.i18n_config_path,
                exc_info=True,
            )
            return None

        if not isinstance(raw_config, dict):
            logger.warning(
                "Ignoring invalid i18n site config in %s",
                self.i18n_config_path,
            )
            return None

        raw_site_language = raw_config.get(
            "siteLanguage",
            raw_config.get("defaultLanguage"),
        )
        if not isinstance(raw_site_language, str):
            return None

        site_language = raw_site_language.strip()
        if not site_language or site_language.lower() == "en":
            return None

        return site_language

    def _logical_relative_path(self, file_path: Path) -> Path | None:
        """Map a source file to the live logical path used for the build output."""
        relative_path = file_path.absolute().relative_to(self.src_dir.absolute())
        if not relative_path.parts:
            return None

        if relative_path.parts[0] != "i18n":
            return relative_path

        if (
            self.site_language is None
            or len(relative_path.parts) < _I18N_TRANSLATED_PATH_PARTS
        ):
            return None

        if relative_path.parts[1] != self.site_language:
            return None

        return Path(*relative_path.parts[2:])

    def _resolve_source_for_logical_path(self, logical_relative: Path) -> Path | None:
        """Resolve the best source file for a logical doc path."""
        if self.site_translation_root is not None:
            translated_path = self.site_translation_root / logical_relative
            if translated_path.is_file():
                return translated_path

        source_path = self.src_dir / logical_relative
        if source_path.is_file():
            return source_path

        return None

    def _build_logical_file(self, file_path: Path, logical_relative: Path) -> None:
        """Build a live doc using its resolved logical path."""
        if logical_relative.parts[0] == "oss":
            self._build_oss_file(file_path, logical_relative)
        elif logical_relative.parts[0] == "langsmith":
            self._build_unversioned_file(file_path, logical_relative)
        elif self.is_shared_relative_path(logical_relative):
            self._build_shared_file(file_path, logical_relative)
        else:
            self._build_simple_file(file_path, logical_relative)

    def _collect_content_logical_paths(self, source_dir: str) -> list[Path]:
        """Collect logical content paths from the source tree and active overlay."""
        logical_paths: set[Path] = set()
        roots = [(self.src_dir / source_dir, Path(source_dir))]

        if self.site_translation_root is not None:
            roots.append((self.site_translation_root / source_dir, Path(source_dir)))

        for root_path, logical_root in roots:
            if not root_path.exists():
                continue

            for file_path in root_path.rglob("*"):
                if not file_path.is_file():
                    continue

                logical_relative = logical_root / file_path.relative_to(root_path)
                if not self.is_shared_relative_path(logical_relative):
                    logical_paths.add(logical_relative)

        return sorted(logical_paths)

    def _collect_shared_logical_paths(self) -> list[Path]:
        """Collect shared logical paths from the source tree and active overlay."""
        logical_paths: set[Path] = set()

        for file_path in self.src_dir.rglob("*"):
            if not file_path.is_file():
                continue

            relative_path = file_path.relative_to(self.src_dir)
            if relative_path.parts and relative_path.parts[0] == "i18n":
                continue

            if self.is_shared_relative_path(relative_path):
                logical_paths.add(relative_path)

        if (
            self.site_translation_root is not None
            and self.site_translation_root.exists()
        ):
            for file_path in self.site_translation_root.rglob("*"):
                if not file_path.is_file():
                    continue

                logical_relative = file_path.relative_to(self.site_translation_root)
                if self.is_shared_relative_path(logical_relative):
                    logical_paths.add(logical_relative)

        return sorted(logical_paths)

    def _finalize_output_path(self, output_path: Path) -> Path:
        """Normalize output paths after builder-side extension rewrites."""
        if output_path.name == "docs.yml" and output_path.suffix.lower() in {
            ".yml",
            ".yaml",
        }:
            return output_path.with_suffix(".json")

        if output_path.suffix.lower() == ".md":
            return output_path.with_suffix(".mdx")

        return output_path

    def _output_paths_for_logical_path(self, logical_relative: Path) -> list[Path]:
        """Return every live build output generated from a logical source path."""
        if not logical_relative.parts:
            return []

        if logical_relative.parts[0] == "oss" and not self.is_shared_relative_path(
            logical_relative
        ):
            if len(logical_relative.parts) > 1 and logical_relative.parts[1] in {
                "python",
                "javascript",
            }:
                sub_path = Path(*logical_relative.parts[2:])
                return [
                    self._finalize_output_path(
                        self.build_dir / "oss" / logical_relative.parts[1] / sub_path
                    ),
                ]

            sub_path = Path(*logical_relative.parts[1:])
            return [
                self._finalize_output_path(
                    self.build_dir / "oss" / "python" / sub_path
                ),
                self._finalize_output_path(
                    self.build_dir / "oss" / "javascript" / sub_path
                ),
            ]

        return [self._finalize_output_path(self.build_dir / logical_relative)]

    def get_output_paths_for_source_file(self, file_path: Path) -> list[Path]:
        """Return every live build output generated for a source file."""
        logical_relative = self._logical_relative_path(file_path)
        if logical_relative is None:
            return []

        return self._output_paths_for_logical_path(logical_relative)

    def handle_deleted_file(self, file_path: Path) -> None:
        """Update build output after a source file is deleted."""
        if file_path.absolute() == self.i18n_config_path.absolute():
            self._refresh_site_language()
            self.build_all()
            return

        logical_relative = self._logical_relative_path(file_path)
        if logical_relative is None:
            return

        source_path = self._resolve_source_for_logical_path(logical_relative)
        if source_path is not None:
            self._build_logical_file(source_path, logical_relative)
            return

        for output_path in self._output_paths_for_logical_path(logical_relative):
            output_path.unlink(missing_ok=True)

    def build_all(self) -> None:
        """Build all documentation files from source to build directory.

        This method clears the build directory and creates version-specific builds
        for both Python and JavaScript documentation.

        The process includes:
        1. Clearing the existing build directory
        2. Building Python version with python/ prefix
        3. Building JavaScript version with javascript/ prefix
        4. Copying shared files (images, configs, etc.)

        Displays:
            Progress bars showing build progress for each version.
        """
        logger.debug(
            "Building versioned documentation from %s to %s",
            self.src_dir,
            self.build_dir,
        )

        # Clear build directory
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
        self.build_dir.mkdir(parents=True, exist_ok=True)

        # Build LangGraph versioned content (oss/ -> oss/python/ and oss/javascript/)
        logger.debug("Building LangGraph Python version...")
        self._build_langgraph_version("oss/python", "python")

        logger.debug("Building LangGraph JavaScript version...")
        self._build_langgraph_version("oss/javascript", "js")

        logger.debug("Building LangSmith content...")
        self._build_unversioned_content("langsmith", "langsmith")

        # Copy shared files (docs.json, images, etc.)
        logger.debug("Copying shared files...")
        self._copy_shared_files()

        # Copy snippet components from @langchain/docs-sandbox npm package
        logger.debug("Copying npm snippet components...")
        self._copy_npm_snippets()

        logger.debug("New structure build complete")

    def _convert_yaml_to_json(self, yaml_file_path: Path, output_path: Path) -> None:
        """Convert a YAML file to JSON format.

        This method loads a docs.yml file using YAML safe_load and writes
        the corresponding docs.json file to the build directory.

        Args:
            yaml_file_path: Path to the source YAML file.
            output_path: Path where the JSON file should be written.
        """
        try:
            # Load YAML content
            with yaml_file_path.open("r", encoding="utf-8") as yaml_file:
                yaml_content = yaml.safe_load(yaml_file)

            # Convert output path from .yml to .json
            json_output_path = output_path.with_suffix(".json")

            # Write JSON content
            with json_output_path.open("w", encoding="utf-8") as json_file:
                json.dump(yaml_content, json_file, indent=2, ensure_ascii=False)

        except yaml.YAMLError:
            logger.exception("Failed to parse YAML file %s", yaml_file_path)
            raise
        except Exception:
            logger.exception("Failed to convert %s to JSON", yaml_file_path)
            raise

    def _rewrite_oss_links(self, content: str, target_language: str | None) -> str:
        """Rewrite /oss/ links to include the target language.

        Args:
            content: The markdown content to process.
            target_language: Target language ("python" or "js") or None to skip.

        Returns:
            Content with rewritten links.
        """
        if not target_language:
            return content

        def rewrite_link(match: re.Match) -> str:
            """Rewrite a single link match."""
            pre = match.group(1)  # Everything before the URL
            url = match.group(2)  # The URL
            post = match.group(3)  # Everything after the URL

            # Only rewrite absolute /oss/ paths that don't contain 'images'
            if url.startswith("/oss/") and "images" not in url:
                parts = url.split("/")
                # Insert full language name after "oss"
                parts.insert(2, self.language_url_names[target_language])
                url = "/".join(parts)

            return f"{pre}{url}{post}"

        # Match markdown links and HTML links/anchors
        # This handles both [text](/oss/path) and <a href="/oss/path">
        pattern = r'(\[.*?\]\(|\bhref="|")(/oss/[^")\s]+)([")\s])'
        return re.sub(pattern, rewrite_link, content)

    def _add_suggested_edits_link(self, content: str, input_path: Path) -> str:
        """Add 'Edit Source' link to the end of markdown content.

        This method appends GitHub links with icons pointing to the source file,
        but only for files that are within the src/ directory.

        Args:
            content: The markdown content to process.
            input_path: Path to the source file.

        Returns:
            The content with the source links appended (if applicable).
        """
        try:
            # Only add links for files in the src/ directory
            relative_path = input_path.absolute().relative_to(self.src_dir.absolute())

            # Do not add source links on the home page (root index.mdx)
            if relative_path.parts == ("index.mdx",):
                return content

            # Construct the GitHub URLs
            edit_url = (
                f"https://github.com/langchain-ai/docs/edit/main/src/{relative_path}"
            )
            issue_url = "https://github.com/langchain-ai/docs/issues/new/choose"

            # Create the callout section with Mintlify Callout component
            source_links_section = (
                "\n\n---\n\n"
                '<div className="source-links">\n'
                '<Callout icon="edit">\n'
                f"    [Edit this page on GitHub]({edit_url}) "
                f"or [file an issue]({issue_url}).\n"
                "</Callout>\n"
                '<Callout icon="terminal-2">\n'
                "    [Connect these docs](/use-these-docs) to Claude, VSCode, and more via MCP for real-time answers.\n"  # noqa: E501
                "</Callout>\n"
                "</div>\n"
            )

            # Append to content
            return content.rstrip() + source_links_section

        except ValueError:
            # File is not within src_dir, return content unchanged
            return content
        except Exception:
            logger.exception("Failed to add source links for %s", input_path)
            # Return original content if there's an error
            return content

    def _process_markdown_content(
        self, content: str, file_path: Path, target_language: str | None = None
    ) -> str:
        """Process markdown content with preprocessing.

        This method applies preprocessing (cross-reference resolution and
        conditional blocks) to markdown content.

        Args:
            content: The markdown content to process.
            file_path: Path to the source file (for error reporting).
            target_language: Target language for conditional blocks ("python" or "js").

        Returns:
            The processed markdown content.
        """
        try:
            # First apply standard markdown preprocessing
            content = preprocess_markdown(
                content, file_path, target_language=target_language
            )

            # Then rewrite /oss/ links to include language
            return self._rewrite_oss_links(content, target_language)

        except Exception:
            logger.exception("Failed to process markdown content from %s", file_path)
            raise

    def _process_markdown_file(
        self, input_path: Path, output_path: Path, target_language: str | None = None
    ) -> None:
        """Process a markdown file with preprocessing and copy to output.

        This method reads a markdown file, applies preprocessing (cross-reference
        resolution and conditional blocks), and writes the processed content to
        the output path.

        Args:
            input_path: Path to the source markdown file.
            output_path: Path where the processed file should be written.
            target_language: Target language for conditional blocks ("python" or "js").
        """
        try:
            # Read the source markdown content
            with input_path.open("r", encoding="utf-8") as f:
                content = f.read()

            # Apply markdown preprocessing
            processed_content = self._process_markdown_content(
                content, input_path, target_language
            )

            # Add "Edit Source" link for files in src/ directory
            processed_content = self._add_suggested_edits_link(
                processed_content, input_path
            )

            # Convert .md to .mdx if needed
            if input_path.suffix.lower() == ".md":
                output_path = output_path.with_suffix(".mdx")

            # Write the processed content
            with output_path.open("w", encoding="utf-8") as f:
                f.write(processed_content)

        except Exception:
            logger.exception("Failed to process markdown file %s", input_path)
            raise

    def build_file(self, file_path: Path) -> None:
        """Build a single file to the appropriate location(s) in the build directory.

        This method handles versioned building for OSS content (creates both Python
        and JavaScript versions) and single-version building for other content.
        The directory structure and version-specific preprocessing are preserved.

        Args:
            file_path: Path to the source file to be built. Must be within
                the source directory.

        Raises:
            AssertionError: If the file does not exist.
        """
        if not file_path.is_file():
            msg = f"File does not exist: {file_path} this is likely a programming error"
            raise AssertionError(msg)

        if file_path.absolute() == self.i18n_config_path.absolute():
            self._refresh_site_language()
            self.build_all()
            return

        logical_relative = self._logical_relative_path(file_path)
        if logical_relative is None:
            return

        source_path = self._resolve_source_for_logical_path(logical_relative)
        if source_path is None:
            return

        self._build_logical_file(source_path, logical_relative)

    def _build_oss_file(self, file_path: Path, relative_path: Path) -> None:
        """Build an OSS file for both Python and JavaScript versions.

        Args:
            file_path: Path to the source file.
            relative_path: Relative path from src_dir.
        """
        # Skip shared files - they're handled separately
        if self.is_shared_file(file_path):
            self._build_shared_file(file_path, relative_path)
            return

        # Build for both Python and JavaScript versions
        oss_relative = relative_path.relative_to(Path("oss"))  # Remove 'oss/' prefix

        # Build Python version
        python_output = self.build_dir / "oss" / "python" / oss_relative
        if self._build_single_file_to_path(file_path, python_output, "python"):
            logger.debug("Built Python version: oss/python/%s", oss_relative)

        # Build JavaScript version
        js_output = self.build_dir / "oss" / "javascript" / oss_relative
        if self._build_single_file_to_path(file_path, js_output, "js"):
            logger.debug("Built JavaScript version: oss/javascript/%s", oss_relative)

    def _build_unversioned_file(self, file_path: Path, relative_path: Path) -> None:
        """Build an unversioned file (langsmith).

        Args:
            file_path: Path to the source file.
            relative_path: Relative path from src_dir.
        """
        output_path = self.build_dir / relative_path
        if self._build_single_file_to_path(file_path, output_path, "python"):
            logger.debug("Built: %s", relative_path)

    def _build_shared_file(self, file_path: Path, relative_path: Path) -> None:
        """Build a shared file (images, docs.json, JS/CSS files).

        Args:
            file_path: Path to the source file.
            relative_path: Relative path from src_dir.
        """
        output_path = self.build_dir / relative_path
        if self._build_single_file_to_path(file_path, output_path, None):
            logger.debug("Built shared file: %s", relative_path)

    def _build_simple_file(self, file_path: Path, relative_path: Path) -> None:
        """Build a simple file (root-level files).

        Args:
            file_path: Path to the source file.
            relative_path: Relative path from src_dir.
        """
        output_path = self.build_dir / relative_path
        if self._build_single_file_to_path(file_path, output_path, None):
            logger.debug("Built: %s", relative_path)

    def _build_single_file_to_path(
        self, file_path: Path, output_path: Path, target_language: str | None
    ) -> bool:
        """Build a single file to a specific output path.

        Args:
            file_path: Path to the source file.
            output_path: Full output path where the file should be written.
            target_language: Target language for conditional blocks ("python" or "js").

        Returns:
            True if the file was built successfully, False if skipped.
        """
        # Skip template files
        if file_path.name == "TEMPLATE.mdx":
            return False

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle special case for docs.yml files
        if file_path.name == "docs.yml" and file_path.suffix.lower() in {
            ".yml",
            ".yaml",
        }:
            self._convert_yaml_to_json(file_path, output_path)
            return True

        # Handle supported file extensions
        if file_path.suffix.lower() in self.copy_extensions:
            # Handle markdown files with preprocessing
            if file_path.suffix.lower() in {".md", ".mdx"}:
                self._process_markdown_file(file_path, output_path, target_language)
                return True
            shutil.copy2(file_path, output_path)
            return True

        # File was skipped
        return False

    def _build_file_with_progress(self, file_path: Path, pbar: tqdm) -> bool:
        """Build a single file with progress bar integration.

        This method is similar to build_file but integrates with tqdm progress
        bar and returns a boolean result instead of printing messages.

        Args:
            file_path: Path to the source file to be built. Must be within
                the source directory.
            pbar: tqdm progress bar instance for updating the description.

        Returns:
            True if the file was copied, False if it was skipped.
        """
        logical_relative = self._logical_relative_path(file_path)
        if logical_relative is None:
            return False

        pbar.set_postfix_str(f"{logical_relative}")
        self.build_file(file_path)
        return True

    def build_files(self, file_paths: list[Path]) -> None:
        """Build specific files by copying them to the build directory.

        This method processes a list of specific files, building only those
        that exist. Shows a progress bar when processing multiple files.

        Args:
            file_paths: List of Path objects pointing to files to be built.
                Only existing files will be processed.
        """
        existing_files = list(file_paths)

        if not existing_files:
            logger.info("No files to build")
            return

        if len(existing_files) == 1:
            # For single file, just build directly without progress bar
            self.build_file(existing_files[0])
            return

        # For multiple files, show progress bar
        copied_count = 0
        skipped_count = 0

        with tqdm(
            total=len(existing_files),
            desc="Building files",
            unit="file",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            dynamic_ncols=True,
            leave=False,
            disable=_IS_CI,
        ) as pbar:
            for file_path in existing_files:
                result = self._build_file_with_progress(file_path, pbar)
                if result:
                    copied_count += 1
                else:
                    skipped_count += 1
                pbar.update(1)

        logger.info(
            "✅ Build complete: %d files copied, %d files skipped",
            copied_count,
            skipped_count,
        )

    def _build_langgraph_version(self, output_dir: str, target_language: str) -> None:
        """Build LangGraph (oss/) content for a specific version.

        Args:
            output_dir: Output directory (e.g., "langgraph/python").
            target_language: Target language for conditional blocks ("python" or "js").
        """
        all_files = self._collect_content_logical_paths("oss")
        if not all_files:
            logger.info("No files found in oss/ directory for %s", output_dir)
            return

        # Process files with progress bar
        copied_count: int = 0
        skipped_count: int = 0

        with tqdm(
            total=len(all_files),
            desc=f"Building {output_dir} files",
            unit="file",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            dynamic_ncols=True,
            leave=False,
            disable=_IS_CI,
        ) as pbar:
            for logical_relative in all_files:
                relative_path = logical_relative.relative_to(Path("oss"))

                if relative_path.parts:
                    first_part = relative_path.parts[0]
                    if first_part in ("python", "javascript"):
                        # Map target_language to expected directory name
                        expected_dir = (
                            "python" if target_language == "python" else "javascript"
                        )
                        # Skip files that are for a different language
                        # (i.e. if we're building for python and we encounter
                        #  `oss/javascript/...`, skip it)
                        if first_part != expected_dir:
                            pbar.update(1)
                            continue
                        # Remove the language-specific directory from the path
                        # e.g., "python/concepts/low_level.md" > "concepts/low_level.md"
                        relative_path = Path(*relative_path.parts[1:])

                # Build to output_dir/ (not `output_dir/oss/`)
                output_path = self.build_dir / output_dir / relative_path
                source_path = self._resolve_source_for_logical_path(logical_relative)
                if source_path is None:
                    skipped_count += 1
                    pbar.update(1)
                    continue

                result = self._build_single_file(
                    source_path,
                    output_path,
                    target_language,
                    pbar,
                    f"{output_dir}/{relative_path}",
                )
                if result:
                    copied_count += 1
                else:
                    skipped_count += 1
                pbar.update(1)

        logger.info(
            "✅ %s complete: %d files copied, %d files skipped",
            output_dir,
            copied_count,
            skipped_count,
        )

    def _build_unversioned_content(self, source_dir: str, output_dir: str) -> None:
        """Build unversioned content (langsmith/).

        Args:
            source_dir: Source directory name (e.g., "langsmith").
            output_dir: Output directory name (same as source_dir).
        """
        all_files = self._collect_content_logical_paths(source_dir)
        if not all_files:
            logger.info("No files found in %s/ directory", source_dir)
            return

        # Process files with progress bar
        copied_count: int = 0
        skipped_count: int = 0

        with tqdm(
            total=len(all_files),
            desc=f"Building {output_dir} files",
            unit="file",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            dynamic_ncols=True,
            leave=False,
            disable=_IS_CI,
        ) as pbar:
            for logical_relative in all_files:
                relative_path = logical_relative.relative_to(Path(source_dir))
                # Build directly to output_dir/
                output_path = self.build_dir / output_dir / relative_path
                source_path = self._resolve_source_for_logical_path(logical_relative)
                if source_path is None:
                    skipped_count += 1
                    pbar.update(1)
                    continue

                result = self._build_single_file(
                    source_path,
                    output_path,
                    "python",
                    pbar,
                    f"{output_dir}/{relative_path}",
                )
                if result:
                    copied_count += 1
                else:
                    skipped_count += 1
                pbar.update(1)

        logger.info(
            "✅ %s complete: %d files copied, %d files skipped",
            output_dir,
            copied_count,
            skipped_count,
        )

    def _build_single_file(
        self,
        file_path: Path,
        output_path: Path,
        target_language: str,
        pbar: tqdm,
        display_path: str,
    ) -> bool:
        """Build a single file with progress bar integration.

        Args:
            file_path: Path to the source file to be built.
            output_path: Full output path for the file.
            target_language: Target language for conditional blocks ("python" or "js").
            pbar: tqdm progress bar instance for updating the description.
            display_path: Path to display in progress bar.

        Returns:
            True if the file was copied, False if it was skipped.
        """
        # Skip template files
        if file_path.name == "TEMPLATE.mdx":
            return False

        # Update progress bar description with current file
        pbar.set_postfix_str(display_path)

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle special case for docs.yml files
        if file_path.name == "docs.yml" and file_path.suffix.lower() in {
            ".yml",
            ".yaml",
        }:
            self._convert_yaml_to_json(file_path, output_path)
            return True
        # Copy other supported files
        if file_path.suffix.lower() in self.copy_extensions:
            # Handle markdown files with preprocessing
            if file_path.suffix.lower() in {".md", ".mdx"}:
                self._process_markdown_file(file_path, output_path, target_language)
                return True
            shutil.copy2(file_path, output_path)
            return True
        return False

    def _build_version_file_with_progress(
        self, file_path: Path, version_dir: str, target_language: str, pbar: tqdm
    ) -> bool:
        """Build a single file for a specific version with progress bar integration.

        Args:
            file_path: Path to the source file to be built.
            version_dir: Directory name for this version (e.g., "python", "javascript").
            target_language: Target language for conditional blocks ("python" or "js").
            pbar: tqdm progress bar instance for updating the description.

        Returns:
            True if the file was copied, False if it was skipped.
        """
        relative_path = file_path.absolute().relative_to(self.src_dir.absolute())
        # Add version prefix to the output path
        output_path = self.build_dir / version_dir / relative_path

        # Update progress bar description with current file
        pbar.set_postfix_str(f"{version_dir}/{relative_path}")

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle special case for docs.yml files
        if file_path.name == "docs.yml" and file_path.suffix.lower() in {
            ".yml",
            ".yaml",
        }:
            self._convert_yaml_to_json(file_path, output_path)
            return True
        # Copy other supported files
        if file_path.suffix.lower() in self.copy_extensions:
            # Handle markdown files with preprocessing
            if file_path.suffix.lower() in {".md", ".mdx"}:
                self._process_markdown_file(file_path, output_path, target_language)
                return True
            shutil.copy2(file_path, output_path)
            return True
        return False

    def is_shared_relative_path(self, relative_path: Path) -> bool:
        """Check if a logical path should be shared between versions."""
        if not relative_path.parts:
            return False

        if relative_path.name == "docs.json":
            return True

        if len(relative_path.parts) == 1 and relative_path.name in {
            "index.mdx",
            "use-these-docs.mdx",
            "playground.mdx",
            "i18n-config.json",
        }:
            return True

        shared_dirs = {"images", "snippets", ".well-known", "fonts"}
        if shared_dirs & set(relative_path.parts):
            return True

        return relative_path.suffix.lower() in {".js", ".css"}

    def is_shared_file(self, file_path: Path) -> bool:
        """Check if a file should be shared between versions rather than duplicated.

        Args:
            file_path: Path to check.

        Returns:
            True if the file should be shared, False if it should be version-specific.
        """
        logical_relative = self._logical_relative_path(file_path)
        if logical_relative is None:
            return False

        return self.is_shared_relative_path(logical_relative)

    def _copy_shared_files(self) -> None:
        """Copy files that should be shared between versions."""
        shared_files = self._collect_shared_logical_paths()

        if not shared_files:
            logger.info("No shared files found")
            return

        copied_count = 0
        for relative_path in shared_files:
            file_path = self._resolve_source_for_logical_path(relative_path)
            if file_path is None:
                continue

            output_path = self.build_dir / relative_path

            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if file_path.suffix.lower() in self.copy_extensions:
                # Handle markdown files with preprocessing for /oss/ link resolution
                if file_path.suffix.lower() in {".md", ".mdx"}:
                    # For snippet files, we need to handle URL rewriting differently
                    # Use a special marker-based approach for dynamic URL resolution
                    if "snippets" in relative_path.parts:
                        logger.debug(
                            "Processing snippet file with URL rewriting: %s",
                            relative_path,
                        )
                        self._process_snippet_markdown_file(file_path, output_path)
                    else:
                        # Regular markdown processing without language-specific rewrite
                        self._process_markdown_file(file_path, output_path, None)
                    copied_count += 1
                else:
                    shutil.copy2(file_path, output_path)
                    copied_count += 1

        logger.info("✅ Shared files copied: %d files", copied_count)

    # Maps npm dist filenames to their output names in build/snippets/
    _NPM_SNIPPET_FILES: ClassVar[dict[str, str]] = {
        "ChatLangChainEmbed.jsx": "chat-langchain-embed.jsx",
        "PatternEmbed.jsx": "pattern-embed.jsx",
        "ExampleEmbed.jsx": "example-embed.jsx",
    }

    # Maps npm dist filenames to their output names in build/ (served at site root).
    _NPM_BUILD_FILES: ClassVar[dict[str, str]] = {
        "ChatLangChainEmbed.js": "ChatLangChainEmbed.js",
    }

    def _rewrite_chat_langchain_embed(self, embed_path: Path) -> None:
        """Force the chat embed to use the hosted app, even on localhost.

        The upstream npm bundle points localhost docs builds at a separate local
        embed dev server on port 4100. Our local docs workflow does not run that
        server, so we rewrite the copied bundle to use the hosted embed URL.
        """
        content = embed_path.read_text(encoding="utf-8")
        rewritten_content = content.replace(
            _CHAT_LANGCHAIN_LOCAL_EMBED_URL,
            _CHAT_LANGCHAIN_PROD_EMBED_URL,
        )

        if rewritten_content != content:
            embed_path.write_text(rewritten_content, encoding="utf-8")
            logger.info("Rewrote ChatLangChain embed to use hosted URL: %s", embed_path)

    def _restore_snippet_code_fence(
        self, content: str, input_path: Path, logical_relative: Path
    ) -> str:
        """Restore fenced code blocks for translated snippets when needed.

        Some translated snippet files under i18n contain only the inner code block
        content, while the source snippet keeps the surrounding markdown fence.
        Mint parses imported snippet MDX files directly, so bare code causes MDX
        parsing failures. When the translated snippet lacks a fence, copy the fence
        marker from the source snippet at the same logical path.
        """
        stripped_content = content.lstrip()
        if stripped_content.startswith(("```", "~~~")):
            return content

        source_path = self.src_dir / logical_relative
        if source_path == input_path or not source_path.is_file():
            return content

        source_content = source_path.read_text(encoding="utf-8").lstrip()
        first_line = source_content.splitlines()[0] if source_content else ""
        fence_match = re.match(r"^(```|~~~)([^\r\n]*)$", first_line)
        if fence_match is None:
            return content

        fence = fence_match.group(1)
        fence_suffix = fence_match.group(2)
        normalized_content = content.strip("\n")
        if not normalized_content:
            return content

        logger.warning(
            "Snippet missing code fence in %s; restored %s fence from %s",
            input_path,
            f"{fence}{fence_suffix}",
            source_path,
        )

        return f"{fence}{fence_suffix}\n{normalized_content}\n{fence}\n"

    def _copy_npm_snippets(self) -> None:
        """Copy snippet components from the @langchain/docs-sandbox npm package.

        Overwrites any source-tree versions already copied by _copy_shared_files
        so the build always uses the latest published component.
        """
        pkg_dist = (
            self.src_dir.parent
            / "node_modules"
            / "@langchain"
            / "docs-sandbox"
            / "dist"
        )
        if not pkg_dist.is_dir():
            logger.warning(
                "@langchain/docs-sandbox not installed — run `npm install` first"
            )
            return

        snippets_dir = self.build_dir / "snippets"
        snippets_dir.mkdir(parents=True, exist_ok=True)

        for src_name, dest_name in self._NPM_SNIPPET_FILES.items():
            src_file = pkg_dist / src_name
            if not src_file.is_file():
                logger.warning("Expected file not found in npm package: %s", src_file)
                continue
            dest_file = snippets_dir / dest_name
            shutil.copy2(src_file, dest_file)
            logger.debug("Copied npm snippet: %s → snippets/%s", src_name, dest_name)

        for src_name, dest_name in self._NPM_BUILD_FILES.items():
            src_file = pkg_dist / src_name
            if not src_file.is_file():
                logger.warning("Expected file not found in npm package: %s", src_file)
                continue
            dest_file = self.build_dir / dest_name
            shutil.copy2(src_file, dest_file)
            if dest_name == "ChatLangChainEmbed.js":
                self._rewrite_chat_langchain_embed(dest_file)
            logger.info("Copied npm build file: %s → build/%s", src_name, dest_name)

    def _process_snippet_markdown_file(
        self, input_path: Path, output_path: Path
    ) -> None:
        """Process a snippet markdown file with language-aware URL resolution.

        For snippet files that contain /oss/ links, we need to create versions
        that work properly when included in different language contexts.
        We'll modify the URLs to use relative paths that resolve correctly.

        Args:
            input_path: Path to the source snippet markdown file.
            output_path: Path where the processed file should be written.
        """
        try:
            # Read the source markdown content
            with input_path.open("r", encoding="utf-8") as f:
                content = f.read()

            logical_relative = self._logical_relative_path(input_path)
            if logical_relative is None:
                logical_relative = input_path.relative_to(self.src_dir)

            content = self._restore_snippet_code_fence(
                content,
                input_path,
                logical_relative,
            )

            # Apply standard markdown preprocessing
            processed_content = preprocess_markdown(
                content, input_path, target_language=None
            )

            # Convert /oss/ links to relative paths that work from any language context
            def convert_oss_link(match: re.Match) -> str:
                """Convert /oss/ links to language-agnostic relative paths.

                IMPORTANT: the conversion creates relative paths that resolve from the
                parent page's directory.
                - /oss/providers/groq → ../providers/groq
                """
                pre = match.group(1)  # Everything before the URL
                url = match.group(2)  # The URL
                post = match.group(3)  # Everything after the URL

                # Only convert absolute /oss/ paths that don't contain 'images'
                # or '/oss/python' or '/oss/javascript'
                if (
                    url.startswith("/oss/")
                    and "images" not in url
                    and "/oss/python" not in url
                    and "/oss/javascript" not in url
                ):
                    # Convert to relative path that works from oss/python/* or oss/js/*
                    # e.g., /oss/releases/langchain-v1 becomes ../releases/langchain-v1
                    parts = url.split("/")
                    oss_path = "/".join(parts[2:])  # Remove /oss/ prefix
                    url = f"../{oss_path}"  # Make it relative

                return f"{pre}{url}{post}"

            # Apply URL conversion
            pattern = r'(\[.*?\]\(|\bhref="|")(/oss/[^")\s]+)([")\s])'
            processed_content = re.sub(pattern, convert_oss_link, processed_content)

            # Convert .md to .mdx if needed
            if input_path.suffix.lower() == ".md":
                output_path = output_path.with_suffix(".mdx")

            # Write the processed content
            with output_path.open("w", encoding="utf-8") as f:
                f.write(processed_content)

        except (OSError, UnicodeDecodeError):
            logger.exception(
                "File I/O or decoding error in snippet markdown file %s", input_path
            )
            raise
        except re.error:
            logger.exception("Regex error in snippet markdown file %s", input_path)
            raise
