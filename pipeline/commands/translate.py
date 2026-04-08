"""Translate docs into language-specific pages under src/i18n/."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from tqdm import tqdm

from pipeline.tools.translation import (
    LanguageTarget,
    OpenAICompatibleTranslator,
    TranslationManifest,
    compute_sha256,
    load_dotenv_file,
    parse_target_languages,
    write_language_switch_config,
)

logger = logging.getLogger(__name__)

_SOURCE_DIR_NAME = "src"
_I18N_DIR_NAME = "i18n"
_CONFIG_FILE_NAME = "config.json"
_MANIFEST_FILE_NAME = "translation-hashes.json"
_MIN_PROMPT_TOKENS = 512


@dataclass(frozen=True)
class TranslationSettings:
    """Environment-driven settings required for translations."""

    base_url: str
    api_key: str
    model: str
    target_languages: list[LanguageTarget]
    concurrency: int
    timeout_seconds: float
    max_retries: int
    max_prompt_tokens: int


@dataclass(frozen=True)
class TranslationRunContext:
    """State shared across translation routines."""

    src_root: Path
    i18n_root: Path
    manifest: TranslationManifest
    translator: OpenAICompatibleTranslator | None
    manifest_lock: Lock
    force: bool
    dry_run: bool
    concurrency: int


@dataclass(frozen=True)
class TranslationWorkItem:
    """Prepared translation metadata for a single source file."""

    source_file: Path
    relative_path: str
    output_path: Path
    source_hash: str


def translate_command(args: Any) -> int:  # noqa: ANN401
    """Run incremental translation for Markdown/MDX source files."""
    source_path = Path(getattr(args, "path", _SOURCE_DIR_NAME))
    force = bool(getattr(args, "force", False))
    dry_run = bool(getattr(args, "dry_run", False))
    limit = int(getattr(args, "limit", 0))

    source_path = source_path.resolve()
    if not source_path.exists():
        logger.error("Path not found: %s", source_path)
        return 1

    src_root = _resolve_src_root(source_path)
    if src_root is None:
        logger.error(
            "Could not find '%s' root for path: %s",
            _SOURCE_DIR_NAME,
            source_path,
        )
        return 1

    env_file_arg = getattr(args, "env_file", None)
    dotenv_path = _resolve_dotenv_path(src_root, env_file_arg)
    loaded_variables = load_dotenv_file(dotenv_path, override_existing=False)
    if dotenv_path.exists():
        logger.info(
            "Loaded %d environment variables from %s",
            loaded_variables,
            dotenv_path,
        )

    try:
        settings = _load_settings_from_env()
    except ValueError:
        logger.exception("Failed to load translation settings from environment.")
        return 1

    i18n_root = src_root / _I18N_DIR_NAME
    config_path = i18n_root / _CONFIG_FILE_NAME
    manifest_path = i18n_root / _MANIFEST_FILE_NAME

    if dry_run:
        logger.info("[dry-run] Would update language switch config: %s", config_path)
    else:
        write_language_switch_config(config_path, settings.target_languages)
        logger.info("Updated language switch config: %s", config_path)

    markdown_files = _collect_markdown_files(source_path, src_root)
    if limit > 0:
        markdown_files = markdown_files[:limit]

    if not markdown_files:
        logger.info("No Markdown/MDX files found to translate.")
        return 0

    manifest = TranslationManifest(manifest_path)
    try:
        manifest.load()
    except (TypeError, ValueError):
        logger.exception("Invalid translation manifest format.")
        return 1

    translator = None
    if not dry_run:
        translator = OpenAICompatibleTranslator(
            base_url=settings.base_url,
            api_key=settings.api_key,
            model=settings.model,
            timeout_seconds=settings.timeout_seconds,
            max_retries=settings.max_retries,
            max_prompt_tokens=settings.max_prompt_tokens,
        )

    context = TranslationRunContext(
        src_root=src_root,
        i18n_root=i18n_root,
        manifest=manifest,
        translator=translator,
        manifest_lock=Lock(),
        force=force,
        dry_run=dry_run,
        concurrency=settings.concurrency,
    )

    translated_count = 0
    skipped_count = 0

    for target in settings.target_languages:
        logger.info("Translating for language: %s (%s)", target.label, target.code)
        translated, skipped = _translate_language(
            target_language=target,
            source_files=markdown_files,
            context=context,
        )
        translated_count += translated
        skipped_count += skipped

    logger.info(
        "Translation complete: %d translated, %d skipped.",
        translated_count,
        skipped_count,
    )
    return 0


def _load_settings_from_env() -> TranslationSettings:
    base_url = _get_required_env("DOCS_AI_OPENAI_BASEURL")
    api_key = _get_required_env("DOCS_AI_OPENAI_APIKEY")
    model = _get_required_env("DOCS_AI_OPENAI_MODEL")
    raw_languages = _get_required_env("DOCS_TARGET_LANGUAGES")
    concurrency = _get_optional_int_env("DOCS_TRANSLATE_CONCURRENCY", default=3)
    timeout_seconds = _get_optional_float_env(
        "DOCS_TRANSLATE_TIMEOUT_SECONDS",
        default=120.0,
    )
    max_retries = _get_optional_int_env("DOCS_TRANSLATE_MAX_RETRIES", default=3)
    max_prompt_tokens = _get_optional_int_env(
        "DOCS_TRANSLATE_MAX_PROMPT_TOKENS",
        default=6000,
    )

    target_languages = parse_target_languages(raw_languages)
    if concurrency < 1:
        msg = "DOCS_TRANSLATE_CONCURRENCY must be >= 1"
        raise ValueError(msg)
    if timeout_seconds <= 0:
        msg = "DOCS_TRANSLATE_TIMEOUT_SECONDS must be > 0"
        raise ValueError(msg)
    if max_retries < 1:
        msg = "DOCS_TRANSLATE_MAX_RETRIES must be >= 1"
        raise ValueError(msg)
    if max_prompt_tokens < _MIN_PROMPT_TOKENS:
        msg = (
            "DOCS_TRANSLATE_MAX_PROMPT_TOKENS must be >= "
            f"{_MIN_PROMPT_TOKENS}"
        )
        raise ValueError(msg)

    return TranslationSettings(
        base_url=base_url,
        api_key=api_key,
        model=model,
        target_languages=target_languages,
        concurrency=concurrency,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        max_prompt_tokens=max_prompt_tokens,
    )


def _get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    msg = f"Missing required environment variable: {name}"
    raise ValueError(msg)


def _get_optional_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        msg = f"Environment variable {name} must be an integer."
        raise ValueError(msg) from exc


def _get_optional_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        msg = f"Environment variable {name} must be a number."
        raise ValueError(msg) from exc


def _resolve_src_root(path: Path) -> Path | None:
    if path.name == _SOURCE_DIR_NAME and path.is_dir():
        return path

    for parent in path.parents:
        if parent.name == _SOURCE_DIR_NAME:
            return parent

    return None


def _resolve_dotenv_path(src_root: Path, env_file: Path | None) -> Path:
    if env_file is not None:
        return env_file.resolve()
    return src_root.parent / ".env"


def _collect_markdown_files(path: Path, src_root: Path) -> list[Path]:
    if path.is_file():
        if _is_markdown_file(path) and _is_translatable_file(path, src_root):
            return [path]
        return []

    files = [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and _is_markdown_file(candidate)
        and _is_translatable_file(candidate, src_root)
    ]
    return sorted(files)


def _is_markdown_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in {".md", ".mdx"}


def _is_translatable_file(file_path: Path, src_root: Path) -> bool:
    relative = file_path.relative_to(src_root)
    if not relative.parts:
        return False

    blocked_roots = {"i18n", "images", "fonts", "code-samples"}
    return relative.parts[0] not in blocked_roots


def _translate_language(
    *,
    target_language: LanguageTarget,
    source_files: list[Path],
    context: TranslationRunContext,
) -> tuple[int, int]:
    pending_items, initial_skipped = _prepare_translation_work_items(
        target_language=target_language,
        source_files=source_files,
        context=context,
    )

    if context.dry_run or context.concurrency <= 1:
        return _translate_language_sequential(
            target_language=target_language,
            total_files=len(source_files),
            pending_items=pending_items,
            initial_skipped=initial_skipped,
            context=context,
        )

    return _translate_language_parallel(
        target_language=target_language,
        total_files=len(source_files),
        pending_items=pending_items,
        initial_skipped=initial_skipped,
        context=context,
    )


def _prepare_translation_work_items(
    *,
    target_language: LanguageTarget,
    source_files: list[Path],
    context: TranslationRunContext,
) -> tuple[list[TranslationWorkItem], int]:
    pending_items: list[TranslationWorkItem] = []
    skipped_count = 0

    for source_file in source_files:
        source_content = source_file.read_text(encoding="utf-8")
        source_hash = compute_sha256(source_content)
        relative_path = source_file.relative_to(context.src_root).as_posix()
        output_path = context.i18n_root / target_language.code / relative_path
        stored_hash = context.manifest.get_hash(target_language.code, relative_path)

        if not context.force and stored_hash == source_hash and output_path.exists():
            skipped_count += 1
            continue

        pending_items.append(
            TranslationWorkItem(
                source_file=source_file,
                relative_path=relative_path,
                output_path=output_path,
                source_hash=source_hash,
            ),
        )

    return pending_items, skipped_count


def _translate_language_sequential(
    *,
    target_language: LanguageTarget,
    total_files: int,
    pending_items: list[TranslationWorkItem],
    initial_skipped: int,
    context: TranslationRunContext,
) -> tuple[int, int]:
    translated_count = 0
    skipped_count = initial_skipped

    with tqdm(
        total=total_files,
        initial=initial_skipped,
        desc=f"Translating {target_language.code}",
        unit="file",
        leave=False,
        dynamic_ncols=True,
    ) as pbar:
        for work_item in pending_items:
            translated = _translate_single_file(
                work_item=work_item,
                target_language=target_language,
                context=context,
            )
            if translated:
                translated_count += 1
            else:
                skipped_count += 1
            pbar.update(1)

    return translated_count, skipped_count


def _translate_language_parallel(
    *,
    target_language: LanguageTarget,
    total_files: int,
    pending_items: list[TranslationWorkItem],
    initial_skipped: int,
    context: TranslationRunContext,
) -> tuple[int, int]:
    translated_count = 0
    skipped_count = initial_skipped

    with tqdm(
        total=total_files,
        initial=initial_skipped,
        desc=f"Translating {target_language.code}",
        unit="file",
        leave=False,
        dynamic_ncols=True,
    ) as pbar, ThreadPoolExecutor(max_workers=context.concurrency) as executor:
        futures = [
            executor.submit(
                _translate_single_file,
                work_item=work_item,
                target_language=target_language,
                context=context,
            )
            for work_item in pending_items
        ]

        for future in as_completed(futures):
            translated = future.result()
            if translated:
                translated_count += 1
            else:
                skipped_count += 1
            pbar.update(1)

    return translated_count, skipped_count


def _translate_single_file(
    *,
    work_item: TranslationWorkItem,
    target_language: LanguageTarget,
    context: TranslationRunContext,
) -> bool:
    if context.dry_run:
        logger.info(
            "[dry-run] Would translate: %s -> %s",
            work_item.relative_path,
            work_item.output_path,
        )
        return True

    if context.translator is None:
        msg = "Translator is required when dry_run is False."
        raise RuntimeError(msg)

    source_content = work_item.source_file.read_text(encoding="utf-8")
    source_hash = compute_sha256(source_content)
    stored_hash = context.manifest.get_hash(
        target_language.code,
        work_item.relative_path,
    )

    if (
        not context.force
        and stored_hash == source_hash
        and work_item.output_path.exists()
    ):
        return False

    translated_content = context.translator.translate_markdown(
        source_content,
        target_language=target_language.label,
    )

    work_item.output_path.parent.mkdir(parents=True, exist_ok=True)
    work_item.output_path.write_text(translated_content, encoding="utf-8")

    with context.manifest_lock:
        context.manifest.set_hash(
            target_language.code,
            work_item.relative_path,
            source_hash,
        )
        context.manifest.save()
    logger.debug(
        "Translated: %s -> %s",
        work_item.relative_path,
        work_item.output_path,
    )
    return True
