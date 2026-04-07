"""Utilities for translating Markdown/MDX docs into multiple languages."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx

logger = logging.getLogger(__name__)

_LANGUAGE_CODE_PATTERN = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_DEFAULT_LANGUAGE_CODE = "en"
_DEFAULT_LANGUAGE_LABEL = "English"
_FENCED_BLOCK_MIN_LINES = 3
_QUOTED_VALUE_MIN_LEN = 2

if TYPE_CHECKING:
    from pathlib import Path

_COMMON_LANGUAGE_LABELS: dict[str, str] = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
}
_ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class LanguageTarget:
    """A language target used for translation and UI display."""

    code: str
    label: str


def compute_sha256(content: str) -> str:
    """Compute a stable SHA256 hash for text content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_dotenv_file(path: Path, *, override_existing: bool = False) -> int:
    """Load environment variables from a dotenv file.

    Args:
        path: Path to the dotenv file.
        override_existing: Whether to override existing process env values.

    Returns:
        Number of variables loaded into ``os.environ``.
    """
    if not path.exists():
        return 0

    loaded_count = 0
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()

        if "=" not in stripped:
            logger.warning("Ignoring invalid .env line %d in %s", line_number, path)
            continue

        raw_key, raw_value = stripped.split("=", 1)
        key = raw_key.strip()
        value = _parse_dotenv_value(raw_value.strip())

        if not _ENV_KEY_PATTERN.fullmatch(key):
            logger.warning(
                "Ignoring invalid .env key '%s' on line %d in %s",
                key,
                line_number,
                path,
            )
            continue

        if not override_existing and key in os.environ:
            continue

        os.environ[key] = value
        loaded_count += 1

    return loaded_count


def parse_target_languages(raw_value: str) -> list[LanguageTarget]:
    """Parse target languages from an environment variable string.

    Supported token formats:
    - `zh-CN`
    - `zh-CN:Chinese (Simplified)`
    - `zh-CN|Chinese (Simplified)`

    Tokens are separated by commas or newlines.
    """
    tokens = [
        token.strip() for token in re.split(r"[,;\n]+", raw_value) if token.strip()
    ]
    if not tokens:
        msg = "No target languages were provided."
        raise ValueError(msg)

    parsed: list[LanguageTarget] = []
    seen_codes: set[str] = set()

    for token in tokens:
        code, label = _parse_single_language_token(token)
        normalized_code = code.lower()

        if normalized_code in seen_codes:
            continue

        seen_codes.add(normalized_code)
        parsed.append(LanguageTarget(code=code, label=label))

    if not parsed:
        msg = "No valid target languages were parsed."
        raise ValueError(msg)

    return parsed


def write_language_switch_config(
    config_path: Path,
    target_languages: list[LanguageTarget],
) -> None:
    """Write the language switcher configuration for frontend usage."""
    config_path.parent.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = [
        {"code": _DEFAULT_LANGUAGE_CODE, "label": _DEFAULT_LANGUAGE_LABEL},
    ]
    existing_codes = {_DEFAULT_LANGUAGE_CODE}

    for target in target_languages:
        normalized_code = target.code.lower()
        if normalized_code in existing_codes:
            continue
        existing_codes.add(normalized_code)
        entries.append({"code": target.code, "label": target.label})

    payload = {
        "defaultLanguage": _DEFAULT_LANGUAGE_CODE,
        "prefix": "i18n",
        "languages": entries,
        "updatedAt": datetime.now(UTC).isoformat(),
    }

    config_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


class TranslationManifest:
    """Stores source content hashes per language and page."""

    def __init__(self, path: Path) -> None:
        """Initialize a manifest at the given file path."""
        self.path = path
        self.data: dict[str, Any] = {
            "version": 1,
            "updatedAt": datetime.now(UTC).isoformat(),
            "languages": {},
        }

    def load(self) -> None:
        """Load manifest content from disk if it exists."""
        if not self.path.exists():
            return

        raw = self.path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            msg = f"Translation manifest must be an object: {self.path}"
            raise TypeError(msg)

        self.data = parsed
        self.data.setdefault("version", 1)
        self.data.setdefault("languages", {})
        self.data.setdefault("updatedAt", datetime.now(UTC).isoformat())

    def save(self) -> None:
        """Persist the current manifest to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updatedAt"] = datetime.now(UTC).isoformat()
        self.path.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def get_hash(self, language_code: str, source_relative_path: str) -> str | None:
        """Get a stored source hash for a language/path pair."""
        normalized_path = _normalize_relative_path(source_relative_path)
        language_entries = self._get_language_entries(language_code)
        entry = language_entries.get(normalized_path)
        if not isinstance(entry, dict):
            return None
        source_hash = entry.get("sourceHash")
        return source_hash if isinstance(source_hash, str) else None

    def set_hash(
        self,
        language_code: str,
        source_relative_path: str,
        source_hash: str,
    ) -> None:
        """Set a source hash for a language/path pair."""
        normalized_path = _normalize_relative_path(source_relative_path)
        language_entries = self._get_or_create_language_entries(language_code)
        language_entries[normalized_path] = {
            "sourceHash": source_hash,
            "updatedAt": datetime.now(UTC).isoformat(),
        }

    def _get_language_entries(self, language_code: str) -> dict[str, Any]:
        languages = self.data.get("languages")
        if not isinstance(languages, dict):
            return {}
        entries = languages.get(language_code)
        return entries if isinstance(entries, dict) else {}

    def _get_or_create_language_entries(self, language_code: str) -> dict[str, Any]:
        languages = self.data.setdefault("languages", {})
        if not isinstance(languages, dict):
            msg = "Invalid manifest format: 'languages' must be an object."
            raise TypeError(msg)
        entries = languages.setdefault(language_code, {})
        if not isinstance(entries, dict):
            msg = f"Invalid manifest format for language: {language_code}"
            raise TypeError(msg)
        return entries


class OpenAICompatibleTranslator:
    """Translate Markdown/MDX via an OpenAI-compatible chat completions API."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize the OpenAI-compatible translator client settings."""
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.endpoint = _build_chat_completions_url(base_url)

    def translate_markdown(self, content: str, target_language: str) -> str:
        """Translate Markdown/MDX content into the target language."""
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a technical translator for Markdown and MDX docs. "
                        "Translate only natural-language prose. Keep Markdown syntax, "
                        "MDX components, links, code blocks, inline code, and all "
                        "identifiers unchanged."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Translate the following Markdown/MDX to "
                        f"{target_language}.\n"
                        "Return only translated Markdown/MDX with the exact same "
                        "structure.\n\n"
                        f"{content}"
                    ),
                },
            ],
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(1, self.max_retries + 1):
            response = self._send_request(payload, headers)
            if response.status_code in _RETRYABLE_STATUS_CODES:
                if attempt >= self.max_retries:
                    response.raise_for_status()
                retry_delay = float(attempt)
                logger.warning(
                    "Translation request failed with HTTP %d. Retrying in %.1fs.",
                    response.status_code,
                    retry_delay,
                )
                time.sleep(retry_delay)
                continue

            response.raise_for_status()
            translated = _extract_message_text(response.json())
            return _strip_code_fence_wrapper(translated)

        msg = "Translation failed after all retries."
        raise RuntimeError(msg)

    def _send_request(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> httpx.Response:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            return client.post(self.endpoint, json=payload, headers=headers)


def _parse_single_language_token(token: str) -> tuple[str, str]:
    if "|" in token:
        raw_code, raw_label = token.split("|", 1)
    elif ":" in token:
        raw_code, raw_label = token.split(":", 1)
    else:
        raw_code = token
        raw_label = ""

    code = raw_code.strip()
    if not _LANGUAGE_CODE_PATTERN.fullmatch(code):
        msg = (
            "Invalid language code in DOCS_TARGET_LANGUAGES: "
            f"'{code}'. Use values like 'zh-CN' or 'ja'."
        )
        raise ValueError(msg)

    if raw_label.strip():
        return code, raw_label.strip()

    default_label = _COMMON_LANGUAGE_LABELS.get(code.lower())
    return code, default_label if default_label else code


def _build_chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _extract_message_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        msg = "Translation response does not contain choices."
        raise ValueError(msg)

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        msg = "Translation response choice has invalid format."
        raise TypeError(msg)

    message = first_choice.get("message")
    if not isinstance(message, dict):
        msg = "Translation response does not contain a message."
        raise TypeError(msg)

    content = message.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    else:
        msg = "Translation response content is missing."
        raise TypeError(msg)

    if not text.strip():
        msg = "Translation response content is empty."
        raise ValueError(msg)

    return text


def _strip_code_fence_wrapper(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) < _FENCED_BLOCK_MIN_LINES:
        return stripped

    if lines[-1].strip() != "```":
        return stripped

    return "\n".join(lines[1:-1]).strip()


def _normalize_relative_path(path_value: str) -> str:
    normalized = path_value.replace("\\", "/")
    return normalized.lstrip("./")


def _parse_dotenv_value(raw_value: str) -> str:
    value_without_comment = _strip_unquoted_comment(raw_value).strip()
    if len(value_without_comment) >= _QUOTED_VALUE_MIN_LEN and (
        value_without_comment[0] == value_without_comment[-1]
        and value_without_comment[0] in {"'", '"'}
    ):
        quote = value_without_comment[0]
        inner = value_without_comment[1:-1]
        if quote == '"':
            inner = (
                inner.replace(r"\\", "\\")
                .replace(r"\"", '"')
                .replace(r"\n", "\n")
                .replace(r"\r", "\r")
                .replace(r"\t", "\t")
            )
        return inner

    return value_without_comment


def _strip_unquoted_comment(value: str) -> str:
    in_single_quote = False
    in_double_quote = False
    escaped = False

    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue

        if char == "\\" and in_double_quote:
            escaped = True
            continue

        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue

        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue

        if (
            char == "#"
            and not in_single_quote
            and not in_double_quote
            and (index == 0 or value[index - 1].isspace())
        ):
            return value[:index].rstrip()

    return value.rstrip()
