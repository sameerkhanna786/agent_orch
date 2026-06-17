"""Helpers for safely materializing generated test artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:/")
_TEST_DIR_NAMES = {
    "__tests__",
    "spec",
    "specs",
    "test",
    "testing",
    "tests",
}
_TEST_EXTENSIONS = {
    ".bats",
    ".c",
    ".cc",
    ".cmake",
    ".cpp",
    ".cs",
    ".csv",
    ".golden",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".json",
    ".js",
    ".jsx",
    ".kt",
    ".kts",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".snap",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


_APPENDABLE_TEST_FRAGMENT_RE = re.compile(
    r"(^|\n)\s*(?:"
    r"(?:async\s+)?def\s+test_[A-Za-z0-9_]+\s*\(|"
    r"class\s+Test[A-Za-z0-9_]*\b|"
    r"func\s+Test[A-Za-z0-9_]+\s*\(|"
    r"(?:describe|it|test)\s*\(|"
    r"o\.spec\s*\(|"
    r"o\s*\(\s*['\"]"
    r")",
    re.MULTILINE,
)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _has_known_test_name(path: str) -> bool:
    name = Path(path).name
    lower_name = name.lower()
    lower_stem = Path(lower_name).stem
    return bool(
        lower_name == "conftest.py"
        or lower_name.startswith("test_")
        or lower_name.startswith("tests_")
        or lower_name.endswith("_test.py")
        or lower_name.endswith("_tests.py")
        or lower_name.endswith("_test.go")
        or lower_name.endswith("_test.rs")
        or ".test." in lower_name
        or ".spec." in lower_name
        or lower_stem.endswith("test")
    )


def looks_like_generated_test_path(path: str) -> bool:
    """Return whether ``path`` is plausibly a generated test artifact path."""

    normalized = str(path or "").strip().replace("\\", "/")
    if not normalized:
        return False
    suffix = Path(normalized).suffix.lower()
    parts = [part.lower() for part in normalized.split("/") if part]
    if not parts:
        return False
    if any(part in _TEST_DIR_NAMES for part in parts[:-1]) and suffix in _TEST_EXTENSIONS:
        return True
    return _has_known_test_name(normalized)


def normalize_generated_test_path(path: Any) -> str:
    """Normalize a generated test path or return ``""`` when unsafe.

    Generated test paths must be repository-relative, stay within the
    repository root, and look like tests. This rejects absolute paths,
    parent traversal, and source-file overwrite attempts.
    """

    text = str(path or "").strip().replace("\\", "/")
    if not text or text.startswith("/") or _WINDOWS_ABSOLUTE_PATH.match(text):
        return ""
    parts = [part for part in text.split("/") if part and part != "."]
    if not parts or any(part == ".." for part in parts):
        return ""
    normalized = "/".join(parts)
    if not looks_like_generated_test_path(normalized):
        return ""
    return normalized


def normalize_generated_test_content(content: Any) -> str:
    """Normalize source text emitted by model/tool JSON boundaries."""
    text = str(content or "").replace("\r\n", "\n")
    # Some degraded CLI recoveries return JSON-escaped source as literal
    # ``\n``/``\t`` sequences. If real line breaks are mostly absent, decode
    # the common source escapes so downstream static quality and F2P runners
    # see actual code.
    if text.count("\\n") >= 2 and text.count("\n") <= 1:
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    return text


def _should_auto_append_to_existing_test(
    *,
    target: Path,
    raw_materialization_mode: Any,
    content: str,
) -> bool:
    """Default omitted-mode snippets to append for existing test files.

    Test-generation models often produce only the new test block for an
    existing repository test file and forget ``materialization_mode=append``.
    Replacing the file drops imports, fixtures, and runner setup, which turns
    an otherwise valid regression into an F2F syntax/import failure. Explicit
    ``replace`` remains honored for callers that intentionally return a full
    file rewrite.
    """
    if raw_materialization_mode not in (None, "", "auto"):
        return False
    if not target.exists() or not target.is_file():
        return False
    stripped = str(content or "").strip()
    if not stripped:
        return False
    try:
        existing = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        existing = target.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    if not existing.strip() or stripped in existing:
        return False
    # If the model emitted an entire file-sized rewrite, appending is more
    # likely to duplicate declarations. Shorter test-block snippets are the
    # failure mode this guard is meant to recover.
    if len(stripped) >= max(len(existing.strip()) * 0.8, 4000):
        return False
    return bool(_APPENDABLE_TEST_FRAGMENT_RE.search(stripped))


def safe_materialize_test_artifact(root: Path, artifact: dict[str, Any]) -> Optional[str]:
    """Safely write one generated test artifact under ``root``.

    Returns the normalized repo-relative path when materialized, otherwise
    returns ``None``. Existing symlinks that escape ``root`` are rejected.
    """

    rel_path = normalize_generated_test_path(artifact.get("path"))
    content = normalize_generated_test_content(artifact.get("content"))
    if not rel_path or not content:
        return None

    try:
        root_resolved = root.resolve(strict=True)
    except OSError:
        return None

    target = root / rel_path
    try:
        if target.exists() or target.is_symlink():
            resolved_target = target.resolve(strict=True)
            if not _is_relative_to(resolved_target, root_resolved):
                return None
        resolved_parent = target.parent.resolve(strict=False)
        if not _is_relative_to(resolved_parent, root_resolved):
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        resolved_parent = target.parent.resolve(strict=True)
        if not _is_relative_to(resolved_parent, root_resolved):
            return None
        raw_materialization_mode = artifact.get("materialization_mode")
        materialization_mode = str(raw_materialization_mode or "replace").strip().lower()
        auto_append = _should_auto_append_to_existing_test(
            target=target,
            raw_materialization_mode=raw_materialization_mode,
            content=content,
        )
        if (materialization_mode == "append" or auto_append) and target.exists():
            try:
                existing = target.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                existing = target.read_text(encoding="utf-8", errors="ignore")
            rendered = existing
            if rendered and not rendered.endswith("\n"):
                rendered += "\n"
            rendered += content
        else:
            rendered = content
        if rendered and not rendered.endswith("\n"):
            rendered += "\n"
        target.write_text(rendered, encoding="utf-8")
    except OSError:
        return None
    return rel_path


def safe_materialize_test_artifacts(
    root: Path,
    artifacts: list[dict[str, Any]],
) -> list[str]:
    """Safely write generated test artifacts and return materialized paths."""

    materialized: list[str] = []
    seen: set[str] = set()
    for artifact in artifacts or []:
        if not isinstance(artifact, dict):
            continue
        rel_path = safe_materialize_test_artifact(root, artifact)
        if rel_path and rel_path not in seen:
            seen.add(rel_path)
            materialized.append(rel_path)
    return materialized
