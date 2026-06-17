"""Patch hygiene and artifact quarantine helpers."""

from __future__ import annotations

import fnmatch
import re
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Optional

from .localizer_scope import is_test_path


class PatchPathCategory(str, Enum):
    SOLUTION_FILE = "solution_file"
    GOLD_PROTECTED_TEST = "gold_protected_test"
    GENERATED_ARTIFACT = "generated_artifact"
    TEMPORARY_ARTIFACT = "temporary_artifact"
    APEX_CONTROL_FILE = "apex_control_file"
    DEPENDENCY_ARTIFACT = "dependency_artifact"
    # Part C: an extracted upstream release tree / archive / installed-package
    # copy that the agent vendored into the workspace (e.g. ``pytest-8.3.5/...``,
    # ``*.tar.gz``, a ``site-packages``-shaped dir). It is never a legitimate
    # solution edit; it is stripped like a dependency artifact and, crucially,
    # is kept DISTINCT from GOLD_PROTECTED_TEST so a vendored ``testing/`` tree
    # can never be conflated with editing the repo's own scored tests.
    VENDORED_UPSTREAM_ARTIFACT = "vendored_upstream_artifact"
    UNKNOWN = "unknown"


_APEX_CONTROL_NAMES = {
    ".apex_expected_test_ids.txt",
    "_apex_expected_test_ids.txt",
    "_apex_expected_ids_filter.py",
    "_apex_run_expected_ids.py",
    "apex_result.json",
    "task_live_state.json",
    "task_state_graph.json",
    "run_manifest.json",
    "apex_run_manifest.json",
}

# WS3A: APEX-written per-rollout agent guidance files (lowercased for
# case-insensitive matching). Quarantined only in gold-suite mode.
_AGENT_CONTEXT_FILENAMES = {"agents.md", "claude.md"}

_AGENT_RUNTIME_METADATA_PATHS = {
    ".claude/settings.json",
    ".claude/settings.local.json",
}

_GENERATED_NAMES = {
    "rollout_report.json",
    "report.json",
    "targeted_report.json",
    "coverage.json",
    ".coverage",
    "coverage.xml",
    "junit.xml",
}

_GENERATED_GLOBS = (
    "*.pyc",
    "*.pyo",
    "*.pytest_cache/*",
    "__pycache__/*",
    ".mypy_cache/*",
    ".ruff_cache/*",
    ".hypothesis/*",
    ".apex_verification_reports/*",
    "htmlcov/*",
    "task_output/*",
    "rollout_status/*",
    "trajectories/*",
)

_TEMPORARY_GLOBS = (
    "*.patch",
    "*.diff",
    "*.rej",
    "*.orig",
    "*.bak",
    "*~",
    "original_*",
    "backup_*",
    "tmp_*",
    "temp_*",
    "scratch_*",
    "patch.py",
    "fix_*.py",
    "clean_*.py",
    # POSIX crash dumps can be emitted as root-level binary files by native
    # runtimes; they are candidate artifacts, never source repairs.
    "core",
    "core.*",
    "vgcore.*",
)

_DEPENDENCY_GLOBS = (
    "node_modules/*",
    ".venv/*",
    "venv/*",
    "build/*",
    "dist/*",
    "*.egg-info/*",
)

_DEPENDENCY_DIR_NAMES = frozenset(
    {
        ".venv",
        "venv",
        "node_modules",
        "build",
        "dist",
    }
)

_TEST_DIR_PARTS = frozenset({"tests", "test", "__tests__"})


@dataclass
class PatchManifest:
    solution_files: list[str] = field(default_factory=list)
    gold_protected_test_files: list[str] = field(default_factory=list)
    generated_artifacts: list[str] = field(default_factory=list)
    temporary_artifacts: list[str] = field(default_factory=list)
    apex_control_files: list[str] = field(default_factory=list)
    dependency_artifacts: list[str] = field(default_factory=list)
    vendored_upstream_artifacts: list[str] = field(default_factory=list)
    unknown_files: list[str] = field(default_factory=list)

    @property
    def excluded_files(self) -> list[str]:
        return (
            list(self.generated_artifacts)
            + list(self.temporary_artifacts)
            + list(self.apex_control_files)
            + list(self.dependency_artifacts)
            + list(self.vendored_upstream_artifacts)
            + list(self.gold_protected_test_files)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "solution_files": list(self.solution_files),
            "gold_protected_test_files": list(self.gold_protected_test_files),
            "generated_artifacts": list(self.generated_artifacts),
            "temporary_artifacts": list(self.temporary_artifacts),
            "apex_control_files": list(self.apex_control_files),
            "dependency_artifacts": list(self.dependency_artifacts),
            "vendored_upstream_artifacts": list(self.vendored_upstream_artifacts),
            "unknown_files": list(self.unknown_files),
            "excluded_files": self.excluded_files,
        }


@dataclass(frozen=True)
class SanitizedPatch:
    patch_text: str
    manifest: PatchManifest
    stripped_test_paths: list[str] = field(default_factory=list)
    stripped_collection_critical_paths: list[str] = field(default_factory=list)


def _normalize(path: Any) -> str:
    text = str(path or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _normalized_set(paths: Iterable[Any]) -> set[str]:
    return {_normalize(path) for path in paths if _normalize(path)}


def _is_gold_suite_visible(evidence_mode: str) -> bool:
    return str(evidence_mode or "").strip() == "gold_suite_visible"


def _matches_any(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _path_matches_any_root(path: str, roots: Iterable[Any]) -> bool:
    normalized = _normalize(path)
    if not normalized:
        return False
    for root in _normalized_set(roots):
        root = root.rstrip("/")
        if root and (normalized == root or normalized.startswith(f"{root}/")):
            return True
    return False


def _is_dependency_artifact_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return any(
        part in _DEPENDENCY_DIR_NAMES or part.startswith(".deps") or part.endswith(".egg-info")
        for part in parts
    )


# Part C: vendored-upstream-source detection. These signals are STRUCTURAL
# (path shape only) — no repo/package/language name is hardcoded, so the rule is
# strictly general (Layer A).
_VENDORED_ARCHIVE_SUFFIXES = (
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tar.xz",
    ".tar",
    ".zip",
    ".whl",
)
# Package-metadata filenames that only appear inside a built/installed package
# tree (sdist/wheel/site-packages), never in a hand-written source repair.
_VENDORED_METADATA_NAMES = frozenset(
    {
        "PKG-INFO",
        "METADATA",
        "RECORD",
        "WHEEL",
        "top_level.txt",
        "entry_points.txt",
        "INSTALLER",
    }
)
_VENDORED_DIR_PARTS = frozenset({"site-packages", "dist-packages"})
# A versioned release directory the agent extracted, e.g. ``pytest-8.3.5``,
# ``numpy-1.26.4``, ``foo-2.0``. Name, a dash, then a dotted version.
_VERSIONED_RELEASE_DIR_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.]*-v?\d+(?:\.\d+)+[A-Za-z0-9.\-]*$")


def _is_archive_path(path: str) -> bool:
    lowered = _normalize(path).lower()
    return any(lowered.endswith(suffix) for suffix in _VENDORED_ARCHIVE_SUFFIXES)


def _vendored_upstream_signal(path: str) -> bool:
    """True when the path is part of a vendored upstream release/install tree.

    Structural-only: a ``site-packages``/``dist-packages`` component, a
    package-metadata file (PKG-INFO/RECORD/WHEEL/...), or a versioned release
    directory component (``pytest-8.3.5/...``). Used to strip-and-requalify a
    download-tainted candidate without ever touching the repo's own scored tests
    (the caller gates this on ``path not in protected_test_files``).
    """
    normalized = _normalize(path)
    if not normalized:
        return False
    parts = PurePosixPath(normalized).parts
    for part in parts:
        if part in _VENDORED_DIR_PARTS:
            return True
        if part in _VENDORED_METADATA_NAMES:
            return True
        # A versioned release dir as an interior component (the extracted root),
        # but never the leaf file itself unless it is also metadata above.
        if part is not parts[-1] and _VERSIONED_RELEASE_DIR_RE.match(part):
            return True
    return False


def _is_ambiguous_tests_py_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    if not parts or parts[-1].lower() != "tests.py":
        return False
    return not any(part.lower() in _TEST_DIR_PARTS for part in parts[:-1])


def _is_unrooted_test_filename_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    if not parts:
        return False
    if any(part.lower() in _TEST_DIR_PARTS for part in parts[:-1]):
        return False
    name = parts[-1].lower()
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or name.endswith("_test.go")
        or name.endswith("_test.rs")
        or name.endswith("_spec.rb")
        or ".test." in name
        or ".spec." in name
    )


def _is_heuristic_protected_test_path(path: str) -> bool:
    return is_test_path(path) and not _is_ambiguous_tests_py_path(path)


def classify_patch_path(
    path: Any,
    *,
    evidence_mode: str = "",
    incomplete_test_files: Iterable[Any] = (),
    collection_critical_paths: Iterable[Any] = (),
    protected_test_files: Iterable[Any] = (),
    dependency_artifact_paths: Iterable[Any] = (),
) -> PatchPathCategory:
    normalized = _normalize(path)
    if not normalized:
        return PatchPathCategory.UNKNOWN
    name = Path(normalized).name
    if (
        name in _APEX_CONTROL_NAMES
        or normalized.lower() in _AGENT_RUNTIME_METADATA_PATHS
        or normalized.startswith(".apex")
        or normalized.startswith("_apex")
        or "/.apex" in normalized
        or "/_apex" in normalized
    ):
        return PatchPathCategory.APEX_CONTROL_FILE
    if name in _GENERATED_NAMES or _matches_any(normalized, _GENERATED_GLOBS):
        return PatchPathCategory.GENERATED_ARTIFACT
    if _matches_any(normalized, _TEMPORARY_GLOBS):
        return PatchPathCategory.TEMPORARY_ARTIFACT
    if (
        _is_dependency_artifact_path(normalized)
        or _matches_any(normalized, _DEPENDENCY_GLOBS)
        or _path_matches_any_root(normalized, dependency_artifact_paths)
    ):
        return PatchPathCategory.DEPENDENCY_ARTIFACT
    # Part C: a vendored upstream release/archive/install tree is junk in any
    # mode and is stripped here — BEFORE the gold-protected-test block — so a
    # vendored ``testing/`` file is requalified as VENDORED, not mistaken for an
    # edit to the repo's own scored tests. The ``protected`` gate is the one
    # override boundary: a path that is literally a repo-own scored test still
    # falls through to GOLD_PROTECTED_TEST (reject wins), so an attacker cannot
    # disguise a real-test edit by nesting it under a versioned dir.
    if _is_archive_path(normalized) or _vendored_upstream_signal(normalized):
        if normalized not in _normalized_set(protected_test_files):
            return PatchPathCategory.VENDORED_UPSTREAM_ARTIFACT
    if _is_gold_suite_visible(evidence_mode):
        # WS3A backstop: APEX writes a per-rollout AGENTS.md/CLAUDE.md to feed the
        # wrapped CLI repo guidance; it is deleted before diff. If a CLI backend
        # returns its own diff that touched the file, quarantine it here. Gated to
        # gold mode only — in real usage a user may legitimately edit these.
        if name.lower() in _AGENT_CONTEXT_FILENAMES:
            return PatchPathCategory.APEX_CONTROL_FILE
        incomplete = _normalized_set(incomplete_test_files)
        collection_critical = _normalized_set(collection_critical_paths)
        protected = _normalized_set(protected_test_files)
        heuristic_protected = _is_heuristic_protected_test_path(normalized)
        if protected and _is_unrooted_test_filename_path(normalized):
            heuristic_protected = False
        if normalized not in incomplete and (
            normalized in protected or heuristic_protected or normalized in collection_critical
        ):
            return PatchPathCategory.GOLD_PROTECTED_TEST
    return PatchPathCategory.SOLUTION_FILE


def build_patch_manifest(
    paths: Iterable[Any],
    *,
    evidence_mode: str = "",
    incomplete_test_files: Iterable[Any] = (),
    collection_critical_paths: Iterable[Any] = (),
    protected_test_files: Iterable[Any] = (),
    dependency_artifact_paths: Iterable[Any] = (),
) -> PatchManifest:
    manifest = PatchManifest()
    incomplete = _normalized_set(incomplete_test_files)
    collection_critical = _normalized_set(collection_critical_paths)
    for raw in paths:
        path = _normalize(raw)
        if not path:
            continue
        category = classify_patch_path(
            path,
            evidence_mode=evidence_mode,
            incomplete_test_files=incomplete,
            collection_critical_paths=collection_critical,
            protected_test_files=protected_test_files,
            dependency_artifact_paths=dependency_artifact_paths,
        )
        if category == PatchPathCategory.APEX_CONTROL_FILE:
            manifest.apex_control_files.append(path)
        elif category == PatchPathCategory.GOLD_PROTECTED_TEST:
            manifest.gold_protected_test_files.append(path)
        elif category == PatchPathCategory.GENERATED_ARTIFACT:
            manifest.generated_artifacts.append(path)
        elif category == PatchPathCategory.TEMPORARY_ARTIFACT:
            manifest.temporary_artifacts.append(path)
        elif category == PatchPathCategory.DEPENDENCY_ARTIFACT:
            manifest.dependency_artifacts.append(path)
        elif category == PatchPathCategory.VENDORED_UPSTREAM_ARTIFACT:
            manifest.vendored_upstream_artifacts.append(path)
        elif category == PatchPathCategory.SOLUTION_FILE:
            manifest.solution_files.append(path)
        else:
            manifest.unknown_files.append(path)
    for attr in (
        "solution_files",
        "gold_protected_test_files",
        "generated_artifacts",
        "temporary_artifacts",
        "apex_control_files",
        "dependency_artifacts",
        "vendored_upstream_artifacts",
        "unknown_files",
    ):
        setattr(manifest, attr, sorted(dict.fromkeys(getattr(manifest, attr))))
    return manifest


def filter_solution_paths(
    paths: Iterable[Any],
    *,
    evidence_mode: str = "",
    incomplete_test_files: Iterable[Any] = (),
    collection_critical_paths: Iterable[Any] = (),
    protected_test_files: Iterable[Any] = (),
    dependency_artifact_paths: Iterable[Any] = (),
) -> list[str]:
    return build_patch_manifest(
        paths,
        evidence_mode=evidence_mode,
        incomplete_test_files=incomplete_test_files,
        collection_critical_paths=collection_critical_paths,
        protected_test_files=protected_test_files,
        dependency_artifact_paths=dependency_artifact_paths,
    ).solution_files


def _diff_token_to_path(token: str) -> str:
    value = token.strip()
    if value in {"", "/dev/null"}:
        return ""
    if value.startswith(("a/", "b/")):
        value = value[2:]
    return _normalize(value.strip('"'))


def _patch_block_path(lines: list[str]) -> str:
    for line in lines:
        if line.startswith("diff --git "):
            parts = line.strip().split()
            if len(parts) >= 4:
                return _diff_token_to_path(parts[3]) or _diff_token_to_path(parts[2])
        if line.startswith("+++ ") and not line.startswith("+++ /dev/null"):
            return _diff_token_to_path(line[4:])
        if line.startswith("--- ") and not line.startswith("--- /dev/null"):
            return _diff_token_to_path(line[4:])
    return ""


def sanitize_patch_text(
    patch_text: str,
    *,
    evidence_mode: str = "",
    incomplete_test_files: Iterable[Any] = (),
    collection_critical_paths: Iterable[Any] = (),
    protected_test_files: Iterable[Any] = (),
    dependency_artifact_paths: Iterable[Any] = (),
) -> SanitizedPatch:
    lines = str(patch_text or "").splitlines(keepends=True)
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("diff --git ") and current:
            blocks.append(current)
            current = [line]
            continue
        current.append(line)
    if current:
        blocks.append(current)

    kept: list[str] = []
    paths: list[str] = []
    stripped_test_paths: list[str] = []
    stripped_collection_critical_paths: list[str] = []
    collection_critical = _normalized_set(collection_critical_paths)
    incomplete = _normalized_set(incomplete_test_files)

    for block in blocks:
        path = _patch_block_path(block)
        if not path:
            kept.extend(block)
            continue
        paths.append(path)
        category = classify_patch_path(
            path,
            evidence_mode=evidence_mode,
            incomplete_test_files=incomplete,
            collection_critical_paths=collection_critical,
            protected_test_files=protected_test_files,
            dependency_artifact_paths=dependency_artifact_paths,
        )
        if category in {
            PatchPathCategory.GOLD_PROTECTED_TEST,
            PatchPathCategory.GENERATED_ARTIFACT,
            PatchPathCategory.TEMPORARY_ARTIFACT,
            PatchPathCategory.APEX_CONTROL_FILE,
            PatchPathCategory.DEPENDENCY_ARTIFACT,
            PatchPathCategory.VENDORED_UPSTREAM_ARTIFACT,
        }:
            if path in collection_critical and path not in incomplete:
                stripped_collection_critical_paths.append(path)
            elif category == PatchPathCategory.GOLD_PROTECTED_TEST:
                # Only genuine repo-own scored-test edits land here; this list
                # drives ``protected_test_violation``. Vendored upstream trees
                # are intentionally NOT recorded here (they are stripped silently
                # and the candidate is requalified on its implementation diff).
                stripped_test_paths.append(path)
            continue
        kept.extend(block)

    return SanitizedPatch(
        patch_text="".join(kept),
        manifest=build_patch_manifest(
            paths,
            evidence_mode=evidence_mode,
            incomplete_test_files=incomplete,
            collection_critical_paths=collection_critical,
            protected_test_files=protected_test_files,
            dependency_artifact_paths=dependency_artifact_paths,
        ),
        stripped_test_paths=sorted(dict.fromkeys(stripped_test_paths)),
        stripped_collection_critical_paths=sorted(
            dict.fromkeys(stripped_collection_critical_paths)
        ),
    )


def sanitize_candidate_worktree(
    *,
    candidate_worktree: Path,
    baseline_repo_dir: Optional[Path],
    changed_files: Iterable[Any],
    artifacts_dir: Path,
    evidence_mode: str = "",
    incomplete_test_files: Iterable[Any] = (),
    collection_critical_paths: Iterable[Any] = (),
    protected_test_files: Iterable[Any] = (),
    dependency_artifact_paths: Iterable[Any] = (),
) -> tuple[Path, PatchManifest, list[str]]:
    """Return a worktree with excluded artifacts removed/restored.

    The original worktree is left untouched unless no excluded files are
    present. Excluded files are copied to the rollout artifacts directory for
    audit, then deleted or restored in the sanitized copy.
    """

    manifest = build_patch_manifest(
        changed_files,
        evidence_mode=evidence_mode,
        incomplete_test_files=incomplete_test_files,
        collection_critical_paths=collection_critical_paths,
        protected_test_files=protected_test_files,
        dependency_artifact_paths=dependency_artifact_paths,
    )
    excluded = manifest.excluded_files
    if not excluded:
        return candidate_worktree, manifest, []

    sanitized_root = artifacts_dir / "sanitized_patch_worktree"
    if sanitized_root.exists():
        shutil.rmtree(sanitized_root, ignore_errors=True)
    shutil.copytree(candidate_worktree, sanitized_root, symlinks=True)

    quarantine_dir = artifacts_dir / "quarantined_patch_artifacts"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    actions: list[str] = []
    for rel_path in excluded:
        source = sanitized_root / rel_path
        category = classify_patch_path(
            rel_path,
            evidence_mode=evidence_mode,
            incomplete_test_files=incomplete_test_files,
            collection_critical_paths=collection_critical_paths,
            protected_test_files=protected_test_files,
            dependency_artifact_paths=dependency_artifact_paths,
        )
        if source.exists() and source.is_file():
            target = quarantine_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        baseline = baseline_repo_dir / rel_path if baseline_repo_dir is not None else None
        if (
            category in {PatchPathCategory.APEX_CONTROL_FILE, PatchPathCategory.DEPENDENCY_ARTIFACT}
            and baseline is not None
            and baseline.exists()
            and baseline.is_file()
        ):
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(baseline, source)
            actions.append(f"restored:{rel_path}")
            continue
        if category == PatchPathCategory.GOLD_PROTECTED_TEST:
            if baseline is not None and baseline.exists() and baseline.is_file():
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(baseline, source)
            elif source.exists():
                if source.is_dir():
                    shutil.rmtree(source, ignore_errors=True)
                else:
                    source.unlink(missing_ok=True)
            actions.append(f"stripped_test:{rel_path}")
            continue
        if source.exists():
            if source.is_dir():
                shutil.rmtree(source, ignore_errors=True)
            else:
                source.unlink(missing_ok=True)
            actions.append(f"removed:{rel_path}")
    return sanitized_root, manifest, actions
