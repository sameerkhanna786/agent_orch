"""
Helpers for interpreting changed paths in git worktrees.
"""

from __future__ import annotations

import ast
import logging
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

logger = logging.getLogger("apex.core.git_utils")

from .filesystem import IgnoreCallable, copy_path, remove_path, remove_symlink_escapes

_IGNORED_CHANGE_MARKERS = (
    ".pyc",
    "_apex_",
)
_IGNORED_CHANGE_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".apex_agent_teams",
    ".apex_agent_runtime",
    ".apex_verification_reports",
    "node_modules",
    ".venv",
    "venv",
    "myenv",
    "site-packages",
}
_IGNORED_CHANGE_FILENAMES = {
    "rollout_report.json",
    "pyvenv.cfg",
}
_IGNORED_CHANGE_EXACT_PATHS = {
    # Provider CLIs may create project-local control files while bootstrapping;
    # they are agent runtime metadata, not candidate solution edits.
    ".claude/settings.json",
    ".claude/settings.local.json",
}
_IGNORED_ROOT_CHANGE_GLOBS = (
    "core",
    "core.*",
    "vgcore.*",
)
_GIT_CHANGED_FILES_TIMEOUT_SECONDS = 30.0
_GIT_SNAPSHOT_CLONE_TIMEOUT_SECONDS = 300.0
_GIT_SNAPSHOT_STATUS_TIMEOUT_SECONDS = 60.0
_GIT_SUBMODULE_SYNC_TIMEOUT_SECONDS = 300.0
_GIT_TIMEOUT_RETURN_CODE = 124


def _kill_process_tree(process: subprocess.Popen[str]) -> None:
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
            return
    except ProcessLookupError:
        return
    except OSError:
        pass
    try:
        process.kill()
    except OSError:
        pass


def _run_git(
    args: list[str],
    *,
    cwd: Path,
    timeout: float = _GIT_CHANGED_FILES_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    command = ["git", *args]
    popen_kwargs: dict[str, object] = {
        "cwd": str(cwd),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True
    process = subprocess.Popen(command, **popen_kwargs)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_tree(process)
        try:
            stdout, stderr = process.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        return subprocess.CompletedProcess(
            command,
            _GIT_TIMEOUT_RETURN_CODE,
            stdout or "",
            (stderr or "") + f"\ngit command timed out after {timeout:.1f}s",
        )
    return subprocess.CompletedProcess(command, process.returncode, stdout or "", stderr or "")


# --------------------------------------------------------------------------
# Stale git-lock recovery
# --------------------------------------------------------------------------
#
# Concurrent git operations against the same repository (e.g. several
# WorktreePool rollouts resetting sibling worktrees under one source repo, or
# upstream tooling that uses GitPython) can leave a stale ``*.lock`` file under
# ``.git`` / ``.git/worktrees`` after the writing process dies. A git mutation
# that then trips over the lock fails with a non-zero return code even though
# nothing is actually writing. This generalizes the prior Commit0-audit-only
# recovery (``commit0_benchmark._run_git_for_official_audit``) into a reusable,
# benchmark-agnostic helper: it scans command output for ``*.lock`` paths under
# git administrative state, and either removes a *demonstrably stale* lock
# (mtime age >= threshold => the writer is dead) and retries immediately, or —
# when the lock is fresh (a live writer is quiescing) — backs off and retries
# WITHOUT deleting it. The age gate guarantees a live lock is only ever waited
# on, never deleted. After ``max_attempts`` it degrades to returning the last
# failing result, i.e. exactly today's behavior.
_GIT_LOCK_PATH_RE = re.compile(
    r"['\"]?(?P<path>[^'\"\n]*?[^'\"\s/\\]+\.lock)['\"]?",
    re.IGNORECASE,
)
_GIT_LOCK_STALE_AGE_SECONDS = 2.0
_GIT_LOCK_RETRY_BASE_BACKOFF_SECONDS = 0.25
_GIT_LOCK_RECOVERY_MAX_ATTEMPTS = 4


def git_lock_paths_from_output(output: str, *, cwd: Path | str) -> list[Path]:
    """Extract candidate ``*.lock`` paths under git administrative state.

    Only locks whose resolved path contains a ``.git`` directory or a
    ``worktrees`` component are returned, so this never touches unrelated
    ``*.lock`` files in the working tree.
    """

    if ".lock" not in output:
        return []
    cwd_path = Path(cwd)
    lock_paths: list[Path] = []
    seen: set[str] = set()
    for match in _GIT_LOCK_PATH_RE.finditer(output):
        raw_path = (match.group("path") or "").strip()
        if not raw_path or not raw_path.endswith(".lock"):
            continue
        lock_path = Path(raw_path)
        if not lock_path.is_absolute():
            lock_path = cwd_path / lock_path
        # Scope strictly to git administrative state (``.git`` /
        # ``.git/worktrees/<name>/...``); never delete arbitrary working-tree
        # ``*.lock`` files.
        if ".git" not in lock_path.parts and "worktrees" not in lock_path.parts:
            continue
        key = str(lock_path)
        if key in seen:
            continue
        seen.add(key)
        lock_paths.append(lock_path)
    return lock_paths


def _recover_stale_git_locks(
    lock_paths: Sequence[Path],
    *,
    stale_lock_age_seconds: float,
) -> bool:
    """Delete locks whose mtime age >= threshold. Returns True if any removed.

    A fresh lock (age below the threshold) is left untouched — the caller backs
    off and retries instead, so a live writer is never disturbed.
    """

    recovered = False
    now = time.time()
    for lock_path in lock_paths:
        try:
            stat_result = lock_path.stat()
        except FileNotFoundError:
            # Already gone (the writer released it) — treat as recoverable so
            # the caller retries immediately rather than sleeping.
            recovered = True
            continue
        except OSError:
            logger.debug("Unable to stat git lock %s", lock_path, exc_info=True)
            continue
        age_seconds = now - float(stat_result.st_mtime)
        if age_seconds < stale_lock_age_seconds:
            # Fresh lock — a live writer is probably quiescing. Do NOT delete.
            continue
        try:
            lock_path.unlink()
        except FileNotFoundError:
            recovered = True
            continue
        except OSError:
            logger.debug("Unable to unlink stale git lock %s", lock_path, exc_info=True)
            continue
        logger.warning("Removed stale git lock before retry: %s", lock_path)
        recovered = True
    return recovered


def run_git_with_lock_recovery(
    args: Sequence[str],
    *,
    cwd: Path | str,
    timeout: float = _GIT_CHANGED_FILES_TIMEOUT_SECONDS,
    max_attempts: int = _GIT_LOCK_RECOVERY_MAX_ATTEMPTS,
    stale_lock_age_seconds: float = _GIT_LOCK_STALE_AGE_SECONDS,
    base_backoff: float = _GIT_LOCK_RETRY_BASE_BACKOFF_SECONDS,
    runner: Optional[Callable[..., subprocess.CompletedProcess[str]]] = None,
    sleep: Optional[Callable[[float], None]] = None,
) -> subprocess.CompletedProcess[str]:
    """Run a git command, recovering from stale ``*.lock`` contention.

    ``args`` may include a leading ``"git"`` or not — it is normalized to the
    bare argument list expected by :func:`_run_git`. On a non-zero return whose
    output mentions a git ``*.lock`` under administrative state, a stale lock
    (mtime age >= ``stale_lock_age_seconds``) is deleted and the command retried
    immediately; a fresh lock triggers exponential backoff (``base_backoff *
    2**attempt``) then retry. After ``max_attempts`` the last result is
    returned, matching the pre-existing single-shot behavior.

    ``runner``/``sleep`` are injectable for testing.
    """

    run = runner or _run_git
    nap = sleep or time.sleep
    normalized = list(args)
    if normalized and normalized[0] == "git":
        normalized = normalized[1:]
    attempts = max(1, int(max_attempts))
    last_result: Optional[subprocess.CompletedProcess[str]] = None
    for attempt in range(attempts):
        result = run(normalized, cwd=Path(cwd), timeout=timeout)
        last_result = result
        if result.returncode == 0:
            return result
        if attempt >= attempts - 1:
            return result
        output = "\n".join(part for part in ((result.stdout or ""), (result.stderr or "")) if part)
        lock_paths = git_lock_paths_from_output(output, cwd=cwd)
        if not lock_paths:
            # Failure unrelated to lock contention — do not retry blindly.
            return result
        if _recover_stale_git_locks(
            lock_paths,
            stale_lock_age_seconds=stale_lock_age_seconds,
        ):
            # Stale lock cleared — retry immediately.
            continue
        # Lock is live/fresh; back off and retry.
        nap(max(0.0, float(base_backoff) * (2**attempt)))
    return (
        last_result if last_result is not None else run(normalized, cwd=Path(cwd), timeout=timeout)
    )


def is_ignored_change_path(path: str) -> bool:
    normalized = normalize_changed_path(path)
    if not normalized:
        return False
    lowered = normalized.lower()
    if lowered in _IGNORED_CHANGE_EXACT_PATHS:
        return True
    if any(marker in lowered for marker in _IGNORED_CHANGE_MARKERS):
        return True
    parts = {part.lower() for part in Path(normalized).parts if part}
    if _IGNORED_CHANGE_PARTS.intersection(parts):
        return True
    if "/" not in normalized and any(
        Path(normalized).match(pattern) for pattern in _IGNORED_ROOT_CHANGE_GLOBS
    ):
        return True
    return Path(normalized).name.lower() in _IGNORED_CHANGE_FILENAMES


def ignored_change_pathspecs() -> list[str]:
    return [
        ":(exclude)**/__pycache__/*",
        ":(exclude)*.pyc",
        ":(exclude).pytest_cache/*",
        ":(exclude)**/.pytest_cache/*",
        ":(exclude).mypy_cache/*",
        ":(exclude)**/.mypy_cache/*",
        ":(exclude).apex_agent_teams/*",
        ":(exclude)**/.apex_agent_teams/*",
        ":(exclude).apex_agent_runtime/*",
        ":(exclude)**/.apex_agent_runtime/*",
        ":(exclude).apex_verification_reports/*",
        ":(exclude)**/.apex_verification_reports/*",
        ":(exclude)node_modules/*",
        ":(exclude)**/node_modules/*",
        ":(exclude).venv/*",
        ":(exclude)**/.venv/*",
        ":(exclude)venv/*",
        ":(exclude)**/venv/*",
        ":(exclude)myenv/*",
        ":(exclude)**/myenv/*",
        ":(exclude)**/site-packages/*",
        ":(exclude)rollout_report.json",
        ":(exclude)**/rollout_report.json",
        ":(exclude)pyvenv.cfg",
        ":(exclude)**/pyvenv.cfg",
        ":(exclude).claude/settings.json",
        ":(exclude)**/.claude/settings.json",
        ":(exclude).claude/settings.local.json",
        ":(exclude)**/.claude/settings.local.json",
        ":(exclude)_apex_*",
        ":(exclude)**/_apex_*",
        # Pytest plugin + ids file we stage into the worktree for ID-direct
        # evaluation. Excluded so the rollout's diff doesn't carry our
        # scoring scaffolding into the patch artifact.
        ":(exclude).apex_expected_test_ids.txt",
        ":(exclude)**/.apex_expected_test_ids.txt",
        ":(exclude)_apex_expected_test_ids.txt",
        ":(exclude)**/_apex_expected_test_ids.txt",
        # Agent scratch files at repo root. Jedi rollouts repeatedly created
        # patch_*.py / patch_blockers*.py / fix_all.py / test_my_*.py at the
        # repo root as throwaway diagnostic scripts, then submitted them as
        # part of the patch. They never ran in tests and inflated the patch.
        ":(exclude)patch_*.py",
        ":(exclude)patch_blockers*.py",
        ":(exclude)fix_all*.py",
        ":(exclude)fix_my_*.py",
        ":(exclude)test_my_*.py",
        ":(exclude)scratch_*.py",
        # POSIX crash dumps can appear as root-level binary files after failed
        # native/extension tests; they are runtime artifacts, not solution code.
        ":(exclude)core",
        ":(exclude)core.*",
        ":(exclude)vgcore.*",
    ]


def cleanup_provider_project_metadata(worktree: Path) -> list[str]:
    """Remove untracked provider CLI project metadata from a candidate worktree."""

    removed: list[str] = []
    root = Path(worktree)
    if not root.exists() or not root.is_dir():
        return removed
    top_level = _run_git(["rev-parse", "--show-toplevel"], cwd=root)
    if top_level.returncode != 0:
        return removed
    try:
        if Path(top_level.stdout.strip()).resolve() != root.resolve():
            return removed
    except OSError:
        return removed

    for rel_path in sorted(_IGNORED_CHANGE_EXACT_PATHS):
        target = root / rel_path
        if not target.exists() and not target.is_symlink():
            continue
        tracked = _run_git(["ls-files", "--error-unmatch", "--", rel_path], cwd=root)
        if tracked.returncode == 0:
            continue
        try:
            remove_path(target)
            removed.append(rel_path)
        except OSError:
            logger.debug("Failed to remove provider project metadata %s", target, exc_info=True)
            continue

        parent = target.parent
        while parent != root:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    return removed


def parse_porcelain_path(line: str) -> str:
    if len(line) < 4:
        return ""

    payload = line[3:].strip()
    if " -> " in payload:
        payload = payload.split(" -> ", 1)[1].strip()

    if payload.startswith('"') and payload.endswith('"'):
        try:
            parsed = ast.literal_eval(payload)
        except (SyntaxError, ValueError):
            parsed = payload
        if isinstance(parsed, str):
            payload = parsed

    return payload.strip()


def normalize_changed_path(path: str) -> str:
    normalized = path.strip()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.rstrip("/")


def relativize_git_worktree_metadata(worktree: Path) -> bool:
    """Make native git-worktree metadata portable across root remounts.

    Docker target runtimes mount the whole task sandbox at a different root
    (for example `/workspace`). Native `git worktree add` writes absolute
    host paths into the worktree `.git` file and the admin `gitdir` backlink;
    rewriting both to relative paths keeps `git` usable inside either root.
    """

    gitfile = Path(worktree) / ".git"
    try:
        content = gitfile.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    prefix = "gitdir:"
    if not content.lower().startswith(prefix):
        return False

    raw_gitdir = content[len(prefix) :].strip()
    if not raw_gitdir:
        return False
    gitdir_path = Path(raw_gitdir)
    if not gitdir_path.is_absolute():
        gitdir_path = gitfile.parent / gitdir_path
    try:
        admin_dir = gitdir_path.resolve(strict=False)
        rel_admin = os.path.relpath(admin_dir, gitfile.parent)
        gitfile.write_text(f"gitdir: {Path(rel_admin).as_posix()}\n", encoding="utf-8")
        admin_gitdir = admin_dir / "gitdir"
        if admin_gitdir.exists() or admin_gitdir.is_symlink():
            rel_worktree_git = os.path.relpath(gitfile, admin_dir)
            admin_gitdir.write_text(f"{Path(rel_worktree_git).as_posix()}\n", encoding="utf-8")
    except OSError:
        return False
    return True


def repo_declares_git_submodules(repo: Path) -> bool:
    gitmodules = Path(repo) / ".gitmodules"
    try:
        return gitmodules.is_file() and bool(gitmodules.read_text(encoding="utf-8").strip())
    except OSError:
        return False


def sync_git_submodules(
    repo: Path,
    *,
    timeout: float = _GIT_SUBMODULE_SYNC_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str] | None:
    """Initialize declared git submodules for ``repo`` if it has any."""

    repo = Path(repo)
    if not repo_declares_git_submodules(repo):
        return None
    sync = _run_git(
        ["submodule", "sync", "--recursive"],
        cwd=repo,
        timeout=min(float(timeout or _GIT_SUBMODULE_SYNC_TIMEOUT_SECONDS), 300.0),
    )
    if sync.returncode != 0:
        return sync
    return _run_git(
        ["submodule", "update", "--init", "--recursive"],
        cwd=repo,
        timeout=float(timeout or _GIT_SUBMODULE_SYNC_TIMEOUT_SECONDS),
    )


def expand_changed_paths(
    worktree: Path,
    paths: list[str],
    *,
    ignored_predicate: Callable[[str], bool] | None = None,
) -> list[str]:
    predicate = ignored_predicate or is_ignored_change_path
    changed: list[str] = []

    for raw_path in paths:
        rel_path = normalize_changed_path(raw_path)
        if not rel_path or predicate(rel_path):
            continue

        absolute_path = worktree / rel_path
        exists = _changed_path_exists(absolute_path)
        if exists is None:
            continue
        if exists and _changed_path_is_dir(absolute_path):
            for child in sorted(_changed_path_regular_files(absolute_path)):
                child_rel_path = normalize_changed_path(str(child.relative_to(worktree)))
                if child_rel_path and not predicate(child_rel_path):
                    changed.append(child_rel_path)
            continue

        changed.append(rel_path)

    return sorted(dict.fromkeys(changed))


def _changed_path_exists(path: Path) -> bool | None:
    try:
        return path.exists()
    except OSError as exc:
        logger.debug("Skipping unstatable changed path %s: %s", path, exc)
        return None


def _changed_path_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError as exc:
        logger.debug("Treating changed path as non-directory after stat failure %s: %s", path, exc)
        return False


def _changed_path_regular_files(path: Path) -> list[Path]:
    children: list[Path] = []
    try:
        iterator = path.rglob("*")
        for child in iterator:
            try:
                if child.is_file():
                    children.append(child)
            except OSError as exc:
                logger.debug("Skipping unreadable changed child path %s: %s", child, exc)
    except OSError as exc:
        logger.debug("Skipping unreadable changed directory tree %s: %s", path, exc)
    return children


def list_changed_files(
    worktree: Path,
    *,
    baseline_ref: str | None = None,
    ignored_predicate: Callable[[str], bool] | None = None,
) -> list[str]:
    predicate = ignored_predicate or is_ignored_change_path
    resolved_worktree = worktree.resolve()
    top_level = _run_git(["rev-parse", "--show-toplevel"], cwd=worktree)
    if top_level.returncode != 0:
        return []
    try:
        resolved_top_level = Path(top_level.stdout.strip()).resolve()
    except OSError:
        return []
    if resolved_top_level != resolved_worktree:
        return []
    add_result = run_git_with_lock_recovery(["add", "-N", "."], cwd=worktree)
    if add_result.returncode == _GIT_TIMEOUT_RETURN_CODE:
        return []
    if baseline_ref:
        result = run_git_with_lock_recovery(
            [
                "diff",
                "--name-only",
                "--relative",
                baseline_ref,
                "--",
                ".",
                *ignored_change_pathspecs(),
            ],
            cwd=worktree,
        )
        if result.returncode == 0:
            raw_paths = [
                normalize_changed_path(line) for line in result.stdout.splitlines() if line.strip()
            ]
            return expand_changed_paths(worktree, raw_paths, ignored_predicate=predicate)
        if result.returncode == _GIT_TIMEOUT_RETURN_CODE:
            return []

    result = run_git_with_lock_recovery(["status", "--porcelain"], cwd=worktree)
    if result.returncode != 0:
        return []
    raw_paths = [parse_porcelain_path(line) for line in result.stdout.splitlines()]
    return expand_changed_paths(worktree, raw_paths, ignored_predicate=predicate)


def is_git_repo(path: Path) -> bool:
    result = _run_git(["rev-parse", "--show-toplevel"], cwd=path)
    return result.returncode == 0


def clone_git_repo_with_overlay(
    source: Path,
    destination: Path,
    *,
    ignore: IgnoreCallable = None,
    ignored_predicate: Callable[[str], bool] | None = None,
    restrict_symlinks_to_root: bool = False,
) -> bool:
    """Clone ``source`` into ``destination`` and overlay dirty worktree state.

    The clone materializes the committed ``HEAD`` state into ``destination`` and
    then reapplies any modified, deleted, renamed, copied, or untracked files
    from ``source`` so callers get an isolated snapshot of the *current*
    worktree, not just the committed baseline.
    """

    source_path = Path(source)
    destination_path = Path(destination)
    predicate = ignored_predicate or is_ignored_change_path
    clone = _run_git(
        ["clone", "--quiet", str(source_path), str(destination_path)],
        cwd=source_path.parent,
        timeout=_GIT_SNAPSHOT_CLONE_TIMEOUT_SECONDS,
    )
    if clone.returncode != 0:
        remove_path(destination_path)
        return False

    status = _run_git(
        ["status", "--porcelain"],
        cwd=source_path,
        timeout=_GIT_SNAPSHOT_STATUS_TIMEOUT_SECONDS,
    )
    if status.returncode != 0:
        remove_path(destination_path)
        return False

    for raw_line in status.stdout.splitlines():
        line = raw_line.rstrip()
        if len(line) < 3:
            continue
        status_code = line[:2]
        payload = line[3:].strip()
        old_path = payload
        new_path = payload
        if " -> " in payload:
            old_path, new_path = payload.split(" -> ", 1)
        old_path = normalize_changed_path(old_path)
        new_path = normalize_changed_path(new_path)

        if status_code == "??":
            if new_path and not predicate(new_path):
                copy_path(source_path / new_path, destination_path / new_path, ignore=ignore)
            continue

        if "D" in status_code:
            if old_path and not predicate(old_path):
                remove_path(destination_path / old_path)
            if new_path and new_path != old_path and not predicate(new_path):
                copy_path(source_path / new_path, destination_path / new_path, ignore=ignore)
            continue

        if "R" in status_code or "C" in status_code:
            if old_path and old_path != new_path and not predicate(old_path):
                remove_path(destination_path / old_path)
            if new_path and not predicate(new_path):
                copy_path(source_path / new_path, destination_path / new_path, ignore=ignore)
            continue

        if new_path and not predicate(new_path):
            copy_path(source_path / new_path, destination_path / new_path, ignore=ignore)

    if restrict_symlinks_to_root:
        remove_symlink_escapes(destination_path)
    return True
