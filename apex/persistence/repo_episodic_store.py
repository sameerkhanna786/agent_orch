"""Per-repo episodic memory (Decisive-Edge C.3).

Where :pymod:`apex.persistence.episodic_store` stores per-task episodes
(keyed by ``repo_signature::task_id``), this module stores per-REPO
patterns that hold across many tasks on the same repository, e.g.

    "this repo's winning patches are pytest-fixture-heavy"
    "this repo uses tox; ``tox -e py311`` is the canonical entrypoint"
    "winning patches on this repo have median diff size <20 lines"

These patterns are mined from completed solve runs and folded into the
agents' system-prompt "Repo conventions" section so subsequent solves
on the SAME repo don't re-discover well-trodden facts about the
codebase.

Design notes
------------

* **Append-only JSONL** with **dedup by pattern_type** — re-observing
  ``pytest_fixture_heavy`` doesn't add a duplicate; it merges
  confidences (max) and unions ``extracted_from_runs``.
* **Atomic appends via fcntl.flock** — same cross-process safety as
  Phase 6.2's :class:`apex.persistence.episodic_store.EpisodicStore`.
* **Storage layout** — ``~/.apex/repo_episodic/<repo_signature>/episodes.jsonl``.
* **Conservative confidence** — heuristic patterns start at 0.6-0.7
  confidence; downstream callers can re-weight when synthesising
  prompts. We never auto-decay; pattern relevance is reassessed each
  load.
* **Heuristic pattern set is INTENTIONALLY small** — easier to extend
  via :pyfunc:`RepoEpisodicStore.extract_patterns_from_run` than to
  ship 20 noisy heuristics on day one.

Public types
------------

* :class:`RepoEpisode` — one durable per-repo pattern record.
* :class:`RepoEpisodicStore` — load / append / extract API.
* :func:`render_repo_episodes_prompt_block` — prompt-side renderer used
  by the agents to inject "Repo conventions" into the system prompt.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional

logger = logging.getLogger("apex.persistence.repo_episodic_store")


# fcntl is POSIX-only; the storage layer degrades to a thread lock with a
# logged warning on Windows. Benchmark workers always run on Linux/macOS.
try:
    import fcntl  # type: ignore[unused-ignore]

    _FCNTL_AVAILABLE = True
except ImportError:  # pragma: no cover — Windows fallback
    fcntl = None  # type: ignore[assignment]
    _FCNTL_AVAILABLE = False


_REPO_EPISODIC_SCHEMA_VERSION = 1
_DEFAULT_STORE_DIRNAME = ".apex/repo_episodic"
_LOCK_TIMEOUT_SECONDS = 30.0


def _now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _resolve_store_root(directory: Optional[str | Path]) -> Path:
    if directory:
        return Path(directory).expanduser().resolve()
    return Path(os.path.expanduser("~")) / _DEFAULT_STORE_DIRNAME


def _sanitize_repo_signature(repo_signature: str) -> str:
    """Defensive cleanup so a malicious repo signature can't escape the root.

    Mirrors :pyfunc:`apex.persistence.episodic_store._sanitize_task_signature`
    one-for-one so the two stores share a hardening surface.
    """
    raw = (repo_signature or "").strip() or "unknown"
    parts = raw.replace("\\", "/").split("/")
    safe_parts = [p for p in parts if p not in ("", ".", "..")]
    safe = "_".join(safe_parts) if safe_parts else "unknown"
    return safe[:128]


@dataclass
class RepoEpisode:
    """One per-repo durable pattern record.

    Fields
    ------
    repo_signature
        Hash from :pyfunc:`apex.persistence.repo_memory.repo_signature_for_path`.
        Stamped on the record so the JSONL is self-describing if it leaks
        past its directory.
    pattern_type
        Stable identifier for the pattern, e.g. ``"pytest_fixture_heavy"``,
        ``"uses_tox"``. Pattern types are the dedup key.
    pattern_summary
        1-2 sentence human-readable description folded into the agent's
        "Repo conventions" prompt block.
    confidence
        Float in [0, 1]. Heuristic patterns start at 0.6-0.7; merge takes
        the max so re-observation strengthens the prior.
    extracted_from_runs
        Task ids that contributed evidence. Unioned on merge, capped at
        ``_MAX_RUNS_PER_PATTERN`` to keep the JSONL bounded.
    last_updated_iso
        UTC iso timestamp of the most recent observation.
    """

    repo_signature: str
    pattern_type: str
    pattern_summary: str
    confidence: float = 0.6
    extracted_from_runs: list[str] = field(default_factory=list)
    last_updated_iso: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "v": _REPO_EPISODIC_SCHEMA_VERSION,
            "repo_signature": str(self.repo_signature or ""),
            "pattern_type": str(self.pattern_type or ""),
            "pattern_summary": str(self.pattern_summary or ""),
            "confidence": float(self.confidence),
            "extracted_from_runs": [str(r) for r in self.extracted_from_runs],
            "last_updated_iso": str(self.last_updated_iso or ""),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RepoEpisode":
        return cls(
            repo_signature=str(data.get("repo_signature") or ""),
            pattern_type=str(data.get("pattern_type") or ""),
            pattern_summary=str(data.get("pattern_summary") or ""),
            confidence=float(data.get("confidence") or 0.0),
            extracted_from_runs=[str(r) for r in (data.get("extracted_from_runs") or [])],
            last_updated_iso=str(data.get("last_updated_iso") or ""),
        )


_MAX_RUNS_PER_PATTERN = 64


# ---------------------------------------------------------------------------
# Heuristic extraction primitives
# ---------------------------------------------------------------------------


# Each heuristic is a callable that takes (changed_files, patch_diff,
# rollout_summary) and returns either None (heuristic does not fire) or
# a (pattern_summary, confidence) tuple. Keeping the signature tight
# means ``extract_patterns_from_run`` stays readable as the heuristic
# set grows.

_PYTEST_FIXTURE_LINE_RE = re.compile(r"^\s*@(?:pytest\.fixture|fixture)\b")
_MONKEYPATCH_RE = re.compile(r"\bmonkeypatch\b")


def _is_python_file(path: str) -> bool:
    return str(path or "").endswith(".py")


def _patch_lines_added(diff: str) -> list[str]:
    """Lines added by the patch (no leading '+', no '+++' header lines)."""
    lines: list[str] = []
    for raw in (diff or "").splitlines():
        if raw.startswith("+++"):
            continue
        if raw.startswith("+"):
            lines.append(raw[1:])
    return lines


def _diff_added_line_count(diff: str) -> int:
    return sum(
        1 for raw in (diff or "").splitlines() if raw.startswith("+") and not raw.startswith("+++")
    )


def _diff_removed_line_count(diff: str) -> int:
    return sum(
        1 for raw in (diff or "").splitlines() if raw.startswith("-") and not raw.startswith("---")
    )


def _diff_total_changed_lines(diff: str) -> int:
    return _diff_added_line_count(diff) + _diff_removed_line_count(diff)


def _detect_pytest_fixture_heavy(diffs: list[str]) -> Optional[tuple[str, float]]:
    """Fires when >50% of winning Python diffs touch a pytest fixture decorator."""
    if not diffs:
        return None
    fixture_count = 0
    for diff in diffs:
        added = _patch_lines_added(diff)
        if any(_PYTEST_FIXTURE_LINE_RE.search(line) for line in added):
            fixture_count += 1
    if fixture_count == 0:
        return None
    fraction = fixture_count / len(diffs)
    if fraction <= 0.5:
        return None
    summary = (
        "Winning patches on this repo are pytest-fixture-heavy "
        f"({fixture_count}/{len(diffs)} winning diffs introduce or modify "
        "an @pytest.fixture). Prefer reusing existing fixtures and adding "
        "new ones at conftest scope rather than constructing test state inline."
    )
    confidence = min(0.6 + 0.2 * fraction, 0.9)
    return summary, round(confidence, 3)


def _detect_small_focused_prs(diffs: list[str]) -> Optional[tuple[str, float]]:
    """Fires when median winning diff has <20 changed lines."""
    if not diffs:
        return None
    sizes = sorted(_diff_total_changed_lines(d) for d in diffs)
    median = sizes[len(sizes) // 2]
    if median >= 20:
        return None
    summary = (
        f"Winning patches on this repo are typically small (median "
        f"{median} changed lines). Prefer the smallest possible diff and "
        "do not bundle adjacent refactors with the bug fix."
    )
    confidence = 0.7
    return summary, confidence


def _detect_monkeypatch_pattern(diffs: list[str]) -> Optional[tuple[str, float]]:
    """Fires when ≥50% of winning diffs reference ``monkeypatch``."""
    if not diffs:
        return None
    hits = 0
    for diff in diffs:
        added = _patch_lines_added(diff)
        if any(_MONKEYPATCH_RE.search(line) for line in added):
            hits += 1
    if hits == 0:
        return None
    fraction = hits / len(diffs)
    if fraction < 0.5:
        return None
    summary = (
        "Winning tests on this repo lean on the pytest ``monkeypatch`` "
        f"fixture ({hits}/{len(diffs)} winning diffs use it). Prefer "
        "monkeypatch.setattr/setitem over manual try/finally swaps."
    )
    confidence = min(0.6 + 0.2 * fraction, 0.85)
    return summary, round(confidence, 3)


def _detect_uses_tox(repo_root: Optional[Path]) -> Optional[tuple[str, float]]:
    """Fires when ``tox.ini`` exists at the repo root."""
    if repo_root is None:
        return None
    if not (repo_root / "tox.ini").exists():
        return None
    summary = (
        "This repo uses tox. The canonical test entrypoint is ``tox`` "
        "(or ``tox -e <env>`` for a specific environment). Prefer running "
        "tox over invoking pytest directly when you need the canonical "
        "test environment."
    )
    return summary, 0.85


def _detect_imports_collected_in_init(repo_root: Optional[Path]) -> Optional[tuple[str, float]]:
    """Fires when a top-level ``__init__.py`` has ≥4 import lines (re-export pattern)."""
    if repo_root is None:
        return None
    candidate_inits: list[Path] = []
    # Look at top-level package __init__.py files (up to 3 deep).
    try:
        for entry in repo_root.iterdir():
            if entry.is_dir() and (entry / "__init__.py").is_file():
                candidate_inits.append(entry / "__init__.py")
            if len(candidate_inits) >= 5:
                break
    except OSError:
        return None
    re_export_count = 0
    examined = 0
    for init_path in candidate_inits:
        examined += 1
        try:
            text = init_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Count top-level "from X import Y" / "import X" lines.
        import_lines = sum(
            1
            for line in text.splitlines()
            if line.startswith("from ") or line.startswith("import ")
        )
        if import_lines >= 4:
            re_export_count += 1
    if examined == 0 or re_export_count == 0:
        return None
    if re_export_count / examined < 0.5:
        return None
    summary = (
        f"This repo uses ``__init__.py`` modules as re-export hubs "
        f"({re_export_count}/{examined} top-level packages re-export "
        "via __init__). When adding new public API, mirror the existing "
        "re-export pattern instead of forcing callers to import from "
        "the implementation module."
    )
    return summary, 0.7


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class RepoEpisodicStore:
    """Per-repo (NOT per-task) episodic patterns.

    Loaded by the orchestrator at solve start, augmented at solve end via
    :py:meth:`extract_patterns_from_run`. JSONL-backed; cross-process
    safe via fcntl.flock. Mirrors the resilience contract of
    :class:`apex.persistence.episodic_store.EpisodicStore`.

    The store is intentionally small: it holds the *durable* patterns
    that survive across solves on the same repo, NOT the per-task
    discoveries (those live in ``EpisodicStore``).
    """

    def __init__(self, base_dir: Optional[str | Path] = None) -> None:
        self._root = _resolve_store_root(base_dir)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def store_path(self, repo_signature: str) -> Path:
        safe = _sanitize_repo_signature(repo_signature)
        return self._root / safe / "episodes.jsonl"

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get_episodes(self, repo_signature: str) -> List[RepoEpisode]:
        """All persisted episodes for ``repo_signature``, sorted by confidence desc.

        Reads do NOT take the cross-process lock (best-effort snapshot,
        same convention as Phase 6.2's EpisodicStore).
        """
        path = self.store_path(repo_signature)
        if not path.is_file():
            return []
        episodes: dict[str, RepoEpisode] = {}
        try:
            with open(path, "rb") as fd:
                self._acquire_flock_locked(fd, exclusive=False)
                try:
                    raw = fd.read().decode("utf-8", errors="replace")
                finally:
                    self._release_flock_locked(fd)
        except OSError as exc:
            logger.warning("RepoEpisodicStore: failed to read %s: %s", path, exc)
            return []

        for line_no, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                logger.debug(
                    "RepoEpisodicStore: skipping unparseable line %d in %s",
                    line_no,
                    path,
                )
                continue
            if not isinstance(obj, dict):
                continue
            version = int(obj.get("v") or 0)
            if version != _REPO_EPISODIC_SCHEMA_VERSION:
                logger.debug(
                    "RepoEpisodicStore: skipping line %d in %s with unsupported version %r",
                    line_no,
                    path,
                    obj.get("v"),
                )
                continue
            episode = RepoEpisode.from_dict(obj)
            if not episode.pattern_type:
                continue
            # Dedup by pattern_type — JSONL is append-only, so the
            # latest-seen entry wins on the merged confidence.
            existing = episodes.get(episode.pattern_type)
            if existing is None:
                episodes[episode.pattern_type] = episode
            else:
                merged_runs = list(
                    dict.fromkeys(
                        list(existing.extracted_from_runs) + list(episode.extracted_from_runs)
                    )
                )[-_MAX_RUNS_PER_PATTERN:]
                episodes[episode.pattern_type] = RepoEpisode(
                    repo_signature=episode.repo_signature or existing.repo_signature,
                    pattern_type=episode.pattern_type,
                    pattern_summary=episode.pattern_summary or existing.pattern_summary,
                    confidence=max(existing.confidence, episode.confidence),
                    extracted_from_runs=merged_runs,
                    last_updated_iso=(episode.last_updated_iso or existing.last_updated_iso),
                )
        sorted_eps = sorted(
            episodes.values(),
            key=lambda ep: (ep.confidence, ep.last_updated_iso),
            reverse=True,
        )
        return sorted_eps

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def add_episode(self, repo_signature: str, episode: RepoEpisode) -> RepoEpisode:
        """Append-only with dedup (merges by pattern_type at read time).

        Returns the persisted episode (with merged confidence + runs)
        so callers can chain.
        """
        if not episode.pattern_type or not str(episode.pattern_type).strip():
            raise ValueError("RepoEpisode.pattern_type must be non-empty")
        # Stamp the repo_signature + timestamp defensively.
        merged = RepoEpisode(
            repo_signature=str(repo_signature or episode.repo_signature or ""),
            pattern_type=str(episode.pattern_type).strip(),
            pattern_summary=str(episode.pattern_summary or ""),
            confidence=max(0.0, min(float(episode.confidence), 1.0)),
            extracted_from_runs=list(episode.extracted_from_runs),
            last_updated_iso=episode.last_updated_iso or _now_utc_iso(),
        )
        # Merge with any existing episode of the same pattern_type so the
        # JSONL log carries the cumulative state on the latest line.
        existing_map = {ep.pattern_type: ep for ep in self.get_episodes(repo_signature)}
        prior = existing_map.get(merged.pattern_type)
        if prior is not None:
            merged_runs = list(
                dict.fromkeys(list(prior.extracted_from_runs) + list(merged.extracted_from_runs))
            )[-_MAX_RUNS_PER_PATTERN:]
            merged = RepoEpisode(
                repo_signature=merged.repo_signature or prior.repo_signature,
                pattern_type=merged.pattern_type,
                pattern_summary=merged.pattern_summary or prior.pattern_summary,
                confidence=max(prior.confidence, merged.confidence),
                extracted_from_runs=merged_runs,
                last_updated_iso=merged.last_updated_iso or prior.last_updated_iso,
            )

        path = self.store_path(repo_signature)
        line = json.dumps(merged.to_dict(), sort_keys=True) + "\n"
        with self._lock:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning(
                    "RepoEpisodicStore: failed to create parent dir %s: %s",
                    path.parent,
                    exc,
                )
                return merged
            try:
                fd = open(path, "ab")
            except OSError as exc:
                logger.warning(
                    "RepoEpisodicStore: failed to open %s for append: %s",
                    path,
                    exc,
                )
                return merged
            try:
                self._acquire_flock_locked(fd)
                try:
                    fd.write(line.encode("utf-8"))
                    fd.flush()
                    try:
                        os.fsync(fd.fileno())
                    except OSError:
                        # tmpfs / CI hosts where fsync isn't available — the
                        # data is still in the page cache, accept the risk.
                        pass
                finally:
                    self._release_flock_locked(fd)
            finally:
                fd.close()
        return merged

    # ------------------------------------------------------------------
    # Heuristic extraction
    # ------------------------------------------------------------------

    def extract_patterns_from_run(
        self,
        run_dir: Path,
        *,
        repo_signature: Optional[str] = None,
        repo_root: Optional[Path] = None,
    ) -> List[RepoEpisode]:
        """Mine the per-run artifacts for repo-level patterns.

        Looks at ``apex_result.json`` in ``run_dir`` for the selected
        rollout's diff + the supporting rollout summaries, then runs the
        registered heuristics. Returns the ``RepoEpisode`` objects that
        fired; the caller is responsible for persisting them via
        :py:meth:`add_episode`.

        ``repo_signature`` defaults to the value in ``apex_result.json``
        (when present) so the caller doesn't have to know it ahead of
        time. ``repo_root`` is the absolute path to the repo on disk
        (used for filesystem-shape heuristics like ``uses_tox``).
        """
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            return []
        apex_result_path = run_dir / "apex_result.json"
        rollout_summaries: list[dict[str, Any]] = []
        winning_diffs: list[str] = []
        contributed_run_id: Optional[str] = None
        loaded: dict[str, Any] = {}
        if apex_result_path.is_file():
            try:
                loaded = json.loads(apex_result_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.warning(
                    "RepoEpisodicStore: failed to parse %s: %s",
                    apex_result_path,
                    exc,
                )
        if isinstance(loaded, dict):
            contributed_run_id = str(loaded.get("task_id") or loaded.get("instance_id") or "")
            if not contributed_run_id:
                contributed_run_id = run_dir.name
            rs = loaded.get("rollout_summaries")
            if isinstance(rs, list):
                rollout_summaries = [item for item in rs if isinstance(item, dict)]
            top_patch = loaded.get("patch")
            if isinstance(top_patch, str) and top_patch.strip():
                winning_diffs.append(top_patch)
        # Fall back to scanning rollout summaries when no top-level patch
        # is present (e.g. salvage-only outcomes still produce a diff).
        for summary in rollout_summaries:
            if not isinstance(summary, dict):
                continue
            if not summary.get("success"):
                continue
            patch = summary.get("patch")
            if isinstance(patch, str) and patch.strip() and patch not in winning_diffs:
                winning_diffs.append(patch)
        # Resolve repo_signature: caller wins, then apex_result, then "".
        if not repo_signature:
            repo_signature = (
                str(loaded.get("repo_signature") or "") if isinstance(loaded, dict) else ""
            )

        episodes: list[RepoEpisode] = []
        # Diff-driven heuristics.
        for detector in (
            _detect_pytest_fixture_heavy,
            _detect_small_focused_prs,
            _detect_monkeypatch_pattern,
        ):
            outcome = detector(winning_diffs)
            if outcome is None:
                continue
            summary_text, confidence = outcome
            episodes.append(
                RepoEpisode(
                    repo_signature=str(repo_signature or ""),
                    pattern_type=detector.__name__.removeprefix("_detect_"),
                    pattern_summary=summary_text,
                    confidence=float(confidence),
                    extracted_from_runs=[contributed_run_id] if contributed_run_id else [],
                    last_updated_iso=_now_utc_iso(),
                )
            )
        # Filesystem-shape heuristics.
        for detector in (_detect_uses_tox, _detect_imports_collected_in_init):
            outcome = detector(repo_root)
            if outcome is None:
                continue
            summary_text, confidence = outcome
            episodes.append(
                RepoEpisode(
                    repo_signature=str(repo_signature or ""),
                    pattern_type=detector.__name__.removeprefix("_detect_"),
                    pattern_summary=summary_text,
                    confidence=float(confidence),
                    extracted_from_runs=[contributed_run_id] if contributed_run_id else [],
                    last_updated_iso=_now_utc_iso(),
                )
            )
        return episodes

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def repo_signatures(self) -> list[str]:
        """List all repo signatures currently in the store."""
        if not self._root.is_dir():
            return []
        out: list[str] = []
        try:
            for entry in self._root.iterdir():
                if entry.is_dir() and (entry / "episodes.jsonl").is_file():
                    out.append(entry.name)
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("RepoEpisodicStore: failed to list root %s: %s", self._root, exc)
            return []
        return sorted(out)

    def clear(self, repo_signature: str) -> bool:
        """Delete the episodes file for ``repo_signature``."""
        path = self.store_path(repo_signature)
        if not path.is_file():
            return False
        try:
            path.unlink()
            return True
        except OSError as exc:
            logger.warning("RepoEpisodicStore: failed to delete %s: %s", path, exc)
            return False

    # ------------------------------------------------------------------
    # Locking (mirrors EpisodicStore)
    # ------------------------------------------------------------------

    def _acquire_flock_locked(self, fd: Any, *, exclusive: bool = True) -> None:
        if not _FCNTL_AVAILABLE or fcntl is None:
            return
        flag = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        deadline = time.time() + _LOCK_TIMEOUT_SECONDS
        while True:
            try:
                fcntl.flock(fd.fileno(), flag | fcntl.LOCK_NB)
                return
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                    logger.warning(
                        "RepoEpisodicStore: flock unexpected error %s; proceeding without lock",
                        exc,
                    )
                    return
                if time.time() > deadline:
                    logger.warning(
                        "RepoEpisodicStore: flock timeout after %.1fs; proceeding without lock",
                        _LOCK_TIMEOUT_SECONDS,
                    )
                    return
                time.sleep(0.05)

    def _release_flock_locked(self, fd: Any) -> None:
        if not _FCNTL_AVAILABLE or fcntl is None:
            return
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        except OSError:  # pragma: no cover - defensive
            pass


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


_PROMPT_HEADER = "# Repo conventions"
_PROMPT_FOOTER_NOTE = (
    "These conventions were learned from previous solves on this repo. "
    "Treat them as priors — strong evidence in the current task overrides "
    "them. Do NOT cite them verbatim back to the user."
)


def render_repo_episodes_prompt_block(
    episodes: Iterable[RepoEpisode],
    *,
    max_entries: int = 6,
    min_confidence: float = 0.55,
) -> str:
    """Render a prompt-ready ``# Repo conventions`` block.

    Returns the empty string when no episode clears the confidence floor;
    callers should ``if block: prompt += "\\n\\n" + block`` so they don't
    pollute the prompt with an empty section.
    """
    eligible = [ep for ep in episodes if ep.pattern_summary and ep.confidence >= min_confidence]
    if not eligible:
        return ""
    eligible.sort(key=lambda ep: ep.confidence, reverse=True)
    eligible = eligible[: max(1, int(max_entries))]
    lines = [_PROMPT_HEADER]
    for ep in eligible:
        # Each entry is a single bullet so the agent can scan them
        # quickly. Confidence is rendered as a 2-decimal so the model
        # can weight them even though we don't expose the raw float.
        lines.append(
            f"- [{ep.pattern_type} | conf={ep.confidence:.2f}] {ep.pattern_summary.strip()}"
        )
    lines.append("")
    lines.append(_PROMPT_FOOTER_NOTE)
    return "\n".join(lines)


__all__ = [
    "RepoEpisode",
    "RepoEpisodicStore",
    "render_repo_episodes_prompt_block",
]
