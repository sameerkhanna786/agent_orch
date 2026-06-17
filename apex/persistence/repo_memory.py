"""
Per-repository persistent insight memory.

APEX collects rollout discoveries inside a single solve via
:class:`apex.rollout.engine.EpisodicMemoryBus`. That memory is volatile —
the next solve on the same repo restarts from a blank slate. This module
adds a small JSON-backed store keyed to the repo signature so that
high-confidence, multi-rollout-supported discoveries can be carried
forward as priors on later solves.

Design notes
------------

This is a **deliberately conservative** memory:

- Only insights that meet a configurable confidence floor and (optionally)
  multi-rollout support threshold are persisted. Single-rollout
  speculation is dropped.
- On reload, each insight's confidence is decayed by a factor that
  depends on the **commit distance** between the signature-creation
  commit and the current HEAD (Phase 5.8). This replaces the older
  always-fixed time-based decay so a bursty solve cluster doesn't
  artificially fade priors that are still on the same commit.
- Insights are deduplicated by a stable signature derived from
  ``insight_type``, sorted file paths, sorted symbols, and a normalised
  description fragment. Re-observation bumps ``support_count`` and updates
  the last-seen timestamp.
- The store caps the total number of persisted insights to keep payloads
  small and the prior set focused on the most-supported beliefs.

Repo signature (Phase 5.8)
--------------------------

The repo signature combines:

* ``realpath`` — collapses symlink farms.
* ``git remote get-url origin`` — distinguishes two checkouts of the
  same upstream repo at different worktree paths.
* ``git rev-parse --abbrev-ref HEAD`` — distinguishes branches /
  worktrees of the same upstream so memory from `main` doesn't bleed
  into a feature branch.

If the directory isn't a git repo, we fall back to realpath-only with a
logged warning. The legacy ``realpath``-only signature key is migrated
on first load: the old store file is copied/promoted into the new key
once, then the old key is left in place for one cycle before being
deprecated. See ``RepoMemoryStore._maybe_promote_legacy_signature``.

Benchmark integrity
-------------------

When the store is loaded, callers should record ``repo_memory_loaded``
and ``prior_insight_count`` in their result artifact so that any
benchmark report can be audited for prior-knowledge bleed-through.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger("apex.persistence.repo_memory")


_PERSISTED_SCHEMA_VERSION = 2
_LEGACY_SCHEMA_VERSION = 1
_DEFAULT_STORE_DIRNAME = ".apex/repo_memory"

_DISABLE_ENV_VAR = "APEX_DISABLE_REPO_MEMORY"
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on", "enabled"})

# Phase 5.9: orphan ``.tmp`` files older than this are deleted on load.
# Concurrent writers might still be using fresher files, so we err on the
# safe side with a one-hour floor.
_ORPHAN_TMP_AGE_SECONDS = 3600.0


def is_repo_memory_disabled_via_env(env: Optional[dict[str, str]] = None) -> bool:
    """Whether the ablation env override forces repo_memory off.

    Reviewers of any persistent-memory result need an audit-friendly way
    to re-run the same benchmark with memory disabled, *without*
    rewriting the config (which would also drop other knobs and make
    A/B comparison harder). Setting ``APEX_DISABLE_REPO_MEMORY=1`` (or
    "true"/"yes"/"on"/"enabled") forces the gate off regardless of what
    the YAML/JSON config says. The override is one-way: there is no
    counterpart that *force-enables* memory, since the inverse ablation
    (memory on when config says off) would silently change the contract
    of an air-gapped or contamination-controlled run.
    """
    source = env if env is not None else os.environ
    raw = str(source.get(_DISABLE_ENV_VAR) or "").strip().lower()
    return raw in _TRUTHY_ENV_VALUES


def _now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _normalize_strings(values: Optional[Iterable[Any]]) -> list[str]:
    if not values:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _normalize_description_fragment(description: str, *, max_chars: int = 80) -> str:
    collapsed = re.sub(r"\s+", " ", str(description or "")).strip().lower()
    return collapsed[:max_chars]


def _git_query(
    repo_path: str,
    args: list[str],
    *,
    timeout: float = 5.0,
) -> Optional[str]:
    """Run a git query inside ``repo_path``; return stripped stdout or None.

    Failures (non-git repo, timeout, missing git binary) all collapse to
    ``None``. Caller is responsible for graceful degradation.
    """
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    out = (result.stdout or "").strip()
    return out or None


def _git_origin_url(repo_path: str) -> Optional[str]:
    return _git_query(repo_path, ["remote", "get-url", "origin"])


def _git_current_branch(repo_path: str) -> Optional[str]:
    return _git_query(repo_path, ["rev-parse", "--abbrev-ref", "HEAD"])


def _git_current_commit(repo_path: str) -> Optional[str]:
    return _git_query(repo_path, ["rev-parse", "HEAD"])


def _git_commit_distance(repo_path: str, from_commit: str) -> Optional[int]:
    """Return the number of commits between ``from_commit`` and HEAD.

    Uses ``git rev-list --count A..HEAD`` semantics. If ``from_commit``
    is not reachable from HEAD (e.g. the branch was rebased) or the
    repo is not a git repo, returns ``None``.
    """
    if not from_commit:
        return None
    out = _git_query(repo_path, ["rev-list", "--count", f"{from_commit}..HEAD"])
    if out is None:
        return None
    try:
        return max(0, int(out))
    except (TypeError, ValueError):
        return None


def repo_signature_for_path(
    repo_path: str,
    *,
    git_origin_url: Optional[str] = None,
    git_branch: Optional[str] = None,
) -> str:
    """Derive a stable signature from an absolute repo path.

    Phase 5.8: signature now combines realpath with the upstream origin
    URL and the current branch. This stops two checkouts of the same
    repo at different branches from sharing memory and silently
    contaminating each other's priors.

    For backward compatibility with callers that don't have the git
    metadata handy (typically tests on bare ``tmp_path`` directories),
    omitting ``git_origin_url`` / ``git_branch`` falls back to looking
    them up via subprocess. If the directory isn't a git repo, both
    components collapse to empty strings and the signature degrades to
    a realpath-only digest with a warning.
    """

    abs_path = os.path.realpath(str(repo_path or ""))
    if not abs_path:
        abs_path = "<empty>"

    # Auto-resolve only when caller didn't pass values. ``None`` means
    # "look it up"; ``""`` means "caller asserts no git metadata".
    origin = git_origin_url
    branch = git_branch
    if origin is None or branch is None:
        if origin is None:
            origin = _git_origin_url(abs_path) or ""
        if branch is None:
            branch = _git_current_branch(abs_path) or ""
        if not origin and not branch:
            logger.warning(
                "repo_signature_for_path: %s is not a git repo (or git metadata "
                "unavailable); falling back to realpath-only signature.",
                abs_path,
            )

    composite = f"{abs_path}@{origin or ''}@{branch or ''}"
    digest = hashlib.sha256(composite.encode("utf-8")).hexdigest()[:16]
    return digest


def repo_signature_legacy_for_path(repo_path: str) -> str:
    """Pre-Phase-5.8 signature: realpath-only.

    Kept for one-cycle migration of legacy stores. See
    :meth:`RepoMemoryStore._maybe_promote_legacy_signature`.
    """
    abs_path = os.path.realpath(str(repo_path or ""))
    if not abs_path:
        abs_path = "<empty>"
    digest = hashlib.sha256(abs_path.encode("utf-8")).hexdigest()[:16]
    return digest


@dataclass
class PersistedInsight:
    """One durable insight record carried across solves."""

    insight_type: str
    description: str
    confidence: float = 0.7
    file_paths: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)
    stage_name: str = ""
    negative: bool = False
    support_count: int = 1
    first_seen_utc: str = ""
    last_seen_utc: str = ""

    def signature(self) -> str:
        """Stable identity key used for de-duplication on merge."""

        components = (
            str(self.insight_type or "").strip().upper(),
            "|".join(sorted(_normalize_strings(self.file_paths))),
            "|".join(sorted(_normalize_strings(self.symbols))),
            "|".join(sorted(_normalize_strings(self.test_ids))),
            "1" if self.negative else "0",
            _normalize_description_fragment(self.description),
        )
        return hashlib.sha1("␟".join(components).encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "insight_type": str(self.insight_type or ""),
            "description": str(self.description or ""),
            "confidence": round(float(self.confidence), 4),
            "file_paths": list(self.file_paths),
            "symbols": list(self.symbols),
            "test_ids": list(self.test_ids),
            "stage_name": str(self.stage_name or ""),
            "negative": bool(self.negative),
            "support_count": int(self.support_count),
            "first_seen_utc": str(self.first_seen_utc or ""),
            "last_seen_utc": str(self.last_seen_utc or ""),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersistedInsight":
        return cls(
            insight_type=str(data.get("insight_type") or ""),
            description=str(data.get("description") or ""),
            confidence=float(data.get("confidence") or 0.0),
            file_paths=_normalize_strings(data.get("file_paths")),
            symbols=_normalize_strings(data.get("symbols")),
            test_ids=_normalize_strings(data.get("test_ids")),
            stage_name=str(data.get("stage_name") or ""),
            negative=bool(data.get("negative")),
            support_count=int(data.get("support_count") or 1),
            first_seen_utc=str(data.get("first_seen_utc") or ""),
            last_seen_utc=str(data.get("last_seen_utc") or ""),
        )


def _resolve_store_root(directory: Optional[str]) -> Path:
    if directory:
        return Path(directory).expanduser().resolve()
    return Path(os.path.expanduser("~")) / _DEFAULT_STORE_DIRNAME


class RepoMemoryStore:
    """JSON-backed per-repository insight cache.

    The store is intentionally process-safe via a per-instance lock plus
    atomic file writes. It is NOT designed for high concurrency across
    processes; the expected pattern is one solve at a time per repo.
    """

    def __init__(
        self,
        repo_path: str,
        *,
        directory: Optional[str] = None,
        max_persisted_insights: int = 64,
        decay_factor: float = 0.85,
        min_confidence_to_persist: float = 0.65,
        prefer_high_support_threshold: int = 2,
    ) -> None:
        self.repo_path = str(repo_path or "")
        # Phase 5.8: capture git metadata at instantiation time so the
        # signature is stable for the life of the store, even if HEAD or
        # the remote URL changes mid-solve.
        abs_path = os.path.realpath(self.repo_path) if self.repo_path else ""
        self._git_origin_url = _git_origin_url(abs_path) if abs_path else None
        self._git_branch = _git_current_branch(abs_path) if abs_path else None
        self.repo_signature = repo_signature_for_path(
            self.repo_path,
            git_origin_url=self._git_origin_url or "",
            git_branch=self._git_branch or "",
        )
        self._legacy_signature = repo_signature_legacy_for_path(self.repo_path)
        self._root = _resolve_store_root(directory)
        self._max_persisted_insights = max(1, int(max_persisted_insights))
        self._decay_factor = max(0.0, min(1.0, float(decay_factor)))
        self._min_confidence_to_persist = max(0.0, min(1.0, float(min_confidence_to_persist)))
        self._prefer_high_support_threshold = max(0, int(prefer_high_support_threshold))
        self._lock = threading.Lock()
        self._loaded = False
        self._payload: dict[str, Any] = self._empty_payload()

    def _empty_payload(self) -> dict[str, Any]:
        now = _now_utc_iso()
        # Phase 5.8: record the commit at signature-creation time so we
        # can compute commit distance on later loads.
        creation_commit = ""
        if self.repo_path:
            creation_commit = _git_current_commit(os.path.realpath(self.repo_path)) or ""
        return {
            "version": _PERSISTED_SCHEMA_VERSION,
            "repo_signature": self.repo_signature,
            "repo_path": self.repo_path,
            "git_origin_url": self._git_origin_url or "",
            "git_branch": self._git_branch or "",
            "signature_creation_commit": creation_commit,
            "first_seen_utc": now,
            "last_updated_utc": now,
            "solve_count": 0,
            "insights": [],
        }

    @property
    def store_path(self) -> Path:
        return self._root / self.repo_signature / "insights.json"

    @property
    def legacy_store_path(self) -> Path:
        return self._root / self._legacy_signature / "insights.json"

    def load(self) -> list[PersistedInsight]:
        """Return the currently persisted insights for this repo.

        Insight confidences are decayed on load. Phase 5.8: decay scales
        with commit distance from the signature-creation commit, not
        wall-clock time. Decay does not mutate the on-disk record — only
        re-observation and persistence does.
        """

        with self._lock:
            self._load_locked()
            insights = [
                PersistedInsight.from_dict(dict(item))
                for item in list(self._payload.get("insights") or [])
                if isinstance(item, dict)
            ]
            decay_multiplier = self._compute_decay_multiplier_locked()
        decayed: list[PersistedInsight] = []
        for insight in insights:
            decayed.append(
                PersistedInsight(
                    insight_type=insight.insight_type,
                    description=insight.description,
                    confidence=max(0.0, min(1.0, insight.confidence * decay_multiplier)),
                    file_paths=list(insight.file_paths),
                    symbols=list(insight.symbols),
                    test_ids=list(insight.test_ids),
                    stage_name=insight.stage_name,
                    negative=insight.negative,
                    support_count=insight.support_count,
                    first_seen_utc=insight.first_seen_utc,
                    last_seen_utc=insight.last_seen_utc,
                )
            )
        return decayed

    def _compute_decay_multiplier_locked(self) -> float:
        """Phase 5.8: decay multiplier = decay_factor ** commit_distance.

        Distance 0 (same commit) → no decay. Distance 1 → one application
        of ``decay_factor``. Distance N → ``decay_factor ** N``.

        If commit distance can't be computed (non-git, unreachable
        creation commit, missing git binary), fall back to the legacy
        single-application decay so behaviour is conservative rather
        than no-decay.
        """
        creation_commit = str(self._payload.get("signature_creation_commit") or "")
        if not creation_commit or not self.repo_path:
            return self._decay_factor
        distance = _git_commit_distance(
            os.path.realpath(self.repo_path),
            creation_commit,
        )
        if distance is None:
            return self._decay_factor
        if distance <= 0:
            return 1.0
        # decay_factor ** distance, with a floor for numerical sanity.
        return max(0.0, min(1.0, float(self._decay_factor) ** float(distance)))

    def merge_and_persist(
        self,
        insights: Iterable[PersistedInsight],
        *,
        solve_increment: int = 1,
    ) -> dict[str, Any]:
        """Merge ``insights`` into the store and write it to disk.

        Returns a small summary dict useful for the calling solve's
        result artifact. Insights below ``min_confidence_to_persist``
        are dropped; the merged set is capped to
        ``max_persisted_insights`` favouring high-support, high-confidence
        records.
        """

        candidate_insights = [
            insight
            for insight in insights
            if isinstance(insight, PersistedInsight)
            and float(insight.confidence) >= self._min_confidence_to_persist
        ]
        with self._lock:
            self._load_locked()
            existing_records = {
                str(
                    item.get("signature") or PersistedInsight.from_dict(dict(item)).signature()
                ): dict(item)
                for item in list(self._payload.get("insights") or [])
                if isinstance(item, dict)
            }
            now_iso = _now_utc_iso()
            for insight in candidate_insights:
                signature = insight.signature()
                existing = existing_records.get(signature)
                if existing is None:
                    record = insight.to_dict()
                    record["signature"] = signature
                    record["first_seen_utc"] = insight.first_seen_utc or now_iso
                    record["last_seen_utc"] = now_iso
                    record["support_count"] = max(1, int(insight.support_count))
                    existing_records[signature] = record
                    continue
                merged = PersistedInsight.from_dict(dict(existing))
                merged.support_count = int(merged.support_count) + max(
                    1, int(insight.support_count)
                )
                # Confidence update: take the convex combination weighted
                # by support so a single mid-confidence re-observation
                # cannot pull a well-supported high-confidence belief
                # back to the new value.
                prior_weight = max(1, int(merged.support_count - insight.support_count))
                new_weight = max(1, int(insight.support_count))
                blended = (
                    (prior_weight * float(merged.confidence))
                    + (new_weight * float(insight.confidence))
                ) / float(prior_weight + new_weight)
                merged.confidence = max(0.0, min(1.0, blended))
                merged.file_paths = _normalize_strings(
                    list(merged.file_paths) + list(insight.file_paths)
                )
                merged.symbols = _normalize_strings(list(merged.symbols) + list(insight.symbols))
                merged.test_ids = _normalize_strings(list(merged.test_ids) + list(insight.test_ids))
                merged.stage_name = merged.stage_name or insight.stage_name
                merged.last_seen_utc = now_iso
                if not merged.first_seen_utc:
                    merged.first_seen_utc = insight.first_seen_utc or now_iso
                merged_record = merged.to_dict()
                merged_record["signature"] = signature
                existing_records[signature] = merged_record

            ranked_records = sorted(
                existing_records.values(),
                key=lambda record: (
                    -int(record.get("support_count") or 0),
                    -float(record.get("confidence") or 0.0),
                    str(record.get("last_seen_utc") or ""),
                ),
            )[: self._max_persisted_insights]
            high_support_count = sum(
                1
                for record in ranked_records
                if int(record.get("support_count") or 0) >= self._prefer_high_support_threshold
            )
            self._payload["insights"] = ranked_records
            self._payload["last_updated_utc"] = now_iso
            self._payload["solve_count"] = int(self._payload.get("solve_count") or 0) + max(
                0, int(solve_increment)
            )
            self._payload["version"] = _PERSISTED_SCHEMA_VERSION
            self._payload["repo_signature"] = self.repo_signature
            self._payload["repo_path"] = self.repo_path
            self._payload["git_origin_url"] = self._git_origin_url or ""
            self._payload["git_branch"] = self._git_branch or ""
            # Ensure signature_creation_commit is set even if the legacy
            # promote path didn't fill it in.
            if not self._payload.get("signature_creation_commit") and self.repo_path:
                self._payload["signature_creation_commit"] = (
                    _git_current_commit(os.path.realpath(self.repo_path)) or ""
                )
            self._save_locked()
            return {
                "repo_signature": self.repo_signature,
                "store_path": str(self.store_path),
                "persisted_insight_count": len(ranked_records),
                "candidate_insight_count": len(candidate_insights),
                "solve_count": int(self._payload.get("solve_count") or 0),
                "last_updated_utc": now_iso,
                "high_support_count": high_support_count,
                "high_support_threshold": int(self._prefer_high_support_threshold),
            }

    def summary(self) -> dict[str, Any]:
        """Return store metadata without exposing raw insight bodies."""

        with self._lock:
            self._load_locked()
            insights = list(self._payload.get("insights") or [])
            return {
                "repo_signature": self.repo_signature,
                "store_path": str(self.store_path),
                "exists": self.store_path.is_file(),
                "persisted_insight_count": len(insights),
                "solve_count": int(self._payload.get("solve_count") or 0),
                "first_seen_utc": str(self._payload.get("first_seen_utc") or ""),
                "last_updated_utc": str(self._payload.get("last_updated_utc") or ""),
                "high_support_count": sum(
                    1
                    for item in insights
                    if isinstance(item, dict)
                    and int(item.get("support_count") or 0) >= self._prefer_high_support_threshold
                ),
                "git_origin_url": str(self._payload.get("git_origin_url") or ""),
                "git_branch": str(self._payload.get("git_branch") or ""),
                "signature_creation_commit": str(
                    self._payload.get("signature_creation_commit") or ""
                ),
            }

    def clear(self) -> None:
        with self._lock:
            self._payload = self._empty_payload()
            try:
                if self.store_path.is_file():
                    self.store_path.unlink()
            except OSError as exc:
                logger.warning("Failed to delete repo memory store at %s: %s", self.store_path, exc)
            self._loaded = True

    def _maybe_promote_legacy_signature(self) -> bool:
        """Phase 5.8: migrate a legacy realpath-only store into the new key.

        If the new-signature store doesn't exist but the legacy-signature
        store does, copy it over. The legacy file is left in place for
        one promotion cycle so a benchmark replay against the legacy
        path still sees the old data — the next ``merge_and_persist`` on
        the new key supersedes it. After that cycle the caller may
        remove the legacy file via :meth:`prune_legacy_signature`.

        Returns True if a promotion happened. Caller is responsible for
        re-loading the payload after a successful promotion.
        """
        if self.repo_signature == self._legacy_signature:
            # No migration needed (e.g. non-git directory: the new
            # signature happens to equal the legacy one).
            return False
        if self.store_path.is_file():
            return False
        legacy_path = self.legacy_store_path
        if not legacy_path.is_file():
            return False
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy_path, self.store_path)
            logger.info(
                "repo_memory: promoted legacy signature store %s -> %s "
                "(Phase 5.8 migration; legacy file kept for one cycle).",
                legacy_path,
                self.store_path,
            )
            return True
        except OSError as exc:
            logger.warning(
                "repo_memory: failed to promote legacy store %s -> %s: %s",
                legacy_path,
                self.store_path,
                exc,
            )
            return False

    def prune_legacy_signature(self) -> bool:
        """Delete the legacy realpath-only store after one promotion cycle.

        Callers should invoke this only after verifying the new-signature
        store exists and has been written to at least once. Returns True
        when a legacy file was actually removed.
        """
        if self.repo_signature == self._legacy_signature:
            return False
        legacy_path = self.legacy_store_path
        if not legacy_path.is_file():
            return False
        try:
            legacy_path.unlink()
            return True
        except OSError as exc:
            logger.warning(
                "repo_memory: failed to prune legacy store %s: %s",
                legacy_path,
                exc,
            )
            return False

    def _cleanup_orphan_tmp_files_locked(self) -> int:
        """Phase 5.9: remove ``*.tmp`` orphans older than ``_ORPHAN_TMP_AGE_SECONDS``.

        Atomic writes go through ``insights.json.tmp`` then ``os.replace``.
        A crash between ``write_text`` and ``os.replace`` leaves the .tmp
        behind. We sweep them on every load, but only ones older than an
        hour to avoid stomping on a concurrent writer that's still
        flushing.
        """
        parent = self.store_path.parent
        if not parent.is_dir():
            return 0
        cutoff = time.time() - _ORPHAN_TMP_AGE_SECONDS
        cleaned = 0
        try:
            for entry in parent.iterdir():
                if not entry.is_file():
                    continue
                if not entry.name.endswith(".tmp"):
                    continue
                try:
                    mtime = entry.stat().st_mtime
                except OSError:
                    continue
                if mtime > cutoff:
                    continue
                try:
                    entry.unlink()
                    cleaned += 1
                except OSError as exc:
                    logger.warning(
                        "repo_memory: failed to remove orphan tmp file %s: %s",
                        entry,
                        exc,
                    )
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning(
                "repo_memory: failed to scan parent dir %s for orphan tmp files: %s",
                parent,
                exc,
            )
            return 0
        if cleaned:
            logger.info(
                "repo_memory: cleaned_orphan_tmp count=%d in %s",
                cleaned,
                parent,
            )
        return cleaned

    def _load_locked(self) -> None:
        if self._loaded:
            return
        # Phase 5.9: best-effort orphan tmp cleanup on every load.
        self._cleanup_orphan_tmp_files_locked()
        # Phase 5.8: opportunistic legacy-signature promotion.
        self._maybe_promote_legacy_signature()
        if self.store_path.is_file():
            try:
                raw = self.store_path.read_text()
                payload = json.loads(raw) if raw else {}
                if isinstance(payload, dict):
                    version = int(payload.get("version") or 0)
                    if version in (_PERSISTED_SCHEMA_VERSION, _LEGACY_SCHEMA_VERSION):
                        self._payload = payload
                        # Backfill new fields if loading a legacy v1 file.
                        if version == _LEGACY_SCHEMA_VERSION:
                            self._payload.setdefault(
                                "git_origin_url",
                                self._git_origin_url or "",
                            )
                            self._payload.setdefault(
                                "git_branch",
                                self._git_branch or "",
                            )
                            if self.repo_path:
                                self._payload.setdefault(
                                    "signature_creation_commit",
                                    _git_current_commit(os.path.realpath(self.repo_path)) or "",
                                )
                            self._payload["version"] = _PERSISTED_SCHEMA_VERSION
                    else:
                        logger.warning(
                            "Repo memory store at %s has unsupported version %r; starting fresh.",
                            self.store_path,
                            payload.get("version"),
                        )
            except (OSError, ValueError) as exc:
                logger.warning("Failed to load repo memory store at %s: %s", self.store_path, exc)
        self._loaded = True

    def _save_locked(self) -> None:
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.store_path.with_suffix(self.store_path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(self._payload, indent=2, sort_keys=True))
            os.replace(tmp_path, self.store_path)
        except OSError as exc:
            logger.warning("Failed to write repo memory store at %s: %s", self.store_path, exc)
