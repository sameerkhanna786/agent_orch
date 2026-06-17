"""Cross-rollout episodic learning facade (Phase 6 item 6.2).

This module is a high-level wrapper over
:class:`apex.persistence.episodic_store.EpisodicStore`. The store is the
durable substrate (append-only JSONL, file-locking, etc.); this facade
turns raw episode records into solve-time hypotheses and outcome
broadcasts.

There are three primitives:

* :func:`learn_from_prior_run` — at the START of a rollout, query the
  store for prior broadcasts on the same task signature and synthesise
  them into a list of :class:`Hypothesis` records. Hypotheses carry a
  decayed confidence based on age and prior support.
* :func:`record_outcome` — at the END of a rollout, broadcast the
  outcome (status, patch metadata, mutation score, abstention, …) so
  the next solve on the same task can learn from it.
* :func:`compose_task_signature` — stable signature for the (repo,
  task_id) pair so callers don't have to import
  :func:`apex.persistence.episodic_store.task_signature_for` directly.

Episode types
-------------

A small fixed vocabulary keeps episodes queryable. New episode types
are additive — old readers ignore unknown types so the schema can grow
without invalidating prior log files.

* ``ROLLOUT_OUTCOME`` — terminal status of one rollout (success /
  failure / abstain), with summary metrics.
* ``ROOT_CAUSE_HYPOTHESIS`` — believed bug location and rationale.
* ``DEAD_END`` — approach the agent tried that didn't work; flagged
  negative so the next run avoids it.
* ``RELEVANT_FILE`` — source file the agent decided was relevant.
* ``MUTATION_SURVIVOR_PATTERN`` — pattern of mutants that survived
  testing, so the next testgen run knows where to attack.

These align with the in-memory ``EpisodicMemoryBus`` insight types so
the cross-run flow can transparently extend the per-solve flow.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..persistence.episodic_store import (
    Episode,
    EpisodicStore,
    task_signature_for,
)

logger = logging.getLogger(__name__)


# Episode type vocabulary. Kept as module-level constants so callers
# don't need to remember the strings and so static analysers can spot
# typos.
EPISODE_ROLLOUT_OUTCOME = "ROLLOUT_OUTCOME"
EPISODE_ROOT_CAUSE = "ROOT_CAUSE_HYPOTHESIS"
EPISODE_DEAD_END = "DEAD_END"
EPISODE_RELEVANT_FILE = "RELEVANT_FILE"
EPISODE_MUTATION_SURVIVOR = "MUTATION_SURVIVOR_PATTERN"

ALL_EPISODE_TYPES = (
    EPISODE_ROLLOUT_OUTCOME,
    EPISODE_ROOT_CAUSE,
    EPISODE_DEAD_END,
    EPISODE_RELEVANT_FILE,
    EPISODE_MUTATION_SURVIVOR,
)


# Default age decay: episodes older than this contribute zero by
# default. Callers that want to keep a longer history pass
# ``max_age_seconds=None``.
DEFAULT_MAX_AGE_SECONDS = 30 * 24 * 3600.0  # 30 days


@dataclass
class Hypothesis:
    """One synthesised belief carried in from a prior run.

    A hypothesis is the SOLVE-TIME view: a concise, decayed,
    deduplicated belief the current rollout can act on. The raw
    :class:`Episode` records that produced it are kept under
    ``source_episodes`` for audit.
    """

    episode_type: str
    description: str
    confidence: float = 0.5
    file_paths: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)
    negative: bool = False
    support_count: int = 1
    last_seen_utc: str = ""
    source_episodes: list[Episode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_type": str(self.episode_type),
            "description": str(self.description),
            "confidence": round(float(self.confidence), 4),
            "file_paths": list(self.file_paths),
            "symbols": list(self.symbols),
            "test_ids": list(self.test_ids),
            "negative": bool(self.negative),
            "support_count": int(self.support_count),
            "last_seen_utc": str(self.last_seen_utc),
        }


def compose_task_signature(repo_signature: str, task_id: str) -> str:
    """Stable per-task signature; thin alias kept for ergonomics."""

    return task_signature_for(repo_signature, task_id)


# ---------------------------------------------------------------------------
# Hypothesis synthesis
# ---------------------------------------------------------------------------


def _normalise_strings(values: Any) -> list[str]:
    if not values:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        text = str(v or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _episode_signature(episode: Episode) -> tuple[str, str, tuple[str, ...], tuple[str, ...], bool]:
    """Stable identity tuple for de-dup."""
    payload = episode.payload or {}
    description = str(payload.get("description") or "").strip().lower()
    file_paths = tuple(sorted(_normalise_strings(payload.get("file_paths"))))
    symbols = tuple(sorted(_normalise_strings(payload.get("symbols"))))
    negative = bool(payload.get("negative"))
    return (episode.episode_type, description[:80], file_paths, symbols, negative)


def _age_decay(timestamp: float, *, half_life_seconds: float) -> float:
    """Exponential half-life decay multiplier in [0, 1].

    half_life_seconds determines how fast confidence fades. A 7-day
    half-life is conservative: a week-old success belief still carries
    ~0.5 of its original confidence, and only at ~2 weeks does it drop
    below 0.25.
    """
    if half_life_seconds <= 0:
        return 1.0
    age = max(0.0, time.time() - float(timestamp or 0.0))
    return float(0.5 ** (age / float(half_life_seconds)))


def learn_from_prior_run(
    store: EpisodicStore,
    *,
    repo_signature: str,
    task_id: str,
    episode_types: Optional[list[str]] = None,
    max_age_seconds: Optional[float] = DEFAULT_MAX_AGE_SECONDS,
    half_life_seconds: float = 7 * 24 * 3600.0,
    min_confidence: float = 0.1,
) -> list[Hypothesis]:
    """Query ``store`` and synthesise ``Hypothesis`` records.

    Episodes are grouped by stable signature (type + normalised
    description + file_paths + symbols + negative) so a recurring
    finding accumulates support_count instead of producing duplicate
    hypotheses. Each surviving hypothesis is decayed by age (half-life
    in seconds) and discarded if below ``min_confidence``.

    Returns an empty list if no episodes exist for this task signature
    — the absence of priors is not an error.
    """
    task_signature = compose_task_signature(repo_signature, task_id)
    types = list(episode_types) if episode_types else None

    if types is None:
        episodes = store.query(
            task_signature=task_signature,
            max_age_seconds=max_age_seconds,
        )
    else:
        episodes = []
        for episode_type in types:
            episodes.extend(
                store.query(
                    task_signature=task_signature,
                    episode_type=episode_type,
                    max_age_seconds=max_age_seconds,
                )
            )

    if not episodes:
        return []

    grouped: dict[tuple[Any, ...], list[Episode]] = {}
    for episode in episodes:
        if not episode.episode_type:
            continue
        signature = _episode_signature(episode)
        grouped.setdefault(signature, []).append(episode)

    hypotheses: list[Hypothesis] = []
    for signature, group in grouped.items():
        # Use the most-recent episode as the canonical payload source;
        # older ones contribute to support_count.
        sorted_group = sorted(group, key=lambda e: e.timestamp)
        latest = sorted_group[-1]
        payload = latest.payload or {}
        # Confidence: average per-episode confidence (default 0.7 if
        # the source didn't record one) decayed by age of the LATEST
        # observation. We use the latest age (not mean age) because a
        # reconfirmed-yesterday belief is fresh even if it was first
        # observed weeks ago.
        confidences = [float((e.payload or {}).get("confidence") or 0.7) for e in sorted_group]
        mean_confidence = sum(confidences) / max(1, len(confidences))
        decay = _age_decay(latest.timestamp, half_life_seconds=half_life_seconds)
        confidence = max(0.0, min(1.0, mean_confidence * decay))
        if confidence < min_confidence:
            continue
        # Merge file_paths / symbols / test_ids across all sources so
        # the hypothesis covers the full union.
        all_files: list[str] = []
        all_symbols: list[str] = []
        all_tests: list[str] = []
        for episode in sorted_group:
            ep_payload = episode.payload or {}
            all_files.extend(_normalise_strings(ep_payload.get("file_paths")))
            all_symbols.extend(_normalise_strings(ep_payload.get("symbols")))
            all_tests.extend(_normalise_strings(ep_payload.get("test_ids")))
        hypotheses.append(
            Hypothesis(
                episode_type=latest.episode_type,
                description=str(payload.get("description") or "").strip(),
                confidence=confidence,
                file_paths=_normalise_strings(all_files),
                symbols=_normalise_strings(all_symbols),
                test_ids=_normalise_strings(all_tests),
                negative=bool(payload.get("negative")),
                support_count=len(sorted_group),
                last_seen_utc=str(latest.timestamp_utc or ""),
                source_episodes=list(sorted_group),
            )
        )

    # Highest-support, highest-confidence first.
    hypotheses.sort(key=lambda h: (-h.support_count, -h.confidence, h.episode_type))
    return hypotheses


# ---------------------------------------------------------------------------
# Outcome broadcasting
# ---------------------------------------------------------------------------


def record_outcome(
    store: EpisodicStore,
    *,
    repo_signature: str,
    task_id: str,
    rollout_id: str,
    outcome: dict[str, Any],
) -> Episode:
    """Append a ROLLOUT_OUTCOME episode for this solve.

    ``outcome`` is a free-form dict (status, mutation_score, patch
    fingerprint, …); we only require ``status`` so downstream
    consumers can filter by it. Anything else just rides along in
    ``payload``.
    """
    if not isinstance(outcome, dict):
        raise TypeError("outcome must be a dict")
    payload = dict(outcome)
    payload.setdefault("status", "unknown")
    return store.broadcast(
        task_signature=compose_task_signature(repo_signature, task_id),
        rollout_id=rollout_id,
        episode_type=EPISODE_ROLLOUT_OUTCOME,
        payload=payload,
    )


def record_episode(
    store: EpisodicStore,
    *,
    repo_signature: str,
    task_id: str,
    rollout_id: str,
    episode_type: str,
    payload: dict[str, Any],
) -> Episode:
    """Append a typed episode (root cause, dead end, mutation pattern, …).

    Convenience wrapper so callers don't have to thread the task
    signature; the underlying :meth:`EpisodicStore.broadcast` is what
    actually persists.
    """
    return store.broadcast(
        task_signature=compose_task_signature(repo_signature, task_id),
        rollout_id=rollout_id,
        episode_type=episode_type,
        payload=dict(payload or {}),
    )


__all__ = [
    "ALL_EPISODE_TYPES",
    "DEFAULT_MAX_AGE_SECONDS",
    "EPISODE_DEAD_END",
    "EPISODE_MUTATION_SURVIVOR",
    "EPISODE_RELEVANT_FILE",
    "EPISODE_ROLLOUT_OUTCOME",
    "EPISODE_ROOT_CAUSE",
    "Hypothesis",
    "compose_task_signature",
    "learn_from_prior_run",
    "record_episode",
    "record_outcome",
]
