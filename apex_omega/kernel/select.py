"""Deterministic, execution-authoritative selection (Fusion Ledger A3; plan §13).

Ported as pure functions from v1's selector (selector.py:2944-2960 ranking tuple;
selector.py:3682 ``_apply_evidence_bound_review`` monotone flip).  The Cardinal
Safety Contract is enforced *structurally*:

  1. ``accepted`` starts from EXECUTION evidence (the verifier), never a score
     threshold; soft signals can only flip it ``True -> False`` (downgrade), never
     ``False -> True`` (promote).
  2. The ranking key is a lexicographic tuple where every execution+deterministic
     key sits strictly ABOVE every learned/LLM key, terminating in a content-/
     cluster-derived tiebreak, NEVER insertion order.
  3. With no accepted candidate the selector ABSTAINS (returns None) rather than
     shipping its least-bad guess.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


@dataclass
class Candidate:
    candidate_id: str
    # --- execution-derived keys (authoritative) ---
    accepted: bool = False
    combined_score: float = 0.0
    public_signal_score: float = 0.0
    verification_score: float = 0.0
    # --- deterministic critic (execution-adjacent, allowed above soft) ---
    critic_score: float = 0.0
    size: int = 1                 # cluster size (how many rollouts agree)
    # --- soft signals: STRICTLY below every execution key ---
    eg_critic_tiebreak: float = 0.0   # learned
    perspective_score: float = 0.0    # LLM
    # --- deterministic final tiebreaks ---
    changed_files_len: int = 0
    cluster_id: int = 0
    # content-derived terminal tiebreak (so colliding cluster_ids never fall back
    # to insertion order -> exploration-order-independent, byte-identical replay)
    content_sha: str = ""
    # identity / payload
    diff: str = ""
    rollout_id: int = -1
    meta: dict = field(default_factory=dict)

    # Dict-like access so LLM-authored orchestrate(ctx) code can inspect a
    # candidate as c["accepted"] / c.get("combined_score") as well as c.accepted.
    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    # ---- structural soft-write seam (Backbone 2.2) ----
    # The ONLY sanctioned mutations a pattern may make. They are structural guarantees:
    # set_soft can touch ONLY the two soft keys (both strictly below every execution key
    # in ranking_key), and refute can ONLY downgrade accepted True->False. So no pattern
    # can ever PROMOTE an unverified candidate (Cardinal Contract preserved without an
    # AST guard, since patterns run host-side outside the sandbox).
    def set_soft(self, *, perspective: Optional[float] = None, eg_critic: Optional[float] = None) -> "Candidate":
        if perspective is not None:
            self.perspective_score = float(perspective)
        if eg_critic is not None:
            self.eg_critic_tiebreak = float(eg_critic)
        return self

    def refute(self) -> "Candidate":
        self.accepted = False        # monotone downgrade only; never False->True
        return self


def ranking_key(c: Candidate) -> tuple:
    """The v1 deterministic ranking tuple (higher is better; sort with
    ``reverse=True``).  Execution keys dominate; learned/LLM keys are strictly
    lower; final tiebreak is ``-cluster_id`` (deterministic, not insertion order)."""
    content = c.content_sha or hashlib.sha1((c.diff or c.candidate_id).encode("utf-8")).hexdigest()
    return (
        c.combined_score,        # execution-derived
        int(c.accepted),         # execution gate
        c.public_signal_score,   # execution-derived
        c.critic_score,          # deterministic critic
        c.size,
        c.verification_score,
        c.eg_critic_tiebreak,    # learned  -- below ALL execution keys
        c.perspective_score,     # LLM      -- below learned
        c.changed_files_len,
        -c.cluster_id,
        content,                 # content-derived terminal tiebreak (NEVER insertion order)
    )


@dataclass
class SoftReview:
    verdict: str               # "refute" downgrades; anything else is a no-op
    source: str = ""
    reason: str = ""


def apply_evidence_bound_review(candidate: Candidate, soft_reviews: Sequence[SoftReview]) -> Candidate:
    """Monotone-in-one-direction acceptance review (v1 ``_apply_evidence_bound_review``).

    Soft reviews may ONLY flip ``accepted`` ``True -> False``.  There is no branch
    that sets ``accepted = True`` from a soft signal — that is the structural
    guarantee a soft signal can never promote an unverified candidate."""
    for review in soft_reviews:
        if candidate.accepted and review.verdict == "refute":
            candidate.accepted = False  # downgrade allowed
        # there is NO branch that sets candidate.accepted = True from a soft signal
    return candidate


def rank_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Deterministic descending rank.  Pure function of candidate fields."""
    return sorted(candidates, key=ranking_key, reverse=True)


def select_best(
    candidates: Sequence[Candidate],
    *,
    allow_unaccepted: bool = False,
) -> Optional[Candidate]:
    """Return the top-ranked ACCEPTED candidate, or None (abstain) if none is
    accepted.  ``allow_unaccepted=True`` is provided ONLY for the A11 negative
    control (Cardinal-relaxation) — it must never be the default path."""
    if not candidates:
        return None
    ranked = rank_candidates(candidates)
    for c in ranked:
        if c.accepted:
            return c
    if allow_unaccepted:
        # A11 negative control: ship the least-bad unverified guess (expected to degrade).
        return ranked[0]
    return None  # abstain: no positive execution evidence (first-class outcome)
