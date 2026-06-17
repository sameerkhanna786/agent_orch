"""Verification result contract (plan §13).

``VerificationResult.accepted`` is THE gate — there is no separate ``passed``
field.  ``accepted = positive_execution_evidence(candidate)``.  The cheap-first
cascade and the commit0-specific scoring live in the eval layer (which reuses
v1's ``decide_evaluation`` / ``evaluate_repo``); this module only defines the
shared contract and the candidate-construction bridge so the selector ranks on
execution-derived keys.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

from .select import Candidate


@dataclass
class VerificationResult:
    accepted: bool = False          # THE execution gate
    score: float = 0.0              # execution-derived combined score in [0,1]
    reason: Optional[str] = None
    passed: int = 0
    failed: int = 0
    errors: int = 0
    total: int = 0
    missing_expected: int = 0
    pass_rate: float = 0.0
    taxonomy: Optional[str] = None  # verification_taxonomy label
    indeterminate: bool = False     # harness/launch failure or timeout (not a genuine regression)
    # --- ADVISORY repair signal (NEVER an acceptance input; seeds the repair loop) ---
    failing_nodeids: list = field(default_factory=list)   # failed/errored test node ids
    failure_excerpts: str = ""                            # short tail of failing output

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted, "score": self.score, "reason": self.reason,
            "passed": self.passed, "failed": self.failed, "errors": self.errors,
            "total": self.total, "missing_expected": self.missing_expected,
            "pass_rate": self.pass_rate, "taxonomy": self.taxonomy,
            "indeterminate": self.indeterminate,
            "failing_nodeids": list(self.failing_nodeids)[:50],
            "failure_excerpts": self.failure_excerpts[:3000],
        }

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "VerificationResult":
        """Rebuild from to_dict() — lossless for the ACCEPTANCE fields (accepted /
        counts / pass_rate / indeterminate), so a journaled score replays the SAME
        accept decision through candidate_from_verification (Backbone 1.1). The
        advisory failing_nodeids[:50]/failure_excerpts[:3000] are knowingly lossy."""
        d = d or {}
        return cls(
            accepted=bool(d.get("accepted", False)), score=float(d.get("score", 0.0) or 0.0),
            reason=d.get("reason"), passed=int(d.get("passed", 0) or 0),
            failed=int(d.get("failed", 0) or 0), errors=int(d.get("errors", 0) or 0),
            total=int(d.get("total", 0) or 0), missing_expected=int(d.get("missing_expected", 0) or 0),
            pass_rate=float(d.get("pass_rate", 0.0) or 0.0), taxonomy=d.get("taxonomy"),
            indeterminate=bool(d.get("indeterminate", False)),
            failing_nodeids=list(d.get("failing_nodeids", []) or []),
            failure_excerpts=d.get("failure_excerpts", "") or "",
        )


def candidate_from_verification(
    *,
    candidate_id: str,
    diff: str,
    vr: VerificationResult,
    rollout_id: int = -1,
    cluster_id: int = 0,
    cluster_size: int = 1,
    changed_files_len: int = 0,
    critic_score: float = 0.0,
    eg_critic_tiebreak: float = 0.0,
    perspective_score: float = 0.0,
    meta: Optional[dict] = None,
) -> Candidate:
    """Build a ranking Candidate whose execution-derived keys come from the
    verifier.  Soft keys (eg_critic/perspective) default to 0.0 (fail-open) and
    sit strictly below execution keys in ``ranking_key``."""
    content_sha = hashlib.sha1((diff or candidate_id).encode("utf-8")).hexdigest()
    return Candidate(
        candidate_id=candidate_id,
        accepted=vr.accepted,
        combined_score=vr.score,
        public_signal_score=vr.pass_rate,
        verification_score=vr.score,
        critic_score=critic_score,
        size=max(1, cluster_size),
        eg_critic_tiebreak=eg_critic_tiebreak,
        perspective_score=perspective_score,
        changed_files_len=changed_files_len,
        cluster_id=cluster_id,
        content_sha=content_sha,
        diff=diff,
        rollout_id=rollout_id,
        meta=meta or {},
    )
