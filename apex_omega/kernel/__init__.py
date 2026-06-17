"""Execution-authoritative kernel (Cardinal Safety Contract): the only producer
of ``accepted=True`` is execution evidence; soft signals are downgrade-only."""

from .select import (
    Candidate,
    SoftReview,
    apply_evidence_bound_review,
    rank_candidates,
    ranking_key,
    select_best,
)
from .verify import VerificationResult, candidate_from_verification

__all__ = [
    "Candidate",
    "SoftReview",
    "ranking_key",
    "rank_candidates",
    "apply_evidence_bound_review",
    "select_best",
    "VerificationResult",
    "candidate_from_verification",
]
