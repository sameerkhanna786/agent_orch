"""Cardinal Safety Contract: execution-authoritative selection (plan §13)."""

from __future__ import annotations

from apex_omega.kernel import (
    Candidate,
    SoftReview,
    apply_evidence_bound_review,
    candidate_from_verification,
    rank_candidates,
    ranking_key,
    select_best,
    VerificationResult,
)


def test_execution_keys_dominate_soft_keys():
    # An unverified candidate with a HUGE soft (perspective) score must never beat
    # an accepted candidate with execution evidence.
    accepted = Candidate("acc", accepted=True, combined_score=0.5, verification_score=0.5, cluster_id=1)
    soft_star = Candidate("soft", accepted=False, combined_score=0.0, perspective_score=99.0,
                          eg_critic_tiebreak=99.0, cluster_id=2)
    assert select_best([soft_star, accepted]).candidate_id == "acc"


def test_abstain_when_none_accepted():
    c = Candidate("x", accepted=False, combined_score=0.9)
    assert select_best([c]) is None  # first-class abstention


def test_monotone_downgrade_only():
    c = Candidate("x", accepted=True, combined_score=0.9)
    apply_evidence_bound_review(c, [SoftReview("approve"), SoftReview("approve")])
    assert c.accepted is True  # soft signals cannot promote/keep beyond execution
    apply_evidence_bound_review(c, [SoftReview("refute", "adversarial")])
    assert c.accepted is False  # downgrade allowed
    # there is no path back to True from a soft signal
    apply_evidence_bound_review(c, [SoftReview("approve")])
    assert c.accepted is False


def test_ranking_tuple_tiebreak_deterministic():
    a = Candidate("a", accepted=True, combined_score=0.8, cluster_id=5)
    b = Candidate("b", accepted=True, combined_score=0.8, cluster_id=2)
    # identical execution keys -> final tiebreak is -cluster_id (deterministic), not insertion order
    ranked = rank_candidates([a, b])
    assert ranked[0].candidate_id == "b"  # lower cluster_id wins via -cluster_id


def test_cardinal_relaxation_negative_control():
    # A11: with allow_unaccepted the least-bad unverified guess ships (expected to degrade).
    c = Candidate("x", accepted=False, combined_score=0.7)
    assert select_best([c], allow_unaccepted=False) is None
    assert select_best([c], allow_unaccepted=True).candidate_id == "x"


def test_content_tiebreak_order_independent_same_cluster():
    # Regression (review finding #5): two accepted candidates with identical
    # execution keys AND the same cluster_id must select the same winner
    # regardless of input order (content-derived terminal tiebreak, not insertion).
    a = Candidate("a", accepted=True, combined_score=0.8, verification_score=0.8, cluster_id=1, diff="AAA")
    b = Candidate("b", accepted=True, combined_score=0.8, verification_score=0.8, cluster_id=1, diff="BBB")
    assert select_best([a, b]).candidate_id == select_best([b, a]).candidate_id


def test_candidate_from_verification_maps_execution_keys():
    vr = VerificationResult(accepted=True, score=1.0, passed=10, total=10, pass_rate=1.0)
    c = candidate_from_verification(candidate_id="r0", diff="d", vr=vr, rollout_id=0)
    assert c.accepted and c.combined_score == 1.0 and c.public_signal_score == 1.0
    assert c.eg_critic_tiebreak == 0.0 and c.perspective_score == 0.0  # soft keys fail open
