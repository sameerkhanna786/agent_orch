"""Commit0 scoring wrappers — reuse v1's execution-authoritative scoring verbatim.

The acceptance gate is v1's ``contracts.decide_evaluation`` (pure) and the local
scorer is ``Commit0BenchmarkRunner.evaluate_repo`` with the
``local_pytest_json_report`` backend (host per-repo uv venv pytest, no Docker).
We never re-implement the gate — that would risk diverging from the only
publishable number (plan §20.1).
"""

from __future__ import annotations

from typing import Any, Optional

from ..kernel.verify import VerificationResult

# The scoring_source value v1 sets ONLY on the gold expected-test-id path. An accept from any
# other source (e.g. the "pytest_summary" visible-suite fallback) is not a trustworthy gold solve.
# Named here so a rename of the v1 literal is a one-line change, not a silent downgrade (review L3).
_GOLD_SCORING_SOURCES = frozenset({"commit0_test_ids"})


def load_expected_ids(repo_name: str) -> list[str]:
    """Visible gold expected-test inventory (the scoring universe).  Returns []
    if the commit0 package is not importable (scoring degrades to runner summary)."""
    try:
        from apex.evaluation.commit0_benchmark import _load_expected_test_ids
        return list(_load_expected_test_ids(repo_name) or [])
    except Exception:
        return []


def decide_from_counts(
    *, passed: int, failed: int, errors: int, total: int, missing: int = 0, raw_returncode: int = 0
) -> tuple[bool, str, float]:
    """Apply v1's commit0 acceptance gate to raw counts.  Returns
    ``(accepted, reason, pass_rate)``.  Falls back to the plain commit0 rule
    (total>0, failed==0, errors==0, missing==0, pass_rate>=1.0) if v1 contracts
    are unavailable."""
    pass_rate = (passed / total) if total > 0 else 0.0
    try:
        from apex.evaluation.contracts import (
            decide_evaluation,
            EvaluationContract,
            ScoredCounts,
        )
        decision = decide_evaluation(
            contract=EvaluationContract.commit0_expected_ids(),
            scored=ScoredCounts(passed=passed, failed=failed, errors=errors, total=total, missing=missing),
            raw_returncode=raw_returncode,
        )
        return bool(decision.is_success), str(getattr(decision, "kind", "")), pass_rate
    except Exception:
        ok = total > 0 and failed == 0 and errors == 0 and missing == 0 and pass_rate >= 1.0
        return ok, ("solved" if ok else "unsolved"), pass_rate


def verification_from_commit0_evaluation(evaluation: Any, *, expected_test_count: int = 0) -> VerificationResult:
    """Map a v1 ``Commit0Evaluation`` to APEX-Ω's VerificationResult.  The
    ``accepted`` gate comes from the evaluation's own contract success (execution
    evidence), never a soft score.

    ``expected_test_count`` is the AUTHORITATIVE static gold-id universe size
    (``len(_load_expected_test_ids(repo))`` from the commit0 bz2 inventory). When
    supplied it is the measurement-integrity denominator-of-record (FM-3): if the
    scored gold denominator collapsed below it (a collection-collapse run where
    only a handful of expected ids COLLECTED — the babel ``gold_total=10`` artifact),
    the uncollected ids are MISSING by construction, so the result can NEVER falsely
    accept on the collected subset and the frontier sees the TRUE distance-to-solve."""
    def _g(name: str, default=0):
        return getattr(evaluation, name, default)

    passed = int(_g("passed", 0) or 0)
    failed = int(_g("failed", 0) or 0)
    errors = int(_g("errors", 0) or 0)
    # GOLD COUNT FIX: the v1 Commit0Evaluation field is ``total_tests`` (= number of GOLD
    # expected test ids in the gold path), NOT ``total`` — reading ``total`` returned the
    # default 0 on EVERY cell, which (a) zeroed the gold tier of the cut-losses detector and
    # (b) made failure_ledger mislabel real partial-gold progress as "expected_id_mismatch".
    cov = getattr(evaluation, "expected_test_coverage", {}) or {}
    total = int(_g("total_tests", _g("total", 0)) or 0)
    missing = int(_g("missing_expected",
                    cov.get("missing_expected_test_count", _g("missing", 0))) or 0)
    pass_rate = float(_g("pass_rate", 0.0) or 0.0)
    # FM-3 measurement-integrity reconcile: the gold denominator-of-record is the AUTHORITATIVE
    # static expected-id universe, never the COLLECTED subset. When the scored denominator collapsed
    # below it (collection-collapse: only a few expected ids collected, e.g. babel gold_total=10),
    # the uncollected ids are MISSING by construction. Lift total to the true universe and recount
    # missing so (a) the SPFG+/SPFG++ frontier sees the honest distance-to-solve and (b) a collapsed
    # run can NEVER falsely accept on a tiny denominator (the accept reconciliation happens below,
    # after the contract `accepted` is read). passed/failed/errors are left untouched — they are real
    # per-id outcomes; only the denominator + missing are made honest.
    _collapsed_universe = False
    if expected_test_count and total < expected_test_count:
        _collapsed_universe = True
        missing = max(missing, expected_test_count - passed)
        total = expected_test_count
        pass_rate = (passed / total) if total > 0 else 0.0
    # Prefer the evaluation's own contract decision (execution-authoritative).
    accepted = bool(
        getattr(evaluation, "scored_success", None)
        if getattr(evaluation, "scored_success", None) is not None
        else (evaluation.contract_success() if hasattr(evaluation, "contract_success") else False)
    )
    # review-fix #7: a NON-genuine outcome (harness/parser/environment failure) returns a
    # Commit0Evaluation WITHOUT raising — evaluation_status == "audit_inconclusive" /
    # taxonomy flags it. Map those to INDETERMINATE so _scored journals them as
    # infra_nonresult (re-run on resume) instead of caching a transient as a real failure.
    _status = str(getattr(evaluation, "evaluation_status", "") or "").lower()
    _tax = str(getattr(evaluation, "verification_taxonomy", "") or "").lower()
    # P0 harness fix (defense in depth): a pytest plugin-abort (rc=4 usage error
    # before collection), a collection error, or a native interpreter crash
    # (segfault/abort/signal: rc<0 or 134-139) is an ENVIRONMENT failure, never a
    # genuine 0. The upstream runner already classifies these as harness_failure /
    # parser_error (-> evaluation_status "audit_inconclusive"), but we also gate on
    # the raw diagnostics + returncode here so a crashed interpreter can never flow
    # into the SPFG+ frontier as a real residual. KEEP the all-gold-ids accept gate:
    # only indeterminate is neutralized; a real partial stays a real residual.
    _diag = getattr(evaluation, "diagnostics", None) or {}
    _diag_indet = bool(
        (isinstance(_diag, dict) and (
            _diag.get("harness_failure")
            or _diag.get("parser_error")
            or _diag.get("native_crash_returncode") is not None))
    )
    _rc = getattr(evaluation, "returncode", 0)
    try:
        _rc = int(_rc)
    except (TypeError, ValueError):
        _rc = 0
    _native_crash = _rc < 0 or _rc in (134, 137, 138, 139)
    indeterminate = ("inconclusive" in _status
                     or any(k in _tax for k in ("harness_failure", "parser_error", "environment_failure"))
                     or _diag_indet
                     or _native_crash)
    if indeterminate:
        accepted = False
    # FM-3: an ACCEPT computed on a COLLAPSED gold universe is not a real solve — you cannot be
    # "solved" without having RUN the full expected-id set. Downgrade to a genuine PARTIAL (a real
    # residual, NOT indeterminate: the collected ids really did pass), so the frontier banks honest
    # partial progress while a false full-solve can never slip through on a tiny denominator.
    if _collapsed_universe and accepted:
        accepted = False
    # GOLD-SCORING GUARD (belt-and-suspenders): under the REQUIRED gold contract the gold path
    # sets scoring_source="commit0_test_ids". An ACCEPT produced by any other source (e.g. the
    # "pytest_summary" visible-suite fallback, or an unfilled default) is NOT a trustworthy gold
    # solve — never bank it as accepted; downgrade to indeterminate so it re-runs under gold
    # scoring. (With gold-scoring pinned+asserted upstream this should never fire; it is the last
    # line of defense against a visible-suite false positive ever counting as a solve.)
    scoring_source = str(getattr(evaluation, "scoring_source", "") or "").strip().lower()
    _nongold_downgrade = bool(accepted and scoring_source and scoring_source not in _GOLD_SCORING_SOURCES)
    if _nongold_downgrade:
        accepted = False
        indeterminate = True
        # review L2: a non-gold (visible-suite) result must not feed its raw pass counts into the
        # detector's BEST distance-to-solve accounting — zero the gold-bearing signals.
        passed = 0
        pass_rate = 0.0
    # combined execution score in [0,1]: full credit only on accept, else pass_rate*0.9 cap
    score = 1.0 if accepted else min(0.89, pass_rate)
    # ADVISORY repair signal (best-effort; never an acceptance input). Tolerant of
    # whatever the v1 evaluation surfaces — empty on any miss.
    failing_nodeids: list = []
    for attr in ("failing_test_ids", "failed_test_ids", "failing_tests", "failed_tests", "failures"):
        v = getattr(evaluation, attr, None)
        if v:
            try:
                failing_nodeids = [str(x) for x in v][:50]
            except Exception:
                failing_nodeids = []
            if failing_nodeids:
                break
    excerpts = ""
    for attr in ("failure_excerpts", "output_tail", "stdout_tail", "test_output_tail", "summary"):
        v = getattr(evaluation, attr, None)
        if isinstance(v, str) and v.strip():
            excerpts = v[-3000:]
            break
    return VerificationResult(
        accepted=accepted, score=score, passed=passed, failed=failed, errors=errors,
        total=total, missing_expected=missing, pass_rate=pass_rate,
        taxonomy=str(getattr(evaluation, "verification_taxonomy", "") or ""),
        indeterminate=indeterminate,
        reason=(f"commit0 accept rejected: non-gold scoring_source={scoring_source!r} "
                "(visible-suite fallback, not gold expected-id match)" if _nongold_downgrade
                else ("commit0 scoring inconclusive (harness/parser/env failure)" if indeterminate
                      else (f"commit0 gold universe collapsed: only {passed + failed + errors}/"
                            f"{expected_test_count} expected ids collected (missing={missing})"
                            if _collapsed_universe
                            else (None if accepted else "commit0 visible suite not fully green")))),
        failing_nodeids=failing_nodeids, failure_excerpts=excerpts,
    )
