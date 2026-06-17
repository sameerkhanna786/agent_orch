"""Durable journal crash/resume semantics (plan §15.5 kill-mid-run invariant).

A committed-OK entry is a valid cache hit across a restart; an entry left
``in_flight`` (crashed mid-call) is NOT a hit and re-runs; an ``infra_nonresult``
is not a hit and re-runs.
"""

from __future__ import annotations

import tempfile

from apex_omega.journal.wal import Journal, RESULT_INFRA_NONRESULT, RESULT_OK


def _commit(j, h, payload):
    j.commit(input_hash=h, kind="agent", prompt_canonical="p", model_id="m", vendor="v",
             cli_version="cv", scoped_inputs_hash="s", result_status=RESULT_OK,
             structured_result={"value": payload}, fs_diff_text="diff:" + payload, usage={})


def test_committed_is_hit_across_restart():
    d = tempfile.mkdtemp()
    j = Journal(d, run_id="t")
    _commit(j, "H1", "alpha")
    # new process / journal instance over the same dir
    j2 = Journal(d, run_id="t")
    hit = j2.lookup("H1")
    assert hit is not None and hit.structured_result["value"] == "alpha"
    assert j2.load_diff(hit.fs_diff_ref) == "diff:alpha"


def test_in_flight_crash_is_not_a_hit():
    d = tempfile.mkdtemp()
    j = Journal(d, run_id="t")
    j.begin(input_hash="H2", kind="agent", prompt_canonical="p", model_id="m", vendor="v",
            cli_version="cv", scoped_inputs_hash="s")  # crash before commit
    j2 = Journal(d, run_id="t")
    assert j2.lookup("H2") is None  # in_flight-only -> re-run

    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        from apex_omega.types import ExecResult
        return ExecResult(final_message="recovered", ok=True)

    from apex_omega.journal.resume import resume_or_run_exec
    res, hit = resume_or_run_exec(j2, {"kind": "agent", "prompt": "p", "model": "m", "vendor": "v",
                                       "cli_version": "cv", "scoped_inputs": {}}, fn)
    # note: this uses a different input_hash than H2; the point is in_flight H2 is gone
    assert calls["n"] == 1 and hit is False


def test_infra_nonresult_is_not_a_hit():
    d = tempfile.mkdtemp()
    j = Journal(d, run_id="t")
    j.commit(input_hash="H3", kind="agent", prompt_canonical="p", model_id="m", vendor="v",
             cli_version="cv", scoped_inputs_hash="s", result_status=RESULT_INFRA_NONRESULT,
             structured_result={}, usage={})
    j2 = Journal(d, run_id="t")
    assert j2.lookup("H3") is None  # transport failure must re-run


def test_resume_or_run_json_infra_status_not_cached():
    # Regression (review finding #2): a non-OK status must NOT become a cache hit.
    from apex_omega.journal.resume import resume_or_run_json
    d = tempfile.mkdtemp()
    j = Journal(d, run_id="t")
    calls = {"n": 0}

    def run():
        calls["n"] += 1
        return {"_returncode": 1}  # simulated infra failure

    comps = {"kind": "cell", "arm": "x", "repo": "r", "scoped_inputs": {}}
    status = lambda v: RESULT_INFRA_NONRESULT if v.get("_returncode") else RESULT_OK
    resume_or_run_json(j, comps, run, status_fn=status)
    resume_or_run_json(j, comps, run, status_fn=status)
    assert calls["n"] == 2, "infra-nonresult must re-run, never become a permanent cache hit"
    # and a fresh Journal over the same dir also re-runs it
    j2 = Journal(d, run_id="t")
    assert j2.lookup(__import__("apex_omega.journal.key", fromlist=["canonical_key"]).canonical_key(comps)) is None


def test_seq_is_monotonic_not_wallclock():
    d = tempfile.mkdtemp()
    j = Journal(d, run_id="t")
    _commit(j, "A", "1")
    _commit(j, "B", "2")
    j2 = Journal(d, run_id="t")
    # each commit allocates exactly one monotonic seq (0,1) -> next_seq recovered as 2
    assert j2._next_seq == 2
