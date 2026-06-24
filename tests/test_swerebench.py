"""Tests for the SWE-rebench benchmark integration (Mode-C, gated).

These pin the load-bearing invariants of the SWE-rebench path WITHOUT any paid
agent calls or network:

  (a) commit0 path is BYTE-IDENTICAL when ``APEX_OMEGA_BENCHMARK`` is unset
      (``pin_gold_scoring_contract`` / driver routing default to commit0);
  (b) the gold provider parses FAIL_TO_PASS / PASS_TO_PASS correctly (incl. the
      numpy-ndarray representation the parquet->pandas path produces);
  (c) Cardinal Contract — a Commit0Evaluation-compatible object with NO real
      pytest run NEVER gets ``scoring_source='commit0_test_ids'`` / accept;
  (d) the swerebench registry's ``local_runnable == not forces_docker``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from apex_omega.eval import commit0_driver
from apex_omega.eval import swerebench_runner as swr
from apex_omega.eval import swerebench_registry as swreg


# --------------------------------------------------------------------------- #
# fixtures: a tiny pinned slice written to a temp file (no HF fetch)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def tiny_slice(tmp_path: Path) -> Path:
    slice_path = tmp_path / "swerebench_slice.json"
    artifact = {
        "_schema": "swerebench_slice/v1",
        "instances": {
            "demo__lib-1": {
                "instance_id": "demo__lib-1",
                "repo": "demo/lib",
                "base_commit": "0" * 40,
                "python": "3.11",
                "install": "pip install -e .",
                "reqs_path": [],
                "packages": [],
                "pip_packages": ["pytest"],
                "test_cmd": "pytest",
                "test_patch": "",
                "created_at": "2024-09-01T00:00:00",
                "stratum": "fresh",
                "fail_to_pass": ["tests/test_a.py::test_x"],
                "pass_to_pass": ["tests/test_a.py::test_y", "tests/test_a.py::test_z"],
                "gold_ids": ["tests/test_a.py::test_x", "tests/test_a.py::test_y",
                             "tests/test_a.py::test_z"],
            },
            "demo__lib-2": {
                "instance_id": "demo__lib-2",
                "repo": "demo/lib",
                "base_commit": "1" * 40,
                "python": "3.10",
                "install": "pip install -e .[dev]",
                "test_cmd": "pytest",
                "created_at": "2022-01-01T00:00:00",
                "stratum": "older",
                "fail_to_pass": ["tests/test_b.py::test_p"],
                "pass_to_pass": ["tests/test_b.py::test_q"],
                "gold_ids": ["tests/test_b.py::test_p", "tests/test_b.py::test_q"],
            },
        },
    }
    slice_path.write_text(json.dumps(artifact), encoding="utf-8")
    return slice_path


# --------------------------------------------------------------------------- #
# (a) commit0 path byte-identical when the selector is unset
# --------------------------------------------------------------------------- #
def test_active_benchmark_defaults_to_commit0(monkeypatch):
    monkeypatch.delenv("APEX_OMEGA_BENCHMARK", raising=False)
    assert commit0_driver._active_benchmark() == "commit0"
    monkeypatch.setenv("APEX_OMEGA_BENCHMARK", "swerebench")
    assert commit0_driver._active_benchmark() == "swerebench"
    monkeypatch.setenv("APEX_OMEGA_BENCHMARK", "")
    assert commit0_driver._active_benchmark() == "commit0"


def test_pin_gold_scoring_contract_commit0_unchanged(monkeypatch):
    """When the selector is unset, pin_gold_scoring_contract still pins the commit0
    gold contract (byte-identical commit0 behavior)."""
    monkeypatch.delenv("APEX_OMEGA_BENCHMARK", raising=False)
    cfg = {"benchmark": {}}
    pinned = commit0_driver.pin_gold_scoring_contract(cfg)
    # The commit0 branch merges the gold evaluation_contract into benchmark.
    assert "evaluation_contract" in pinned.get("benchmark", {}), \
        "commit0 path must still pin the gold contract when selector unset"


def test_pin_gold_scoring_contract_swerebench_is_noop(monkeypatch):
    """The swerebench path must NOT pin the commit0 contract (no-op pass-through)."""
    monkeypatch.setenv("APEX_OMEGA_BENCHMARK", "swerebench")
    cfg = {"benchmark": {"some": "value"}, "llm_configs": [{"backend": "codex_cli"}]}
    out = commit0_driver.pin_gold_scoring_contract(cfg)
    assert out is cfg, "swerebench pin_gold_scoring_contract must be an identity no-op"
    assert "evaluation_contract" not in out.get("benchmark", {})


# --------------------------------------------------------------------------- #
# (b) the gold provider parses FAIL_TO_PASS / PASS_TO_PASS correctly
# --------------------------------------------------------------------------- #
def test_gold_provider_parses_fail_and_pass(tiny_slice, monkeypatch):
    monkeypatch.setattr(swr, "SLICE_PATH", tiny_slice)
    gold = swr.gold_ids_for("demo__lib-1", path=tiny_slice)
    assert gold == sorted({
        "tests/test_a.py::test_x", "tests/test_a.py::test_y", "tests/test_a.py::test_z"})
    # gold_ids = sorted(set(FAIL_TO_PASS) | set(PASS_TO_PASS))
    rec = swr.slice_instances(tiny_slice)["demo__lib-1"]
    assert sorted(set(rec["fail_to_pass"]) | set(rec["pass_to_pass"])) == gold


def test_build_script_parse_handles_numpy_array():
    """The build-script parser normalizes the parquet->pandas numpy-ndarray
    representation that v1's _parse_literal_list mishandles.

    The build script runs with the EVAL venv (numpy + apex present); the test venv
    may lack either, so skip cleanly when its module-level imports are unavailable.
    """
    np = pytest.importorskip("numpy")
    try:
        from scripts.build_swerebench_slice import _parse_literal_list
    except Exception as exc:  # apex not importable in this venv
        pytest.skip(f"build script deps unavailable: {exc}")

    arr = np.array(["tests/a.py::test_x", "tests/a.py::test_y"], dtype=object)
    assert _parse_literal_list(arr) == ["tests/a.py::test_x", "tests/a.py::test_y"]
    # JSON / python-literal string forms still delegate to the v1 parser.
    assert _parse_literal_list('["tests/a.py::test_z"]') == ["tests/a.py::test_z"]
    assert _parse_literal_list(["tests/a.py::test_w"]) == ["tests/a.py::test_w"]
    assert _parse_literal_list(None) == []


def test_task_gold_ids_from_record(tiny_slice):
    rec = swr.slice_instances(tiny_slice)["demo__lib-2"]
    task = swr.SweRebenchTask.from_record(rec)
    assert task.gold_ids == ["tests/test_b.py::test_p", "tests/test_b.py::test_q"]
    assert task.repo_name == "lib"
    assert set(task.fail_to_pass) == {"tests/test_b.py::test_p"}
    assert set(task.pass_to_pass) == {"tests/test_b.py::test_q"}


# --------------------------------------------------------------------------- #
# (c) Cardinal Contract: no real run => never commit0_test_ids / accept
# --------------------------------------------------------------------------- #
def test_no_real_run_never_sets_gold_scoring_source():
    """An evaluation produced WITHOUT a real pytest run carries the default
    pytest_summary source and NEVER contract_success() (the Cardinal gate)."""
    # The harness-failure return path the runner uses when no report is produced.
    ev = swr._make_evaluation(
        returncode=1, output="boom", total_tests=3,
        evaluation_backend="swerebench_local_pytest_json",
        diagnostics={"harness_failure": True, "reason": "no json report produced"})
    assert ev.scoring_source != "commit0_test_ids"
    assert ev.scoring_source == "pytest_summary"
    assert ev.contract_success() is False


def test_real_full_gold_run_accepts():
    """A real full-gold run (all f2p flipped + all p2p preserved, no failed/errors/
    missing) is the ONLY shape that sets commit0_test_ids AND accepts."""
    ev = swr._make_evaluation(
        returncode=0, raw_returncode=0, output="ok",
        passed=3, failed=0, errors=0, skipped=0, total_tests=3,
        scoring_source="commit0_test_ids",
        evaluation_backend="swerebench_local_pytest_json",
        expected_test_coverage={
            "expected_test_count": 3, "matched_expected_test_count": 3,
            "missing_expected_test_count": 0, "skipped_expected_test_count": 0,
            "coverage_preserved": True},
        diagnostics={"accept": True})
    assert ev.scoring_source == "commit0_test_ids"
    assert ev.contract_success() is True


def test_partial_gold_run_does_not_accept():
    ev = swr._make_evaluation(
        returncode=1, raw_returncode=1, output="x",
        passed=2, failed=1, errors=0, skipped=0, total_tests=3,
        scoring_source="commit0_test_ids",
        evaluation_backend="swerebench_local_pytest_json",
        expected_test_coverage={
            "expected_test_count": 3, "matched_expected_test_count": 3,
            "missing_expected_test_count": 0, "skipped_expected_test_count": 0,
            "coverage_preserved": True})
    assert ev.contract_success() is False


def test_evaluate_repo_empty_gold_is_harness_failure(tmp_path):
    """No gold ids => harness_failure diagnostics, never a false accept."""
    rec = {"instance_id": "x", "repo": "demo/lib", "base_commit": "0" * 40,
           "python": "3.11", "install": "pip install -e .", "test_cmd": "pytest",
           "fail_to_pass": [], "pass_to_pass": [], "gold_ids": []}
    task = swr.SweRebenchTask.from_record(rec)
    runner = swr.SweRebenchRunner.__new__(swr.SweRebenchRunner)
    ev = runner.evaluate_repo(task, tmp_path, artifacts_dir=tmp_path / "art",
                              label="x", python_executable="python", env={},
                              expected_test_ids=[], use_expected_test_scoring=True)
    assert ev.scoring_source != "commit0_test_ids"
    assert ev.contract_success() is False
    assert (ev.diagnostics or {}).get("harness_failure") is True


def test_scoring_maps_harness_failure_to_indeterminate():
    """The UNCHANGED scoring.verification_from_commit0_evaluation maps a runner
    harness failure to INDETERMINATE (not a false zero)."""
    from apex_omega.eval.scoring import verification_from_commit0_evaluation
    ev = swr._make_evaluation(
        returncode=1, output="boom", total_tests=3,
        evaluation_backend="swerebench_local_pytest_json",
        diagnostics={"harness_failure": True})
    vr = verification_from_commit0_evaluation(ev, expected_test_count=3)
    assert vr.indeterminate is True
    assert vr.accepted is False

    ev2 = swr._make_evaluation(
        returncode=0, raw_returncode=0, output="ok", passed=3, failed=0, errors=0,
        skipped=0, total_tests=3, scoring_source="commit0_test_ids",
        evaluation_backend="swerebench_local_pytest_json",
        expected_test_coverage={"expected_test_count": 3, "matched_expected_test_count": 3,
                                "missing_expected_test_count": 0, "skipped_expected_test_count": 0,
                                "coverage_preserved": True},
        diagnostics={"accept": True})
    vr2 = verification_from_commit0_evaluation(ev2, expected_test_count=3)
    assert vr2.accepted is True
    assert vr2.indeterminate is False
    assert vr2.score == 1.0


# --------------------------------------------------------------------------- #
# (d) registry local_runnable == not forces_docker
# --------------------------------------------------------------------------- #
def test_registry_local_runnable_is_not_forces_docker(tiny_slice):
    specs = swreg.all_specs(tiny_slice)
    assert set(specs) == {"demo__lib-1", "demo__lib-2"}
    for iid, spec in specs.items():
        assert spec.local_runnable == (not spec.forces_docker)
        # curated swerebench instances are Docker-free by construction.
        assert spec.forces_docker is False
        assert spec.local_runnable is True
    assert swreg.local_runnable_targets(tiny_slice) == ["demo__lib-1", "demo__lib-2"]


def test_registry_strata_split(tiny_slice):
    assert swreg.fresh_targets(tiny_slice) == ["demo__lib-1"]
    assert swreg.older_targets(tiny_slice) == ["demo__lib-2"]
    assert swreg.get("demo__lib-1", slice_path=tiny_slice).python_version == "3.11"


def test_registry_reuses_frozen_repospec(tiny_slice):
    """The swerebench registry reuses the SAME frozen RepoSpec dataclass."""
    from apex_omega.eval.registry import RepoSpec
    spec = swreg.get("demo__lib-1", slice_path=tiny_slice)
    assert isinstance(spec, RepoSpec)
    with pytest.raises(Exception):
        spec.name = "mutated"  # frozen
