#!/usr/bin/env python3
"""VALIDATION GATE for the SWE-rebench slice (the linchpin).

For each candidate instance: run the GOLD ``patch`` through
``swerebench_runner._prepare_repo`` -> apply the patch -> ``evaluate_repo`` and
ASSERT the evaluation ACCEPTS (every FAIL_TO_PASS flips + every PASS_TO_PASS
preserved, ``scoring_source=='commit0_test_ids'``). Instances that fail the gate
(install failed / node-id mismatch / patch reject) are reported and dropped.

Run with the EVAL venv (has uv + huggingface_hub):
    /Users/.../apex/.venv/bin/python scripts/validate_swerebench_slice.py \
        --pool /tmp/swerebench_pool.json --workdir /tmp/swe_gate --limit 8

The candidate pool carries the gold ``patch`` (added here from the dataset). The
emitted slice (``configs/swerebench_slice.json``) does NOT carry the gold patch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from apex_omega.eval.swerebench_runner import SweRebenchRunner, SweRebenchTask  # noqa: E402


def _load_patches(instance_ids: list[str]) -> dict[str, str]:
    """Fetch the gold patch for each instance id directly from the parquet shards."""
    from huggingface_hub import hf_hub_download  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
    import pandas as pd  # type: ignore

    frames = []
    for shard in ("data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet"):
        frames.append(pq.read_table(
            hf_hub_download("nebius/SWE-rebench", shard, repo_type="dataset")).to_pandas())
    df = pd.concat(frames, ignore_index=True)
    wanted = set(instance_ids)
    out: dict[str, str] = {}
    for _, r in df.iterrows():
        iid = str(r["instance_id"])
        if iid in wanted:
            out[iid] = str(r["patch"] or "")
    return out


def _apply_patch(repo_dir: Path, patch_text: str) -> tuple[bool, str]:
    """Apply the gold patch to the worktree (git apply, with fallbacks)."""
    patch_file = repo_dir / ".gold.patch"
    patch_file.write_text(patch_text, encoding="utf-8")
    for args in (
        ["git", "apply", "--whitespace=nowarn", str(patch_file)],
        ["git", "apply", "-3", "--whitespace=nowarn", str(patch_file)],
        ["patch", "-p1", "-i", str(patch_file)],
    ):
        proc = subprocess.run(args, cwd=str(repo_dir), text=True, capture_output=True)
        if proc.returncode == 0:
            return True, " ".join(args)
    return False, ((proc.stdout or "") + (proc.stderr or ""))[-1500:]


def validate_instance(rec: dict, patch_text: str, workdir: Path) -> dict:
    iid = rec["instance_id"]
    cell = workdir / iid
    repo_dir = cell / "repo"
    runtime_dir = cell / "runtime"
    result = {"instance_id": iid, "repo": rec["repo"], "stratum": rec["stratum"],
              "gold": len(rec["gold_ids"]), "passed_gate": False}
    runner = SweRebenchRunner()
    task = SweRebenchTask.from_record(rec)
    try:
        env = runner._prepare_repo(task, repo_dir, runtime_dir)
    except Exception as exc:
        result["stage"] = "prepare"
        result["error"] = str(exc)[-1200:]
        return result
    venv_python = str(Path(env["VIRTUAL_ENV"]) / "bin" / "python")
    if not patch_text.strip():
        result["stage"] = "patch"
        result["error"] = "empty gold patch"
        return result
    ok, detail = _apply_patch(repo_dir, patch_text)
    if not ok:
        result["stage"] = "apply_patch"
        result["error"] = detail
        return result
    result["patch_apply"] = detail
    try:
        ev = runner.evaluate_repo(
            task, repo_dir, artifacts_dir=cell / "evals", label=iid,
            python_executable=venv_python, env=env,
            expected_test_ids=rec["gold_ids"], use_expected_test_scoring=True,
            timeout_seconds=900)
    except Exception as exc:
        result["stage"] = "evaluate"
        result["error"] = str(exc)[-1200:]
        return result
    cov = getattr(ev, "expected_test_coverage", {}) or {}
    result.update({
        "scoring_source": ev.scoring_source,
        "passed": ev.passed, "failed": ev.failed, "errors": ev.errors,
        "skipped": ev.skipped, "total_tests": ev.total_tests,
        "missing": cov.get("missing_expected_test_count"),
        "f2p_flipped": cov.get("fail_to_pass_flipped"),
        "f2p_total": cov.get("fail_to_pass_total"),
        "p2p_preserved": cov.get("pass_to_pass_preserved"),
        "p2p_total": cov.get("pass_to_pass_total"),
        "contract_success": bool(ev.contract_success()),
        "diagnostics": {k: v for k, v in (getattr(ev, "diagnostics", {}) or {}).items()
                        if k != "per_id_outcomes"},
    })
    result["passed_gate"] = bool(
        ev.contract_success() and ev.scoring_source == "commit0_test_ids")
    if not result["passed_gate"]:
        result["stage"] = "gate"
        per_id = (getattr(ev, "diagnostics", {}) or {}).get("per_id_outcomes")
        if per_id:
            result["per_id_outcomes"] = per_id
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="candidate pool json (from build script --pool-out)")
    ap.add_argument("--workdir", default=str(Path(tempfile.gettempdir()) / "swe_gate"))
    ap.add_argument("--limit", type=int, default=0, help="cap instances validated (0=all)")
    ap.add_argument("--ids", default="", help="only validate these comma-separated ids")
    ap.add_argument("--out", default="", help="write the validation report json here")
    args = ap.parse_args()

    pool = json.loads(Path(args.pool).read_text())["instances"]
    ids = [i.strip() for i in args.ids.split(",") if i.strip()] or sorted(pool.keys())
    if args.limit:
        ids = ids[:args.limit]
    patches = _load_patches(ids)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    results = []
    for iid in ids:
        rec = pool.get(iid)
        if rec is None:
            print(f"[gate] SKIP {iid}: not in candidate pool", file=sys.stderr, flush=True)
            continue
        print(f"[gate] validating {iid} ({rec['stratum']}, gold={len(rec['gold_ids'])}) ...",
              file=sys.stderr, flush=True)
        res = validate_instance(rec, patches.get(iid, ""), workdir)
        tag = "PASS" if res["passed_gate"] else f"FAIL@{res.get('stage','?')}"
        print(f"[gate]   -> {tag}  "
              f"f2p={res.get('f2p_flipped')}/{res.get('f2p_total')} "
              f"p2p={res.get('p2p_preserved')}/{res.get('p2p_total')} "
              f"passed={res.get('passed')}/{res.get('total_tests')} "
              f"src={res.get('scoring_source')}",
              file=sys.stderr, flush=True)
        if not res["passed_gate"] and res.get("error"):
            print(f"[gate]      err: {str(res['error'])[:300]}", file=sys.stderr, flush=True)
        results.append(res)

    passed = [r["instance_id"] for r in results if r["passed_gate"]]
    report = {"validated": len(results), "passed": passed,
              "n_passed": len(passed), "results": results}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"n_validated": len(results), "n_passed": len(passed),
                      "passed_ids": passed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
