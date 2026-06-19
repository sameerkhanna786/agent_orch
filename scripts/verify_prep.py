#!/usr/bin/env python3
"""Verify a commit0 repo PREPS + collects tests LOCALLY (no Docker, no codex burn) before we
commit heavy eval cells to it. Replicates the eval's prep path: build the v1 runner (forced
local) -> discover_tasks -> _prepare_repo (clone + uv venv + editable install) -> pytest
--collect-only. Prints RUNNABLE / NOT-RUNNABLE so we don't waste cells on a Docker-only repo.

Usage: python scripts/verify_prep.py <repo>
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from apex.core.config import ApexConfig  # noqa: E402
from apex.evaluation.commit0_benchmark import Commit0BenchmarkRunner, _load_expected_test_ids  # noqa: E402
from apex_omega.eval import registry  # noqa: E402
from apex_omega.eval.commit0_autogen import _force_local_config_dict  # noqa: E402


def main() -> int:
    repo = sys.argv[1]
    spec = registry.get(repo)
    print(f"[{repo}] registry: forces_docker={spec.forces_docker} local_runnable={spec.local_runnable} py={spec.python_version}", flush=True)
    if spec.forces_docker:
        print(f"[{repo}] RESULT: NOT-RUNNABLE (registry forces_docker)", flush=True)
        return 0
    base = json.loads(Path("configs/base_commit0_local.json").read_text())
    config = ApexConfig.from_dict(_force_local_config_dict(base))
    out = Path(tempfile.mkdtemp(prefix=f"prepverify_{repo}_"))
    fb = registry.DATASET_FALLBACK_REVISIONS.get(repo)
    runner = Commit0BenchmarkRunner(config=config, output_dir=str(out / "v1"), dataset_split="test",
                                    dataset_fallback_revisions=[fb] if fb else None, split=repo)
    tasks = runner.discover_tasks(repos=[repo], limit=1)
    print(f"[{repo}] tasks discovered: {len(tasks)}", flush=True)
    if not tasks:
        print(f"[{repo}] RESULT: NOT-RUNNABLE (no task)", flush=True)
        return 1
    try:
        env = runner._prepare_repo(tasks[0], out / "repo", out / "runtime")
    except Exception as exc:
        print(f"[{repo}] RESULT: NOT-RUNNABLE (prep failed: {type(exc).__name__}: {str(exc)[:200]})", flush=True)
        return 1
    vp = Path(env["VIRTUAL_ENV"]) / "bin" / "python"
    exp = _load_expected_test_ids(repo) or []
    print(f"[{repo}] PREP OK venv={vp.exists()} expected_gold_ids={len(exp)}", flush=True)
    try:
        r = subprocess.run([str(vp), "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
                           cwd=str(out / "repo"), capture_output=True, text=True, timeout=900,
                           env={**os.environ, **env})
    except Exception as exc:
        print(f"[{repo}] RESULT: PREP-OK but collect raised {type(exc).__name__}: {exc}", flush=True)
        return 0
    tail = ((r.stdout or "") + (r.stderr or ""))
    m = re.search(r"(\d+) tests? collected", tail)
    n = m.group(1) if m else "?"
    print(f"[{repo}] collect rc={r.returncode} collected={n}\n  tail: {tail[-240:]}", flush=True)
    print(f"[{repo}] RESULT: RUNNABLE (collected {n} tests)" if r.returncode in (0, 5)
          else f"[{repo}] RESULT: PREP-OK but collect rc={r.returncode} (modular stubs may not import yet — still runnable)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
