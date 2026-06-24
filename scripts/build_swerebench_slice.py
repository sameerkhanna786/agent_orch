#!/usr/bin/env python3
"""OFFLINE build script for the pinned SWE-rebench evaluation slice.

Run with the EVAL venv (``/Users/sameertkhanna/Documents/apex/apex/.venv/bin/python``)
which has ``huggingface_hub`` + ``pyarrow``:

    /Users/.../apex/.venv/bin/python scripts/build_swerebench_slice.py

It reads the two ``nebius/SWE-rebench`` parquet shards DIRECTLY (the eval venv's
``datasets`` torch auto-format is broken — bypass it), filters to Docker-free,
pure-pip, small-gold instances stratified by recency (FRESH >=2024-06 vs OLDER
<2023), and EMITS a checked-in pinned artifact ``configs/swerebench_slice.json``
mapping ``instance_id -> {repo, base_commit, python, install, requirements,
test_cmd, created_at, stratum, gold_ids}``.

This is the DETERMINISTIC registry + inventory source for the SWE-rebench Mode-C
path — it is NEVER re-fetched at eval time.  ``swerebench_runner`` /
``swerebench_registry`` read this file only.

IMPORTANT design note (verified against the real data):
  * The dataset's ``requirements`` column is a conda-style ``@ file:///croot/...``
    pin list built on a Linux conda box — NOT installable via uv on a mac.  We
    record it ONLY for provenance.  The runner installs from ``install`` (e.g.
    ``pip install -e .[dev]``) + the in-repo ``reqs_path`` + ``pip_packages``,
    and skips any ``file://`` / ``@`` local pins.
  * Gold ids = ``sorted(set(FAIL_TO_PASS) | set(PASS_TO_PASS))`` parsed via
    ``apex.evaluation.swebench_benchmark._parse_literal_list``.

A SECOND optional pass (``--validate``) is the build-time validation gate driver:
it runs each candidate's GOLD patch through ``swerebench_runner`` and keeps only
instances whose gold patch makes every FAIL_TO_PASS flip + every PASS_TO_PASS
preserved.  By default the build only emits the *candidate pool*; pass
``--keep <id,id,...>`` to pin the validated subset after running the gate
separately (see ``scripts/validate_swerebench_slice.py`` invoked by the runner's
self-test).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np  # type: ignore

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Reuse the v1 SWE-bench list parser for FAIL_TO_PASS / PASS_TO_PASS ONLY.
from apex.evaluation.swebench_benchmark import _parse_literal_list as _v1_parse_literal_list  # noqa: E402


def _parse_literal_list(value) -> list[str]:
    """Parse FAIL_TO_PASS / PASS_TO_PASS via v1's ``_parse_literal_list``.

    The parquet->pandas representation hands these columns as numpy ``ndarray``s,
    which v1's parser does NOT special-case (it would ``str()`` the WHOLE array
    into a single mangled element). Normalize an ndarray (or any non-str
    sequence) to a plain Python list of strings FIRST, then delegate to v1 so the
    JSON/python-literal-string and bare-string cases still go through the shared
    parser unchanged.
    """
    if isinstance(value, np.ndarray):
        return [str(x) for x in value.tolist() if x is not None]
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if x is not None]
    return _v1_parse_literal_list(value)

SLICE_PATH = _REPO / "configs" / "swerebench_slice.json"

FRESH_CUTOFF = "2024-06-01"
OLDER_CUTOFF = "2023-01-01"

# A curated candidate POOL of lightweight, pure-Python, pip-installable repos that
# install apt-free under uv.  The validation gate (run separately) drops any that
# fail to install or whose pinned gold-ids don't match collected node-ids.  These
# were chosen from the Docker-free / pure-pip / small-gold pool (see the module
# docstring); sqlglot/narwhals are FRESH (>=2024-06), pyupgrade/faker are OLDER.
DEFAULT_REPOS = (
    "asottile/pyupgrade",       # OLDER: single-file AST rewriter, pip install -e .[dev]
    "joke2k/faker",             # OLDER: pure-python lib
    "tobymao/sqlglot",          # FRESH: pure-python SQL parser
    "narwhals-dev/narwhals",    # FRESH: pure-python dataframe-compat shim
    "neogeny/TatSu",            # OLDER: parser generator
    "reata/sqllineage",         # OLDER: pure-python
)


def _empty(v) -> bool:
    if v is None:
        return True
    if isinstance(v, (list, tuple, np.ndarray)):
        return len(v) == 0
    if isinstance(v, str):
        return not v.strip()
    return False


def _ic_get(ic, key):
    try:
        return ic[key]
    except Exception:
        return None


def _as_str_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple, np.ndarray)):
        return [str(x) for x in v if x is not None]
    if isinstance(v, str):
        return [v] if v.strip() else []
    return [str(v)]


def _load_dataframe():
    from huggingface_hub import hf_hub_download  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
    import pandas as pd  # type: ignore

    frames = []
    for shard in ("data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet"):
        path = hf_hub_download("nebius/SWE-rebench", shard, repo_type="dataset")
        frames.append(pq.read_table(path).to_pandas())
    return pd.concat(frames, ignore_index=True)


def _stratum(created_at: str) -> str | None:
    c = str(created_at or "")[:10]
    if c >= FRESH_CUTOFF:
        return "fresh"
    if c < OLDER_CUTOFF:
        return "older"
    return None  # mid-range (2023..2024-05) is neither stratum


def _candidate_record(row) -> dict | None:
    ic = row["install_config"]
    if not hasattr(ic, "__getitem__"):
        return None
    # Docker-free filter: empty pre_install (the verified Docker-free signal).
    if not _empty(_ic_get(ic, "pre_install")):
        return None
    install = str(_ic_get(ic, "install") or "").strip()
    if "apt" in install.lower() or "pip" not in install:
        return None  # pure-pip only
    f2p = sorted(set(_parse_literal_list(row.get("FAIL_TO_PASS"))))
    p2p = sorted(set(_parse_literal_list(row.get("PASS_TO_PASS"))))
    if not (1 <= len(f2p) <= 15):
        return None
    gold_ids = sorted(set(f2p) | set(p2p))
    if not (2 <= len(gold_ids) <= 40):
        return None  # small gold universe -> fast, deterministic validation
    stratum = _stratum(row["created_at"])
    if stratum is None:
        return None
    # uv must be able to provision the interpreter locally. Pythons < 3.8 (e.g.
    # 3.6/3.7) are not available as uv-managed downloads on this mac (uv venv
    # rc=2 "No interpreter found"), so they fail the runnability gate by
    # construction — exclude them up front so the curated pool is uv-installable.
    pyv = str(_ic_get(ic, "python") or "")
    try:
        major, minor = (int(x) for x in pyv.split(".")[:2])
        if (major, minor) < (3, 8):
            return None
    except Exception:
        return None
    # reqs_path: in-repo requirements file(s) the runner may install (apt-free).
    reqs_path = [str(x) for x in _as_str_list(_ic_get(ic, "reqs_path"))]
    packages = [str(x) for x in _as_str_list(_ic_get(ic, "packages"))]
    pip_packages = [str(x) for x in _as_str_list(_ic_get(ic, "pip_packages"))]
    return {
        "instance_id": str(row["instance_id"]),
        "repo": str(row["repo"]),
        "base_commit": str(row["base_commit"]),
        "python": str(_ic_get(ic, "python") or "3.11"),
        "install": install,
        # in-repo requirements paths (installable apt-free); the raw conda pins in
        # the `requirements` column are recorded for provenance only, NOT installed.
        "reqs_path": reqs_path,
        "packages": packages,
        "pip_packages": pip_packages,
        "requirements": str(row.get("requirements") or "")[:4000],  # provenance only
        "test_cmd": str(_ic_get(ic, "test_cmd") or "pytest"),
        "log_parser": str(_ic_get(ic, "log_parser") or ""),
        # The gold TEST state = base + test_patch (SWE-bench semantics): the
        # FAIL_TO_PASS tests are DEFINED by test_patch, so the runner applies it
        # during prep (tests are the read-only spec the agent solves against). This
        # is the gold tests, NOT the gold code solution.
        "test_patch": str(row.get("test_patch") or ""),
        "created_at": str(row["created_at"]),
        "stratum": stratum,
        "fail_to_pass": f2p,
        "pass_to_pass": p2p,
        "gold_ids": gold_ids,
    }


def build_candidate_pool(*, repos: tuple[str, ...], per_repo: int) -> dict:
    df = _load_dataframe()
    pool: dict[str, dict] = {}
    counts: dict[str, int] = {}
    # Deterministic order: sort by instance_id so re-runs are reproducible.
    df = df.sort_values("instance_id", kind="stable")
    for _, row in df.iterrows():
        if str(row["repo"]) not in repos:
            continue
        rec = _candidate_record(row)
        if rec is None:
            continue
        key = (rec["repo"], rec["stratum"])
        if counts.get(key, 0) >= per_repo:
            continue
        pool[rec["instance_id"]] = rec
        counts[key] = counts.get(key, 0) + 1
    return pool


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the pinned SWE-rebench slice")
    ap.add_argument("--repos", default=",".join(DEFAULT_REPOS),
                    help="comma-separated org/name repos to draw candidates from")
    ap.add_argument("--per-repo", type=int, default=3,
                    help="max candidate instances per (repo, stratum)")
    ap.add_argument("--keep", default="",
                    help="comma-separated instance_ids to PIN as the validated slice "
                         "(if set, only these survive into the artifact)")
    ap.add_argument("--out", default=str(SLICE_PATH))
    ap.add_argument("--pool-out", default="",
                    help="also write the full candidate pool here (for the gate)")
    args = ap.parse_args()

    repos = tuple(r.strip() for r in args.repos.split(",") if r.strip())
    pool = build_candidate_pool(repos=repos, per_repo=args.per_repo)
    print(f"[build] candidate pool: {len(pool)} instances across {len(repos)} repos",
          file=sys.stderr)
    by_stratum: dict[str, int] = {}
    for rec in pool.values():
        by_stratum[rec["stratum"]] = by_stratum.get(rec["stratum"], 0) + 1
    print(f"[build] strata: {by_stratum}", file=sys.stderr)

    if args.pool_out:
        Path(args.pool_out).write_text(
            json.dumps({"instances": pool}, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[build] wrote candidate pool -> {args.pool_out}", file=sys.stderr)

    keep = [k.strip() for k in args.keep.split(",") if k.strip()]
    if keep:
        missing = [k for k in keep if k not in pool]
        if missing:
            print(f"[build] WARNING: --keep ids not in pool (re-deriving from full data): {missing}",
                  file=sys.stderr)
            # Re-derive any kept id that fell outside the per-repo cap directly.
            df = _load_dataframe()
            for _, row in df.iterrows():
                if str(row["instance_id"]) in missing:
                    rec = _candidate_record(row)
                    if rec is not None:
                        pool[rec["instance_id"]] = rec
        selected = {k: pool[k] for k in keep if k in pool}
    else:
        selected = pool

    artifact = {
        "_schema": "swerebench_slice/v1",
        "_source": "nebius/SWE-rebench",
        "_strata": {"fresh": f">= {FRESH_CUTOFF}", "older": f"< {OLDER_CUTOFF}"},
        "_note": ("Pinned offline registry+inventory; never re-fetched at eval time. "
                  "gold_ids = sorted(set(FAIL_TO_PASS)|set(PASS_TO_PASS)). The raw "
                  "`requirements` is conda file:// pins kept for provenance ONLY; the "
                  "runner installs from `install` + in-repo reqs_path + pip_packages."),
        "instances": dict(sorted(selected.items())),
    }
    Path(args.out).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"[build] wrote {len(selected)} instances -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
