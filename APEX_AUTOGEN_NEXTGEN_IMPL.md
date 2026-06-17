# APEX-Ω Autogen Next-Gen — File-Level Implementation Plan

# FILE-LEVEL IMPLEMENTATION PLAN (gap-closed; absolute paths)

Phases land in order; `pytest tests/` must be green (re-confirm the real baseline on the apex venv first — static grep counts ~78 `def test_`; the plan's "79" must be empirically verified) before the next phase.

## Phase P0 — harness correctness (land + prove green FIRST)

### Step 1 — P0.1 race-safe editable resolution. `/Users/sameertkhanna/Documents/agent_orch/apex_omega/eval/commit0_autogen.py::score_fn`
Add helpers and rewrite `score_fn` (currently 237-260, reuses shared `env` at line 249):
```python
import subprocess, os
from pathlib import Path
from typing import Optional

def _detect_src_pkg(worktree_path: Path) -> Optional[str]:
    src = worktree_path / "src"
    if not src.is_dir():
        return None
    for child in sorted(src.iterdir()):
        if child.is_dir() and (child / "__init__.py").exists():
            return child.name
    return None

def _resolve_pkg_file(call_env: dict, venv_python: str, pkg: str) -> Optional[str]:
    try:
        r = subprocess.run([venv_python, "-c",
            f"import {pkg},sys; sys.stdout.write({pkg}.__file__ or '')"],
            env=call_env, capture_output=True, text=True, timeout=60)
        return (r.stdout or "").strip() or None
    except Exception:
        return None
```
In `score_fn`, before `evaluate_repo`:
```python
wt = Path(worktree_path)
call_env = dict(env)                       # per-call: NEVER mutate the shared env
src_pkg = _detect_src_pkg(wt)              # None for flat repos -> FAST-PATH NO-OP
if src_pkg is not None:
    prior = call_env.get("PYTHONPATH", "")
    call_env["PYTHONPATH"] = str(wt / "src") + (os.pathsep + prior if prior else "")
    resolved = _resolve_pkg_file(call_env, str(venv_python), src_pkg)
    if resolved is None or not str(Path(resolved).resolve()).startswith(str(wt.resolve())):
        from ..kernel.verify import VerificationResult
        engine.log(f"P0.1 misconfig: {src_pkg} -> {resolved}, not under {wt}")
        return VerificationResult(accepted=False, score=0.0, indeterminate=True,
            reason=f"editable resolution outside worktree: {resolved}")
# pass env=call_env (NOT env) into evaluate_repo; pass artifacts_dir through (Step 5)
```
Fallback note: if a true PEP-660 finder repo is found where PYTHONPATH does not win, call the existing `_verify_editable_target_inside_repo` (`/Users/sameertkhanna/Documents/agent_orch/apex/evaluation/commit0_benchmark.py:14033`) per-tree under a per-venv `threading.Lock`. Flat repos never reach any of this.

### Step 2 — P0.2 memray + parity. `/Users/sameertkhanna/Documents/agent_orch/apex/evaluation/commit0_benchmark.py`
At `_COMMIT0_PYTEST_OPTION_PLUGIN_PACKAGE_MODULES` (line 1588) add `"pytest-memray": "pytest_memray",`. Add a generic guard so any inferred package whose `--<opt>` is present gets `-p <module>`. In `commit0_autogen.py::prompt_builder`, surface the exact authoritative gate command for self-verify.

### Step 3 — P0.3 reflog/branch scrub. `/Users/sameertkhanna/Documents/agent_orch/apex_omega/eval/repo_sanitize.py::scrub_upstream_identifiers`
After tag/remote scrub: `git reflog expire --all --expire=now`, `git gc --prune=now`, delete every branch != `apex-base`.

### Step 4 — P0.4 de-seed. `/Users/sameertkhanna/Documents/agent_orch/apex/evaluation/commit0_benchmark.py::build_issue_description` (line 3868)
Add `_neutralize_upstream_spec(self.specification)` (strip `https?://\S+` and `v?\d+\.\d+(\.\d+)?` literals) before the `Task objective:` line (3882).

**Gate: re-run jinja under TODAY's unchanged orchestration to prove P0.1 alone flips fail->solve. Then `pytest tests/` green.**

## Phase E — advisory failing-test signal
### Step 5 — `/Users/sameertkhanna/Documents/agent_orch/apex_omega/kernel/verify.py`: add `failing_nodeids`, `failure_excerpts`, `finalization_status` to `VerificationResult` + `to_dict`; thread into `Candidate.meta` in `candidate_from_verification` (line 44). `/Users/sameertkhanna/Documents/agent_orch/apex_omega/eval/scoring.py::verification_from_commit0_evaluation`: add `artifacts_dir` kwarg + tolerant `_extract_failures` (glob pytest-json, collect failed/error nodeids + longrepr tails capped ~3KB; try/except -> `([], "")`). `score_fn` passes `artifacts_dir`.

## Phase L — lineage + ladder
### Step 6 — `/Users/sameertkhanna/Documents/agent_orch/apex_omega/autogen/context.py`
- `ANTI_FETCH_POLICY` const (bans external-package acquisition; PERMITS `urllib.parse`, `pip install -e .`, pytest, compileall).
- Line 128: `task_prompt = (prompt or self._prompt_builder(self, aid, strat)) + ANTI_FETCH_POLICY`.
- `solve_attempt`: add `suppress_approach` param (sets `self._suppress_approach` read by prompt_builder); thread `res.fetch_attempted`, `finalization_status`, `diff`, `pass_rate` into meta.
- `repair_attempt(parent, ...)`: fresh `acquire` + `apply_diff(wt, parent_diff)`; **on False, retry `git apply --3way`; on continued failure return None/indeterminate** (never repair from base); prompt = failing nodeids + excerpts + prior diff + ANTI_FETCH_POLICY; `scoped_inputs` includes `parent_diff_sha` + failing set; one journaled `engine.agent`; re-score; cannot set accepted.
- `should_repair(cand)`: genuine + improvable (not fetch_attempted, not policy_violation, not indeterminate, pass_rate>0 OR collection-error count decreasing).
- `solve_and_repair(...)`: base then <= max_iters repairs; stop on accept / plateau / ceiling; vendor-cycle repairs.
- `make_repairing_attempt(i)`, `repair_depth()`, `raise_cap(reason, to, min_delta=0.02)` (never exceeds `engine.max_total_agents`; 3-strike enforced by caller; tied to token-headroom).
- `localize(candidates)` (SBFL Ochiai -> `repo_map["edit_targets"]`; fallback to surfacing failing_nodeids when no contrast).
- `select`: provenance refute (`fetch_attempted` OR post-session dist-info hit) + test-file immutability via `apply_evidence_bound_review` (monotone, downgrade-only — `select.py:85`).

### Step 7 — `/Users/sameertkhanna/Documents/agent_orch/apex_omega/autogen/templates.py`: rewrite `DEFAULT_ORCHESTRATION` (14-42) to R0(suppress_approach)->R1(solve_and_repair)->R2(diverse repair-capable fan, plateau stop). Repurpose `DECOMPOSE_EXEMPLAR` as the R3 authored exemplar.

### Step 8 — `/Users/sameertkhanna/Documents/agent_orch/apex_omega/autogen/architect.py`
- **Fix line 35** so R0-R2 read STATIC `build_repo_map` difficulty; gate scout (line 28-41) + author behind static==hard (R3).
- `build_repo_map`: add `edit_targets`/`skeletons` slots; **add pre-eval smoke gate hook + collection-error-count metric** and a broad-deficit detector (low pass_rate + high error diversity -> decompose).
- `API_REFERENCE` += new ctx surface; `INVARIANTS` += rule 6.
- Keep `solved` = autogen-only; add separate `pooled_winner`.

## Phase A — anti-shortcut enforcement
### Step 9 — `/Users/sameertkhanna/Documents/agent_orch/apex_omega/executor/v1_executor.py` + `/Users/sameertkhanna/Documents/agent_orch/apex/core/cli_backend.py`
- **Build the NEW jail command detector** (does not exist): in the process-tree jail, when `internet=False` + target package known, flag/deny `pip|uv install <target>` (exact name match, so `pydantic-core` is allowed when target is `pydantic`), `pip download <target>`, `curl|wget` of PyPI, `git clone <upstream>`; ALLOWLIST `pip install -e .`, pytest, compileall.
- Surface `fetch_attempted` into `ExecResult` (`/Users/sameertkhanna/Documents/agent_orch/apex_omega/types.py` add field + to_dict/from_dict) and thread to `Candidate.meta`.
- Add the **post-session dist-info provenance check** (compare venv dist-info against the prepared base snapshot) feeding the §5.4 refute.

### Step 10 — `/Users/sameertkhanna/Documents/agent_orch/apex_omega/autogen/sandbox.py::lint_source`: ban fetch-acquisition string literals only (negative-lookahead on `-e .`; do NOT match `restoration`/`urllib`). `/Users/sameertkhanna/Documents/agent_orch/apex_omega/autogen/__init__.py`: export `ANTI_FETCH_POLICY`.

## Phase R — variance + budget + run wiring
### Step 11 — `/Users/sameertkhanna/Documents/agent_orch/apex_omega/eval/commit0_driver.py`: `--seeds N` (3 for hard repos); separate `autogen_solved` vs `pooled_solved`; agents/solve + tokens/solve.
### Step 12 — **RESTORE hard token ceiling**: set `Budget(total=...)` for the cell (currently `total=None` -> `can_start()` always True, `budget.py:53`); `raise_cap` checks token headroom. `/Users/sameertkhanna/Documents/agent_orch/scripts/run_ladder.py`: keep template pinned at `--autogen-max-agents 8`; for autogen drop the pin so difficulty+raise_cap+token-ceiling govern; `--seeds 3` for jinja/mimesis/pydantic; size per-attempt budget vs ~1921s observed single-solve, decoupled from `CELL_TIMEOUT` (currently 2400 at `run_ladder.py:34` — likely too low for pydantic/mimesis single solves).

## Phase T — tests (write as each step lands; keep all green)
New `/Users/sameertkhanna/Documents/agent_orch/tests/test_ladder.py` + extensions, all FakeExecutor:
1. `test_score_fn_flat_layout_noop` (voluptuous-shaped: `_detect_src_pkg`->None, NO PYTHONPATH inject, green eval NOT flipped to indeterminate) — **floor-protection blocker**.
2. `test_score_fn_src_layout_asserts_worktree` (resolve-outside-worktree -> indeterminate, never false-accept).
3. `test_score_fn_concurrent_no_crosstalk` (2+ parallel score_fn on src-layout each scored against its own worktree — catches the race).
4. `test_memray_plugin_registered` + built command has `-p pytest_memray`.
5. `test_repair_forks_parent_diff` (apply_diff spied; prompt has failing nodeids; journal key differs; repair never sets accepted).
6. `test_repair_apply_diff_3way_fallback_then_indeterminate`.
7. `test_solve_and_repair_{accept,plateau,policy_violation_skip}`.
8. `test_select_refutes_fetch_provenance` + `test_select_refutes_dist_info_provenance` + `test_select_refutes_test_file_modification` (all monotone downgrade-only).
9. `test_anti_fetch_policy_always_concatenated` (custom prompt still gets suffix; suffix permits urllib + `-e .`).
10. `test_floor_lineage_suppresses_approach` (omits scout approach, retains v1 anti-cheat guardrails).
11. `test_lint_bans_fetch_literals` (`git clone`/`pip install mimesis` reject; `pip install -e .`/`minimal verified restoration` pass).
12. `test_scout_gated_off_for_non_hard` (static governs R0-R2; scout only at static==hard).
13. `test_jail_internet_off_target_deny_allows_self_verify` (deny `pip install <target>`, allow `pip install -e .`/`pydantic-core`).
14. `test_collection_error_climb_metric` (collection-blocked repo: repair allowed to climb on decreasing error count, not abandoned by pass_rate plateau).
15. `test_raise_cap_hysteresis` (never exceeds ceiling; token-headroom gated).
16. `test_reflog_and_branch_scrub` + `test_build_issue_description_deseeds_version_url`.
17. `test_ladder_floor_degrades_to_best_of_n` (author crashes -> floor still produces verified winner) — **floor blocker**.

CI markers: tests 1, 17 (and a real voluptuous re-run gate `agents_used<=1 AND solved=True`) are release blockers.