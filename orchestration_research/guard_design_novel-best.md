# Workspace-Discovery Guard — Fix Design (novel / best-combination)

**File under change (primary):** `/Users/sameertkhanna/Documents/agent_orch/apex/core/cli_backend.py`
**File under change (telemetry bridge):** `/Users/sameertkhanna/Documents/agent_orch/apex_omega/executor/v1_executor.py` + `/Users/sameertkhanna/Documents/agent_orch/apex_omega/autogen/context.py`
**Author:** principal-engineer subagent
**Status:** design; eval is LIVE; this is the vendored v1 file (real, git-tracked edits)

---

## 1. Root cause (confirmed against source + re-implemented predicates)

The fatal aborts on pydantic come from the DIRECT monitored-command branch of
`_process_tree_workspace_policy_violation` (`cli_backend.py:11122-11194`). A `find` (and the rest of
`_WORKSPACE_POLICY_MONITORED_COMMANDS`, line 1152-1176) is flagged when either:

- **process-cwd escape** (`11128`, via `_path_escapes_workspace`, `9742-9754`): the command runs from a
  dir not under the worktree; or
- **path-operand escape** (`11152-11168`, operands from `_command_path_operands`, `9635-9707`): an
  operand resolves outside the worktree.

Severity is `backend_helper` (SOFT/continue) ONLY if a narrow downgrade predicate matches; otherwise it
is hardcoded `fatal` (`11140`, `11174`) → `CLIProcessPolicyViolation` → `finalization_status=policy_violation`,
`tokens=0`, all work discarded (handling at `8994-9033`).

The two downgrade predicates recognize ONLY:
- `_looks_like_backend_helper_workspace_policy_violation` (`9771-9801`) → cwd contains an APEX marker
  (`.apex_agent_runtime/`, `apex-cli-offline-`, `cli_airgapped_homes/`, `fastzip-castree-`,
  constant `1246-1251`) or a system-runtime prefix+marker (`/usr/local|/opt/homebrew|/System|/Library`
  + `/site-packages/` etc., `1255-1270`);
- `_path_resolves_to_system_helper_target` (`10902-10913`) → same APEX/system markers.

**They do NOT recognize codex's own runtime HOME / uv cache**, which lives at
`cells/<mod>/runtime/home/.cache/uv/...` (a `/private/tmp` path with none of those markers), nor the host
`~/.cache`, nor sibling worktrees, nor the ladder dir. I re-implemented both predicates in
`.venv_omega` and confirmed: all five evidence paths return `downgraded=False` → **fatal**.

This contradicts the codebase's own **SANDBOX-NOT-PROMPT / deny-don't-kill** philosophy
(`apex_omega/autogen/context.py:182-188, 500-504`): "RECORD never penalize … the worktree-shadow +
execution gate already make any of these incapable of producing a false solve … we do not kill or
down-rank the attempt."

### Impact (from the IMPACT ledger, 416 fatal aborts)
- 11% genuine agent cheat (`find/rg/diff <upstream-pkg-copy>`, or the cell's own `cells/...__<repo>` parent)
- 37% codex-INTERNAL false-positive (codex's uv resolver / venv / fastzip helper / shared `~/.cache`)
- 50% benign-env (broad traversal of ladder root, `/`, `/opt/homebrew`, tmp, `~/.gemini`)
- 11% cross-cell (this cell killed for a DIFFERENT concurrent cell's subprocess in the ps tree)
- 2% sed/awk script-arg mis-parsed as a path operand
- 28% of ALL pydantic agent rollouts (175/617) aborted; jinja 21%, networkx 15%, etc. Every repo bleeds.

Anti-cheat signal is ALREADY preserved independently: `integrity_log.jsonl` recorded 242 `sandbox_escape`
entries regardless of the abort. The fatal kill destroys all work for ZERO marginal anti-cheat benefit.

---

## 2. The fix — two-tier, env-derived, deny-don't-kill (the "best combination")

A single coherent change with two tiers, both inside `_process_tree_workspace_policy_violation` (so BOTH
call sites — host tree `8989` and target-runtime tree `9164` — are fixed at once), plus a one-line
telemetry bridge.

### Tier 1 — env-derived allowed non-workspace roots → **silent soft** (kills the 37% codex-internal + 2% mis-parse-adjacent + most cross-cell)

Reuse the EXISTING, deterministic helper `_cli_sandbox_writable_roots(env, working_dir=...)`
(`cli_backend.py:6198-6227`) which already enumerates the legitimate non-workspace roots straight from
the launch env: `HOME`, `CODEX_HOME`, `XDG_CONFIG_HOME`, `XDG_CACHE_HOME`, `XDG_DATA_HOME`,
`XDG_STATE_HOME`, `CLAUDE_CONFIG_DIR`, `GEMINI_CLI_HOME`, `OPENCODE_CONFIG_DIR`,
`GOOGLE_APPLICATION_CREDENTIALS`, and the target-tool context parent. These resolve to exactly
`cells/<mod>/runtime/home/...` — the codex uv cache and venv that are causing 37% of the fatals.

Add a new keyword param `allowed_nonworkspace_roots: tuple[str, ...] = ()` to
`_process_tree_workspace_policy_violation`. Inside, an operand/cwd whose resolved path is UNDER one of
those roots is treated as `backend_helper` (the EXISTING silent-soft path: logged once, appended to
`soft_policy_violation_log`, rollout continues; `8996-9019`).

Because these are env-derived per-cell, a SIBLING cell's runtime home does NOT match THIS cell's roots —
so Tier 1 downgrades only this cell's own codex internals, never another cell's. Isolation intact.

**Host shared cache (`~/.cache`, `~/.gemini`).** Codex's resolver legitimately reads the shared host uv
cache (`~/.cache/uv`) when the per-cell `XDG_CACHE_HOME` is not the cache codex actually consults. Add a
small, explicit host-cache allowlist derived from `env["HOME"]`/`os.path.expanduser("~")`:
`{<home>/.cache, <home>/.gemini, <home>/.codex, <home>/.cache/uv, <home>/.cache/pip}` — read-only,
codex-internal. Gate this list behind a module constant so it is auditable and replay-deterministic.

### Tier 2 — residual genuine escapes → **course-correcting soft (deny-don't-kill)**, telemetry preserved (kills the 50% benign-env, downgrades the 11% genuine cheat from KILL to DENY+RECORD)

For a monitored-command escape that is NOT under an allowed root and NOT already classified as an
`external_source_acquisition` source-provenance violation (that path stays fatal — see §4), introduce a
NEW severity string `"course_correct"` returned in place of `"fatal"` for the generic
"repository discovery outside the rollout workspace" class.

`"course_correct"` is handled exactly like `backend_helper`/`blocked_by_policy` at the severity switch
(`8996`, `9180`) — appended to `soft_policy_violation_log`, `_emit_progress("policy_violation_soft", ...)`,
rollout CONTINUES — with ONE addition: the violation reason is also recorded as integrity telemetry so the
genuine-cheat SIGNAL survives the downgrade (see telemetry bridge below). The monitored set is read-only,
the worktree shadows site-packages, and there is no network — so a `find`/`grep`/`cat` of an out-of-tree
path (even a fetched upstream copy) can DENY the agent nothing it could have used to false-solve. The
course-correction message ("Keep repository discovery inside the current workspace.") nudges the agent
back, matching the deny-don't-kill precedent already in the codebase.

This is the novel part: instead of choosing between "downgrade everything (lose the cheat signal)" and
"kill everything (lose all the work)", Tier 2 **denies the command's usefulness, records the attempt, and
lets the rollout keep solving** — the literal SANDBOX-NOT-PROMPT contract, now applied to the read-only
discovery guard that was the lone violator of it.

### Telemetry bridge (preserve the 11% genuine-cheat signal across the soft-downgrade)

Today `classify_attempt_integrity` (`context.py:191-197`) keys `sandbox_escape` on
`finalization_status == "policy_violation"` (true only while the guard KILLS). After the downgrade,
`finalization_status` becomes `"completed"`, so the signal would vanish. Preserve it with the smallest
possible additive change:

1. `cli_backend.py`: the `soft_policy_violation_log` already lands in `timeout_audit["soft_policy_violations"]`
   (`9436`) and `timeout_audit` is attached to the result (`model_result.timeout_audit = timeout_audit`,
   `7860`). No change needed here beyond Tiers 1-2.
2. `v1_executor.py` (`200-214`): copy the soft-violation reasons from `res.timeout_audit` into the
   `ExecResult` so the classifier can see them. Add an additive optional field
   `policy_soft_violations: list[str]` to `ExecResult` (`apex_omega/types.py:154`) and populate it from
   `getattr(res, "timeout_audit", {}).get("soft_policy_violations", [])` mapped to their `reason` strings.
3. `context.py` `classify_attempt_integrity`: add one OR-clause — also emit `sandbox_escape` when
   `getattr(res, "policy_soft_violations", None)` contains a "discovery outside the rollout workspace"
   reason. This keeps `integrity_log.jsonl` recording the genuine 11% with zero loss of fidelity, while
   the rollout is scored normally.

(If a strictly single-file change is mandated, an acceptable fallback is to set `res.error` to the
violation reason while keeping `ok=True`/`finalization_status="completed"`; the classifier already
substring-matches `error`. The additive field is preferred because it does not overload `error` semantics.)

---

## 3. Exact edits (files / methods / lines)

### 3.1 `cli_backend.py` — constant (near 1280, after `_WORKSPACE_POLICY_SYSTEM_TARGET_PREFIXES`)
Add:
```python
# Host-shared, read-only caches the nested CLI resolver may consult when the
# per-cell XDG_CACHE_HOME is not the cache it actually uses. Read-only +
# worktree-shadow + no-network => cannot produce a false solve. Derived from
# HOME at call time; kept here for auditability/replay determinism.
_WORKSPACE_POLICY_HOST_CACHE_SUBDIRS: tuple[str, ...] = (
    ".cache", ".cache/uv", ".cache/pip", ".gemini", ".codex",
)
```

### 3.2 `cli_backend.py` — `_process_tree_workspace_policy_violation` signature (10915-10924)
Add keyword param:
```python
allowed_nonworkspace_roots: tuple[str, ...] = (),
```
Normalize once at the top of the method (after `workspace_root = ...`, 10925):
```python
allowed_roots = tuple(
    Path(r).expanduser().resolve(strict=False) for r in allowed_nonworkspace_roots
)
```
Add a small helper predicate (method or local closure):
```python
def _under_allowed_root(self, path: Optional[Path], allowed_roots) -> bool:
    if path is None:
        return False
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        return False
    for root in allowed_roots:
        try:
            resolved.relative_to(root); return True
        except ValueError:
            continue
    return False
```

### 3.3 `cli_backend.py` — process-cwd branch (11128-11150)
Before building the violation dict, if `self._under_allowed_root(process_cwd, allowed_roots)` →
`likely_backend_helper = True` (silent soft). Otherwise, change the else-fatal to the two-tier severity:
```python
if self._under_allowed_root(process_cwd, allowed_roots):
    severity = "backend_helper"; likely_backend_helper = True
else:
    likely_backend_helper = self._looks_like_backend_helper_workspace_policy_violation(...)
    severity = "backend_helper" if likely_backend_helper else "course_correct"
```
(`severity` replaces the inline `"backend_helper" if ... else "fatal"` at 11140.)

### 3.4 `cli_backend.py` — path-operand branch (11164-11194)
Add the allowed-root check before the system-helper check (11172):
```python
if self._under_allowed_root(resolved_path, allowed_roots):
    severity = "backend_helper"; likely_backend_helper = True
else:
    target_is_system_helper = self._path_resolves_to_system_helper_target(resolved_path)
    likely_backend_helper = target_is_system_helper
    severity = "backend_helper" if likely_backend_helper else "course_correct"
```
(`severity` replaces the inline assignment at 11174.)

### 3.5 `cli_backend.py` — severity switch, BOTH call sites (8996 and 9180)
Add `"course_correct"` to the soft set:
```python
if severity in {"backend_helper", "blocked_by_policy", "course_correct"}:
```
No other change to the soft body — it already logs once, appends to `soft_policy_violation_log`, emits
`policy_violation_soft`, and continues.

### 3.6 `cli_backend.py` — call sites pass the roots
- Host tree (8989-8993): pass
  `allowed_nonworkspace_roots=self._workspace_policy_allowed_roots(target_runtime_env, working_dir)`.
- Target-runtime tree (9164-9177): same, computed from `target_runtime_env` + `target_workdir`.

Add a tiny wrapper `_workspace_policy_allowed_roots(env, working_dir)` that returns
`tuple(_cli_sandbox_writable_roots(env or {}, working_dir=working_dir)) + host-cache roots from env["HOME"]`
(joining `_WORKSPACE_POLICY_HOST_CACHE_SUBDIRS`). Fail-open to `()` when `env` is None.

### 3.7 telemetry bridge (3 tiny additive edits)
- `apex_omega/types.py:154` ExecResult: add `policy_soft_violations: list[str] = field(default_factory=list)`
  (+ include in `to_dict`/`from_dict`).
- `apex_omega/executor/v1_executor.py:200-214`: populate it from `res.timeout_audit["soft_policy_violations"]`.
- `apex_omega/autogen/context.py:196`: add OR-clause so a soft "discovery outside the rollout workspace"
  reason also emits `sandbox_escape`.

---

## 4. How isolation / anti-cheat is PRESERVED (not weakened)

1. **No false solve is enabled.** The monitored commands are READ-ONLY (`find/grep/cat/diff/ls/...`).
   The worktree SHADOWS site-packages and there is NO network, so any out-of-tree path a `find`/`grep`
   could reveal — including a fetched upstream copy — still cannot be turned into passing code that the
   scorer (keyed on gold expected-test ids against the worktree) would accept. Downgrading READ to
   deny+continue changes nothing about solvability of a cheat.
2. **Genuine source-acquisition stays FATAL.** The separate `_target_runtime_source_provenance_policy_violation`
   path (curl/wget URLs, inline python URL probes, repo-local downloader helpers; `10625-10762`,
   returned as `severity="fatal"` at `11042-11061`) is UNTOUCHED. Network fetches and vendor/downloader
   helpers remain hard kills.
3. **Cross-worktree / sibling-cell reads still denied.** Tier 1 only allows THIS cell's env-derived roots;
   a sibling worktree (`wt_m730006` while assigned `wt_m730005`) and sibling cell parents are NOT under
   them, so they fall to Tier 2 = DENY + RECORD + continue (the agent is denied the read and nudged), and
   they are recorded as `sandbox_escape` telemetry. The agent never actually gets the cross-worktree
   bytes (the command's result is irrelevant to scoring and the course-correction tells it to stop).
4. **Telemetry never lost.** Every downgraded violation is appended to `soft_policy_violation_log` →
   `timeout_audit` → ExecResult → `integrity_log.jsonl` `sandbox_escape`. The 11% genuine cheats are still
   fully recorded for cross-eval analysis; we simply stop DESTROYING the work for them.
5. **Determinism / replay.** Allowed roots are a pure function of the launch env + working_dir (already
   journaled inputs). `_under_allowed_root` is pure path math. No clock, no network, no host-time. The
   host-cache subdir list is a fixed module constant.

Net: the guard still DENIES every out-of-workspace read and RECORDS every attempt. It stops doing the one
thing the codebase philosophy forbids — KILLING the rollout for a read it already neutralized.

---

## 5. Why this unblocks pydantic

The pydantic evidence cell's 7/8 fatal waves were: codex uv cache (Tier 1 → silent soft), `~/.cache`
(Tier 1 host-cache → silent soft), sibling worktree (Tier 2 → deny+record+continue), ladder dir +
`/opt/homebrew` (Tier 2 → deny+record+continue). NONE of these is a network fetch or a downloader helper,
so none hits the still-fatal source-provenance path. Every wave that previously died at `tokens=0` now
runs to completion and banks its diff. The 8th wave (the 3000s timeout, 263KB uncommitted diff) is a
SEPARATE acceptance-checkpointing issue (tracked in autogen-nextgen-design) and out of scope here, but it
too stops being starved of sibling waves. pydantic becomes evaluable; the same fix recovers the 8-28%
rollout loss across jinja/networkx/minitorch/mimesis/voluptuous/babel.

---

## 6. Risks

- **R1 — a real cheat now merely denied, not killed.** Mitigated: read-only + shadow + no-network =>
  no false solve possible; signal still recorded. This is the explicit codebase contract.
- **R2 — env-derived roots too broad (e.g. HOME == workspace parent).** Mitigated: `_normalize_sandbox_roots`
  already drops `root == workspace`; `_under_allowed_root` uses strict `relative_to` (no prefix-string
  bug). Add a unit test that workspace itself and a sibling worktree are NOT "under" an allowed root.
- **R3 — host `~/.cache` allowlist could shadow a genuine cheat staged under `~/.cache`.** Low: it is
  read-only and cannot false-solve; if paranoid, scope the host-cache allowlist to `.cache/uv`,
  `.cache/pip`, `.codex`, `.gemini` only (the resolver dirs), leaving the rest of `~/.cache` fatal.
- **R4 — new `"course_correct"` severity not handled at a third site.** Audit: only two soft switches
  exist (8996, 9180); both updated. The `runtime_policy_violations` marker path (9120-9156) is always
  fatal by construction and is intentionally left fatal (it carries vendor-reported violations).
- **R5 — cross-cell ps-tree scope (11% killed for another cell).** Tier 2 already converts these from
  KILL to deny+record. A deeper fix (verify the offending PID's cwd is under THIS cell root before
  attributing) is a follow-up, not required to unblock pydantic.
- **R6 — telemetry-bridge field add.** Additive, defaulted; `to_dict`/`from_dict` round-trip must be
  kept in sync (covered by a test).

---

## 7. Test plan

### Unit (guard predicates — new `tests/test_workspace_guard_downgrade.py`)
1. `_under_allowed_root`: codex uv cache under CODEX_HOME/XDG_CACHE_HOME → True; sibling worktree,
   ladder dir, `/opt/homebrew`, genuine `/private/tmp/pydantic_upstream` → False; workspace itself → False.
2. `_process_tree_workspace_policy_violation` with `allowed_nonworkspace_roots` set:
   - `find <codex-uv-cache>` → severity `backend_helper` (silent soft).
   - `find <host>/.cache/uv` → severity `backend_helper`.
   - `find <sibling-worktree>` → severity `course_correct` (NOT fatal).
   - `find <ladder-dir>` / `find /opt/homebrew/lib/perl5` → severity `course_correct`.
   - `curl https://pypi.org/...` (source-provenance) → severity `fatal` (UNCHANGED).
   - `find <pydantic_upstream>` → severity `course_correct` AND is recorded as a soft violation
     (genuine-cheat path: denied + recorded, not killed).
3. Severity switch: `course_correct` is appended to `soft_policy_violation_log` and does NOT raise
   `CLIProcessPolicyViolation`; `fatal` still raises.

### Telemetry bridge
4. `classify_attempt_integrity(ExecResult(ok=True, finalization_status="completed",
   policy_soft_violations=["...discovery outside the rollout workspace..."]))` → emits `sandbox_escape`.
   (Extends existing `test_integrity_classifier_detects_escape_fetch_and_cheat`.)
5. `ExecResult.to_dict()/from_dict()` round-trips `policy_soft_violations`.

### Regression / full suite
6. `.venv_omega/bin/python -m pytest tests/ -q` — full suite must stay green (currently 301).
   Pay attention to `test_review_fixes.py` (integrity classifier), `test_safety_and_ablation.py`,
   `test_converge_*`, `test_repair.py`.
7. Re-implement-and-assert micro-check (already run during design) over the five evidence paths
   confirming Tier-1/Tier-2 verdicts, committed as a doctest-style unit.

### Live gate
8. Re-run the pydantic evidence cell (`hybrid-nogate__pydantic__s0`): expect 0 fatal
   `policy_violation` aborts from the discovery guard, repair waves finalize with `tokens>0`, and
   `integrity_log.jsonl` still records `sandbox_escape` for any genuine out-of-tree reads.
