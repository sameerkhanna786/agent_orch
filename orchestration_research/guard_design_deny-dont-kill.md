# Workspace-Discovery Guard: deny-don't-kill design

**File under change:** `apex/core/cli_backend.py` (vendored APEX v1, git-tracked)
**Author:** principal-engineer design pass
**Status:** design (ready to implement)

---

## 1. Problem (one paragraph)

When a codex coding-agent (or one of its repair waves) runs a monitored discovery command
(`find`/`grep`/`rg`/`ls`/`diff`/`cp`/…) whose CWD or a path operand resolves **outside** the agent's
worktree, the workspace guard in `_process_tree_workspace_policy_violation`
(`apex/core/cli_backend.py:10915-11195`) returns a violation dict with `severity="fatal"`. The
severity handler at `apex/core/cli_backend.py:8994-9033` raises `CLIProcessPolicyViolation`, which
sets `finalization_status=policy_violation` and `tokens=0` — the **entire rollout is discarded with
zero useful work**. On pydantic (95 files / 5091 tests) every repair wave triggers this (7/8 in the
evidence cell `hybrid-nogate__pydantic__s0`) because pydantic's solve naturally walks broad paths and
because codex's own runtime HOME / uv cache live **outside** `workspace_root` (at
`cells/<mod>/runtime/home/.cache/…`, a sibling of `repo/`). pydantic is therefore un-evaluable, and
every other repo bleeds 5-28% of rollouts to the same kill.

This **fatal-abort contradicts the codebase's own SANDBOX-NOT-PROMPT philosophy**
(`apex_omega/autogen/context.py:181-197, 500-514`): *"we do not kill or down-rank the attempt; the
sandbox simply denies the action and the agent is left to solve the real task… the worktree-shadow +
execution gate already make any of these incapable of producing a false solve."* A SOFT
deny-don't-kill path **already exists** in the same handler (severity `backend_helper` /
`blocked_by_policy`, lines 8996-9019) — it logs the violation, emits `policy_violation_soft`
telemetry, and **continues** the rollout. The fix routes workspace-discovery violations through that
existing soft path instead of the fatal one.

---

## 2. Root cause (exact code)

The fatal severity is assigned at **two** decision points inside the direct monitored-command loop of
`_process_tree_workspace_policy_violation`:

| # | Branch | Lines | Decision |
|---|--------|-------|----------|
| A | **process-CWD escape** (`find` running *from* an outside dir) | `11128-11150` | `severity = "backend_helper" if likely_backend_helper else "fatal"` (11139-11140) where `likely_backend_helper = _looks_like_backend_helper_workspace_policy_violation(...)` |
| B | **path-operand escape** (`find <PATH>` targeting an outside dir) | `11152-11194` | `severity = "backend_helper" if likely_backend_helper else "fatal"` (11173-11174) where `likely_backend_helper = _path_resolves_to_system_helper_target(resolved_path)` |

The downgrade-to-`backend_helper` predicates only recognize **APEX-managed helper roots**
(`.apex_agent_runtime/`, `apex-cli-offline-`, `cli_airgapped_homes/`, `fastzip-castree-`, constants
at `1246-1280`) and **system site-packages** under `/usr/local|/opt/homebrew|/System|/Library` +
`/site-packages/` markers (`_path_text_looks_like_backend_runtime_helper`, `9803-9808`). They do
**not** recognize: the user's `~/.cache`; codex's own `CODEX_HOME`/uv cache at
`.../runtime/home/.cache/uv/` (a `/private/tmp` path with none of those markers); sibling worktrees;
or the ladder dir. So **all five evidence paths fall through to `"fatal"`** — confirmed empirically by
re-running the guard against the evidence inputs via `.venv_omega/bin/python`:

```
user_cache         -> severity=fatal     # /Users/sameertkhanna/.cache
codex_uv_cache     -> severity=fatal     # .../runtime/home/.cache/uv/archive-v0
sibling_worktree   -> severity=fatal     # .../worktrees/wt_m730006 (assigned wt_m730005)
ladder_dir         -> severity=fatal     # /private/tmp/omega_phase_ab_n3max
homebrew_perl      -> fatal under enforce # /opt/homebrew/lib/perl5
GENUINE_upstream   -> severity=fatal     # /private/tmp/pydantic_upstream  (a real cheat)
```

The impact analysis quantifies this: of 416 fatal aborts, **only ~11% (45/416) are genuine
agent-cheat** (discovery against a *fetched/extracted upstream package copy*). 89% are codex-internal
false-positives, benign-env traversal, sed/awk arg mis-parses, or **cross-cell** false-positives (one
cell killed for another concurrent cell's subprocess). And the anti-cheat signal is **already
preserved independently** of the kill: `integrity_log.jsonl` recorded 242 `sandbox_escape` telemetry
entries (LIVE) regardless of abort — so the fatal kill destroys all work for **zero marginal
anti-cheat benefit**.

---

## 3. Design: route workspace-discovery violations through the SOFT path

### 3.1 Principle

A monitored discovery command (`find`, `grep`, `ls`, …) is **read-only** and the sandbox **already
denies the actual access** — a path outside `workspace_root` is unreadable (jail), site-packages is
shadowed by the worktree, and there is no network. So the command **cannot mutate state, cannot read
contraband, and cannot produce a false solve**. The correct response is exactly the codebase's
existing soft contract: **deny the access, record telemetry, tell the agent the path is outside its
workspace, and continue.** We therefore downgrade the workspace-discovery (CWD-escape and
path-operand-escape) cases from `"fatal"` to a soft severity, **with one narrow fatal carve-out**: a
discovery command whose target resolves to a **known fetched/extracted upstream-package copy** stays
fatal, because that is the genuine "cheat by reading the reference implementation" signal and it is
cheap and high-precision to detect by path marker.

### 3.2 New soft severity value

Add a new severity string `"workspace_discovery"` and make the SOFT branch at line 8996 recognize it,
alongside the existing `backend_helper` / `blocked_by_policy`. This keeps the new behavior **clearly
attributable** in the audit (distinct from the pre-existing `backend_helper` downgrades) and is a
strictly additive change to the severity-handling set.

### 3.3 New predicate: genuine-cheat carve-out (stays fatal)

Add a small predicate that returns `True` only when a resolved path looks like a fetched/extracted
**upstream reference copy** of the package under test — the genuine cheat. Use the exact markers the
impact analysis identified in the cheat evidence:

```python
# New module constant near _WORKSPACE_POLICY_SYSTEM_TARGET_PREFIXES (~line 1280).
# Path-substring markers that identify a FETCHED/EXTRACTED upstream reference copy
# of the package-under-test (the genuine "read the reference implementation" cheat).
# A monitored discovery command targeting one of these stays FATAL — everything
# else outside the workspace is a soft deny-and-continue.
_WORKSPACE_POLICY_UPSTREAM_REFERENCE_MARKERS: tuple[str, ...] = (
    "_upstream/",
    "_upstream.",
    "_wheel/",
    "_restore/",
    "_spec.txt",
)
```

```python
# New staticmethod near _path_resolves_to_system_helper_target (~line 10913).
@staticmethod
def _path_resolves_to_upstream_reference_copy(path: Optional[Path]) -> bool:
    """True if `path` looks like a fetched/extracted upstream reference copy of the
    package-under-test (e.g. /tmp/pydantic_upstream, /tmp/mimesis_wheel,
    .../site-packages/<pkg> reached via an extracted copy). Reading these is the
    genuine 'crib from the reference implementation' cheat and stays FATAL."""
    if path is None:
        return False
    text = str(path)
    if any(marker in text for marker in _WORKSPACE_POLICY_UPSTREAM_REFERENCE_MARKERS):
        return True
    # Discovery aimed straight at an installed copy of the package via host
    # site-packages (not the worktree-shadowed one) is also a reference-crib.
    return "/site-packages/" in text and not CLIModelClient._path_text_looks_like_backend_runtime_helper(text)
```

> Note: the `/site-packages/` clause is deliberately conservative. The worktree shadows
> site-packages, so a *successful* read there is already harmless; this clause only keeps the **fatal
> cheat-signal** for the explicit case where the agent points a discovery command directly at a
> site-packages copy of the target package. If desired this clause can be omitted to be even more
> permissive (everything outside workspace → soft); it is included to keep the genuine-cheat fatal
> signal strong per constraint (3). It does **not** affect codex's own runtime venv, which is matched
> by `_path_text_looks_like_backend_runtime_helper` and excluded.

### 3.4 Edits at the two decision points

**Branch A — process-CWD escape (`apex/core/cli_backend.py:11128-11150`).** Replace the
binary `backend_helper`/`fatal` with a three-way decision. The CWD here is *where the command runs
from*, not what it reads, so there is no operand to check against the upstream-reference markers; a
CWD escape is always soft (deny-and-continue) unless it is already a backend-helper:

```python
            if self._path_escapes_workspace(process_cwd, working_dir=workspace_root):
                likely_backend_helper = self._looks_like_backend_helper_workspace_policy_violation(
                    process_cwd=process_cwd,
                    path_tokens=path_tokens,
                    working_dir=workspace_root,
                )
                severity = "backend_helper" if likely_backend_helper else "workspace_discovery"
                return {
                    "pid": pid,
                    "command_name": command_name,
                    "command": command,
                    "cwd": str(process_cwd) if process_cwd is not None else None,
                    "likely_backend_helper": likely_backend_helper,
                    "severity": severity,
                    "reason": (
                        (
                            "CLI backend helper executed repository discovery outside the rollout workspace: "
                            if likely_backend_helper
                            else "CLI subprocess ran a discovery command from outside the rollout workspace (denied; continuing): "
                        )
                        + f"`{command_name}` is running from `{process_cwd}` instead of under "
                        f"`{workspace_root}`. Keep repository discovery inside the current workspace."
                    ),
                }
```

**Branch B — path-operand escape (`apex/core/cli_backend.py:11152-11194`).** Keep the
genuine-cheat carve-out fatal; downgrade everything else to soft:

```python
                target_is_system_helper = self._path_resolves_to_system_helper_target(resolved_path)
                target_is_upstream_reference = self._path_resolves_to_upstream_reference_copy(
                    resolved_path
                )
                likely_backend_helper = target_is_system_helper
                if target_is_upstream_reference:
                    severity = "fatal"
                elif likely_backend_helper:
                    severity = "backend_helper"
                else:
                    severity = "workspace_discovery"
                if severity == "fatal":
                    reason_prefix = "CLI subprocess attempted to read an upstream reference copy of the package under test: "
                elif likely_backend_helper:
                    reason_prefix = "CLI backend helper executed repository discovery outside the rollout workspace: "
                else:
                    reason_prefix = "CLI subprocess ran a discovery command outside the rollout workspace (denied; continuing): "
                return {
                    "pid": pid,
                    "command_name": command_name,
                    "command": command,
                    "cwd": str(process_cwd) if process_cwd is not None else None,
                    "path_token": token,
                    "resolved_path": str(resolved_path) if resolved_path is not None else None,
                    "likely_backend_helper": likely_backend_helper,
                    "severity": severity,
                    "reason": (
                        reason_prefix
                        + f"`{command_name}` targeted `{token}` which resolves to `{resolved_path}`. "
                        "Keep repository discovery inside the current workspace."
                    ),
                }
```

### 3.5 Severity handler — recognize the new soft severity (`apex/core/cli_backend.py:8996`)

```python
                    if severity in {"backend_helper", "blocked_by_policy", "workspace_discovery"}:
```

The body of that branch is unchanged: it dedups by `(pid, command_name, path_token|cwd)`, appends to
`soft_policy_violation_log`, logs a warning, and emits `policy_violation_soft`. The
`soft_policy_violation_log` is already surfaced into the completion audit at lines 9023, 9144-9145,
9211-9212, and 9435-9436, so the workspace-discovery denials become first-class audit telemetry.

### 3.6 What is NOT changed (kept fatal)

- **Source-provenance / external-source-acquisition** (`11033-11061`,
  `_target_runtime_source_provenance_policy_violation`): curl/wget URLs, inline-URL probes, and
  repo-local downloader/vendor helpers — untouched, stays `severity="fatal"`. This is the real
  "fetch the upstream package" cheat and is independent of the discovery guard.
- **git-history discovery** (`11014-11032`, `11090-11121`), **host dynamic execution bypass**
  (`11001-11013`), and the **target-runtime shell payload boundary** (`11062-11089`) — all untouched.
- **Upstream-reference-copy discovery** (new carve-out, §3.3) — explicitly kept fatal.

---

## 4. How isolation / anti-cheat is preserved (constraints 1-3)

**(1) Isolation / anti-cheat is NOT weakened.** The guard never *granted* the access — the sandbox
jail does. The agent's worktree is `workspace_root`; reads outside it are denied at the OS/sandbox
layer regardless of this guard's severity. Three independent mechanisms make a discovery command
incapable of a false solve, none of which this change touches:
  - the worktree **shadows site-packages**, so even a successfully-read fetched package cannot become
    the imported implementation at test time;
  - there is **no network**, so nothing new can be fetched;
  - the **execution/acceptance gate** is keyed on gold expected-test ids, so a weakened or
    cross-contaminated read cannot manufacture a green.
  The genuine read-the-reference cheat (`*_upstream/`, `*_wheel/`, `/site-packages/<pkg>`) is **kept
  fatal** by the new carve-out (§3.3), and the *acquisition* cheat (curl/wget/downloader) stays fatal
  in the untouched source-provenance path (§3.6). So the high-precision cheat signals remain
  enforced; only the low-precision "ran a read-only command that happened to point outside the jail"
  case is downgraded — and that case the sandbox already denied.

**(2) Determinism / replay preserved.** The change is a pure severity reclassification computed from
the same inputs (`process_entries`, `working_dir`) with no new I/O, randomness, time, or env reads.
The new predicate is a deterministic substring test over the already-resolved path. Replay of the same
process-tree snapshot yields the same severity.

**(3) Cheat telemetry preserved.** Nothing is silenced. Soft violations are appended to
`soft_policy_violation_log` and surfaced in every completion/timeout audit; `policy_violation_soft`
progress events still fire. Upstream `apex_omega` integrity classification
(`classify_attempt_integrity`, `context.py:181-197`) records `sandbox_escape` telemetry from the
denied action / "outside the root" signal exactly as before — and now it does so **without also
zeroing the rollout**, which is the explicit intent of that module's docstring.

---

## 5. Does it unblock pydantic? (yes — with reasoning)

Yes. The pydantic aborts are precisely the CWD-escape and path-operand-escape cases this change
downgrades:
  - `find /Users/sameertkhanna/.cache …` → operand escape, not an upstream-reference marker → **soft**.
  - `find …/runtime/home/.cache/uv/archive-v0 …` (codex's own uv cache) → operand/CWD escape, no
    upstream marker → **soft** (and codex's runtime venv is additionally excluded from the fatal
    site-packages clause via `_path_text_looks_like_backend_runtime_helper`).
  - sibling worktree `wt_m730006`, ladder dir `/private/tmp/omega_phase_ab_n3max`,
    `/opt/homebrew/lib/perl5` → escapes with no upstream marker → **soft**.

After the change, each of these returns `severity="workspace_discovery"`, the handler logs it and
**continues**, so the rollout proceeds to do real repair work instead of finalizing with
`tokens=0`. The 7/8 pydantic repair waves that were aborted will run to completion (subject to the
normal progress/hard-timeout budget). The one hard-timeout wave (3000s, 263KB diff) is a *separate*
budget issue, not a guard issue, and is out of scope here. Cross-repo, the impact analysis shows
5-28% of rollouts across all repos were being killed by this guard; routing the ~89% non-cheat
fraction to soft recovers that work.

---

## 6. Risks

1. **A genuinely-cheating discovery slips through if its target lacks an upstream marker.** Mitigation:
   harmless by construction — the worktree-shadow + no-network + gold-id acceptance gate prevent any
   read from producing a false solve (this is the whole codebase philosophy). The high-precision cheat
   paths (upstream-reference markers + source-provenance acquisition) stay fatal, and the
   `sandbox_escape` telemetry still records the attempt for offline analysis. *Residual: low.*
2. **Marker list drift** — a future fetched-copy naming convention not in
   `_WORKSPACE_POLICY_UPSTREAM_REFERENCE_MARKERS` would be soft, not fatal. Mitigation: the markers
   are derived from the observed cheat evidence; the list is one centralized constant, easy to extend;
   and the soft path still records telemetry so drift is observable. *Residual: low.*
3. **Soft-log volume** on a broad-walking solve (pydantic). Mitigation: the handler already dedups by
   `(pid, command_name, path_token|cwd)` (8002-8011) and logs once per unique tuple. *Residual:
   negligible.*
4. **`/site-packages/` fatal clause over-triggers** on a legitimate read of the *shadowed* worktree
   site-packages. Mitigation: that path is *inside* `workspace_root`, so `_path_escapes_workspace`
   returns False and the operand loop `continue`s before severity is ever computed (11164-11168) —
   the clause is only reached for an *escaping* site-packages target. The codex runtime venv is
   additionally excluded via `_path_text_looks_like_backend_runtime_helper`. If any residual concern,
   drop the `/site-packages/` clause (the markers alone still catch the observed cheats). *Residual:
   low.*
5. **Cross-cell false-positive still soft-logs the wrong cell.** This change converts those from a
   fatal cross-cell kill (44/416) into a harmless soft log — a strict improvement — but does not *fix*
   the underlying process-tree-scope bug. Out of scope; tracked separately. *Residual: none (net
   positive).*

---

## 7. Test plan

### 7.1 Unit test on the guard predicate (new file `tests/test_workspace_discovery_guard.py`)

Instantiate the client without running `__init__` (the guard methods only use `self` to call sibling
helpers): `client = CLIModelClient.__new__(CLIModelClient)`. Build minimal `process_entries`
(`{pid: {"command": ..., "argv": [...], "cwd": ...}}`) and assert on the returned `severity`.

Cases (all with `working_dir = .../worktrees/wt_m730005`):

| Input | Expected severity |
|---|---|
| `find /Users/u/.cache -name x` | `workspace_discovery` (soft) |
| `find .../runtime/home/.cache/uv/archive-v0 -name x` | `workspace_discovery` (soft) |
| `find .../worktrees/wt_m730006 -name x` (sibling) | `workspace_discovery` (soft) |
| `find /private/tmp/omega_phase_ab_n3max -name x` (ladder dir) | `workspace_discovery` (soft) |
| `grep -r x /opt/homebrew/lib/perl5` | `workspace_discovery` or `backend_helper` (soft) |
| `find /private/tmp/pydantic_upstream -name x` | **`fatal`** (genuine cheat, carve-out) |
| `find /tmp/mimesis_wheel -name x` | **`fatal`** (genuine cheat) |
| `find . -name x` (CWD inside worktree) | `None` (no violation) |
| `find <worktree>/sub -name x` (operand inside worktree) | `None` |
| APEX helper root (`.apex_agent_runtime/`) | `backend_helper` (unchanged) |

Plus a handler-level assertion: feed a `severity="workspace_discovery"` dict to the severity-set
membership check used at 8996 and assert it is treated as soft (membership test on the new set).

### 7.2 Regression: severity-handler set

Assert `"workspace_discovery"` is in the soft set and that the fatal-raise branch is **not** taken for
it; assert `"fatal"` still raises.

### 7.3 Full suite

```
cd /Users/sameertkhanna/Documents/agent_orch
PYTHONPATH=. .venv_omega/bin/python -m pytest tests/ -q
```
The suite is `-n auto` parallel (pytest.ini). Expect the existing ~301 green to stay green plus the
new unit tests. No existing test imports the guard predicate, so there is no existing-behavior
collision; the only behavioral change is severity for the previously-fatal non-cheat discovery cases.

### 7.4 Live A/B gate (the real signal)

Re-run the evidence cell `hybrid-nogate__pydantic__s0` (or a fresh n=1 pydantic cell) and confirm:
  - repair-wave `finalization_status` is no longer `policy_violation` with `tokens=0`;
  - `soft_policy_violations` appear in the completion audit;
  - `integrity_log.jsonl` still records `sandbox_escape` entries (telemetry intact);
  - per-repo fatal-abort fraction drops toward the ~11% genuine-cheat floor.

---

## 8. Summary of exact edits

| Location | Change |
|---|---|
| `apex/core/cli_backend.py` ~1280 (after `_WORKSPACE_POLICY_SYSTEM_TARGET_PREFIXES`) | **add** `_WORKSPACE_POLICY_UPSTREAM_REFERENCE_MARKERS` constant |
| `apex/core/cli_backend.py` ~10913 (after `_path_resolves_to_system_helper_target`) | **add** staticmethod `_path_resolves_to_upstream_reference_copy` |
| `apex/core/cli_backend.py:11139-11148` (Branch A, process-CWD escape) | **edit** severity → `"workspace_discovery"` when not backend_helper; soften reason text |
| `apex/core/cli_backend.py:11172-11192` (Branch B, path-operand escape) | **edit** three-way severity: fatal iff upstream-reference, else backend_helper, else `"workspace_discovery"`; soften reason text |
| `apex/core/cli_backend.py:8996` | **edit** soft-severity set to include `"workspace_discovery"` |
| `tests/test_workspace_discovery_guard.py` | **new** unit tests (§7.1-7.2) |
