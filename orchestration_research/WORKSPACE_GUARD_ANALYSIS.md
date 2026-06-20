# Workspace-Discovery Guard — Root Cause, Decision, and Fix

**Deciding architect report.** Scope: the strategy-side workspace-discovery guard in
`apex/core/cli_backend.py` that fatally aborts codex rollouts (finalization_status=
`policy_violation`, tokens=0) when a monitored discovery command (`find`/`grep`/…) touches a path
outside the rollout worktree. On pydantic this aborts essentially every repair wave, making the repo
un-evaluable.

All claims below are grounded in the live file and validated empirically in `.venv_omega` against the
literal evidence paths.

---

## 1. WHY this happened (root cause)

The fatal aborts come from the **direct monitored-command loop** in
`_process_tree_workspace_policy_violation` (`cli_backend.py:11122-11194`), NOT the bash-payload or
git-history paths. For each process in the agent's tree:

1. `find` is in `_WORKSPACE_POLICY_MONITORED_COMMANDS` (line 1159), so the process is NOT skipped at
   `11122`.
2. Two independent triggers fire `fatal`:
   - **Process-CWD escape** (`11128`): `_path_escapes_workspace(process_cwd, working_dir=workspace_root)`
     — `find` is running *from* a dir outside the worktree.
   - **Path-operand escape** (`11152-11168`): an operand from `_command_path_operands`
     (`9635`) resolves outside the worktree via `_path_escapes_workspace` (`9742-9754`,
     `relative_to` → `ValueError` → escapes=True).
3. Severity is `fatal` **unless** a narrow downgrade predicate matches:
   - CWD branch → `_looks_like_backend_helper_workspace_policy_violation` (`9771`).
   - Operand branch → `_path_resolves_to_system_helper_target` (`10902`).

Both downgrade predicates recognize **only** APEX-managed helper roots
(`_WORKSPACE_POLICY_SYSTEM_TARGET_PREFIXES` = `.apex_agent_runtime/`, `apex-cli-offline-`,
`cli_airgapped_homes/`, `fastzip-castree-`; lines 1275-1280) and **system** `site-packages` reads
(`_path_text_looks_like_backend_runtime_helper` requires a `/usr/local|/opt/homebrew|/System|/Library`
prefix AND a `/site-packages/`-class marker; lines 1255-1270, 9803-9808).

They do **not** recognize the codex runtime's own layout, which is the crux: under the cell the agent's
`CODEX_HOME` + `uv` cache live at `cells/<mod>/runtime/home/.cache/uv/...` — a `/private/tmp` path with
none of those markers. Nor do they recognize the host `~/.cache`, sibling worktrees, the ladder dir,
or read-only OS dirs like `/opt/homebrew/lib/perl5`.

### Empirical confirmation (`.venv_omega`)

Reproducing `_path_resolves_to_system_helper_target` against the five evidence paths returns `False`
for ALL of them except a genuine system `site-packages` path:

```
False  /Users/sameertkhanna/.cache                                  -> FATAL
False  .../runtime/home/.cache/uv/archive-v0/...  (codex uv cache)  -> FATAL
False  .../worktrees/wt_m730006/repo              (sibling worktree)-> FATAL
False  /private/tmp/omega_phase_ab_n3max          (ladder dir)      -> FATAL
False  /opt/homebrew/lib/perl5                                       -> FATAL
True   /opt/homebrew/lib/python3.14/site-packages/faker             -> soft
```

So the root cause is a **downgrade allow-list that omits the agent's own runtime infra** (CODEX_HOME /
XDG cache / TMPDIR), the host `~/.cache`, and read-only OS dirs. These benign reads fall through to the
hardcoded `fatal` branch (`11140`, `11174`) which raises `CLIProcessPolicyViolation` →
`finalization_status=policy_violation`, tokens=0 (`8995-9033`).

### Why it CONTRADICTS the codebase philosophy

`apex_omega/autogen/context.py:170-178, 499-517` (SANDBOX-NOT-PROMPT): *"we do NOT … kill an attempt
for trying to fetch/cheat … the worktree SHADOWS site-packages so a fetched package can NEVER produce
a false solve … a blocked escape is RECORDED as telemetry, not punished; the agent is left to solve
the real task."* The fatal-abort path kills the rollout instead of denying-and-continuing. The
existing soft path (`8996`) is the in-repo deny-don't-kill precedent the guard already has — it just
isn't reached for these classes.

---

## 2. SHOULD anything be done? (broader vs pydantic-only; FP vs correct)

**Yes. It is broad, not pydantic-only, and ~89% false-positive.** From the IMPACT data (416 fatal
aborts across LIVE + ladder_n5 + ladder_n5_framed):

- **Only 11% (45/416) are genuine agent-cheat** — agent/repair waves reading a fetched/extracted
  upstream copy (`/tmp/{pydantic,mimesis,jinja,voluptuous}_upstream`, `*_wheel`, `*_restore`,
  `*_spec.txt`) or the agent's own cell parent.
- **37% codex-internal false-positives** — the codex resolver/venv (`runtime/home/.cache/uv`,
  `runtime/.venv/.../site-packages`, `fastzip-castree-uid-0`, host `~/.cache`).
- **50% benign-env** — broad traversal of the ladder root, `/`, `/opt/homebrew`, `/usr`,
  `/private/var/folders` tmp.
- **2% guard mis-parse** — a `sed`/`awk` script arg (`/^$/d`, `/dev/fd/15`) mistaken for a path operand.
- **(separately) 11% cross-cell** — one cell killed for a *different* cell's subprocess (process-tree
  scope bug).

Per-repo fraction of ALL agent rollouts fatally aborted (tokens=0): **pydantic 28%, jinja 21%,
networkx 15%, minitorch 11%, mimesis 8%, voluptuous 6%, babel 5%.** Collectable repos lose up to
50% of rollouts (omega_template_k8 mimesis/jinja/pydantic = 9/18). This is destroying real solve work
across the suite.

**Is the guard "correct"?** It is correct in *intent* (block reads of another worktree / a real
upstream copy that could cross-contaminate) but **over-broad in mechanism**: it treats "any monitored
read whose path isn't under an APEX marker" as fatal, which sweeps in the agent's own infra and the OS.
The genuine-cheat carve-out should be **structural** (location-based), not a leftover of a too-narrow
allow-list. So: **fix it — reclassify, do not remove.**

---

## 3. BEST fix (graft) + isolation argument

**Chosen: `fix-classification` (env-aware allow-list / SOFT downgrade), grafted with two pieces from
`deny-dont-kill` and one new telemetry-preservation edit the candidates both missed.**

### Why this combination

- **`fix-classification` is the isolation-correct base** (judge: isolation best, 84). It is an
  ALLOW-LIST: downgrade ONLY paths under *this* agent's own env-infra roots
  (`HOME`/`CODEX_HOME`/`XDG_CACHE_HOME`/`TMPDIR`), the host `~/.cache`, and read-only OS roots — and
  `/tmp`+`/private/tmp` are **deliberately excluded** from the OS set. Every genuine cross-isolation
  read (sibling worktree, another cell, own cell parent, ladder root, `/tmp/*_upstream`) stays FATAL
  *by construction* — the fatal carve-out does NOT depend on the cheat being conveniently named.
  `deny-dont-kill` is a deny-LIST (default = soft, fatal only on a magic marker tuple), so an
  adversary who does `cp -r sibling /tmp/x; find /tmp/x` evades the fatal path entirely (judge:
  isolation 46). Allow-list wins the explicit isolation constraint.

- **Graft G1 from `deny-dont-kill`: the upstream-reference precedence rule.** Add an explicit
  `_path_resolves_to_upstream_reference_copy` check that takes **precedence over** the infra check.
  This closes the one adversarial hole in a pure allow-list: an upstream copy planted *under* TMPDIR
  (`$TMPDIR/pydantic_upstream/…`). Validated: with the precedence rule that path stays **FATAL** even
  though it is under an infra root.

- **Graft G2 from `deny-dont-kill`: a `workspace_discovery` soft severity label** (instead of
  overloading `backend_helper`) so soft logs are self-describing for cross-eval analysis. Routed
  through the SAME existing soft handler.

- **New edit T1 (telemetry preservation) — the gap both candidates and both judges flagged but
  neither fixed.** `classify_attempt_integrity` (`context.py:181-204`) keys `sandbox_escape` on
  `res.finalization_status == "policy_violation"` OR error-text. After a soft downgrade the rollout
  COMPLETES (status `completed`, `error=None`), and the soft record lives only in
  `model_result.timeout_audit["soft_policy_violations"]` (`cli_backend.py:9435-9436`, set on the
  result at `7860`). **The v1→omega adapter (`v1_executor.py:200-213`) drops `timeout_audit`**, so the
  integrity-log `sandbox_escape` signal would silently vanish for every downgraded read — violating
  constraint (3) "don't weaken the cheat signal." T1 surfaces a compact soft-violation summary into
  `ExecResult.error` (an already-classified field) so the existing classifier keeps firing.

### Isolation / anti-cheat argument (constraints 1-3)

1. **Anti-cheat / no false solve preserved.** This is a severity *reclassification*, not a grant — the
   guard never granted access; the OS sandbox denies the read regardless of severity. Downgraded
   commands are read-only (`find/grep/diff/ls/cat/head/tail/du/tree`). Three independent structural
   mechanisms make any downgraded read incapable of a false solve, all UNCHANGED:
   (a) the worktree **shadows** site-packages so a read package can never be the imported impl;
   (b) **no network**; (c) acceptance keyed on **gold expected-test ids**. The one residual the shadow
   does NOT neutralize — reading a **sibling worktree's source** and hand-copying it into your own
   worktree (real cross-task contamination) — stays **FATAL**, because a sibling worktree is not under
   the assigned env-infra roots and not under a system root. Validated: sibling worktree, another
   cell, own cell parent, ladder root, `/tmp/*_upstream`, `/tmp/*_wheel`, and `$TMPDIR/*_upstream`
   (adversarial) all remain FATAL.

2. **Determinism / replay preserved.** Infra roots are a pure function of the fixed per-rollout env,
   resolved with the same `Path.resolve()` the guard already uses; default `()` makes every
   un-threaded caller bit-identical to today. T1 is a deterministic string summary of the already-
   deterministic soft log.

3. **Cheat signal NOT weakened — strengthened.** The v1-side `soft_policy_violation_log` already
   records every downgraded escape (`9009`/`9192`); T1 additionally restores the apex_omega
   `integrity_log.jsonl` `sandbox_escape` signal that the adapter would otherwise drop. So the cheat
   signal is retained on BOTH sides, while the fatal kill (which IMPACT shows gave ZERO marginal
   anti-cheat benefit — telemetry already fired on all 242 LIVE escapes) is removed.

### Empirical split (validated in `.venv_omega`, per-rollout pydantic env)

```
SOFT   ~/.cache (host)            SOFT   codex uv cache (XDG_CACHE_HOME/uv)
SOFT   TMPDIR scratch             SOFT   /opt/homebrew/perl5   /usr/lib   /var/folders tmp
SOFT   system site-packages
FATAL  sibling worktree           FATAL  another cell          FATAL  own cell parent
FATAL  ladder root                FATAL  /tmp/pydantic_upstream FATAL  /tmp/mimesis_wheel
FATAL  $TMPDIR/pydantic_upstream  (adversarial under-infra upstream copy -> still FATAL via G1)
```

This is exactly the desired split: the pydantic cell's aborts (`~/.cache`, codex uv cache,
`/opt/homebrew/lib/perl5`, ladder traversal of OS dirs) become SOFT so the rollout continues and
emits real diffs; deliberate upstream reads stay FATAL.

---

## 4. ORDERED implementation plan

All edits in `apex/core/cli_backend.py` unless noted. Anchor on method names + literal strings
(`target_is_system_helper`, `likely_backend_helper`), NOT raw line numbers (14k-line file, actively
edited). `Mapping`/`Path` already imported.

**Step 1 — New constants** (after `_WORKSPACE_POLICY_SYSTEM_TARGET_PREFIXES`, ~line 1280):
```python
_WORKSPACE_POLICY_SYSTEM_READONLY_ROOTS: tuple[str, ...] = (
    "/usr/", "/opt/homebrew/", "/opt/local/", "/System/", "/Library/",
    "/etc/", "/bin/", "/sbin/", "/private/var/folders/", "/var/folders/",
)  # deliberately EXCLUDES /tmp and /private/tmp
_WORKSPACE_POLICY_UPSTREAM_REFERENCE_MARKERS: tuple[str, ...] = (
    "_upstream/", "_upstream.", "_wheel/", "_restore/", "_spec.txt",
)
_WORKSPACE_POLICY_AGENT_RUNTIME_ENV_KEYS: tuple[str, ...] = (
    "HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME",
    "CODEX_HOME", "CLAUDE_CONFIG_DIR", "GEMINI_CLI_HOME", "OPENCODE_CONFIG_DIR", "TMPDIR",
)
```

**Step 2 — `_agent_runtime_infra_roots(env)` `@staticmethod`** (near `_path_resolves_to_system_helper_target`):
resolve each present env key to `Path(...).resolve()` (swallow `OSError`), append `Path.home()/'.cache'`,
drop `'/'`, de-dupe by string; return `tuple[Path,...]`. With `env=None` returns just `(~/.cache,)`.

**Step 3 — `_path_resolves_to_upstream_reference_copy(path)` `@staticmethod`** (G1; takes precedence):
True if any `_WORKSPACE_POLICY_UPSTREAM_REFERENCE_MARKERS` substring is in `str(path)` OR
(`'/site-packages/'` in text AND NOT `_path_text_looks_like_backend_runtime_helper(text)` — excludes
the codex *system* venv; a copied site-packages outside a system root is treated as a reference copy).

**Step 4 — `_path_is_agent_runtime_infra(path, runtime_infra_roots=())` `@classmethod`**:
`if path is None: return False`; **`if cls._path_resolves_to_upstream_reference_copy(path): return False`**
(G1 precedence); `True` if `path.resolve()` is `relative_to` any infra root, OR `str(path)` starts with
any `_WORKSPACE_POLICY_SYSTEM_READONLY_ROOTS`, OR `cls._path_resolves_to_system_helper_target(path)`
(existing downgrade kept as a strict subset = pure widening). Swallow `ValueError`/`OSError`.

**Step 5 — thread `runtime_infra_roots` through the guard.**
- `_process_tree_workspace_policy_violation` (sig ~10915): add kwarg
  `runtime_infra_roots: tuple[Path, ...] = ()`.
- **Operand branch** (the `target_is_system_helper`/`likely_backend_helper` lines ~11172-11174):
  replace with a three-way:
  `if self._path_resolves_to_upstream_reference_copy(resolved_path): severity = "fatal"`
  `elif self._path_is_agent_runtime_infra(resolved_path, runtime_infra_roots=runtime_infra_roots): severity = "workspace_discovery"; likely_backend_helper = True`
  `else: severity = "fatal"`. Keep reason text; reason_prefix uses the helper wording when not fatal.
- **CWD branch** (`11128-11140`): compute
  `likely_backend_helper = self._looks_like_backend_helper_workspace_policy_violation(process_cwd=…, path_tokens=…, working_dir=…, runtime_infra_roots=runtime_infra_roots)`;
  set `"severity": "workspace_discovery" if likely_backend_helper else "fatal"`.

**Step 6 — `_looks_like_backend_helper_workspace_policy_violation`** (`9771`): add
`runtime_infra_roots: tuple[Path, ...] = ()`. In the explicit-escape veto loop, `continue` (skip the
hard-veto `return False`) when `self._path_is_agent_runtime_infra(resolved, runtime_infra_roots=…)` is
True. Before the final `return False`, add:
`if self._path_is_agent_runtime_infra(process_cwd, runtime_infra_roots=runtime_infra_roots): return True`.
(The upstream-precedence in Step 4 keeps `find $TMPDIR/x_upstream` from an infra CWD fatal: its operand
vetoes.)

**Step 7 — call sites compute + pass roots.**
- In `_communicate_with_progress_timeout` (sig ~8542, `target_runtime_env` param at 8551): once near
  the top compute `runtime_infra_roots = self._agent_runtime_infra_roots(target_runtime_env)`; pass it
  to BOTH `_process_tree_workspace_policy_violation` calls (~8989 and ~9164).
- Standalone audit call ~11273 (`_target_runtime_completion_policy_audit`, has `env`): pass
  `runtime_infra_roots=self._agent_runtime_infra_roots(env)`.

**Step 8 — severity dispatcher** (`cli_backend.py:8996` AND `9180`): change the soft set
`{"backend_helper", "blocked_by_policy"}` → `{"backend_helper", "blocked_by_policy", "workspace_discovery"}`
(both occurrences). Also add `"workspace_discovery"` to the early-return soft set in
`_target_runtime_completion_policy_audit` (`11283-11286`) so the completion-audit path also continues.
Body unchanged (dedup, append to `soft_policy_violation_log`, emit `policy_violation_soft`).

**Step 9 — telemetry preservation (T1)** in `apex_omega/executor/v1_executor.py:200-213`: before
building the returned `ExecResult`, read `getattr(res, "timeout_audit", {}) or {}`; if it has a
non-empty `soft_policy_violations`, build a compact summary string (count + first reason, which already
contains "repository discovery outside the rollout workspace") and fold it into the `error` field:
`error = res.error or ("soft policy: " + summary)`. This keeps `classify_attempt_integrity`
(`context.py:196`, matches "outside the root"/"policy" text) firing `sandbox_escape` for downgraded
reads. (Optional follow-up: add a first-class `soft_policy_violations` field to `ExecResult` +
`classify_attempt_integrity`; the `error`-fold is the minimal, contained version.)

**Out of scope (tracked separately):** the 2% `sed`/`awk` operand mis-parse (`_command_path_operands`)
and the 11% cross-cell process-tree-scope kill (`_collect_process_tree_entries`). This fix narrows both
populations but their structural fixes are separate.

---

## 5. Test plan

**Unit — new `tests/test_workspace_policy_classification.py`:**
1. `_agent_runtime_infra_roots`: resolves `{HOME, XDG_CACHE_HOME, CODEX_HOME, TMPDIR}`, drops `'/'`,
   `None` env → just `~/.cache`.
2. `_path_is_agent_runtime_infra` True: under HOME / XDG_CACHE_HOME (`uv/archive-v0`) / TMPDIR,
   `~/.cache`, `/opt/homebrew/lib/perl5`, `/usr/lib/x`, `/private/var/folders/.../T/tmpX`, system
   `site-packages/faker`.
3. `_path_is_agent_runtime_infra` False: sibling worktree, another cell, own cell parent, ladder root,
   `/tmp/pydantic_upstream`, `/tmp/mimesis_wheel`, AND `$TMPDIR/pydantic_upstream` (G1 precedence).
4. `_process_tree_workspace_policy_violation` with synthetic process_entries:
   `find <XDG_CACHE_HOME>/uv` + `runtime_infra_roots` → severity `workspace_discovery`;
   operand = sibling worktree → `fatal`.
5. `process_cwd` inside XDG_CACHE_HOME → `workspace_discovery`; `find /tmp/pydantic_upstream` from an
   infra CWD → `fatal` (veto preserved).
6. **Regression guard:** call the guard WITHOUT `runtime_infra_roots` → reproduces pre-fix severities.
7. Pin the dispatcher mapping at `8996`/`9180` (`workspace_discovery` → SOFT append+continue;
   `fatal` → raise).
8. **T1:** an `ExecResult` derived from a `CLIModelResult` whose `timeout_audit` carries
   `soft_policy_violations` → `classify_attempt_integrity` still returns a `sandbox_escape` signal.

**Regression — full suite:** `.venv_omega/bin/python -m pytest tests/ -q` (expect current 325 green +
new tests). Watch `test_review_fixes.py` escape/integrity tests confirm telemetry still records escapes.

**Live A/B gate:** re-run `hybrid-nogate__pydantic__s0` editable-installed; confirm cache/system aborts
no longer give tokens=0, `soft_policy_violation_log` now contains them, `integrity_log.jsonl` still
records `sandbox_escape`, and a deliberate `*_upstream` read still aborts FATALLY.

---

## 6. Risks & mitigations

- **R1 — a genuine cheat hides under an infra root.** Mitigated: infra roots are read-only
  caches/config that cannot become a false solve (shadow + no-network); telemetry still records; G1
  upstream-precedence keeps a planted upstream copy under TMPDIR FATAL; mutating-into-worktree commands
  are unaffected.
- **R2 — env not threaded at a call site.** Mitigated by default `()` (fail-safe to current behavior)
  and by always adding host `~/.cache` even for `None` env (the dominant host-cache FP covered
  everywhere).
- **R3 — broadening `/usr`,`/opt/homebrew`,`/var/folders` downgrades a real system-lib read.**
  Acceptable: system `site-packages` was already downgraded; perl/stdlib/OS-scratch are the same
  read-only-OS class with no false-solve capability; `/tmp`+`/private/tmp` excluded so real source-copy
  cheats stay fatal.
- **R4 — `Path.resolve()` on a non-existent operand.** Handled like existing code
  (`OSError`/`ValueError` swallowed).
- **R5 — telemetry adapter drift.** The T1 `error`-fold is contained; the optional first-class
  `soft_policy_violations` field is the durable follow-up.
- **R6 — line drift.** Anchor on method names + the `target_is_system_helper`/`likely_backend_helper`
  literals.
