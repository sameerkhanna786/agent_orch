# Fix: classify codex-internal / env discovery as `backend_helper` (SOFT), not `fatal`

**File:** `apex/core/cli_backend.py` (vendored apex v1, git-tracked)
**Owner method to fix:** `_process_tree_workspace_policy_violation` + its two severity branches and the
classifier `_looks_like_backend_helper_workspace_policy_violation` / `_path_resolves_to_system_helper_target`.
**Goal:** A `find`/`grep`/`diff`/`cp`/… that *originates from* or *targets* the codex runtime HOME, the
uv/pip cache, the host `~/.cache`, the rollout's own TMPDIR, or read-only system dirs is reclassified
`backend_helper` (SOFT course-correction, rollout CONTINUES). Only a genuine agent command reading
**another worktree, another cell, the cell parent, the whole ladder dir, or an extracted upstream/source
package copy** stays `fatal`. This is the deny-don't-kill behavior the codebase's own SANDBOX-NOT-PROMPT
policy already mandates everywhere else.

---

## 1. Root cause (confirmed by re-implementing the predicates in `.venv_omega`)

The fatal aborts come from the DIRECT monitored-command loop in
`_process_tree_workspace_policy_violation` (`cli_backend.py:11122-11194`), NOT the shell-payload path.
A monitored command (`find` ∈ `_WORKSPACE_POLICY_MONITORED_COMMANDS`, line 1159) is flagged when either:

- **process-CWD branch** (11128-11150): the process CWD escapes the worktree
  (`_path_escapes_workspace`, 9742-9754), severity downgraded only if
  `_looks_like_backend_helper_workspace_policy_violation` (9771-9801) is True; else `fatal` (11140).
- **path-operand branch** (11152-11194): an operand from `_command_path_operands` (9635-9707) resolves
  outside the worktree, severity downgraded only if `_path_resolves_to_system_helper_target` (10902-10913)
  is True; else `fatal` (11174).

Both downgrade predicates only recognize **APEX-managed helper markers**
(`_WORKSPACE_POLICY_BACKEND_HELPER_MARKERS` = `.apex_agent_runtime/`, `apex-cli-offline-`,
`cli_airgapped_homes/`, `fastzip-castree-`; lines 1246-1251) and **system site-packages** under
`/usr/local|/opt/homebrew|/System|/Library` containing `/site-packages/` etc.
(`_path_text_looks_like_backend_runtime_helper`, 9803-9808; prefixes 1255-1260, markers 1261-1270).

They do NOT recognize the codex runtime's own paths, which on this host live at:

```
.../cells/<mod>/runtime/home            = HOME / CODEX_HOME
.../cells/<mod>/runtime/home/.cache     = XDG_CACHE_HOME  (uv archive-v0, pip, codex caches live here)
.../cells/<mod>/runtime/tmp             = TMPDIR
/Users/<user>/.cache                    = host shared uv/pip/codex cache (when not relocated)
/opt/homebrew/lib/perl5, /usr/..., /private/var/folders/...  = read-only system / OS scratch
```

None of these contain an APEX marker and none start with a system prefix+`/site-packages/`, so every one
of the five evidence paths falls through to `fatal` → `CLIProcessPolicyViolation` →
`finalization_status=policy_violation`, `tokens=0` (8995-9029). Empirically: of 416 fatal aborts, only
**11% are genuine agent cheat**; **89% are codex-internal / benign-env / mis-parse / cross-cell false
positives**, and pydantic loses **28% of all rollouts** to this (worst evidence cell = 17 aborts).

### Why an env-aware classifier is the correct cut line

The agent's OWN runtime infrastructure roots are *already known at guard time* — they are the env keys
`HOME`, `XDG_CACHE_HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_STATE_HOME`, `CODEX_HOME`,
`CLAUDE_CONFIG_DIR`, `GEMINI_CLI_HOME`, `TMPDIR` (set at `cli_backend.py:8502-8512`, propagated through
`launch_env` → `target_runtime_env` param of `_communicate_with_progress_timeout`, 8551 / 7672-7674).
Anything *under one of those roots* is the agent's own backend infrastructure; anything *under the host
`~/.cache`* is the shared package cache; anything under read-only system dirs is the OS. None of these can
produce a false solve (worktree-shadow + no-network already guarantee that). Everything else outside the
worktree — sibling worktrees, sibling cells, the cell parent, the ladder root, `*_upstream/*_wheel/*_restore`
copies — is a genuine cross-contamination read and STAYS fatal. Because worktrees are minted as siblings
under a shared `worktrees/` dir (`apex_omega/isolation/worktree.py:78`), a sibling worktree is *naturally*
not under any env-infra root and not a system dir, so it stays fatal with **no special-casing**.

---

## 2. Exact changes

### 2a. New constant: system / OS read-only roots (after line 1280)

`_WORKSPACE_POLICY_SYSTEM_TARGET_PREFIXES` (1275-1280) is too narrow (APEX markers only). Add a separate,
explicit set of **read-only host system roots** that are backend/OS noise — deliberately EXCLUDING bare
`/tmp` and `/private/tmp` (those host the genuine `*_upstream` copies and sibling cells, so they must stay
fatal):

```python
# Read-only host/OS roots a backend resolver or interpreter legitimately walks
# (perl libs, python stdlib, OS scratch). EXCLUDES bare /tmp + /private/tmp on
# purpose: extracted upstream package copies and sibling task cells live there
# and must stay fatal. APEX-managed markers stay in the set above.
_WORKSPACE_POLICY_SYSTEM_READONLY_ROOTS: tuple[str, ...] = (
    "/usr/",
    "/opt/homebrew/",
    "/opt/local/",
    "/System/",
    "/Library/",
    "/etc/",
    "/bin/",
    "/sbin/",
    "/private/var/folders/",   # macOS per-user OS scratch (mkstemp default)
    "/var/folders/",
)
```

### 2b. New env-aware helper roots builder + classifier (static/classmethods, near line 9802)

Add two small methods to `CLIModelClient` so they are trivially unit-testable and pure:

```python
@staticmethod
def _agent_runtime_infra_roots(env: Optional[Mapping[str, str]]) -> tuple[Path, ...]:
    """Resolved roots that belong to the agent's OWN runtime infrastructure.

    These are the relocated CLI HOME / caches / config / scratch declared in the
    rollout env, plus the shared host ~/.cache package cache. A monitored read
    that stays inside one of these is backend/env noise, never a cross-task read.
    """
    roots: list[Path] = []
    keys = (
        "HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
        "XDG_STATE_HOME", "CODEX_HOME", "CLAUDE_CONFIG_DIR",
        "GEMINI_CLI_HOME", "OPENCODE_CONFIG_DIR", "TMPDIR",
    )
    for key in keys:
        value = str((env or {}).get(key) or "").strip()
        if not value:
            continue
        try:
            roots.append(Path(value).resolve())
        except OSError:
            continue
    # Shared host package cache (uv/pip/codex) when HOME is NOT relocated.
    try:
        roots.append((Path.home() / ".cache").resolve())
    except (OSError, RuntimeError):
        pass
    # De-dup while preserving order; drop "/" if it ever sneaks in.
    seen: set[str] = set()
    out: list[Path] = []
    for r in roots:
        t = str(r)
        if t in {"", "/"} or t in seen:
            continue
        seen.add(t)
        out.append(r)
    return tuple(out)

@classmethod
def _path_is_agent_runtime_infra(
    cls,
    path: Optional[Path],
    *,
    runtime_infra_roots: tuple[Path, ...] = (),
) -> bool:
    """True if `path` is the agent's own runtime infra, host cache, or a
    read-only OS/system root — i.e. backend noise, not a cross-task read."""
    if path is None:
        return False
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    for root in runtime_infra_roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    text = str(resolved)
    for prefix in _WORKSPACE_POLICY_SYSTEM_READONLY_ROOTS:
        if text == prefix.rstrip("/") or text.startswith(prefix):
            return True
    # Preserve the existing APEX-managed-helper + system site-packages downgrade.
    return cls._path_resolves_to_system_helper_target(resolved)
```

`Mapping` is already imported (used widely, e.g. 7090). `_path_resolves_to_system_helper_target` is the
existing classmethod (10902); calling it last means the **existing** downgrade set is a strict subset of
the new behavior — pure widening, never tightening.

### 2c. Thread the env-derived roots into the guard (one param, default `()`)

`_process_tree_workspace_policy_violation` (10915-10924) gains one optional kwarg:

```python
def _process_tree_workspace_policy_violation(
    self,
    process_entries: dict[int, dict[str, Any]],
    working_dir: str,
    *,
    target_runtime_enforced: bool = False,
    target_runtime_git_history_policy: str = "blocked",
    target_runtime_source_network_policy: str = "unspecified",
    target_runtime_filesystem_boundary_policy: str = "policy_enforced",
    runtime_infra_roots: tuple[Path, ...] = (),   # NEW
) -> Optional[dict[str, Any]]:
```

Default `()` keeps all existing callers byte-for-byte equivalent (the new downgrade simply never fires
without roots, so behavior is unchanged where roots aren't supplied).

### 2d. Use the classifier in BOTH severity branches

**Process-CWD branch (11128-11140).** Replace the `likely_backend_helper` computation with an OR against
the new env-aware check (the existing classifier still runs first so its explicit-escape-operand veto is
preserved — see §3):

```python
likely_backend_helper = self._looks_like_backend_helper_workspace_policy_violation(
    process_cwd=process_cwd,
    path_tokens=path_tokens,
    working_dir=workspace_root,
    runtime_infra_roots=runtime_infra_roots,   # NEW kwarg, see below
)
```

and extend `_looks_like_backend_helper_workspace_policy_violation` (9771-9801) to (a) accept
`runtime_infra_roots=()`, (b) use the broader infra test inside its explicit-escape veto so an operand
that points at infra does NOT veto, and (c) add an infra check on the CWD itself before returning False:

```python
def _looks_like_backend_helper_workspace_policy_violation(
    self, *, process_cwd, path_tokens, working_dir, runtime_infra_roots=(),
) -> bool:
    if process_cwd is None:
        return False
    explicit_escape_tokens = [t for t in path_tokens
                              if self._path_token_is_explicit_workspace_escape(t)]
    if explicit_escape_tokens:
        for token in explicit_escape_tokens:
            if token.startswith("/") and self._path_text_looks_like_backend_runtime_helper(token):
                continue
            resolved = self._resolve_monitored_path_token(
                token, working_dir=working_dir, process_cwd=process_cwd)
            # NEW: an operand that resolves into the agent's own infra / host
            # cache / system root is benign and must not veto the downgrade.
            if self._path_is_agent_runtime_infra(
                resolved, runtime_infra_roots=runtime_infra_roots):
                continue
            if self._path_escapes_workspace(resolved, working_dir=working_dir):
                return False   # genuine cross-task operand -> stays fatal
    cwd_text = str(process_cwd)
    if any(m in cwd_text for m in _WORKSPACE_POLICY_BACKEND_HELPER_MARKERS):
        return True
    if self._path_text_looks_like_backend_runtime_helper(cwd_text):
        return True
    # NEW: CWD inside the agent's own runtime infra / host cache / system root.
    if self._path_is_agent_runtime_infra(process_cwd, runtime_infra_roots=runtime_infra_roots):
        return True
    return False
```

**Path-operand branch (11172-11174).** Replace the narrow target test:

```python
# OLD:
target_is_system_helper = self._path_resolves_to_system_helper_target(resolved_path)
likely_backend_helper = target_is_system_helper
# NEW:
likely_backend_helper = self._path_is_agent_runtime_infra(
    resolved_path, runtime_infra_roots=runtime_infra_roots)
```

### 2e. Build & pass the roots at the call sites

At the monitor call site (8989) inside `_communicate_with_progress_timeout`, compute the roots ONCE from
the env this function already receives (`target_runtime_env`, param at 8551) and pass them in:

```python
# computed once, above the monitor loop (deterministic for the rollout):
runtime_infra_roots = self._agent_runtime_infra_roots(target_runtime_env)
...
violation = self._process_tree_workspace_policy_violation(
    process_entries,
    working_dir,
    target_runtime_enforced=target_runtime_enforced,
    runtime_infra_roots=runtime_infra_roots,   # NEW
)
```

Also pass it at the second in-function call site (9164, `target_runtime_enforced=True`) and the standalone
call at 11273 (the completion-audit path) — for 11273, pass `_agent_runtime_infra_roots(env)` using the env
in that scope, or `()` if no env is in scope (safe default = unchanged behavior). Note `target_runtime_env`
is `None` for non-target-runtime rollouts; `_agent_runtime_infra_roots(None)` still returns the host
`~/.cache` root, which correctly downgrades the most common host-cache false positive even off-target.

---

## 3. How isolation / anti-cheat is preserved (NOT weakened)

1. **Still deny + record every out-of-workspace read.** This is a *severity reclassification*, not a
   removal. The violation is still detected, still logged: SOFT violations append to
   `soft_policy_violation_log` and emit `policy_violation_soft` (8996-8019); the integrity layer’s
   `sandbox_escape` telemetry (`integrity_log.jsonl`, 242 entries observed) fires **independently of
   severity**, so the genuine-cheat signal is fully retained. The IMPACT data already shows the fatal
   kill produced ZERO marginal anti-cheat benefit over telemetry.
2. **No false solve is possible from a downgraded read.** Per the codebase policy
   (`apex_omega/autogen/context.py:182-188,503-504`): the worktree shadows site-packages and there is no
   network, so reading the uv cache, host `~/.cache`, or system site-packages can never inject a passing
   implementation. The monitored commands are read-only (`find/grep/diff/ls/cat/head/tail/du/tree`), and
   the mutating ones (`cp/mv/rsync/tee/tar`) writing *into* the worktree are out of scope of this change.
3. **Genuine cross-contamination STAYS fatal.** The cut line is precise (validated, §4): sibling worktree,
   sibling cell, own cell parent, the ladder root, and `/tmp` `*_upstream/*_wheel/*_restore` copies are
   NOT under any env-infra root, NOT a system root, and NOT a system site-packages helper → `fatal`,
   unchanged. The `/tmp` exclusion from `_WORKSPACE_POLICY_SYSTEM_READONLY_ROOTS` is deliberate so the
   real-source-copy cheats keep aborting.
4. **The explicit-escape veto is preserved and even strengthened.** `find /tmp/pydantic_upstream` run
   *from* an infra CWD still vetoes (operand resolves outside workspace and is not infra) → fatal.
5. **Determinism / replay.** The roots are derived from the fixed per-rollout env and resolved with the
   same `Path.resolve()` already used by `_path_escapes_workspace`; the decision is a pure function of
   (command, cwd, env) with no clock/network/random input. Default `()` keeps every non-wired caller bit-
   identical.

---

## 4. Why this unblocks pydantic (and every repo)

The pydantic evidence cell's 7/8 aborts were exactly: `~/.cache`, the codex own `runtime/home/.cache/uv`
cache, `/opt/homebrew/lib/perl5`, and the ladder dir. Under this fix the first three become SOFT
(rollout continues, integrity still recorded), removing the dominant abort source. Validated split
(prototype run in `.venv_omega` against the literal evidence paths):

| target | new severity | correct? |
|---|---|---|
| `~/.cache` | SOFT | yes (host cache) |
| codex own `runtime/home/.cache/uv/...` | SOFT | yes (own infra) |
| `runtime/tmp/...` (TMPDIR) | SOFT | yes (own scratch) |
| `/opt/homebrew/lib/perl5`, `/usr/...`, `/var/folders/...` | SOFT | yes (read-only system/OS) |
| system `/opt/homebrew/.../site-packages/faker` | SOFT | yes — unchanged from today; cannot false-solve |
| **sibling worktree `wt_*`** | **FATAL** | yes (cross-task) |
| **another cell `cells/<other>/...`** | **FATAL** | yes (cross-task) |
| **own cell parent `cells/<mod>`** | **FATAL** | yes (escapes worktree) |
| **ladder root `/private/tmp/<ladder>`** | **FATAL** | yes (reaches sibling cells) |
| **`/tmp/pydantic_upstream`, `/tmp/mimesis_wheel`** | **FATAL** | yes (extracted source cheat) |

This removes the 89% false-positive aborts (codex-internal 37% + benign-env 50% + mis-parse 2%) and the
cross-cell kills, while the 11% genuine cheats and all cross-task reads stay fatal. pydantic repair waves
that today record `tokens=0` will instead run to completion and emit real diffs.

The remaining known issue *not* fixed here (out of scope, noted for follow-up): the 2% sed/awk
mis-parse (operands like `/^$/d`, `/class`) — those are a `_command_path_operands` parsing bug, separate
from classification; and the cross-cell process-tree-scope false positive (one cell killed for another
cell's subprocess, 11%) — that is a `_collect_process_tree_entries` scoping issue. This design narrows
both populations substantially (most sed/awk and cross-cell targets resolve to infra/system and now
downgrade) but the structural fixes are tracked separately.

---

## 5. Test plan

### Unit (new `tests/test_workspace_policy_classification.py`)

Construct a client without running `__init__` side effects via a minimal `LLMConfig` for a CLI backend,
or call the pure classmethods directly (they need no instance state):

1. `test_infra_roots_from_env` — `_agent_runtime_infra_roots({"HOME": h, "XDG_CACHE_HOME": c, "CODEX_HOME": h, "TMPDIR": t})` returns resolved `{h, c, t, ~/.cache}`; empty/`None` env → just `~/.cache`; `/` is dropped.
2. `test_infra_classifier_downgrades` — `_path_is_agent_runtime_infra` returns True for: a path under HOME; under XDG_CACHE_HOME (uv archive-v0); under TMPDIR; `~/.cache/...`; `/opt/homebrew/lib/perl5`; `/usr/lib/x`; `/private/var/folders/ab/T/tmpX`; and `/opt/.../site-packages/faker`.
3. `test_infra_classifier_keeps_fatal` — returns False for: sibling worktree `.../worktrees/wt_other`; another cell `.../cells/other/...`; own cell parent `.../cells/<mod>`; ladder root; `/private/tmp/pydantic_upstream/pydantic`; `/private/tmp/mimesis_wheel/x`.
4. `test_process_tree_operand_downgrade` — feed a synthetic `process_entries` with `find <XDG_CACHE_HOME>/uv ...`, `working_dir=<worktree>`, `runtime_infra_roots=<env roots>` → violation dict has `severity == "backend_helper"`. Same harness with operand `= sibling worktree` → `severity == "fatal"`.
5. `test_process_cwd_downgrade_and_veto` — `find .` whose `process_cwd` is inside XDG_CACHE_HOME → `backend_helper`; but `find /tmp/pydantic_upstream` from an infra CWD → `fatal` (veto preserved).
6. `test_default_roots_preserve_legacy` — calling `_process_tree_workspace_policy_violation` WITHOUT `runtime_infra_roots` reproduces the pre-fix severities for the same inputs (regression guard).
7. `test_severity_routing_unchanged` — assert the dispatcher mapping at 8995-9024 still treats `backend_helper`→SOFT (append + continue) and `fatal`→raise; this code path is untouched but pin it.

Run: `/Users/sameertkhanna/Documents/agent_orch/.venv_omega/bin/python -m pytest tests/test_workspace_policy_classification.py -q`.

### Regression / full suite

`/Users/sameertkhanna/Documents/agent_orch/.venv_omega/bin/python -m pytest tests/ -q`
(expect the current 301 green to stay green + the 7 new). Pay attention to `test_review_fixes.py`
(`test_escape_attempt_is_recorded_but_attempt_still_scored`, `test_integrity_classifier_detects_escape_fetch_and_cheat`) to confirm telemetry still records escapes.

### Live A/B (the real gate)

Re-run one pydantic cell (`hybrid-nogate__pydantic__s0`) under the editable install and confirm:
`finalization_status` is no longer `policy_violation`/`tokens=0` for the cache/system aborts; the
`soft_policy_violation_log` now contains those entries; `integrity_log.jsonl` still records
`sandbox_escape` telemetry; and any deliberate `*_upstream` read still aborts fatally.

---

## 6. Risks

- **R1 — a genuine cheat hides under an infra root.** Mitigated: infra roots are read-only caches/config
  the agent can't turn into a false solve (worktree-shadow + no-network); telemetry still records the
  escape; mutating commands into the worktree are unaffected.
- **R2 — env not passed at some call site → no downgrade.** Mitigated by default `()` (fail-safe to
  current behavior) and `~/.cache` still added even for `None` env, so the dominant host-cache FP is
  covered everywhere.
- **R3 — broadening `/usr` `/opt/homebrew` etc. downgrades a real system-lib read.** Acceptable and
  consistent: system site-packages were ALREADY downgraded; broadening to perl/stdlib/OS-scratch is the
  same class (read-only OS, no false-solve capability). `/tmp` and `/private/tmp` are deliberately
  excluded so the real-source-copy cheats stay fatal.
- **R4 — `Path.resolve()` on a non-existent operand.** Already handled the same way as the existing code
  (`_resolve_monitored_path_token`/`_path_escapes_workspace` swallow `OSError`/`ValueError`); the new
  helpers mirror that.
- **R5 — line numbers drift** (file is 14176 lines, actively edited). Anchor edits on method names and the
  literal `target_is_system_helper = ...` / `likely_backend_helper = ...` strings, not raw line numbers.
