# Guard fix design — `contain-env`: contain HOME/cache under (or whitelist-adjacent to) the rollout workspace

**File under change:** `/Users/sameertkhanna/Documents/agent_orch/apex/core/cli_backend.py` (vendored v1; edits are real + git-tracked)
**New test file:** `/Users/sameertkhanna/Documents/agent_orch/tests/test_workspace_policy_guard.py`
**Problem owner doc:** memory `converge-reduce-merge-bug.md` / pydantic repair-wave investigation (task #6)
**Date:** 2026-06-20

---

## 1. One-paragraph summary

The WORKSPACE-DISCOVERY GUARD fatally aborts a codex rollout (`finalization_status=policy_violation`, tokens=0)
whenever a monitored read-only command (`find`/`grep`/`rg`/…) runs *from* or *targets* a path outside the
agent's worktree, **unless** that path matches a narrow set of APEX-managed helper markers. The agent's own
provisioned runtime HOME, CODEX_HOME, XDG cache, and uv cache live **outside** `workspace_root` (sibling to
`repo/` at `cells/<mod>/runtime/home/.cache/uv`), and they carry **none** of the recognized markers — so
codex's own internal resolver/extraction discovery (37% of all 416 aborts) and benign env traversal (50%)
go fatal. This kills all work for **zero** marginal anti-cheat benefit, because the worktree-shadow +
no-network already make any out-of-workspace read incapable of producing a false solve, and the
`integrity_log.jsonl` `sandbox_escape` telemetry is recorded **independently of the abort**. The fix has two
parts: **(A)** thread the rollout's *own* HOME/CODEX_HOME/cache/uv/TMPDIR roots (already known at the launch
site as `launch_env`) into the guard as an explicit `workspace_policy_allowed_roots` set and **downgrade**
(not skip) a violation whose cwd/target resolves under one of them to the existing `backend_helper` SOFT
path; and **(B)** at the env layer, pin `UV_CACHE_DIR` + `TMPDIR` **inside** the apex-managed CLI home so
codex's resolver/extraction/temp activity stays in-bounds under an already-recognized marker. The genuine-cheat
signal (a fetched `*_upstream`/`*_wheel`/site-packages copy, a sibling worktree, the ladder root) stays **fatal**
exactly as today because those roots are NOT in the allowed set.

This fix targets the **codex-internal (37%)** and **benign-env-under-own-home** subset that share the root
cause "the agent's own provisioned home/cache is outside the worktree." It is intentionally **narrow**: it
does NOT touch sibling-worktree, ladder-dir, `~/.cache`, or fetched-package handling — those remain fatal /
are handled by the companion `allowed-roots-broaden` and `process-scope` designs. Containing the env is the
*highest-leverage, lowest-risk, replay-safe* slice and removes the false-positive **source** rather than
widening the guard's blast radius.

---

## 2. Root cause (verified, with exact lines)

Guard entry: `_process_tree_workspace_policy_violation` (`cli_backend.py:10915`), called every
`activity_check_interval` from `_communicate_with_progress_timeout` at **`cli_backend.py:8989`**.

The two fatal triggers for a monitored command (`find` ∈ `_WORKSPACE_POLICY_MONITORED_COMMANDS`, line 1159):

1. **PROCESS-CWD escape** (`11128`): `if self._path_escapes_workspace(process_cwd, working_dir=workspace_root)`
   → severity = `backend_helper` iff `_looks_like_backend_helper_workspace_policy_violation(...)` (`11129-11140`),
   else **`fatal`**. Emits the evidence string `` `find` is running from `<cwd>` instead of under `<workspace_root>` ``.
2. **PATH-OPERAND escape** (`11152-11194`): for each operand from `_command_path_operands` (`11127`), if it
   resolves outside the worktree (`11164`) → severity = `backend_helper` iff
   `_path_resolves_to_system_helper_target(resolved_path)` (`11172-11174`), else **`fatal`**. Emits
   `` `find` targeted `<token>` which resolves to `<resolved_path>` ``.

Severity handling at **`8995-9033`**: `severity ∈ {backend_helper, blocked_by_policy}` ⇒ **SOFT** (append to
`soft_policy_violation_log`, `_emit_progress("policy_violation_soft")`, **rollout CONTINUES**). `severity == "fatal"`
⇒ **ABORT** (`raise CLIProcessPolicyViolation` ⇒ `finalization_status=policy_violation`, tokens=0). The SOFT
path is the existing **deny-don't-kill** precedent we extend.

**Why the agent's own home goes fatal.** The downgrade predicates only recognize:
- `_WORKSPACE_POLICY_BACKEND_HELPER_MARKERS` / `_WORKSPACE_POLICY_SYSTEM_TARGET_PREFIXES` (lines 1246-1251,
  1275-1280): `.apex_agent_runtime/`, `apex-cli-offline-`, `cli_airgapped_homes/`, `fastzip-castree-`.
- `_path_text_looks_like_backend_runtime_helper` (`9803-9808`): path must **start with**
  `/usr/local/ | /opt/homebrew/ | /System/ | /Library/` (`1255-1260`) **and** contain `/site-packages/`,
  `/dist-packages/`, etc. (`1261-1270`).

The live codex home is at `…/cells/<mod>/runtime/home/` (a `/private/tmp` path) with uv cache at
`…/runtime/home/.cache/uv/archive-v0/…`. It matches **no** marker, so both predicates return False ⇒ fatal.
Verified empirically (`.venv_omega`, re-implementing both predicates against the 5 evidence paths):

```
FATAL      /Users/sameertkhanna/.cache
FATAL      …/cells/autogen_orchestrator__pydantic/runtime/home/.cache/uv/archive-v0/…   ← codex's OWN uv cache
FATAL      …/cells/autogen_orchestrator__pydantic/worktrees/wt_m730006                  ← sibling worktree
FATAL      /private/tmp/omega_phase_ab_n3max                                            ← ladder dir
FATAL      /opt/homebrew/lib/perl5                                                      ← no /site-packages/ marker
DOWNGRADE  …/.apex_agent_runtime/repo-<digest>/.cli_homes/codex_cli/.cache             ← apex-managed home (works today)
DOWNGRADE  /opt/homebrew/lib/python3.14/site-packages/faker                            ← system site-packages
```

The contrast on the last two lines is the entire bug: when the relocation path
(`_relocate_cli_home_for_target_runtime`, `cli_backend.py:6117`) is used, HOME lands under `.apex_agent_runtime/`
and is correctly downgraded. The live eval's codex home lives under the cell's `runtime/home`, which is not a
recognized root. Part A removes the dependence on *where* the home happens to live by deriving the allowed roots
from the launch env; Part B additionally pins the home's temp/cache under a recognized root.

---

## 3. The fix

### Part A — env-derived allowed-roots (primary; removes 37% codex-internal + benign-under-own-home false positives)

**A1. New helper to compute the rollout's own runtime roots.** Add next to `_cli_sandbox_writable_roots`
(`cli_backend.py:6198`). It reuses the exact set of env keys that already define the nested-CLI sandbox's
writable home, plus the uv/temp keys:

```python
# cli_backend.py — new module-level constant near line 6198
_WORKSPACE_POLICY_OWN_RUNTIME_ENV_KEYS: tuple[str, ...] = (
    "HOME",
    "CODEX_HOME",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
    "CLAUDE_CONFIG_DIR",
    "GEMINI_CLI_HOME",
    "OPENCODE_CONFIG_DIR",
    "UV_CACHE_DIR",
    "TMPDIR",
)

def _workspace_policy_own_runtime_roots(
    env: Optional[dict[str, str]],
    *,
    working_dir: str,
) -> frozenset[str]:
    """Resolved absolute roots that belong to THIS rollout's own provisioned
    runtime (its CLI HOME, caches, uv cache, temp). A monitored read-only command
    that stays inside one of these is the agent's own backend resolver/extraction,
    never cross-task contamination — it is downgraded to the soft deny path, not
    a fatal abort. The worktree itself is intentionally EXCLUDED (handled by the
    in-workspace path); only the OUT-of-workspace own-runtime roots are listed."""
    if not env:
        return frozenset()
    try:
        workspace = Path(working_dir).expanduser().resolve(strict=False)
    except OSError:
        workspace = Path(working_dir).expanduser().absolute()
    roots: set[str] = set()
    for key in _WORKSPACE_POLICY_OWN_RUNTIME_ENV_KEYS:
        value = str(env.get(key) or "").strip()
        if not value:
            continue
        try:
            resolved = Path(value).expanduser().resolve(strict=False)
        except OSError:
            continue
        # Never allow the workspace itself or the filesystem root to be smuggled
        # in as an "own runtime" root (defensive: an empty/"/" env value).
        if resolved == workspace or str(resolved) in ("/", ""):
            continue
        roots.add(str(resolved))
    return frozenset(roots)
```

**A2. New predicate: is a path under one of the rollout's own runtime roots?** Add next to
`_path_resolves_to_system_helper_target` (`cli_backend.py:10902`):

```python
@staticmethod
def _path_under_own_runtime_root(
    path: Optional[Path],
    own_runtime_roots: frozenset[str],
) -> bool:
    """True if `path` resolves under one of THIS rollout's own provisioned
    runtime roots (its CLI HOME / caches / uv cache / temp)."""
    if path is None or not own_runtime_roots:
        return False
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        resolved = path
    for root_text in own_runtime_roots:
        try:
            resolved.relative_to(Path(root_text))
            return True
        except ValueError:
            continue
    return False
```

**A3. Thread `own_runtime_roots` into the guard.** Change the signature of
`_process_tree_workspace_policy_violation` (`cli_backend.py:10915-10924`) to accept a keyword:

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
        own_runtime_roots: frozenset[str] = frozenset(),   # NEW
    ) -> Optional[dict[str, Any]]:
```

**A4. Apply the downgrade in BOTH escape branches** (only where a fatal would otherwise be emitted):

- *PROCESS-CWD branch* (`cli_backend.py:11128-11150`): after computing `likely_backend_helper`, OR-in the
  own-runtime check before choosing severity:

```python
            if self._path_escapes_workspace(process_cwd, working_dir=workspace_root):
                likely_backend_helper = self._looks_like_backend_helper_workspace_policy_violation(
                    process_cwd=process_cwd,
                    path_tokens=path_tokens,
                    working_dir=workspace_root,
                )
                # NEW: a monitored command running from the rollout's OWN runtime
                # home/cache is the agent's own backend resolver — soft-deny, not
                # fatal. Genuine cross-task/sibling/upstream cwds are not in this set.
                own_runtime_cwd = self._path_under_own_runtime_root(
                    process_cwd, own_runtime_roots
                )
                downgrade = likely_backend_helper or own_runtime_cwd
                return {
                    "pid": pid,
                    "command_name": command_name,
                    "command": command,
                    "cwd": str(process_cwd) if process_cwd is not None else None,
                    "likely_backend_helper": downgrade,
                    "own_runtime_root": own_runtime_cwd,            # NEW telemetry field
                    "severity": "backend_helper" if downgrade else "fatal",
                    "reason": ( ... unchanged ... ),
                }
```

- *PATH-OPERAND branch* (`cli_backend.py:11172-11194`): mirror it for the resolved target token:

```python
                target_is_system_helper = self._path_resolves_to_system_helper_target(resolved_path)
                # NEW: targets inside the rollout's OWN runtime home/cache are the
                # agent's own resolver/extraction discovery — soft-deny, not fatal.
                target_is_own_runtime = self._path_under_own_runtime_root(
                    resolved_path, own_runtime_roots
                )
                likely_backend_helper = target_is_system_helper or target_is_own_runtime
                severity = "backend_helper" if likely_backend_helper else "fatal"
                ...
                return {
                    ...
                    "likely_backend_helper": likely_backend_helper,
                    "own_runtime_root": target_is_own_runtime,      # NEW telemetry field
                    "severity": severity,
                    "reason": ( ... unchanged ... ),
                }
```

> Note: the in-code comment at `11169-11171` ("Sibling task tempdirs, arbitrary /tmp traversal, and broad
> system paths remain fatal") stays **true** — `own_runtime_roots` contains only THIS rollout's resolved
> home/cache/uv/temp, never a sibling worktree, the ladder dir, or a fetched-package dir.

**A5. Compute and pass the roots at the single call site** (`_communicate_with_progress_timeout`).
Add a parameter and compute the roots once before the loop.

- Signature (`cli_backend.py:8542-8557`): add `workspace_policy_allowed_roots: frozenset[str] = frozenset(),`.
- Before the activity loop, hoist (so it is computed once, not every interval):
  `own_runtime_roots = workspace_policy_allowed_roots`.
- At the guard call (`cli_backend.py:8989-8993`), pass it:

```python
                violation = self._process_tree_workspace_policy_violation(
                    process_entries,
                    working_dir,
                    target_runtime_enforced=target_runtime_enforced,
                    own_runtime_roots=own_runtime_roots,            # NEW
                )
```

- At the **invocation** of `_communicate_with_progress_timeout` (`cli_backend.py:7662-7682`), compute from
  `launch_env` (always in scope at 7469; carries HOME/CODEX_HOME/XDG_CACHE_HOME unconditionally — not only in
  target-runtime mode, so this also fixes the **non-Docker local** pydantic case where `target_runtime_env`
  is None):

```python
                        communication_result = self._communicate_with_progress_timeout(
                            process,
                            working_dir=working_dir,
                            ...
                            workspace_policy_allowed_roots=_workspace_policy_own_runtime_roots(
                                launch_env, working_dir=working_dir
                            ),                                       # NEW
                        )
```

### Part B — env-layer containment (secondary hardening; keeps codex's resolver/temp in a recognized root)

Two cache/temp env keys are not currently pinned, so codex/uv fall back to `~/.cache` and the system temp,
which escape the workspace AND the own-runtime set unless we set them. Pin them inside the apex-managed home
in `_relocate_cli_home_for_target_runtime` right after the XDG block (`cli_backend.py:6133-6143`):

```python
    env.update(
        {
            "HOME": str(home_path),
            "XDG_CONFIG_HOME": str(home_path / ".config"),
            "XDG_CACHE_HOME": str(home_path / ".cache"),
            "XDG_DATA_HOME": str(home_path / ".local/share"),
            "XDG_STATE_HOME": str(home_path / ".local/state"),
            # NEW: pin uv cache + temp inside the apex-managed home so codex's
            # package resolver/extraction and any scratch tempfiles stay under a
            # recognized runtime root (home_path is under .apex_agent_runtime/),
            # never ~/.cache or the system /tmp. setdefault so an operator-set
            # value still wins.
        }
    )
    (home_path / ".cache" / "uv").mkdir(parents=True, exist_ok=True)
    (home_path / "tmp").mkdir(parents=True, exist_ok=True)
    env.setdefault("UV_CACHE_DIR", str(home_path / ".cache" / "uv"))
    env.setdefault("TMPDIR", str(home_path / "tmp"))
```

This makes Part A's job easy even when the home *isn't* under `.apex_agent_runtime/` (the apex_omega cell
layout), because (a) `UV_CACHE_DIR`/`TMPDIR` are now explicit env values Part A picks up, and (b) the codex
backend's `fastzip-castree-` extraction helper already matches an existing marker. Note Part B's
`setdefault` means it never overrides the apex_omega cell wiring if that layer chooses to set these itself.

---

## 4. How isolation / anti-cheat is preserved (not weakened)

1. **No new readable surface that could false-solve.** The downgrade is applied ONLY to paths that resolve
   under THIS rollout's own provisioned HOME/cache/uv/temp — directories Apex itself created for the agent.
   They contain no candidate source and no upstream package. The **worktree-shadow** (the worktree shadows
   `site-packages`, per `apex_omega/autogen/context.py:182-188`) plus **no-network** already guarantee that
   even a *fully* fetched package cannot produce a false solve; reading the agent's own uv cache is strictly
   weaker than that already-tolerated case.
2. **Sibling worktrees, the ladder dir, `~/.cache`, and fetched `*_upstream`/`*_wheel`/site-packages copies
   stay FATAL.** None of those are in `own_runtime_roots` (which is exactly the resolved env-key set, with
   the workspace and `/` defensively excluded). The 45/416 = 11% genuine-cheat aborts (e.g.
   `find /private/tmp/pydantic_upstream …`) are unaffected — they keep aborting.
3. **Deny, don't grant.** A downgraded violation is still a *recorded deny*: it routes to the existing SOFT
   path (`8995-9019`) — appended to `soft_policy_violation_log`, logged
   (`Workspace policy soft-violation (downgraded)`), and surfaced via `_emit_progress("policy_violation_soft")`.
   The command is read-only and the rollout simply continues. This is the **same** mechanism already used for
   `backend_helper`/`blocked_by_policy` today; we are widening *which paths* qualify, not inventing a new
   relaxation.
4. **Telemetry untouched.** The independent `integrity_log.jsonl` `sandbox_escape` records (242 in the LIVE
   cell) are written regardless of severity, so the genuine-cheat signal is preserved at full fidelity. We
   add a `"own_runtime_root": bool` field to the violation dict so downgraded-vs-genuine is now *machine-
   distinguishable* in the audit (a net improvement for cheat analysis).
5. **Determinism / replay.** `own_runtime_roots` is a pure function of `launch_env` + `working_dir`, both
   fixed for a given rollout; `Path.resolve(strict=False)` is deterministic. No clock, no network, no
   randomness, no filesystem-state dependence (we resolve paths textually; we do not stat for existence in
   the predicate). Replaying the same env + process tree yields the same verdict.

---

## 5. Why this unblocks pydantic

The IMPACT analysis attributes the 416 fatal aborts as: 11% genuine cheat, **37% codex-internal**
(codex's own uv cache `/cells/<mod>/runtime/home/.cache/uv`, the runtime `.venv/site-packages`,
`fastzip-castree`, and `~/.cache`), 50% benign-env, 11% cross-cell, 2% mis-parse. Pydantic's evidence cell
(`hybrid-nogate__pydantic__s0`, 7/8 repair waves fatal) is dominated by exactly the codex-internal class:
the aborting `find` is "running from" / targeting the codex uv cache and runtime home that sit at
`cells/<mod>/runtime/home/…`.

- **Part A** moves every codex-internal `find`/`grep` that operates inside the rollout's own HOME/CODEX_HOME/
  XDG cache/uv cache/TMPDIR from `fatal` → `backend_helper` (SOFT). Those waves now **continue** instead of
  aborting at tokens=0, so pydantic repair rollouts can actually run and accumulate diff/work.
- **Part B** additionally captures the uv-cache + temp that aren't yet pinned, so even a codex resolver run
  that would have spilled to `~/.cache` now stays inside a root Part A recognizes.

`~/.cache` (the user home cache, part of the codex-internal 37%) is reached **only if** codex ignores
`XDG_CACHE_HOME`/`UV_CACHE_DIR`; Part B's `UV_CACHE_DIR` + the existing `XDG_CACHE_HOME` redirect that
traffic into the contained cache, and if a stray `~/.cache` read still occurs Part A will *not* cover it
(it's not in the rollout's env roots) — that residual is intentionally left to the companion
`allowed-roots-broaden` design (which decides whether the shared host cache is allowlisted). So this design
unblocks the **dominant** pydantic failure mode (codex-internal own-home discovery) deterministically, and
explicitly scopes out the cross-cell / sibling / ladder-dir classes so it cannot accidentally weaken those.

Expected effect on the headline metric: pydantic's 175/617 = 28% tokens=0 abort rate should drop to roughly
the residual non-own-home fraction; the other repos (jinja 21%, networkx 15%, …) benefit proportionally to
their codex-internal share.

---

## 6. Risks and mitigations

| # | Risk | Likelihood | Mitigation |
|---|------|-----------|------------|
| 1 | An env value is empty or `/` and accidentally whitelists the whole FS. | Low | `_workspace_policy_own_runtime_roots` explicitly drops `""` and `/`, and drops the workspace itself; `resolve(strict=False)` of `""` → cwd, also excluded. Unit-tested. |
| 2 | A genuine cheat happens to live *inside* the rollout's own home (agent writes an upstream copy into `$HOME` then reads it). | Very low | Even then it cannot false-solve (worktree-shadow + no-network). The write itself is recorded; and a fetched copy under `$HOME` is functionally identical to today's already-tolerated `site-packages` shadow case. |
| 3 | Symlink in the worktree points into the home, blurring the boundary. | Low | `resolve(strict=False)` canonicalizes symlinks before the `relative_to` check, so the *real* target governs; consistent with the existing `_path_escapes_workspace` logic. |
| 4 | New kwargs break an external caller of `_process_tree_workspace_policy_violation` / `_communicate_with_progress_timeout`. | Low | Both new params are keyword-only with `frozenset()` defaults → 100% backward compatible; default behavior (no allowed roots) is byte-identical to today. |
| 5 | Part B's `TMPDIR` relocation breaks a tool that needs the host temp. | Low | `setdefault` (operator/cell wiring wins); the dir is created first; codex already runs with a relocated XDG cache, so a relocated temp is in the same regime. Covered by the full suite. |
| 6 | Vendored-file drift: a future v1 re-vendor overwrites the edit. | Medium | Edits are git-tracked on this branch; the change is localized to 1 file and 4 small sites, easy to re-apply. Document in memory. |

Out of scope (explicitly NOT addressed here, to keep the change minimal and the cheat-signal intact):
sibling-worktree cross-cell aborts (11%), ladder-dir / `/` broad traversal (part of benign-env 50%),
`~/.cache` host shared cache, and the sed/awk arg mis-parse (2%). Those are separate designs.

---

## 7. Test plan

### 7.1 New unit test — `tests/test_workspace_policy_guard.py` (the guard predicate, no subprocess)

Import the real class: `from apex.core.cli_backend import CLIModelClient, _workspace_policy_own_runtime_roots`
(import verified working under `.venv_omega`). Predicates are static / pure, so we test them directly
against synthetic process-tree dicts — no codex, no network, fast, replay-safe.

1. **`test_own_runtime_root_set_excludes_workspace_and_root`** — `_workspace_policy_own_runtime_roots`
   with `HOME=/tmp/cell/runtime/home`, `XDG_CACHE_HOME=/tmp/cell/runtime/home/.cache`,
   `UV_CACHE_DIR=/tmp/cell/runtime/home/.cache/uv`, `TMPDIR=/tmp/cell/runtime/home/tmp`,
   `working_dir=/tmp/cell/repo`. Assert the four roots are present and the workspace + `/` are absent;
   assert empty env → empty set.
2. **`test_path_under_own_runtime_root`** — `_path_under_own_runtime_root` True for
   `…/runtime/home/.cache/uv/archive-v0/x`, False for `…/worktrees/wt_m730006`, `/private/tmp/<ladder>`,
   `/Users/x/.cache`, `/private/tmp/pydantic_upstream/y`.
3. **`test_find_in_own_uv_cache_downgrades_to_soft`** — build a `process_entries` dict with a `find`
   process whose `cwd` is the codex uv cache; call `_process_tree_workspace_policy_violation(...,
   own_runtime_roots=<set including that cache>)`; assert `severity == "backend_helper"` and
   `own_runtime_root is True`. Then call it with `own_runtime_roots=frozenset()` and assert
   `severity == "fatal"` (proves the downgrade is gated on the new set and the default is unchanged).
4. **`test_find_targeting_own_cache_operand_downgrades`** — same but the escape is via a **path operand**
   (`find <uv_cache_path> -name '*.py'`) with cwd inside the worktree → exercises the PATH-OPERAND branch.
5. **`test_sibling_worktree_stays_fatal`** — `find` targeting `…/worktrees/wt_m730006` while
   `own_runtime_roots` lists only this rollout's home → `severity == "fatal"` (anti-cross-contamination
   intact).
6. **`test_fetched_upstream_stays_fatal`** — `find /private/tmp/pydantic_upstream …` → `fatal` (genuine
   cheat unaffected).
7. **`test_default_kwargs_byte_identical`** — call the guard with NO `own_runtime_roots` kwarg on a fatal
   case and assert identical verdict to the explicit-empty-set call (backward-compat guarantee).

### 7.2 Integration / env-layer assertions

8. **`test_relocate_pins_uv_and_tmpdir`** — call `_relocate_cli_home_for_target_runtime` (or assert on the
   resulting `env`) for a `CODEX_CLI` config and confirm `UV_CACHE_DIR` and `TMPDIR` resolve **under** HOME,
   the dirs exist, and `setdefault` does not clobber a pre-set value.
9. **End-to-end smoke (manual / one cell):** re-run the pydantic evidence cell
   `hybrid-nogate__pydantic__s0` (or a 1-repo `LADDER_DIR=/tmp` slice) and assert: (a) repair-wave
   `finalization_status=policy_violation` aborts on codex-internal paths drop to ~0; (b)
   `soft_policy_violation_log` now contains the corresponding `own_runtime_root: true` entries; (c)
   `integrity_log.jsonl` still records the `sandbox_escape` telemetry (signal preserved); (d) at least one
   pydantic repair wave produces a committed diff (tokens > 0).

### 7.3 Full regression

Run the whole suite (the harness reports ~194-301 green depending on branch state):

```
cd /Users/sameertkhanna/Documents/agent_orch
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_omega/bin/python -m pytest tests/ -q -p xdist -n auto -p no:cacheprovider -o addopts=
```

Expect all existing tests green (the change is additive + default-inert) plus the new
`tests/test_workspace_policy_guard.py` cases.

---

## 8. Change inventory (exact edit sites)

| Site | File:lines | Change |
|------|-----------|--------|
| Constant + helper | `cli_backend.py:~6198` (near `_cli_sandbox_writable_roots`) | add `_WORKSPACE_POLICY_OWN_RUNTIME_ENV_KEYS` + `_workspace_policy_own_runtime_roots(env, working_dir)` |
| Predicate | `cli_backend.py:~10902` (near `_path_resolves_to_system_helper_target`) | add staticmethod `_path_under_own_runtime_root(path, own_runtime_roots)` |
| Guard signature | `cli_backend.py:10915-10924` | add kw-only `own_runtime_roots: frozenset[str] = frozenset()` |
| Guard PROCESS-CWD branch | `cli_backend.py:11128-11150` | OR-in own-runtime downgrade; add `own_runtime_root` telemetry field |
| Guard PATH-OPERAND branch | `cli_backend.py:11172-11194` | OR-in own-runtime downgrade; add `own_runtime_root` telemetry field |
| Communicate signature | `cli_backend.py:8542-8557` | add kw-only `workspace_policy_allowed_roots: frozenset[str] = frozenset()` |
| Communicate guard call | `cli_backend.py:8989-8993` | pass `own_runtime_roots=...` |
| Call site | `cli_backend.py:7662-7682` | pass `workspace_policy_allowed_roots=_workspace_policy_own_runtime_roots(launch_env, working_dir=working_dir)` |
| Env containment | `cli_backend.py:6133-6143` | `setdefault` `UV_CACHE_DIR` + `TMPDIR` under `home_path`; mkdir both |
| New test | `tests/test_workspace_policy_guard.py` | 9 cases (§7) |

Total: one production file, ~60 net new lines (mostly the helper + predicate + test), all additive and
default-inert.
