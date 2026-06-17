# NEXT STEPS — Fair Re-run of the commit0 Eval Suite

> **Goal:** re-run the **entire** (arm × repo) matrix under the **final code**, so every
> result is a proper **apples-to-apples** comparison. Prior results were produced under a
> mix of pre/post-fix conditions (some before container sanitization, before the uv-python
> repair, etc.), so they are **not** directly comparable. Wipe and re-run clean.

**State as of 2026-06-15 (all green, ready to run):**
- Disk: **~291 GB free (34% used)** — the chronic disk crisis is fixed (it was a `watchman`
  deleted-open-file leak holding ~290 GB; restarting `watchman`+`scribe_ca` reclaimed it).
- uv `python3.10` exec-bit bug: **ROOT-CAUSED + FIXED.** `_strip_checkout` was doing
  `chmod(0o600)` on venv symlinks, which `Path.chmod` follows into the SHARED uv interpreter,
  stripping its exec bit and fast-failing all later cells (`total=0`). Fix: never chmod
  symlinks (`scripts/run_ladder.py`) + regression test `tests/test_strip_checkout.py`. The
  old static `chmod +x` band-aid is no longer needed.
- Container sanitizer: **wired** into the shared `_prepare_repo` (all arms).
- Fail-open-to-template: **dropped** (autogen stands alone).
- `SAFE_BUILTINS`: **60** (incl `type`) — autogen sandbox fix.
- Test suite: **76/76** green.

---

## ▶ DO THIS (ordered)

### 1. Pre-flight (30 sec — confirm nothing regressed)
```bash
cd /Users/sameertkhanna/Documents/agent_orch
VENV=/Users/sameertkhanna/Documents/apex/apex/.venv/bin/python

df -h /System/Volumes/Data | tail -1                 # want >20 GB free
stat -f '%Sp' /Users/sameertkhanna/.local/share/uv/python/cpython-3.10.20-macos-aarch64-none/bin/python3.10  # want -rwxr-xr-x
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $VENV -m pytest tests/ -q -p no:cacheprovider -o addopts= | tail -3  # want 74 passed
```

### 2. Clean slate (archive old results, keep accumulated observations)
```bash
cd /Users/sameertkhanna/Documents/agent_orch
TAG=$(date +%Y%m%d-%H%M%S)
mkdir -p runs/archive
mv runs/ladder "runs/archive/ladder_prererun_$TAG"
mkdir -p runs/ladder
# preserve the append-only signal observations across the re-run:
cp "runs/archive/ladder_prererun_$TAG/signals_log.jsonl" runs/ladder/ 2>/dev/null || true
```

### 3. Launch the full re-run (resumable, background)
```bash
cd /Users/sameertkhanna/Documents/agent_orch
APEX_OMEGA_PYTHON=/Users/sameertkhanna/Documents/apex/apex/.venv/bin/python \
LADDER_CONCURRENCY=6 LADDER_MIN_FREE_MB=1200 \
PYTHONPATH=. \
nohup /Users/sameertkhanna/Documents/apex/apex/.venv/bin/python scripts/run_ladder.py \
  >> runs/ladder/runner_stdout.log 2>&1 &
echo "runner PID $!"
```
- **17 cells**: `{B0_codex_1shot, baseline_v1_k8, omega_template_k8, omega_autogen_k8}` ×
  `{voluptuous, jinja, mimesis, pydantic}` + `B2_v1_fullcap16` on voluptuous. K=8.
- Resumable: re-running the same command **skips** finished cells, re-runs the rest.
- **Concurrency 6** is the sweet spot for the 17-cell matrix (ceil(17/6)=3 waves, same as 8,
  but lets all 5 final/heavy cells — 4×pydantic + B2 — run concurrently in the last wave).
  Safe now that the strip bug is fixed (it no longer mutates shared state). 14 cores / 48 GB /
  ~288 GB free; load ~16–23 is fine (agents are I/O-bound on the model API). Drop to 4–5 if
  disk/CPU ever tightens (`pydantic` builds `pydantic-core` with Rust → heaviest cell).

### 4. Monitor
```bash
tail -f runs/ladder/runner_stdout.log          # live cell completions
cat runs/ladder/progress.jsonl                 # one line per cell
PYTHONPATH=. $VENV scripts/track_signals.py | sed -n '1,40p'   # live ledger at any milestone
```

### 5. Finalize (when all 17 cells done)
```bash
PYTHONPATH=. $VENV scripts/track_signals.py            # -> runs/ladder/SIGNALS_LEDGER.md
PYTHONPATH=. $VENV scripts/capture_autogen_evidence.py # -> runs/ladder/autogen_evidence.md
```
Then read **`runs/ladder/SIGNALS_LEDGER.md`** — it has the verdict:
solve-rate per arm, **agents/solve** (efficiency), failure-class mix, and the
**autogen-vs-template head-to-head** (watch the `AUTOGEN_WON` count).

---

## Fairness invariants (what makes this apples-to-apples)
Every cell, every arm, now runs under **identical** conditions:
1. **Sanitized container** — `apex_omega/eval/repo_sanitize.py` runs inside the shared
   `_prepare_repo`, so no arm can see the upstream version / fetch the real release
   (version literals → `0.0.0`, upstream URLs blanked, CHANGELOG/egg-info removed, git
   tags+remotes stripped, committed onto `apex-base` so worktrees inherit it).
2. **Autogen stands alone** — no fail-open-to-template; an abstain is a real autogen failure.
3. **Same budget** — K=8 agents across all orchestrated arms.
4. **Same interpreter** — repaired uv `python3.10`.
5. **Execution-authoritative scoring** — the real pytest gate decides acceptance, never a soft score.

## What changed since the last (partial) run — why the re-run is needed
- **Container sanitization** added (the big one — earlier cells saw upstream version/URLs).
- **uv `python3.10`** exec bit repaired (earlier cells errored `total=0`).
- **`SAFE_BUILTINS`** expanded (autogen `type()` NameError fixed).
- **fail-open-to-template** dropped (autogen comparison now honest).
- **Runner hardened** (disk preflight + per-cell checkout strip).
- **Failure classifier** distinguishes `strategy_sandbox_block` vs `genuine_abstain`.

## File map
- `scripts/run_ladder.py` — the matrix runner (ARMS, REPOS, concurrency, preflight, strip).
- `scripts/track_signals.py` — regenerates `runs/ladder/SIGNALS_LEDGER.md` (+ `--note` to append).
- `scripts/capture_autogen_evidence.py` — per-cell autogen failure evidence.
- `apex_omega/eval/repo_sanitize.py` — the container sanitizer (unit-tested).
- `apex/evaluation/commit0_benchmark.py::_prepare_repo` — where sanitizer is wired (~line 13620).
- `runs/ladder/` — live results; `runs/ladder_pre_sanitize/` — mimesis before-state (kept).
- `runs/archive/` — archived prior runs.
- Memory: `~/.claude/projects/-Users-sameertkhanna-Documents-agent-orch/memory/` (see MEMORY.md).

## Watch-outs / known issues
- **TMPDIR leak**: commit0 prep leaves `apex-commit0-*` dirs in `$TMPDIR`
  (`/private/var/folders/.../T`); ~982 had accumulated (~6 GB). Clear periodically:
  `rm -rf /private/var/folders/7z/_f9zcsy940l2dvfmjwsmd_kc0000gn/T/apex-* 2>/dev/null`
  (only when the runner is stopped). **TODO:** patch prep to auto-clean its temp.
- **networkx + cookiecutter**: dropped — require Docker (not runnable in this local setup).
- **pydantic**: heaviest cell (Rust build); if a cell errors fast (`total=0`/few-sec), read
  the cell's `autogen_cell_error.json`. The old "uv `python3.10` exec bit got stripped" cause
  is fixed (see strip fix above); if `Failed to query Python interpreter` recurs, the
  `_strip_checkout` symlink guard regressed — check `tests/test_strip_checkout.py`. (Note: a
  benign Codex `meta-approved.rules` "Permission denied" appears in `run_manifest.json` health
  snapshots — unrelated, non-fatal, not the uv bug.)
- **Disk daemon leak**: if free space silently drops again, check
  `sudo lsof -nP +L1 | awk '$7~/^[0-9]+$/{s+=$7}END{print s/1e9" GB"}'` — restart
  `watchman` (`watchman shutdown-server`) if it's hoarding deleted files.

## Scope note (full 15-repo set vs this ladder)
Original target set was 15 repos; this ladder uses the 4 cleanly-local-runnable ones
(voluptuous, jinja, mimesis, pydantic) for the interpretable comparison. To widen later,
add repos to `REPOS` in `scripts/run_ladder.py` (Docker-dependent ones need a Docker backend).
