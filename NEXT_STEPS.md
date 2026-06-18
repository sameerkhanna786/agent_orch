# NEXT STEPS — commit0 eval re-run (BLOCKER FIXED, isolation restored)

> **State as of 2026-06-17 (session 3):** the 3-session 0-token blocker is **ROOT-CAUSED + FIXED + VALIDATED**,
> the fix is committed+pushed (`c601e22`, `db706ee` on origin/main), the credential read-jail is **restored**,
> and a clean **n≥3-seed** apples-to-apples re-run is the remaining work. The eval is **runnable**.

## TL;DR status
- **Tests:** ~194 passing (196 `def test_`; ~2 skipped/gated). `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 <venv> -m pytest tests/ -q -p no:cacheprovider -o addopts=`
- **Blocker:** codex rolled out under TWO stacked sandboxes → 0 tokens. `os error 2` (ENOENT) = the outer `sandbox-exec` read-jail wrap **only when the worktree is under `~/Documents`** (macOS TCC × seatbelt). `os error 1` (EPERM, in-process app-server) = the Meta codex *launcher* seatbelt when the launcher flag was missing.
- **Fix:** `c601e22` (bypass branch adds `--dangerously-disable-osx-sandbox`) + `db706ee` (`cell_done()` no longer counts partial Mode-A cells as done). Validated: voluptuous solved 1/1 multiple ways.
- **Isolation:** the read-jail is **fully usable** — run jail-ON with the worktrees **outside `~/Documents`** (`LADDER_DIR` in `/tmp`). Proven: jail-ON + `/tmp` worktree → solved 1/1, `oserr2=0 oserr1=0`, no lockout. (Mode-C codex worktree lives at `LADDER_DIR/worktrees/wt_*`; Mode-A workspace is `mkdtemp` under `$TMPDIR=/var/folders`, already off `~/Documents`.)

## ▶ DO THIS — launch the n≥3 jail-ON re-run (resumable, background)
```bash
cd /Users/sameertkhanna/Documents/agent_orch
VENV=/Users/sameertkhanna/Documents/apex/apex/.venv/bin/python
LADDER_DIR=/tmp/omega_ladder \
APEX_HOST_CLI_READ_JAIL=1 APEX_CODEX_BYPASS_SANDBOX=0 \
APEX_OMEGA_SKIP_AUTH_PREFLIGHT=1 \
LADDER_SEEDS=3 LADDER_CONCURRENCY=2 LADDER_MIN_FREE_MB=2000 \
APEX_OMEGA_PYTHON=$VENV PYTHONPATH=. \
nohup $VENV scripts/run_ladder.py >> /tmp/omega_ladder/runner_stdout.log 2>&1 &
echo "runner PID $!"
```
- **`LADDER_DIR=/tmp/omega_ladder` is required for jail-ON** (keeps Mode-C worktrees off the TCC-protected `~/Documents`). Results live in `/tmp`; **archive evidence to `runs/` at the end** (see Finalize).
- **`APEX_HOST_CLI_READ_JAIL=1 APEX_CODEX_BYPASS_SANDBOX=0`** restores the credential read-jail (denies `~/.ssh .aws .gnupg .azure .config/{gcloud,gh} .git-credentials .netrc .npmrc .pypirc .password-store`, `~/Documents ~/Desktop ~/Downloads`, `/Volumes /Users/Shared`). The read_jail branch wins; the bypass flag is ignored (no broken dual-flag combo).
  - **Simpler alternative (no isolation):** omit the two jail vars and `LADDER_DIR` — `run_cell()` setdefaults jail-OFF + bypass and runs codex fully unsandboxed into `runs/ladder` (persistent). Fine on a single-user dev box; loses the credential read-jail.
- **Concurrency = 2** is deliberate, NOT conservative: the OMEGA arms are UNBOUNDED, so each cell can fan out to the engine's within-cell cap (`max_concurrent` ≈ 12 on 14 cores); 2 cells × 12 ≈ 24 heavy codex procs is the binding constraint on 48 GB RAM. Bump to 3 on resume only if RAM/load headroom is confirmed.
- **n≥3 seeds:** `LADDER_SEEDS=3` → 63 cells (`{label}__{repo}__s{0,1,2}`), one bounded pool (same peak load as 1 seed, just more cells), seed-major so a full matrix lands per seed. Resumable: re-run the same command to skip finished cells.

## Monitor
```bash
tail -f /tmp/omega_ladder/runner_stdout.log        # live cell completions
cat /tmp/omega_ladder/progress.jsonl               # one line per cell
# health probe (no errors, real tokens):
for f in $(find /tmp/omega_ladder -name calls_wal.jsonl); do echo "$f oserr=$(grep -c 'os error' $f)"; done
```

## Finalize (when all 63 cells done)
```bash
LADDER_DIR=/tmp/omega_ladder PYTHONPATH=. $VENV scripts/track_signals.py            # -> SIGNALS_LEDGER.md
LADDER_DIR=/tmp/omega_ladder PYTHONPATH=. $VENV scripts/capture_autogen_evidence.py # -> autogen_evidence.md
# persist results off /tmp:
mkdir -p runs/ladder_n3 && cp -R /tmp/omega_ladder/* runs/ladder_n3/
```
Read `SIGNALS_LEDGER.md`: solve-rate per arm **with variance across the 3 seeds + pass@k**, **agents/solve** (cost), failure-class mix, and the autogen-vs-template head-to-head.

## The matrix (per seed)
5 arms × 4 repos + 1 EXTRA = 21 cells/seed → 63 at n=3.
- **Arms:** `B0_codex_1shot` (1-shot), `baseline_v1_k8` (best-of-8), `ralph_wiggum_loop` (vanilla iterate-until-done), `omega_template_unbounded`, `omega_autogen_unbounded` (`--autogen-author`, scout 3). + `B2_v1_fullcap16` on voluptuous (cost-pathology witness).
- **Repos:** voluptuous (easy), jinja, mimesis, pydantic (hard). networkx/cookiecutter dropped (need Docker).
- **Budget is intentionally UNEQUAL:** omega arms are UNBOUNDED (`_OMEGA_MAX=1000` + plateau governor + 1000-agent backstop) vs B0 1-shot / baseline best-of-8. **Always report agents/solve alongside solve-rate.**

## Scoring fairness (exclude infra non-results from denominators)
Three distinct `infra_nonresult` families exist — do NOT count any as a real failure:
1. codex `os error 2` after AI Gateway = the sandbox blocker (now fixed; should be absent).
2. `repository discovery outside the rollout workspace` = a workspace-jail guard (strategy-side).
3. `heartbeat_timeout` = `runtime.py` per-agent watchdog.
A cell is an apples-to-apples solve only if its **winning candidate had non-zero usage** (`GRID_RECLASSIFIED.md` method).

## Watch-outs
- **`/tmp` ephemerality:** jail-ON results live in `/tmp/omega_ladder`; `/private/tmp` survives the session but a reboot/periodic-cleanup can purge it — finalize/copy to `runs/` when done. (Use jail-OFF → `runs/ladder` if you need persistence without the archive step.)
- **TMPDIR leak:** commit0 prep leaves `apex-commit0-*` + `apexomega_*` dirs in `$TMPDIR`. Clear when the runner is stopped: `rm -rf /var/folders/7z/_f9zcsy940l2dvfmjwsmd_kc0000gn/T/apex-* /var/folders/7z/_f9zcsy940l2dvfmjwsmd_kc0000gn/T/apexomega_* 2>/dev/null`
- **Disk daemon leak:** if free space silently drops, `sudo lsof -nP +L1 | awk '$7~/^[0-9]+$/{s+=$7}END{print s/1e9" GB"}'`; restart `watchman` if it hoards deleted files.

## File map
- `scripts/run_ladder.py` — matrix runner (ARMS/REPOS, jail-OFF setdefaults, `cell_done`, strip, seeds).
- `apex/core/cli_backend.py` — codex launch + the seatbelt read-jail (`_host_cli_read_jail_*` ~:3601-3760, codex branches ~:11564/11578/11586).
- `apex_omega/eval/commit0_autogen.py` — Mode-C autogen cell (WorktreeProvider worktree under run-dir).
- `apex_omega/isolation/worktree.py` — mints worktrees off the source repo.
- `scripts/track_signals.py`, `scripts/capture_autogen_evidence.py` — finalization.
- Memory: `~/.claude/projects/-Users-sameertkhanna-Documents-agent-orch/memory/` (see `MEMORY.md`; `eval-zero-token-blocker` has the full root-cause).
