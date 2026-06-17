# APEX-Ω

A vendor-neutral, deterministic **dynamic-workflow engine** — orchestration-as-code
that spawns isolated coding-agent workers (Codex, Claude Code, or both in one run),
holds intermediate state in script variables + a durable journal (never a chat
window), and converges through execution-grounded verify-and-select — built on
APEX v1's execution-authoritative kernel as the hardened substrate.

This repo implements **Phase 0** of `APEX_NEXTGEN_PLAN.md` (the engine + normalized
executor + durable journal), plus the commit0 **evaluation + ablation harness** over
the 15 target repos, reusing the v1 harness at `~/Documents/apex/apex` as a library.

## Layers (plan §8)

| Layer | Package | What it is |
|---|---|---|
| L0 engine | `apex_omega/engine` | `agent / parallel / pipeline / phase / budget` — orchestration-as-code. `pipeline()` is the net-new per-item streaming primitive (Fusion Ledger A2). |
| L1 executor | `apex_omega/executor` | Normalized `Executor` over codex/claude/gemini/opencode, **wrapping** v1 `CLIModelClient`; ACP-style capability negotiation; degrade-not-crash (A10). |
| L2 kernel | `apex_omega/kernel` | Cardinal Safety Contract: deterministic ranking tuple + monotone (`True→False`-only) acceptance; the only producer of `accepted=True` is execution evidence (A3). |
| journal | `apex_omega/journal` | Durable input-hash WAL resume — survives `kill -9`, re-runs only edited/new calls (A9, genuinely net-new). |
| isolation | `apex_omega/isolation` | Per-rollout git-worktree isolation, lock-before-touch, Cardinal-safe release (A5). |
| ablation | `apex_omega/ablation` | Fail-open flag surface (`AblationConfig`), `SafetyModeConfig` admission rules (§18.4.1), the A1–A11 + baseline + negative-control arm matrix. |
| eval | `apex_omega/eval` | commit0 driver (Mode A: drives v1's proven `Commit0BenchmarkRunner` per arm) + the 15-repo registry + v1 scoring reuse. |
| workflows | `apex_omega/workflows` | The reference best-of-N program (Mode B: engine-native, drives real workers in worktrees, scores by execution, selects under the Contract). |

## Five invariants (carried verbatim)

filesystem-as-source-of-truth · execution-evidence-authoritative selection ·
fail-loud-never-fake · durable resumable journaling · vendor neutrality.

## Running

Use the v1 venv python (it can `import apex` + `import commit0`) with this repo on
`PYTHONPATH`:

```bash
cd /Users/sameertkhanna/Documents/agent_orch
VENV=/Users/sameertkhanna/Documents/apex/apex/.venv/bin/python

# cheap preflight (no paid calls): imports, Docker/vendor status, arm+config loads, discovery
PYTHONPATH=. $VENV -m apex_omega doctor

# list the ablation matrix and the target repos
PYTHONPATH=. $VENV -m apex_omega arms
PYTHONPATH=. $VENV -m apex_omega repos

# free engine-native best-of-N demo (worktree isolation + journal + real scoring + select)
PYTHONPATH=. $VENV -m apex_omega bestofn-demo --k 3

# run the commit0 ablation matrix (Mode A — PAID vendor calls)
PYTHONPATH=. $VENV -m apex_omega eval \
    --arms baseline,A1_adaptive_k,A8_vendor_mix \
    --repos voluptuous,minitorch \
    --limit 1 --rollouts 1 \
    --run-dir runs/exp1 --local-only
```

The eval matrix is **journaled per (arm, repo) cell**, so a killed run resumes and
re-runs only incomplete cells. Each cell forces the local no-Docker scoring path
(`local_pytest_json_report`, `commit0_docker_runtime_mode=never`).

## The 15 target repos

`minitorch jinja voluptuous web3.py statsmodels babel pydantic pytest networkx
mimesis scrapy seaborn sphinx geopandas cookiecutter` — all present in the cached
`wentingzhao/commit0_combined` dataset.

- **Locally runnable (no Docker, 12):** minitorch, jinja, voluptuous, statsmodels,
  babel, pydantic, pytest, networkx, mimesis, seaborn, geopandas, cookiecutter.
- **Need Docker (3, apt-get pre_install):** web3.py (clang), scrapy (libxml2/libxslt),
  sphinx (graphviz). Run these once the Docker daemon is up.
- `pytest` uses dataset fallback revision `afc4d5f9…` (handled by the registry).

## Ablation matrix (`apex_omega/ablation/arms.py`)

Baselines **B0/B2/B4**, ablations **A1–A11** (one mechanism each, fail-open to the
v1/heuristic baseline), and four **negative controls** that deliberately violate an
invariant to demonstrate degradation (static-CTDG-gate, raw share-all,
thin-executor-everywhere, Cardinal-relaxation). Arms with a faithful v1 flag mapping
carry a `v1_overlay` and run via Mode A; the rest are exercised by the engine-native
path. `SafetyModeConfig` makes it *impossible* to silently ship a rejected form
(share-all needs an explicit opt-in; there is no plan-score "gate" enum at all).

## Status

Phase-0 engine, journal, executor, kernel, isolation, ablation, eval driver, and the
reference workflow are implemented and validated: `tests/` (32 tests) covers resume
(same-process + across-restart + edited/changed-snapshot re-run), pipeline
determinism, parallel null-on-fail, budget, the ranking/abstention/monotone-downgrade
contract, SafetyModeConfig admission, and capability negotiation. The free
`bestofn-demo` validates the full engine-native loop; `eval` runs real commit0 slices.

Not yet built (deferred per the plan): the learned controller (Phase 3, default-off /
fail-open), GEPA/RL stages, and the held-out-vendor training harness — the engine
ships as a hardened, vendor-neutral best-of-N substrate regardless.
