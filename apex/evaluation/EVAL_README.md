# APEX TestGenEvalLite — Evaluation Pipeline (V4)

This document is the operator's guide for running TestGenEvalLite end-to-end
through the V4 quality pipeline. Everything described here is checked-in
code; nothing depends on a one-off external orchestrator.

## What "V4" wires up

| ID | Capability | Module | Active by default? |
|----|---|---|---|
| W1 | Final whole-file acceptance gate | `final_acceptance_gate.py`, `_apply_default_final_acceptance_gate` in `testgeneval_benchmark.py` | YES (`APEX_FINAL_ACCEPTANCE_GATE=1`) |
| W2 | Atomic acceptance verifier | `atomic_acceptance.py`, `_verify_deterministic_repair` in `testgeneval_benchmark.py` | YES |
| W3 | AST roundtrip + parse gate | `code_emission.py`, called from `validate_static_artifacts` | YES |
| W4 | Execution-grounded oracle capture | `oracle_capture.py`, `oracle_repair.py` | YES (proactive: `APEX_PROACTIVE_ORACLE_REPAIR=1`; also fires as repair attempt 2) |
| W5 | Import + signature preflight | `import_preflight.py`, `signature_preflight.py`, `import_validate_python_artifacts` | YES |
| W6 | Repo-context probe (fixtures, conftest, isolation markers) | `repo_context.py` | YES |
| W7 | Hierarchical gap-fill (one extra test per uncovered focal symbol) | `hierarchical_gap_fill.py` | OPT-IN (`APEX_HIERARCHICAL_GAP_FILL=1`) |
| W8 | Diversified, scope-shrinking deterministic repair | `repair_strategies.py`, `_try_deterministic_repair` | YES |
| W9 | Mutation-targeted assertion shape (pytest.approx for floats, assert_allclose for ndarrays) | `mutation_targeting.py` consumed by `oracle_repair.py` | YES |
| W10 | Join harness output into per-task records | `run_artifacts.py:join_harness_results_into_records` | YES (called by the runner) |
| W11 | Mutation-timeout exposure to the official harness | `upstream_patches/testgeneval/mutation_timeout_argument.patch` + CLI `--mutation-timeout-seconds` | YES |
| W12 | Stratified report (per-runner, per-repo, mutation-completeness, drop counts) | `run_artifacts.py:write_testgen_run_report` | YES |

The two workstreams that are NOT fully active by default:

- **W7** is opt-in because each gap-fill costs one extra LLM call per
  uncovered focal symbol. Enable with `APEX_HIERARCHICAL_GAP_FILL=1` when
  budget allows.
- **W4 proactive** can be disabled with `APEX_PROACTIVE_ORACLE_REPAIR=0`
  if a project's call sites are too expensive to invoke during generation.

## End-to-end run procedure

The pipeline has three stages: prepare predictions, score against the
official harness, render the run report. Each stage has a dedicated
checked-in entry point.

### Stage 1: Generate predictions through the V4 pipeline

```
python -m apex.evaluation.runners.testgenevallite_generate \
    --output-dir .apex_testgeneval_lite_$(date +%Y%m%d) \
    --model-name apex-v4 \
    --parallelism 8 \
    --generation-timeout-seconds 300 \
    --pytest-timeout-seconds 120 \
    --max-repair-attempts 3 \
    --candidate-count 3 \
    --require-target-environment \
    --docker-official-repo .apex_testgeneval_lite_20260503_085612/official_repo_mount
```

Optional flags:

- `--from-json /path/to/local-tasks.json` — bypass the HuggingFace fetch.
- `--task-id INSTANCE_ID` (repeatable) — restrict to specific tasks.
- `--limit N` — cap task count for smoke runs.
- `--measure-mutation` / `--measure-coverage` — enable extra quality probes
  only when no target adapter is bound; official benchmark scoring remains
  authoritative.
- `--require-target-environment` — fail records instead of falling back to
  host dynamic execution when a benchmark/project environment is not bound.

This stage:

1. Loads the kjain14/testgenevallite tasks.
2. For each task, runs `evaluate_testgeneval_task_with_default_generator`,
   which threads the artifact through W3 → W5 → W6 → W4 → W7 → W8 → W1 with
   W2 verification.
3. Writes `preds/<model>__testgenevallite__0__test.jsonl` in the format
   the official `run_evaluation.py` consumes.
4. Writes per-task `records/<id>.json` for the W10 join.
5. Writes `run_manifest.json` and `generation_summary.json`.

### Stage 2: Score with the official harness (W11 mutation timeout exposed)

```
python -m apex.evaluation.runners.testgenevallite \
    --official-repo /path/to/kjain14/testgeneval-fork \
    --predictions-jsonl .apex_testgeneval_lite_$(date +%Y%m%d)/preds/apex-v4__testgenevallite__0__test.jsonl \
    --output-dir .apex_testgeneval_lite_$(date +%Y%m%d) \
    --task-parallelism 16 \
    --timeout-seconds 900 \
    --mutation-timeout-seconds 14400 \
    --docker-namespace kdjain
```

This:

1. Applies the W11 upstream patch
   (`upstream_patches/testgeneval/mutation_timeout_argument.patch`) so the
   harness honors `--mutation_timeout`. The patch is idempotent.
2. Invokes `run_evaluation.py` with the right host flags. The mutation
   timeout flows through the docker `MUTATION_TIMEOUT` env var and lands
   in `swebench_docker.evaluate_instance.main`'s `TaskEnvContextManager`
   call.
3. After the harness exits, calls
   `join_harness_results_into_records` (W10), which:
   - Joins by `id` first (matches harness keys like
     `astropy__astropy-12907-37`), falling back to `instance_id`.
   - Scrapes mutation-completeness from `official_eval_logs/*.full.eval.log`
     (`total jobs: N` / `complete: K (P%)` / `MutationTimeout`).
4. Writes `RUN_REPORT.md` with the W12 stratified breakdown.

### Stage 3: Read the report

The headline section of `RUN_REPORT.md` reports the SOTA-comparable metric
(unfiltered pass@1) along with the filtered subset. Subsequent sections:

- Pass rate by detected runner (pytest / unittest / sympy-bin-test / ...).
- Pass rate by repository (catches per-project regressions).
- Mutation completeness histogram (>=95% / 80-95% / <80% / unknown). The
  mean mutation score on completeness >=80% is reported separately so the
  long-tail timeout truncation doesn't bias the headline upward.
- Tests dropped by the W1 final acceptance gate.
- Tests rejected by W2 atomic acceptance.

## Verifying the wiring without a full run

The codebase ships a smoke that exercises the report writer and the W10
join against a saved May-5 run dir. To verify the headline reads correctly
(should be ~22.5%, not 0%):

```
python <<'PY'
from pathlib import Path
import json, shutil
src = Path('.apex_testgeneval_lite_20260505_091155_70ba0e8_fresh')
dst = Path('/tmp/apex_smoke_replay')
if dst.exists(): shutil.rmtree(dst)
dst.mkdir()
shutil.copytree(src / 'records', dst / 'records')
(dst / 'official_reports').mkdir()
shutil.copy(src / 'official_reports' / next((src / 'official_reports').glob('*_full.json')).name,
            dst / 'official_reports' / next((src / 'official_reports').glob('*_full.json')).name)
(dst / 'official_eval_logs').symlink_to(src / 'official_eval_logs')

from apex.evaluation.runners.testgenevallite import _write_official_run_report
print(_write_official_run_report(dst))
print((dst / 'RUN_REPORT.md').read_text()[:1200])
PY
```

## Test coverage

The full V4 wiring is locked in by `tests/test_testgeneval_v4_remediation.py`.
Notable tests:

- `test_join_harness_results_uses_id_when_instance_id_does_not_match` — the
  W10 reporting bug regression (headline=0% bug).
- `test_join_harness_results_scrapes_eval_logs_for_completeness` — mutation
  timeout / completeness extraction.
- `test_oracle_repair_*` — W4 oracle capture in the no-LLM repair path.
- `test_oracle_repair_uses_pytest_approx_for_floats` — W9 mutation-shape
  selection.
- `test_proactive_oracle_repair_rewrites_artifact_before_failure` — W4
  proactive (no failure required).
- `test_hierarchical_gap_fill_appends_only_when_atomic_acceptance_passes` —
  W7 + W2 combined.
- `test_v4_pipeline_composition_w4_then_w7` — composition smoke covering
  W4 → W7 → W2.
- `test_testgenevallite_command_prefers_run_evaluation_when_present` — the
  Stage-2 command shape, including the `--mutation_timeout` flag.

Run the suite with:

```
.venv/bin/python -m pytest tests/test_testgeneval_v4_remediation.py \
                          tests/test_testgeneval_remediation.py \
                          tests/test_testgeneval_benchmark.py \
                          tests/test_run_artifacts.py \
                          tests/test_test_minimizer.py -q
```

(Expected: 165+ tests pass.)

## Architectural invariants

These are enforced by review, not by tooling. Violations will surface in
the next deep-dive analysis:

1. **The benchmark adapter is the only benchmark-specific surface.**
   Generation, validation, repair, oracle synthesis, emission, and the
   gap-fill loop contain zero `if benchmark == "testgenevallite":` branches.
2. **The language adapter is the only language-specific surface.** All
   Python-isms (AST roundtrip, import preflight) live behind the
   language emitter / preflight interfaces.
3. **No schema-only PRs.** New `apex_validation` fields must land with
   the runtime that produces a non-default value for them.
4. **Deterministic-first repair.** The LLM is invoked only after every
   deterministic strategy that could fix the same diagnostic has been
   tried.
5. **W2 verifies every deterministic change.** A repair that doesn't
   strictly reduce failures must fall through; the LLM gets the chance to
   try something different.

## What V4 deliberately doesn't fix yet

These are tracked but require larger architectural shifts and are NOT in
scope for the eval-readiness milestone:

- **Per-test prompt schema.** W7 currently reuses the existing single-test
  prompt template. A first-class per-test response schema would let the
  generator stream individual tests with their CallSpec and atomic
  feedback.
- **Cross-language language adapters.** Today the AST roundtrip / import
  preflight / oracle capture are Python-only. Adding JS/TS is one new
  module per concern behind the existing interfaces.

## Running TestGenEval Full (1,210 tasks)

The TestGenEvalLite runner accepts the full TestGenEval dataset by name —
schema is identical between Lite and Full, and the mounted
`swebench_docker` repo's `run_evaluation.py` accepts any HF dataset name
via `get_eval_refs()`. No code changes needed; just a CLI flag flip.

**Launcher script:** `scripts/launch_testgeneval_full.sh`

```bash
bash scripts/launch_testgeneval_full.sh             # uses default timestamp + apex-tge-full
bash scripts/launch_testgeneval_full.sh smoke "smoke-model"  # smoke variant (edit script to add --limit)
```

**Direct invocation** (equivalent):

```bash
.venv/bin/python -m apex.evaluation.runners.testgenevallite_generate \
    --output-dir .apex_testgeneval_full_<stamp> \
    --model-name apex-tge-full \
    --dataset-name kjain14/testgeneval \
    --split test \
    --parallelism 4 \
    --candidate-count 4 \
    --no-v5-patch-surrogate \
    --require-target-environment \
    --docker-gate-enabled \
    --docker-official-repo .apex_testgeneval_lite_20260503_085612/official_repo_mount \
    --docker-namespace kdjain
```

**Scaling notes**

| Resource | TestGenEvalLite | TestGenEval Full |
|---|---|---|
| Tasks | 160 | 1,210 |
| Wall (parallelism 4) | ~6 hours | ~30 hours |
| Scoring wall | ~6 hours | ~6 hours (parallelism 8) |
| Disk | shared kdjain images | shared kdjain images (~120 GB) |

**Resumability**: `--skip-existing` reuses any record file already in
`<output-dir>/records/`. Safe to re-run after a partial completion.

**Aggregation**: same `apex.evaluation.runners.testgenevallite_aggregate`
module, pass `--dataset-name kjain14/testgeneval`.

## Other benchmark drivers (see `BENCHMARK_INTEGRATION_PLAN.md`)

- **SWT-Bench Lite/Verified/Full** — `apex/evaluation/runners/swtbench_generate.py` (test-gen-against-bug; F2P metric; uses APEX V5 voting which is the TEX-T pattern)
- **SWE-Bench classic/Verified/Pro/Multilingual** — `apex/evaluation/runners/swebench_generate.py` (code-edit benchmarks; same shared driver via `--harness-mode`)
- **ProgramBench** — `apex/evaluation/runners/programbench_generate.py` (cleanroom rebuild from spec)
- **SWE-EVO** — `apex/evaluation/runners/swe_evo_generate.py` (multi-commit software evolution; uses `apex/orchestrator/in_container_agent.py`)
