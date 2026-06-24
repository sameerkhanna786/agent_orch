# FrontierCode decision + SWE-rebench integration

Workflow `w20q03tw1` (8 agents, adversarial). Question: is FrontierCode a better/complementary A/B
substrate than commit0 for measuring ORCHESTRATION value? Keep commit0 regardless.

## Verdict: do NOT add FrontierCode; ADD SWE-rebench (recency-split) instead.

**FrontierCode** = Cognition AI benchmark (~June 2026, cognition.com/blog/frontier-code; "probable" id —
brand-new, web-only, vendor-conflicted). 4-bar scoring:
- Bar1 contamination-resistant: PASS (private/held-out, vendor-asserted).
- Bar2 repo-level (exercises our orchestrator): PASS.
- Bar3 difficulty band: PASS.
- **Bar4 harness-runnable: FAIL (decisive).** Gated/private (no local env/test-cmd/gold-ids) AND
  hybrid LLM-judge + maintainer-rubric scored. Our `scoring.py:19,149-153` hard-gates ACCEPT to
  `scoring_source=='commit0_test_ids'` (executable evidence only) — any LLM-judge source is force-
  downgraded to indeterminate. So FrontierCode is DOUBLY incompatible: un-runnable + un-acceptable in our
  execution-grounded no-silent-loss contract. A 3/4-bar benchmark that can't run here can't be our substrate.

## Green-lit alternative: SWE-rebench (Nebius, HF `nebius/SWE-rebench`) — FEASIBLE
Recency-split SWE-task benchmark: issue + repo -> multi-file patch -> FAIL_TO_PASS/PASS_TO_PASS pytest
node-ids (executable-scored, no LLM-judge). Directly attacks commit0's famous-library memorization via
temporal contamination defense, while exercising our decompose->multi-agent->reduce->loop orchestrator.

**Feasibility VERIFIED (this session):** HF reachable; dataset = 21,336 instances / 3,468 repos,
created_at 2014->2025. **9,154 Docker-free (pre_install empty) = 43%** (1,495 FRESH >=2024-06; 5,735 OLDER
<2023; 8,179 with 1-15 FAIL_TO_PASS). Each row ships a PINNED `requirements` + `install_config`
(pip install -e .[...] , python ver) -> uv-installable locally without Docker/apt. So a large recency-
stratified local slice exists. (Read parquet directly via huggingface_hub+pyarrow; the eval venv's
`datasets` torch auto-format is broken — bypass it.)

## Integration (keep commit0 byte-identical; SWE-rebench = Mode-C only)
NO orchestrator/executor/scoring.py changes. New: `swerebench_runner.py` (self-contained LOCAL runner
mirroring v1 _prepare_repo/evaluate_repo/_build_test_command, emitting a Commit0Evaluation-compatible
object: contract_success(), scoring_source='commit0_test_ids', total_tests=gold-universe, diagnostics for
the indeterminate guard — EXECUTION-GROUNDED: real local pytest-json-report only), `swerebench_registry.py`
(instance-id keyed RepoSpec, local_runnable=not forces_docker), `swerebench_autogen.py` (copy of
commit0_autogen.run_autogen_cell swapping the runner + instance-id discovery; drop datasets/pydantic-core
preflight). Edits: commit0_driver.run_cell dispatch on APEX_OMEGA_BENCHMARK==swerebench;
pin_gold_scoring_contract made benchmark-aware (it pins the commit0 contract unconditionally today);
run_ladder LADDER_BENCHMARK switch + a curated Docker-free instance list. Gold ids = sorted(set(FAIL_TO_PASS)
| set(PASS_TO_PASS)) parsed via `apex/evaluation/swebench_benchmark._parse_literal_list`, pinned to a
checked-in inventory (never re-fetched). Full-gold accept = all FAIL_TO_PASS flip AND all PASS_TO_PASS
preserved (failed==errors==missing==0).

## VALIDATION GATE (linchpin, before any orchestration eval)
Run each candidate instance's GOLD `patch` through the runner -> must pass all FAIL_TO_PASS+PASS_TO_PASS
under uv locally. This proves (a) the instance is locally runnable (apt-free) and (b) the runner is
execution-grounded + the pinned gold-ids match collected node-ids. Instances that fail the gate are dropped.

## A/B plan (the de-contaminated orchestration measurement)
tree-search vs hybrid-diag on a curated ~10-15 Docker-free slice, CONTAMINATION-STRATIFIED (FRESH >=2024-06
vs OLDER), n=5, frugal (APEX_CODEX_FAST=0, concurrency 2). Key claim: the tree-search-vs-hybrid delta
should PERSIST on the FRESH stratum (delta = construction, not memory). If it collapses to ~0 on fresh,
that is evidence commit0's deltas were contamination-inflated. Metrics: solve-rate, mean gold pass-fraction,
agents/cell, per-seed variance. Caveat: ~3-5 instances/stratum is low-powered -> directional; widen later.
