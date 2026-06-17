#!/usr/bin/env bash
# ===========================================================================
# APEX Phase A.5 — Smoke sweep + calibration pipeline
# ===========================================================================
#
# REQUIRES
#   * At least one supported LLM CLI on PATH:
#         claude   (Claude Code)
#         codex    (OpenAI Codex CLI)
#         gemini   (Google Gemini CLI)
#         opencode (opencode CLI)
#     Each CLI is itself an agent loop; APEX dispatches commands to the
#     CLI binary and never calls any provider API directly. NO shell-level
#     API key is required — each CLI manages its own auth.
#   * Docker on PATH (benchmark harnesses run in containers).
#
# Use --dry-run to print the planned commands WITHOUT requiring any CLI or
# docker. Safe in CI / on sandboxed dev hosts. NOTE: --dry-run will NOT
# actually train calibrated weights; it only previews the command plan.
# A real calibration requires at least one CLI installed AND docker.
#
# WALL-CLOCK ESTIMATES (real, non-dry-run)
#   * --smoke   step  (run_benchmark_sweep.sh --smoke)    : ~30 minutes
#                     (1 task per benchmark × 3 benchmarks).
#   * --smoke-30 step (run_benchmark_sweep.sh --smoke-30) : ~10–15 hours
#                     PER benchmark on a single host (30 tasks × 3
#                     benchmarks = ~30–45 hours total wall time without
#                     additional task parallelism). Plan accordingly.
#
# WHAT IT DOES
#   1. Runs scripts/run_benchmark_sweep.sh --smoke      (1 task / benchmark)
#   2. Runs scripts/run_benchmark_sweep.sh --smoke-30   (30 tasks / benchmark)
#   3. Harvests controller_decisions.jsonl traces from the smoke-30 run.
#   4. Retrains the controller policy weights into apex/configs/controller_models/
#      (promotes priors from "calibrated-v1-synthetic" to "calibrated-v1").
#   5. Harvests per-candidate ranking outcomes from the smoke-30 run.
#   6. Recalibrates testgen ranking weights into
#      apex/configs/testgen_ranking_weights_calibrated.json.
#   7. Calibrates per-benchmark abstention thresholds into
#      apex/configs/abstention_thresholds_per_benchmark.json.
#   8. Prints a one-screen "what changed" summary.
#
# OUTPUT LAYOUT
#   <out-root>/smoke1/                 ← step 1 sweep dir
#   <out-root>/smoke30/                ← step 2 sweep dir
#   <out-root>/controller_traces.jsonl ← step 3 (harvester)
#   <out-root>/ranking_outcomes.jsonl  ← step 5 (harvester)
#   <out-root>/calibration_summary.json
#
# Usage:
#   bash apex/scripts/calibrate_smoke_sweep.sh [--dry-run] \
#       [--output-dir <path>] [--skip-smoke1] [--skip-smoke30]
#
#   --dry-run        Print the planned commands; do NOT execute them.
#                    Safe without any CLI / docker. Will NOT produce
#                    calibrated weights.
#   --output-dir     Root directory for all calibration artifacts.
#                    Defaults to runs/calibration_smoke_$(date +%Y%m%d_%H%M%S).
#   --skip-smoke1    Skip step 1 (smoke-1 sanity sweep). Use when re-running
#                    only the calibration steps against an existing smoke-30.
#   --skip-smoke30   Skip step 2 (smoke-30 sweep). Useful for re-deriving
#                    weights from a previously captured directory; pass
#                    --output-dir pointing at the existing root.
# ===========================================================================

set -euo pipefail

# shellcheck disable=SC1091
. "$(dirname "$0")/../../scripts/_common.sh"
cd "$APEX_HOME"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

DRY_RUN=0
OUTPUT_DIR=""
SKIP_SMOKE1=0
SKIP_SMOKE30=0

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="${2:-}"
            if [ -z "$OUTPUT_DIR" ]; then
                echo "ERROR: --output-dir requires a path argument" >&2
                exit 2
            fi
            shift 2
            ;;
        --output-dir=*)
            OUTPUT_DIR="${1#--output-dir=}"
            shift
            ;;
        --skip-smoke1)
            SKIP_SMOKE1=1
            shift
            ;;
        --skip-smoke30)
            SKIP_SMOKE30=1
            shift
            ;;
        -h|--help)
            sed -n '1,70p' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            echo "Run with --help for usage." >&2
            exit 2
            ;;
    esac
done

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${APEX_HOME}/runs/calibration_smoke_$(date +%Y%m%d_%H%M%S)"
fi

SMOKE1_DIR="${OUTPUT_DIR}/smoke1"
SMOKE30_DIR="${OUTPUT_DIR}/smoke30"
CONTROLLER_TRACES="${OUTPUT_DIR}/controller_traces.jsonl"
RANKING_OUTCOMES="${OUTPUT_DIR}/ranking_outcomes.jsonl"
RANKING_RUNS_MIRROR="${OUTPUT_DIR}/ranking_runs_mirror"
SUMMARY_JSON="${OUTPUT_DIR}/calibration_summary.json"
CONTROLLER_MODELS_DIR="${APEX_HOME}/apex/configs/controller_models"
TESTGEN_WEIGHTS_OUT="${APEX_HOME}/apex/configs/testgen_ranking_weights_calibrated.json"
ABSTENTION_OUT="${APEX_HOME}/apex/configs/abstention_thresholds_per_benchmark.json"
SWEEP_SCRIPT="${APEX_HOME}/scripts/run_benchmark_sweep.sh"

# ---------------------------------------------------------------------------
# Pre-flight gates (real run only). --dry-run skips them so the script is
# safe to lint/preview on hosts with no API keys + no docker.
# ---------------------------------------------------------------------------

if [ "$DRY_RUN" != "1" ]; then
    AVAILABLE_CLIS=()
    for _cli in claude codex gemini opencode; do
        if command -v "$_cli" >/dev/null 2>&1; then
            AVAILABLE_CLIS+=("$_cli")
        fi
    done
    unset _cli
    if [ "${#AVAILABLE_CLIS[@]}" -eq 0 ]; then
        cat >&2 <<'EOF'
ERROR: no supported LLM CLI found on PATH.

The smoke sweep dispatches to one of these CLI binaries (each is its
own agent loop; APEX does NOT call any provider API directly, so no
shell-level API key is required):

  claude    Claude Code           (https://claude.ai/code)
  codex     OpenAI Codex CLI      (https://github.com/openai/codex)
  gemini    Google Gemini CLI     (https://github.com/google-gemini/gemini-cli)
  opencode  opencode CLI          (https://opencode.ai)

Install at least one and ensure it is on PATH, OR re-run with --dry-run
to preview the planned commands without executing anything.
EOF
        exit 2
    fi
    echo "Detected LLM CLIs on PATH: ${AVAILABLE_CLIS[*]}"
    if ! command -v docker >/dev/null 2>&1; then
        echo "ERROR: docker not on PATH; benchmark harnesses require it." >&2
        echo "       Install docker, or re-run with --dry-run." >&2
        exit 2
    fi
fi

mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# run() / write_file() — print every command; skip on --dry-run.
# ---------------------------------------------------------------------------

run() {
    printf '\n+ %s\n' "$*"
    if [ "$DRY_RUN" = "1" ]; then
        return 0
    fi
    eval "$@"
}

# Capture metadata so the final summary can compare before/after.
PRE_CONTROLLER_VERSION="unknown"
if [ -f "${CONTROLLER_MODELS_DIR}/contract_gap.json" ]; then
    PRE_CONTROLLER_VERSION="$(
        "$APEX_PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1])).get('policy_version',''))" \
            "${CONTROLLER_MODELS_DIR}/contract_gap.json" 2>/dev/null || echo unknown
    )"
fi

PRE_RANKING_PATH=""
if [ -f "$TESTGEN_WEIGHTS_OUT" ]; then
    PRE_RANKING_PATH="$TESTGEN_WEIGHTS_OUT"
fi

PRE_ABSTENTION_PATH=""
if [ -f "$ABSTENTION_OUT" ]; then
    PRE_ABSTENTION_PATH="$ABSTENTION_OUT"
fi

# ---------------------------------------------------------------------------
# Step 1 — Smoke (1 task per benchmark).
# Sanity check: confirms the wiring + per-benchmark configs work before we
# spend the budget on the 30-task sweep.
# ---------------------------------------------------------------------------

if [ "$SKIP_SMOKE1" = "1" ]; then
    echo "(skipping step 1 — smoke-1 sweep)"
else
    run "bash $SWEEP_SCRIPT --smoke --output-dir $SMOKE1_DIR ${DRY_RUN:+--dry-run} || true"
fi

# ---------------------------------------------------------------------------
# Step 2 — Smoke-30 (30 tasks per benchmark).
# This is the corpus the calibrators learn against.
# ---------------------------------------------------------------------------

if [ "$SKIP_SMOKE30" = "1" ]; then
    echo "(skipping step 2 — smoke-30 sweep)"
else
    run "bash $SWEEP_SCRIPT --smoke-30 --output-dir $SMOKE30_DIR ${DRY_RUN:+--dry-run} || true"
fi

# ---------------------------------------------------------------------------
# Step 3 — Harvest controller decision traces.
# ---------------------------------------------------------------------------

run "$APEX_PYTHON $APEX_HOME/apex/scripts/harvest_controller_traces.py \
    $SMOKE30_DIR --summary-json $OUTPUT_DIR/controller_traces_summary.json \
    > $CONTROLLER_TRACES"

# ---------------------------------------------------------------------------
# Step 4 — Retrain controller policy weights.
# ---------------------------------------------------------------------------

run "$APEX_PYTHON $APEX_HOME/apex/scripts/train_controller_policy.py \
    --traces $CONTROLLER_TRACES \
    --output $CONTROLLER_MODELS_DIR \
    --policy-version calibrated-v1"

# ---------------------------------------------------------------------------
# Step 5 — Harvest per-candidate ranking outcomes (incl. flat mirror).
# ---------------------------------------------------------------------------

run "$APEX_PYTHON $APEX_HOME/apex/scripts/harvest_ranking_outcomes.py \
    $SMOKE30_DIR \
    --mirror-runs-dir $RANKING_RUNS_MIRROR \
    --summary-json $OUTPUT_DIR/ranking_outcomes_summary.json \
    > $RANKING_OUTCOMES"

# ---------------------------------------------------------------------------
# Step 6 — Recalibrate testgen ranking weights.
# ---------------------------------------------------------------------------

run "$APEX_PYTHON $APEX_HOME/apex/scripts/calibrate_testgen_ranking.py \
    --runs-dir $RANKING_RUNS_MIRROR \
    --backend grid \
    --grid-step 0.05 \
    > $TESTGEN_WEIGHTS_OUT"

# ---------------------------------------------------------------------------
# Step 7 — Calibrate per-benchmark abstention threshold.
# ---------------------------------------------------------------------------

run "$APEX_PYTHON $APEX_HOME/apex/scripts/calibrate_abstention_threshold.py \
    --runs-dir $SMOKE30_DIR \
    --output $ABSTENTION_OUT \
    --threshold-step 0.05 \
    --calibration-run $(basename $OUTPUT_DIR)"

# ---------------------------------------------------------------------------
# Step 8 — Summary print.
# ---------------------------------------------------------------------------

if [ "$DRY_RUN" = "1" ]; then
    cat <<EOF

==============================================================================
DRY RUN — calibrate_smoke_sweep.sh
==============================================================================
Planned output dir       : $OUTPUT_DIR
Smoke-1 sweep dir        : $SMOKE1_DIR  (skipped: $SKIP_SMOKE1)
Smoke-30 sweep dir       : $SMOKE30_DIR (skipped: $SKIP_SMOKE30)
Controller traces JSONL  : $CONTROLLER_TRACES
Ranking outcomes JSONL   : $RANKING_OUTCOMES
Controller models dir    : $CONTROLLER_MODELS_DIR
Ranking weights output   : $TESTGEN_WEIGHTS_OUT
Abstention thresholds    : $ABSTENTION_OUT

No commands were executed (and no calibrated weights were produced).
Re-run without --dry-run on a host that has docker + at least one of
{claude, codex, gemini, opencode} on PATH. Each CLI manages its own
auth; no shell-level API key is required.
EOF
    exit 0
fi

# Build a JSON summary with what changed.
"$APEX_PYTHON" - <<EOF
import json, os
from pathlib import Path

summary = {
    "output_dir": "$OUTPUT_DIR",
    "controller_models_dir": "$CONTROLLER_MODELS_DIR",
    "testgen_weights_out": "$TESTGEN_WEIGHTS_OUT",
    "abstention_thresholds_out": "$ABSTENTION_OUT",
    "controller_models_pre_version": "$PRE_CONTROLLER_VERSION",
}
post_versions = {}
for name in ("contract_gap", "broad_regression", "high_interface_risk", "importability_blocker"):
    path = Path("$CONTROLLER_MODELS_DIR") / f"{name}.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text())
            post_versions[name] = {
                "policy_version": payload.get("policy_version"),
                "ece": payload.get("training_metadata", {}).get("ece"),
                "ece_uncalibrated": payload.get("training_metadata", {}).get("ece_uncalibrated"),
            }
        except Exception:
            post_versions[name] = {"error": "unparseable"}
summary["controller_models_post"] = post_versions

ranking_path = Path("$TESTGEN_WEIGHTS_OUT")
if ranking_path.exists():
    try:
        summary["testgen_ranking_weights"] = json.loads(ranking_path.read_text())
    except Exception:
        summary["testgen_ranking_weights"] = {"error": "unparseable"}

abst_path = Path("$ABSTENTION_OUT")
if abst_path.exists():
    try:
        summary["abstention_thresholds"] = json.loads(abst_path.read_text())
    except Exception:
        summary["abstention_thresholds"] = {"error": "unparseable"}

Path("$SUMMARY_JSON").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print()
print("=" * 78)
print("Phase A.5 calibration complete")
print("=" * 78)
print(f"Output dir              : $OUTPUT_DIR")
print(f"Controller models       : $CONTROLLER_MODELS_DIR")
pre = "$PRE_CONTROLLER_VERSION"
post = post_versions.get("contract_gap", {}).get("policy_version", "?")
print(f"Controller policy_ver   : {pre} -> {post}")
for name, info in sorted(post_versions.items()):
    ece = info.get("ece")
    ece_un = info.get("ece_uncalibrated")
    print(f"  regime={name:25s} ece={ece}  (uncalibrated {ece_un})")
weights = summary.get("testgen_ranking_weights") or {}
if isinstance(weights, dict) and weights:
    print()
    print("Testgen ranking weights (calibrated):")
    for k, v in sorted(weights.items()):
        if isinstance(v, (int, float)):
            print(f"  {k:30s} {v:.4f}")
abst = summary.get("abstention_thresholds") or {}
if isinstance(abst, dict) and abst:
    print()
    print("Per-benchmark abstention thresholds:")
    meta = abst.get("_metadata", {}) if isinstance(abst.get("_metadata"), dict) else {}
    f1s = meta.get("f1_per_benchmark", {})
    for k, v in sorted(abst.items()):
        if k.startswith("_"):
            continue
        f1 = f1s.get(k)
        f1_str = f"f1={f1:.4f}" if isinstance(f1, (int, float)) else "f1=?"
        if isinstance(v, (int, float)):
            print(f"  {k:20s} threshold={v:.2f}  {f1_str}")
print()
print(f"Summary JSON            : $SUMMARY_JSON")
EOF
