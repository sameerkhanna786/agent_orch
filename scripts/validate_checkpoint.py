#!/usr/bin/env python3
"""Targeted validation for Tier-1.1 acceptance-checkpointing + opt-in repair.

Re-runs the TWO mimesis cells that discarded verified 6052/6052 passes in run-4
(omega_template + omega_autogen), under the run-4-shaped config (repair ON via
APEX_OMEGA_REPAIR_ITERS=2, autogen cap 16) but WITH the checkpoint + recovery fix.
Goal: prove a verified solve is now BANKED (either the cell completes solved, or it
is killed at the wall and recovered from accepted_checkpoint.json) instead of lost.

Runs through run_ladder.run_cell (monkeypatched to a validation dir) so the full
subprocess + kill + _recover_checkpoint path is exercised. Both cells run in parallel.
"""
from __future__ import annotations

import importlib.util
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("APEX_OMEGA_REPAIR_ITERS", "2")  # opt-in repair for this validation

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_ladder", str(REPO / "scripts" / "run_ladder.py"))
rl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl)

VDIR = REPO / "runs" / "validation_checkpoint"
VDIR.mkdir(parents=True, exist_ok=True)
rl.LADDER_DIR = VDIR
rl.PROGRESS = VDIR / "progress.jsonl"

CAP = "16"
CELLS = [
    ("omega_template_k8", ["--arms", "autogen_orchestrator", "--autogen-scout-agents", "0",
                           "--autogen-max-agents", CAP], "mimesis"),
    ("omega_autogen_k8",  ["--arms", "autogen_orchestrator", "--autogen-scout-agents", "3",
                           "--autogen-author", "--autogen-max-agents", CAP], "mimesis"),
]

print(f"validation: repair_iters={os.environ['APEX_OMEGA_REPAIR_ITERS']} cap={CAP} "
      f"cell_timeout={rl.CELL_TIMEOUT} -> {VDIR}", flush=True)
with ThreadPoolExecutor(max_workers=2) as ex:
    futs = [ex.submit(rl.run_cell, *c) for c in CELLS]
    for f in futs:
        f.result()
print("validation cells done", flush=True)
