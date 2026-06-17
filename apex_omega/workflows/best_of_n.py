"""The reference best-of-N / pipeline commit0 workflow (Phase-0 exit criterion).

This is orchestration-as-code on the APEX-Ω engine: it fans out K isolated
coding-agent workers (Codex, Claude, or a mixed fleet — the ``worker_specs``
list), each editing its own git worktree, journals every ``agent()`` call (so the
whole solve is resume-survivable), scores each candidate with an
execution-grounded ``score_fn``, and selects under the Cardinal Safety Contract
(execution-authoritative; abstains rather than shipping an unverified guess).

It is vendor-neutral by construction (the vendor is a field on the worker spec),
demonstrates the net-new ``pipeline()`` primitive, and degrades to verified
best-of-N — the floor it can never do worse than.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from ..ablation.arms import AblationConfig
from ..ablation.safety_modes import validate_safety_modes
from ..engine.runtime import Engine
from ..isolation.worktree import WorktreeProvider, apply_diff
from ..kernel.select import Candidate, select_best
from ..kernel.verify import VerificationResult, candidate_from_verification
from ..types import ScopedTask


@dataclass
class WorkerSpec:
    vendor: str
    model: str
    extra: dict = field(default_factory=dict)   # extra LLMConfig fields (cli_command, cli_model_id, ...)

    # Dual access: LLM-authored orchestrate(ctx) code is unpredictable, so a worker
    # works as a (vendor, model) tuple, a dict ({"vendor":..,"model":..}), an
    # attribute object (.vendor/.model), or via unpacking (v, m = worker).
    def __getitem__(self, key):
        if isinstance(key, int):
            return (self.vendor, self.model, self.extra)[key]
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        yield self.vendor
        yield self.model

    def __len__(self):
        return 2


@dataclass
class SolveResult:
    winner: Optional[Candidate]
    candidates: list[Candidate]
    abstained: bool
    k: int

    def to_dict(self) -> dict:
        return {
            "abstained": self.abstained,
            "k": self.k,
            "n_candidates": len(self.candidates),
            "n_accepted": sum(1 for c in self.candidates if c.accepted),
            "winner": (None if self.winner is None else {
                "candidate_id": self.winner.candidate_id, "rollout_id": self.winner.rollout_id,
                "vendor": self.winner.meta.get("vendor"), "score": self.winner.combined_score,
                "accepted": self.winner.accepted, "diff_bytes": len(self.winner.diff or ""),
            }),
            "candidates": [
                {"candidate_id": c.candidate_id, "accepted": c.accepted, "score": c.combined_score,
                 "vendor": c.meta.get("vendor"), "pass_rate": c.public_signal_score}
                for c in self.candidates
            ],
        }


def best_of_n_solve(
    engine: Engine,
    *,
    source_repo: str,
    executor: Any,                       # V1Executor | FakeExecutor (anything with spawn())
    worker_specs: Sequence[WorkerSpec],
    build_prompt: Callable[[int, str], str],
    score_fn: Callable[[str], VerificationResult],
    k: int = 3,
    base_commit: Optional[str] = None,
    run_scope: str = "bestofn",
    schema: Optional[dict] = None,
    ablation: Optional[AblationConfig] = None,
    sandbox: str = "workspace-write",
    timeout_seconds: Optional[int] = None,
) -> SolveResult:
    """Run K isolated best-of-N rollouts on the engine and select under the
    Cardinal Contract.  ``worker_specs`` is cycled across the K rollouts so a
    single solve can carry a Claude branch, a Codex branch, and a cheap leaf
    simultaneously (the heterogeneous-fleet pattern)."""
    abl = ablation or AblationConfig()
    # Fail loud at wiring time if a rejected form (Cardinal relaxation, share-all,
    # economy-floor disablement) is requested without the explicit research opt-in.
    validate_safety_modes(abl.to_safety_modes(), dynamic_coverage_available=lambda: True)
    provider = WorktreeProvider(
        source_repo, base_commit=base_commit,
        workspace_dir=str(Path(engine.run_dir) / "worktrees"), run_scope=run_scope,
    )
    engine.phase(f"best_of_n[k={k}]")

    def make_thunk(i: int) -> Callable[[], Optional[Candidate]]:
        spec = worker_specs[i % len(worker_specs)]

        def _thunk() -> Optional[Candidate]:
            handle = provider.acquire(i)
            try:
                wt = handle.path
                session = executor.spawn(wt, spec.vendor, spec.model, spec=spec.extra)
                task = ScopedTask(
                    prompt=build_prompt(i, wt), schema=schema, sandbox=sandbox,
                    model=spec.model, vendor=spec.vendor, internet=False,
                    timeout_seconds=timeout_seconds,
                    scoped_inputs={"repo_snapshot_sha": provider.base_commit, "rollout": i},
                )
                res = engine.agent(
                    task, lambda t: session.run(t), node_id=f"rollout{i}",
                    cli_version=getattr(session, "cli_version", ""),
                    materialize=lambda diff, _wt=wt: apply_diff(_wt, diff),
                )
                vr = score_fn(wt)
                cand = candidate_from_verification(
                    candidate_id=f"r{i}", diff=res.fs_diff, vr=vr, rollout_id=i, cluster_id=i,
                    meta={"vendor": spec.vendor, "model": spec.model,
                          "finalization_status": res.finalization_status, "ok": res.ok},
                )
                return cand
            finally:
                provider.release(handle, confirm_patch_extracted=True)

        return _thunk

    results = engine.parallel([make_thunk(i) for i in range(k)])
    candidates = [c for c in results if c is not None]
    # A11 negative control: relaxing the contract lets an unverified candidate ship.
    allow_unaccepted = not abl.cardinal_contract_enforced
    winner = select_best(candidates, allow_unaccepted=allow_unaccepted)
    engine.log(f"best_of_n done: {len(candidates)} candidates, "
               f"{sum(1 for c in candidates if c.accepted)} accepted, "
               f"winner={winner.candidate_id if winner else 'ABSTAIN'}")
    return SolveResult(winner=winner, candidates=candidates,
                       abstained=(winner is None), k=k)


# --- a generic execution-grounded score_fn (pytest) ------------------------
_PYTEST_COUNT_RE = re.compile(r"(?:(\d+) passed)?(?:, )?(?:(\d+) failed)?(?:, )?(?:(\d+) error)?")


def make_pytest_score_fn(
    test_cmd: str,
    *,
    python_executable: Optional[str] = None,
    timeout: int = 600,
    env: Optional[dict] = None,
) -> Callable[[str], VerificationResult]:
    """Score a candidate worktree by running its test command and applying v1's
    commit0 acceptance gate to the parsed counts (execution-authoritative)."""
    from ..eval.scoring import decide_from_counts

    full_env = {**os.environ, **(env or {})}

    def _score(worktree_cwd: str) -> VerificationResult:
        cmd = test_cmd
        proc = subprocess.run(cmd, cwd=worktree_cwd, shell=True, text=True,
                              capture_output=True, timeout=timeout, env=full_env)
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        passed = _count(out, r"(\d+) passed")
        failed = _count(out, r"(\d+) failed")
        errors = _count(out, r"(\d+) error")
        total = passed + failed + errors
        accepted, reason, pass_rate = decide_from_counts(
            passed=passed, failed=failed, errors=errors, total=total,
            missing=0, raw_returncode=proc.returncode,
        )
        return VerificationResult(
            accepted=accepted, score=(1.0 if accepted else min(0.89, pass_rate)),
            passed=passed, failed=failed, errors=errors, total=total,
            pass_rate=pass_rate, reason=None if accepted else reason,
            indeterminate=(total == 0 and proc.returncode not in (0, 1)),
        )

    return _score


def _count(text: str, pattern: str) -> int:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else 0
