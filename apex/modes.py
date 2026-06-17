"""Three first-class APEX usage modes.

APEX has historically been benchmark-shaped: SWE-Bench Pro testgen
hides the gold tests AND gold patch from the agent at generation time.
Real-world callers want explicit modes that treat the available
artifacts as INPUTS (not hidden oracles):

    Mode 1 (testgen_with_fix):
        Caller has a code fix; wants tests that catch the bug it
        addresses. Use case: regression-suite augmentation, code
        review, "I just shipped a fix and want a test for it."

    Mode 2 (codegen_with_tests):
        Caller has a test suite; wants the code change that makes
        them pass. Use case: classic TDD — write tests first, the
        agent writes the implementation.

    Mode 3 (generate_both):
        Caller has only a problem statement. Agent must produce
        BOTH tests and code, chained: testgen first to define the
        contract, then codegen against those tests.

All three modes are thin compositions over the existing testgen + F2P
+ orchestrator infrastructure. They do NOT add new LLM-call surfaces;
they just preset the inputs and route to the right subsystem. The
benefit: a single coherent API surface that's easy to invoke from an
IDE plugin, CI gate, or interactive agent, with no benchmark task
object required.

Public API:
    run_testgen_with_fix(...)      → ModeResult
    run_codegen_with_tests(...)    → ModeResult
    run_generate_both(...)         → ModeResult

Each function takes broken_dir / fixed_dir paths (or a single repo_dir
for the agent to work in), the problem statement, and any caller-
supplied artifacts. The agent invocations themselves are pluggable via
optional callable parameters so the modes module can be tested
without spinning up real LLMs.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from .core.generated_tests import safe_materialize_test_artifacts

logger = logging.getLogger(__name__)


# --- Mode identifiers (string constants for stable serialization) ---

MODE_TESTGEN_WITH_FIX = "testgen_with_fix"
MODE_CODEGEN_WITH_TESTS = "codegen_with_tests"
MODE_GENERATE_BOTH = "generate_both"
# Phase 6 item 6.1: adversarial test-vs-patch self-play mode.
MODE_GENERATE_BOTH_SELF_PLAY = "generate_both_self_play"
# Phase 6 item 6.4: mutation-targeted active-learning testgen mode.
MODE_TESTGEN_WITH_ACTIVE_LEARNING = "testgen_with_active_learning"

ALL_MODES = (
    MODE_TESTGEN_WITH_FIX,
    MODE_CODEGEN_WITH_TESTS,
    MODE_GENERATE_BOTH,
    MODE_GENERATE_BOTH_SELF_PLAY,
    MODE_TESTGEN_WITH_ACTIVE_LEARNING,
)


# --- Agent-mode identifiers (Phase 3.1.a) ---
#
# ``agent_mode`` selects WHICH agent surface produces the patch in
# ``run_codegen_with_tests`` (and, by extension, ``run_generate_both``).
# Defaults to ``scaffolded`` which preserves the legacy MASAI
# Reproducer/Localizer/Patcher path through ApexOrchestrator.

AGENT_MODE_SCAFFOLDED = "scaffolded"
AGENT_MODE_CLI_AGENT = "cli_agent"
AGENT_MODE_IN_CONTAINER_V5 = "in_container_v5"
# Phase 6.5: planner-above-V5. Decomposes the problem into sub-tasks,
# allocates a per-subtask turn budget, runs the V5 in-container agent
# once per sub-task, and rebalances the budget after each. See
# :class:`apex.orchestration.HierarchicalAgent`.
AGENT_MODE_HIERARCHICAL_V5 = "hierarchical_v5"

ALL_AGENT_MODES = (
    AGENT_MODE_SCAFFOLDED,
    AGENT_MODE_CLI_AGENT,
    AGENT_MODE_IN_CONTAINER_V5,
    AGENT_MODE_HIERARCHICAL_V5,
)

AgentMode = Literal["scaffolded", "cli_agent", "in_container_v5", "hierarchical_v5"]


@dataclass
class ModeResult:
    """Unified result shape for all three modes."""

    mode: str
    success: bool
    test_artifacts: list[dict[str, Any]] = field(default_factory=list)
    patch: Optional[str] = None
    f2p_summary: dict[str, Any] = field(default_factory=dict)
    mutation_summary: dict[str, Any] = field(default_factory=dict)
    minimization_summary: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "success": self.success,
            "test_artifacts": list(self.test_artifacts),
            "patch": self.patch,
            "f2p_summary": dict(self.f2p_summary),
            "mutation_summary": dict(self.mutation_summary),
            "minimization_summary": dict(self.minimization_summary),
            "error": self.error,
            "diagnostics": dict(self.diagnostics),
        }


# --- Pluggable callables for test/code generation ---

# A test_generator takes (repo_path, problem_statement) and returns a
# list of test_artifact dicts (each with "path" and "content"). The
# default delegates to the real test_writer; tests pass a fake.
TestGenerator = Callable[[Path, str], list[dict[str, Any]]]

# A code_generator takes (repo_path, problem_statement, test_artifacts)
# and returns a unified-diff patch string (or None on failure).
CodeGenerator = Callable[[Path, str, list[dict[str, Any]]], Optional[str]]

# Phase I.4: a surrogate_patcher takes (repo_path, problem_statement,
# surrogate_index) and returns a unified-diff patch string for the
# bug. The index lets the caller diversify across attempts (different
# seeds, prompts, sampling temperatures). When the caller has no
# gold patch, the modes API generates ``n_surrogates`` candidates via
# this callable and uses them as an ensemble F2P oracle for ranking
# the test portfolio.
SurrogatePatcher = Callable[[Path, str, int], Optional[str]]


# ---------------------------------------------------------------------------
# Mode 1: testgen_with_fix
# ---------------------------------------------------------------------------


# Decisive-Edge C.6: benchmarks whose dataset ships a defined gold oracle.
# When ``run_testgen_with_fix`` is invoked with one of these benchmark ids
# AND a gold_patch, the surrogate-oracle ensemble fan-out is auto-skipped:
# the gold patch IS the oracle. This saves ~24 LLM calls per task
# (n_surrogates=8 × 3 distinct-model surrogate backends).
_BENCHMARKS_WITH_GOLD_ORACLE: frozenset[str] = frozenset(
    {
        "testgeneval",
        "testgeneval_lite",
        "swtbench",
        "swt_bench",
        "swt_bench_lite",
        "swtbench_lite",
    }
)


def run_testgen_with_fix(
    *,
    repo_path: str | Path,
    problem_statement: str,
    gold_patch: Optional[str] = None,
    output_dir: str | Path,
    test_generator: Optional[TestGenerator] = None,
    surrogate_patcher: Optional[SurrogatePatcher] = None,
    n_surrogates: int = 8,
    language: str = "python",
    install_repo: bool = False,
    timeout_seconds: float = 300.0,
    persist_to_repo_memory: bool = True,
    repo_memory_directory: Optional[str] = None,
    focus_files: Optional[list[str]] = None,
    surrogate_models: Optional[list[str]] = None,
    weighted_consensus_threshold: float = 0.5,
    use_surrogate: bool = True,
    benchmark_id: Optional[str] = None,
) -> ModeResult:
    """Mode 1: generate tests that catch a bug.

    Two paths depending on what the caller supplies:

    **Gold-fix path** (``gold_patch`` is non-empty): caller has a known
    fix. We build broken / fixed sandboxes, run ``test_generator`` against
    broken, and score with the gold-patch F2P oracle. Standard flow.

    **Surrogate-fix path** (Phase I.4 — ``gold_patch`` is None and
    ``surrogate_patcher`` is supplied): caller has only a problem
    statement. We invoke the test_generator against broken, then call
    ``surrogate_patcher`` ``n_surrogates`` times to synthesize candidate
    fixes, and score the test portfolio via the multi-surrogate ensemble
    F2P oracle (apex.evaluation.surrogate_oracle). Tests that flip F2P
    under EVERY surrogate are reported as ``consensus_f2p`` and are
    high-precision bug catchers; tests that flip under any surrogate
    enter ``union_f2p`` and are at least relevant.

    Per the project directive ("never reduce model size / power"),
    ``n_surrogates`` defaults to 8 (Phase 4A item 4.7 — raised from 4
    so consensus voting has stronger signal) and surrogates round-robin
    through ``surrogate_models`` (defaults to the
    ``OrchestrationConfig.surrogate_oracle_models`` triple of distinct
    CLI agent backends so the ensemble doesn't sample the same blind
    spots N times). Callers can crank up by passing a larger value.

    Decisive-Edge C.6: ``use_surrogate`` (default True) gates the
    surrogate-fix path. Pass ``use_surrogate=False`` together with a
    non-empty ``gold_patch`` to skip the surrogate ensemble entirely on
    benchmarks that ship a defined oracle (TestGenEval, SWT-Bench).
    Skipping saves ~24 LLM calls per task (n_surrogates=8 × 3 distinct-
    model surrogate backends) without changing F2P signal — the gold
    patch is exact, the surrogates would only add noise. Passing
    ``use_surrogate=False`` *without* a gold_patch raises a clear error
    rather than silently producing zero F2P signal. When
    ``benchmark_id`` matches a known gold-oracle benchmark
    (``testgeneval``, ``testgeneval_lite``, ``swtbench``, ...) and a
    gold_patch is supplied, ``use_surrogate`` auto-flips to False even
    if the caller left the default — explicit ``use_surrogate=True``
    overrides the auto-flip for ablations.

    The returned ModeResult carries the generated test_artifacts plus
    f2p_summary (the single-surrogate or aggregated payload) and
    surrogate_oracle_summary on diagnostics when surrogates were used.

    Generalizes outside benchmarks: no benchmark task object required.
    """
    repo = Path(repo_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sandboxes = output / "_modes_sandboxes"
    broken_dir = sandboxes / "broken"
    fixed_dir = sandboxes / "fixed"

    has_gold_patch = bool(gold_patch and gold_patch.strip())
    has_surrogate_patcher = surrogate_patcher is not None

    # Decisive-Edge C.6: auto-disable the surrogate path on benchmarks
    # that ship a defined oracle. The caller can still force the surrogate
    # fan-out on those benchmarks for ablations by passing use_surrogate=True.
    auto_skipped_surrogate = False
    normalized_benchmark = (benchmark_id or "").strip().lower()
    if use_surrogate and has_gold_patch and normalized_benchmark in _BENCHMARKS_WITH_GOLD_ORACLE:
        # Only auto-flip when the caller didn't explicitly pass
        # surrogate_patcher (an explicit patcher is a clear opt-in to the
        # surrogate path; respect it).
        if not has_surrogate_patcher:
            use_surrogate = False
            auto_skipped_surrogate = True

    # Decisive-Edge C.6: explicit "no surrogate, no gold patch" is a
    # configuration error, not a silently-degraded run.
    if not use_surrogate and not has_gold_patch:
        return ModeResult(
            mode=MODE_TESTGEN_WITH_FIX,
            success=False,
            error=(
                "Cannot run testgen without a gold patch and without "
                "surrogate oracle. Pass use_surrogate=True (with a "
                "surrogate_patcher) or supply a gold_patch."
            ),
        )

    if not has_gold_patch and not has_surrogate_patcher:
        return ModeResult(
            mode=MODE_TESTGEN_WITH_FIX,
            success=False,
            error=(
                "no oracle: pass either gold_patch (known fix) or "
                "surrogate_patcher (synthesizes fix candidates) so the "
                "F2P score can be computed"
            ),
        )

    if has_gold_patch:
        try:
            _prepare_paired_sandboxes(
                repo_path=repo,
                broken_dir=broken_dir,
                fixed_dir=fixed_dir,
                gold_patch=gold_patch or "",
            )
        except _ModeError as exc:
            return ModeResult(
                mode=MODE_TESTGEN_WITH_FIX,
                success=False,
                error=str(exc),
            )
    else:
        # Surrogate path needs only a broken sandbox for the test_generator;
        # surrogate fixes are applied per-surrogate inside the oracle.
        try:
            _clone_repo(repo, broken_dir)
        except _ModeError as exc:
            return ModeResult(
                mode=MODE_TESTGEN_WITH_FIX,
                success=False,
                error=str(exc),
            )
        # Phase 4A item 4.4: pin the broken sandbox state so any
        # capture_oracle invocation against it confirms the workdir
        # actually represents the pre-fix repo (fixes are applied
        # per-surrogate INSIDE the oracle, never to this top-level
        # sandbox).
        try:
            from .evaluation.oracle_capture import write_oracle_state_sentinel

            write_oracle_state_sentinel(broken_dir, "pre_fix")
        except Exception:  # pragma: no cover - defensive
            pass

    # The test_generator works against the BROKEN sandbox; that's the
    # state the agent normally writes tests against in the benchmark
    # flow. The fix (gold or surrogate) is held by the F2P oracle, not
    # shown to the agent (consistent with benchmark integrity even in
    # real-world use).
    generator = test_generator or _default_test_generator
    try:
        test_artifacts = generator(broken_dir, problem_statement)
    except Exception as exc:
        return ModeResult(
            mode=MODE_TESTGEN_WITH_FIX,
            success=False,
            error=f"test_generator raised: {type(exc).__name__}: {exc}",
        )

    if not test_artifacts:
        return ModeResult(
            mode=MODE_TESTGEN_WITH_FIX,
            success=False,
            test_artifacts=[],
            error="test_generator produced no artifacts",
        )

    if has_gold_patch:
        from .evaluation import evaluate_tdd_iteration

        f2p_report = evaluate_tdd_iteration(
            broken_dir=broken_dir,
            fixed_dir=fixed_dir,
            test_artifacts=test_artifacts,
            output_dir=output / "_tdd_report",
            language=language,
            timeout_seconds=timeout_seconds,
            install_repo=install_repo,
        )
        summary = dict(f2p_report.get("summary") or {})
        # Phase I.7: persist cross-task testgen insights so the next
        # run on this same repo gets stronger priors.
        memory_summary = _persist_testgen_run_to_repo_memory(
            repo_path=repo,
            persist=persist_to_repo_memory,
            directory=repo_memory_directory,
            focus_files=focus_files,
            f2p_summary=summary,
        )
        diagnostics = {
            "broken_dir": str(broken_dir),
            "fixed_dir": str(fixed_dir),
            "oracle_kind": "gold_patch",
            "test_runner_adapter": f2p_report.get("test_runner_adapter"),
        }
        # Decisive-Edge C.6: surface whether the surrogate fan-out was
        # explicitly skipped (use_surrogate=False) or auto-skipped because
        # the benchmark ships a defined oracle. Useful for cost auditing.
        if not use_surrogate:
            diagnostics["surrogate_skipped"] = True
            diagnostics["surrogate_skip_reason"] = (
                "auto_benchmark_with_gold_oracle"
                if auto_skipped_surrogate
                else "use_surrogate_false"
            )
            # Estimated cost saved: n_surrogates * len(surrogate_models).
            # The default 8 * 3 = 24 LLM calls per task on the surrogate
            # ensemble (apex.evaluation.surrogate_oracle).
            estimated_models = (
                len(surrogate_models)
                if surrogate_models is not None and len(surrogate_models) > 0
                else 3
            )
            diagnostics["surrogate_calls_skipped_estimate"] = int(n_surrogates) * int(
                estimated_models
            )
        if memory_summary is not None:
            diagnostics["repo_memory_persistence"] = memory_summary
        return ModeResult(
            mode=MODE_TESTGEN_WITH_FIX,
            success=bool(summary.get("any_f2p")),
            test_artifacts=list(test_artifacts),
            f2p_summary=summary,
            diagnostics=diagnostics,
        )

    # Surrogate-fix path
    from .evaluation.surrogate_oracle import evaluate_via_surrogate_oracle

    assert surrogate_patcher is not None  # narrowed by has_surrogate_patcher
    resolved_surrogate_models = list(
        surrogate_models if surrogate_models is not None else _default_surrogate_models()
    )
    oracle_report = evaluate_via_surrogate_oracle(
        broken_repo_path=broken_dir,
        problem_statement=problem_statement,
        test_artifacts=test_artifacts,
        surrogate_patcher=surrogate_patcher,
        output_dir=output / "_surrogate_oracle",
        n_surrogates=n_surrogates,
        language=language,
        install_repo=install_repo,
        timeout_seconds=timeout_seconds,
        surrogate_models=resolved_surrogate_models,
        weighted_consensus_threshold=weighted_consensus_threshold,
    )
    surrogate_summary = oracle_report.to_dict()
    success = oracle_report.consensus_f2p_count > 0
    aggregated_f2p_summary: dict[str, Any] = {
        "any_f2p": success,
        "consensus_f2p_count": oracle_report.consensus_f2p_count,
        "union_f2p_count": oracle_report.union_f2p_count,
        "weighted_consensus_f2p_count": (oracle_report.weighted_consensus_f2p_count),
        "n_surrogates_with_any_f2p": oracle_report.n_surrogates_with_any_f2p,
        "n_surrogates_applied": oracle_report.n_surrogates_applied,
        "f2p_tests": list(oracle_report.consensus_f2p_nodeids),
        "weighted_consensus_f2p_tests": list(oracle_report.weighted_consensus_f2p_nodeids),
        "status": oracle_report.status,
    }
    memory_summary = _persist_testgen_run_to_repo_memory(
        repo_path=repo,
        persist=persist_to_repo_memory,
        directory=repo_memory_directory,
        focus_files=focus_files,
        f2p_summary=aggregated_f2p_summary,
    )
    surrogate_diagnostics: dict[str, Any] = {
        "broken_dir": str(broken_dir),
        "oracle_kind": "surrogate_patches",
        "surrogate_oracle_summary": surrogate_summary,
    }
    if memory_summary is not None:
        surrogate_diagnostics["repo_memory_persistence"] = memory_summary
    return ModeResult(
        mode=MODE_TESTGEN_WITH_FIX,
        success=success,
        test_artifacts=list(test_artifacts),
        f2p_summary=aggregated_f2p_summary,
        diagnostics=surrogate_diagnostics,
    )


# ---------------------------------------------------------------------------
# Mode 2: codegen_with_tests
# ---------------------------------------------------------------------------


def run_codegen_with_tests(
    *,
    repo_path: str | Path,
    problem_statement: str,
    gold_test_artifacts: list[dict[str, Any]],
    output_dir: str | Path,
    code_generator: Optional[CodeGenerator] = None,
    language: str = "python",
    install_repo: bool = False,
    timeout_seconds: float = 300.0,
    agent_mode: AgentMode = AGENT_MODE_SCAFFOLDED,
    docker_image: Optional[str] = None,
    container_supervisor: Optional[Any] = None,
    in_container_v5_runner: Optional[Callable[..., Any]] = None,
    llm_config: Any = None,
    llm_caller: Optional[Callable[..., Any]] = None,
    max_turns: Optional[int] = None,
    # Phase 6.5: hierarchical-V5 budget controls. Only consulted when
    # ``agent_mode == AGENT_MODE_HIERARCHICAL_V5``.
    total_budget: Optional[int] = None,
    n_subtasks: Optional[int] = None,
    rebalance_strategy: str = "feedback",
    hierarchical_v5_runner: Optional[Callable[..., Any]] = None,
) -> ModeResult:
    """Mode 2: generate the code change that makes a given test suite pass.

    The caller supplies the original ``repo_path`` (current/broken state)
    and ``gold_test_artifacts`` (the test cases that define correctness).
    We:

      1. Materialize the gold tests into a "broken" sandbox
      2. Invoke ``code_generator`` to produce a fix patch
      3. Apply the patch to a "fixed" sandbox (a separate clone)
      4. Run :func:`evaluate_tdd_iteration` to verify the gold tests
         transition F2P under the agent's patch

    ``agent_mode`` selects WHICH agent surface produces the patch:

    * ``"scaffolded"`` (default, back-compat) — invokes ``code_generator``
      (defaulting to the MASAI Reproducer/Localizer/Patcher pipeline
      through ApexOrchestrator).
    * ``"cli_agent"`` — same code-generator path; reserved as an explicit
      label for the CLI-backend rollouts that internally are LLM agents.
    * ``"in_container_v5"`` — runs the new
      :class:`apex.orchestrator_in_container_agent.InContainerAgent`
      against the broken sandbox. If ``docker_image`` is set we wrap the
      run in a :class:`apex.core.container_supervisor.ContainerSupervisor`
      for true container isolation; otherwise we fall back to the V1 host
      bash shim with a logged warning. The same ``ModeResult`` shape is
      produced as the other modes — diff is mapped to ``patch_files``
      via ``_apply_patch``, and ``give_up`` / ``parse_failure`` /
      ``llm_failure`` / ``max_turns`` outcomes are reported in
      ``diagnostics`` with status-style markers.

    Generalizes outside benchmarks: classic TDD — tests are the spec,
    agent makes them green.
    """
    if agent_mode not in ALL_AGENT_MODES:
        return ModeResult(
            mode=MODE_CODEGEN_WITH_TESTS,
            success=False,
            error=(f"unknown agent_mode={agent_mode!r}; expected one of {list(ALL_AGENT_MODES)}"),
        )

    repo = Path(repo_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sandboxes = output / "_modes_sandboxes"
    broken_dir = sandboxes / "broken"

    try:
        _clone_repo(repo, broken_dir)
        _materialize_test_artifacts(broken_dir, gold_test_artifacts)
    except _ModeError as exc:
        return ModeResult(
            mode=MODE_CODEGEN_WITH_TESTS,
            success=False,
            error=str(exc),
        )

    if agent_mode == AGENT_MODE_HIERARCHICAL_V5:
        v5_outcome = _invoke_hierarchical_v5_agent(
            broken_dir=broken_dir,
            problem_statement=problem_statement,
            docker_image=docker_image,
            container_supervisor=container_supervisor,
            llm_config=llm_config,
            llm_caller=llm_caller,
            max_turns=max_turns,
            total_budget=total_budget,
            n_subtasks=n_subtasks,
            rebalance_strategy=rebalance_strategy,
            runner_override=hierarchical_v5_runner,
        )
    elif agent_mode == AGENT_MODE_IN_CONTAINER_V5:
        v5_outcome = _invoke_in_container_v5_agent(
            broken_dir=broken_dir,
            problem_statement=problem_statement,
            docker_image=docker_image,
            container_supervisor=container_supervisor,
            llm_config=llm_config,
            llm_caller=llm_caller,
            max_turns=max_turns,
            target_runtime_required=False,
            runner_override=in_container_v5_runner,
        )
    else:
        v5_outcome = None  # type: ignore[assignment]

    if agent_mode in (AGENT_MODE_IN_CONTAINER_V5, AGENT_MODE_HIERARCHICAL_V5):
        assert v5_outcome is not None  # narrows for mypy
        # Bridge the V5 summary onto the codegen path: the V5 patch IS
        # the agent's patch and the rest of this function flows through
        # the F2P / verifier path unchanged.
        if v5_outcome.error:
            return ModeResult(
                mode=MODE_CODEGEN_WITH_TESTS,
                success=False,
                error=v5_outcome.error,
                diagnostics=v5_outcome.diagnostics,
            )
        if v5_outcome.patch is None:
            # V5 ran cleanly but produced no patch (give_up / max_turns /
            # parse_failure). Surface as ABSTAINED via diagnostics; do
            # not try to apply a None patch.
            v5_diag = dict(v5_outcome.diagnostics)
            v5_diag.setdefault("status_marker", "ABSTAINED")
            return ModeResult(
                mode=MODE_CODEGEN_WITH_TESTS,
                success=False,
                error=v5_outcome.error or "in_container_v5 produced no patch",
                diagnostics=v5_diag,
            )
        patch = v5_outcome.patch
        v5_extra_diag = dict(v5_outcome.diagnostics)
    else:
        v5_extra_diag = {}
        generator = code_generator or _default_code_generator
        try:
            patch = generator(broken_dir, problem_statement, gold_test_artifacts)
        except Exception as exc:
            return ModeResult(
                mode=MODE_CODEGEN_WITH_TESTS,
                success=False,
                error=f"code_generator raised: {type(exc).__name__}: {exc}",
            )

    if not patch or not patch.strip():
        return ModeResult(
            mode=MODE_CODEGEN_WITH_TESTS,
            success=False,
            error="code_generator produced no patch",
        )

    fixed_dir = sandboxes / "fixed"
    apply_diagnostics: Optional[dict[str, Any]] = None
    try:
        _clone_repo(repo, fixed_dir)
        _materialize_test_artifacts(fixed_dir, gold_test_artifacts)
        apply_diagnostics = _apply_patch(fixed_dir, patch)
    except _ModeError as exc:
        return ModeResult(
            mode=MODE_CODEGEN_WITH_TESTS,
            success=False,
            patch=patch,
            error=f"could not apply agent patch: {exc}",
            diagnostics={"patch_apply_path": "failed"},
        )

    from .evaluation import evaluate_tdd_iteration

    f2p_report = evaluate_tdd_iteration(
        broken_dir=broken_dir,
        fixed_dir=fixed_dir,
        test_artifacts=gold_test_artifacts,
        output_dir=output / "_tdd_report",
        language=language,
        timeout_seconds=timeout_seconds,
        install_repo=install_repo,
    )
    summary = dict(f2p_report.get("summary") or {})
    diagnostics: dict[str, Any] = {
        "broken_dir": str(broken_dir),
        "fixed_dir": str(fixed_dir),
        "test_runner_adapter": f2p_report.get("test_runner_adapter"),
    }
    if isinstance(apply_diagnostics, dict):
        diagnostics.update(apply_diagnostics)
    if v5_extra_diag:
        diagnostics["agent_mode"] = agent_mode
        if agent_mode == AGENT_MODE_HIERARCHICAL_V5:
            diagnostics["hierarchical_v5"] = v5_extra_diag
        else:
            diagnostics["in_container_v5"] = v5_extra_diag
    else:
        diagnostics["agent_mode"] = agent_mode
    return ModeResult(
        mode=MODE_CODEGEN_WITH_TESTS,
        success=bool(summary.get("any_f2p")),
        patch=patch,
        f2p_summary=summary,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Mode 3: generate_both
# ---------------------------------------------------------------------------


def run_generate_both(
    *,
    repo_path: str | Path,
    problem_statement: str,
    output_dir: str | Path,
    test_generator: Optional[TestGenerator] = None,
    code_generator: Optional[CodeGenerator] = None,
    language: str = "python",
    install_repo: bool = False,
    timeout_seconds: float = 300.0,
) -> ModeResult:
    """Mode 3: chained testgen → codegen from a problem statement only.

    The caller supplies just ``repo_path`` and ``problem_statement``.
    We:

      1. Phase A — invoke ``test_generator`` against the current
         (broken) repo state to write tests for the bug.
      2. Phase B — invoke ``code_generator`` with the generated
         tests as the spec, producing a fix patch.
      3. Run F2P verification (broken vs. broken+patch) using the
         generated tests to confirm the chain produced a coherent
         test+code pair.

    The chain succeeds if the F2P oracle confirms ``any_f2p`` — i.e.
    the agent-generated tests correctly fail on the broken state and
    pass on the agent-generated patch.
    """
    repo = Path(repo_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Phase A: testgen against the broken state
    sandboxes = output / "_modes_sandboxes"
    broken_dir = sandboxes / "broken"
    try:
        _clone_repo(repo, broken_dir)
    except _ModeError as exc:
        return ModeResult(
            mode=MODE_GENERATE_BOTH,
            success=False,
            error=str(exc),
        )

    test_gen = test_generator or _default_test_generator
    try:
        test_artifacts = test_gen(broken_dir, problem_statement)
    except Exception as exc:
        return ModeResult(
            mode=MODE_GENERATE_BOTH,
            success=False,
            error=f"test_generator raised: {type(exc).__name__}: {exc}",
        )
    if not test_artifacts:
        return ModeResult(
            mode=MODE_GENERATE_BOTH,
            success=False,
            test_artifacts=[],
            error="test_generator produced no artifacts (Phase A)",
        )

    # Phase B: codegen against the generated tests
    code_gen = code_generator or _default_code_generator
    try:
        patch = code_gen(broken_dir, problem_statement, test_artifacts)
    except Exception as exc:
        return ModeResult(
            mode=MODE_GENERATE_BOTH,
            success=False,
            test_artifacts=list(test_artifacts),
            error=f"code_generator raised: {type(exc).__name__}: {exc}",
        )
    if not patch or not patch.strip():
        return ModeResult(
            mode=MODE_GENERATE_BOTH,
            success=False,
            test_artifacts=list(test_artifacts),
            error="code_generator produced no patch (Phase B)",
        )

    # Phase C: F2P verification of the chain
    fixed_dir = sandboxes / "fixed"
    apply_diagnostics: Optional[dict[str, Any]] = None
    try:
        _clone_repo(repo, fixed_dir)
        apply_diagnostics = _apply_patch(fixed_dir, patch)
    except _ModeError as exc:
        return ModeResult(
            mode=MODE_GENERATE_BOTH,
            success=False,
            test_artifacts=list(test_artifacts),
            patch=patch,
            error=f"could not apply agent patch in phase C: {exc}",
            diagnostics={"patch_apply_path": "failed"},
        )

    from .evaluation import evaluate_tdd_iteration

    f2p_report = evaluate_tdd_iteration(
        broken_dir=broken_dir,
        fixed_dir=fixed_dir,
        test_artifacts=test_artifacts,
        output_dir=output / "_tdd_report",
        language=language,
        timeout_seconds=timeout_seconds,
        install_repo=install_repo,
    )
    summary = dict(f2p_report.get("summary") or {})
    diagnostics: dict[str, Any] = {
        "broken_dir": str(broken_dir),
        "fixed_dir": str(fixed_dir),
        "test_runner_adapter": f2p_report.get("test_runner_adapter"),
    }
    if isinstance(apply_diagnostics, dict):
        diagnostics.update(apply_diagnostics)
    return ModeResult(
        mode=MODE_GENERATE_BOTH,
        success=bool(summary.get("any_f2p")),
        test_artifacts=list(test_artifacts),
        patch=patch,
        f2p_summary=summary,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Default test/code generators (real LLM-backed implementations)
# ---------------------------------------------------------------------------


def _default_test_generator(repo_path: Path, problem_statement: str) -> list[dict[str, Any]]:
    """Default test generator: a single structured CLI prompt to the
    strongest available LLM (Codex CLI + gpt-5.5 by default) asking
    for a JSON portfolio of test artifacts.

    Wired through ``apex._default_generators.default_test_generator``
    (Phase I.8). Returns ``[]`` (with a logged warning) if the LLM
    call fails, the response can't be parsed, or no CLI tool is
    installed — preserving the prior-version contract that the modes
    API never crashes on a default generator failure.

    For maximum quality, callers should still pass a custom
    ``test_generator`` callable that wraps their full orchestrator
    setup (multi-rollout, F2P-driven, mutation-discriminated). The
    default is a one-shot prompt, suitable for IDE plugins / CI gates
    that don't need the full ensemble.
    """
    try:
        from ._default_generators import default_test_generator as _impl
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning(
            "modes default_test_generator: import failed (%s); returning [].",
            exc,
        )
        return []
    return _impl(repo_path, problem_statement)


def _default_code_generator(
    repo_path: Path,
    problem_statement: str,
    test_artifacts: list[dict[str, Any]],
) -> Optional[str]:
    """Default code generator: invokes ``ApexOrchestrator.solve()``
    with a problem statement augmented to mark the supplied tests as
    success criteria.

    Wired through ``apex._default_generators.default_code_generator``
    (Phase I.8). Returns ``None`` (with a logged warning) when the
    orchestrator fails or doesn't produce a successful patch."""
    try:
        from ._default_generators import default_code_generator as _impl
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning(
            "modes default_code_generator: import failed (%s); returning None.",
            exc,
        )
        return None
    return _impl(repo_path, problem_statement, test_artifacts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _default_surrogate_models() -> list[str]:
    """Return the default per-surrogate model list.

    Phase 4A item 4.7: surrogates round-robin through distinct CLI agent
    backends so the ensemble doesn't sample the same blind spots N
    times. Reads ``OrchestrationConfig.surrogate_oracle_models`` so the
    list is configurable. Falls back to a hard-coded triple when the
    config is unreachable (defensive — modes API must never crash on a
    config import failure)."""

    try:
        from .core.config import OrchestrationConfig

        return list(OrchestrationConfig().surrogate_oracle_models)
    except Exception:  # pragma: no cover - defensive
        return [
            "codex_cli:gpt-5.5",
            "claude_cli:opus",
            "gemini_cli:gemini-3.1-pro",
        ]


class _ModeError(RuntimeError):
    pass


@dataclass
class _V5AgentOutcome:
    """Internal: bridge dataclass between InContainerAgent.solve_with_summary
    and ModeResult.

    ``patch`` is None when the agent gave up / hit max_turns / failed to
    parse / had an LLM exception. ``error`` is set on hard infrastructure
    failures only (workspace missing, supervisor crashed, etc.); a clean
    "no patch" outcome surfaces as ``patch=None`` with an ABSTAINED
    status_marker in ``diagnostics``.
    """

    patch: Optional[str] = None
    error: Optional[str] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _invoke_in_container_v5_agent(
    *,
    broken_dir: Path,
    problem_statement: str,
    docker_image: Optional[str] = None,
    container_supervisor: Optional[Any] = None,
    llm_config: Any = None,
    llm_caller: Optional[Callable[..., Any]] = None,
    max_turns: Optional[int] = None,
    test_command: Optional[str] = None,
    patch_verifier_reject_cap: int = 3,
    target_runtime_required: bool = False,
    runner_override: Optional[Callable[..., Any]] = None,
) -> _V5AgentOutcome:
    """Drive the V5 in-container agent against a prepared sandbox.

    Wires the ``InContainerAgent`` into the modes API. The mapping from
    ``AgentRunSummary.terminated_reason`` to ModeResult is:

        submit_patch                            -> patch in ModeResult.patch
        give_up | parse_failure |
        llm_failure | max_turns                  -> patch=None, status_marker=ABSTAINED

    When ``docker_image`` is provided we wrap the run in a
    :class:`apex.core.container_supervisor.ContainerSupervisor` for true
    container isolation. When neither ``docker_image`` nor an explicit
    ``container_supervisor`` is given, we fall back to V1 host bash
    shim (logged warning). Tests can stub out the run by passing
    ``runner_override``.
    """
    if runner_override is not None:
        try:
            override_outcome = runner_override(
                broken_dir,
                problem_statement,
                {
                    "docker_image": docker_image,
                    "container_supervisor": container_supervisor,
                    "llm_config": llm_config,
                    "llm_caller": llm_caller,
                    "max_turns": max_turns,
                    "test_command": test_command,
                    "target_runtime_required": target_runtime_required,
                },
            )
        except Exception as exc:  # pragma: no cover — defensive
            return _V5AgentOutcome(
                error=f"in_container_v5_runner raised: {type(exc).__name__}: {exc}",
                diagnostics={"runner_override": True},
            )
        return _coerce_runner_override(override_outcome)

    # Lazy imports so the modes module can still be imported on hosts
    # that don't have the V5 agent build deps available.
    try:
        from .orchestrator_in_container_agent import (
            DEFAULT_MAX_TURNS,
            InContainerAgent,
        )
    except Exception as exc:  # pragma: no cover — defensive
        return _V5AgentOutcome(
            error=f"failed to import in-container agent: {exc}",
        )

    supervisor = container_supervisor
    supervisor_owned_locally = False
    diagnostics: dict[str, Any] = {
        "broken_dir": str(broken_dir),
        "container_isolation": "supervisor" if supervisor or docker_image else "v1_host_shim",
        "target_runtime_required": bool(target_runtime_required),
    }
    try:
        if target_runtime_required and supervisor is None and not docker_image:
            diagnostics["container_isolation"] = "required_unavailable"
            return _V5AgentOutcome(
                error=(
                    "target runtime isolation is required, but no docker image "
                    "or container supervisor was supplied"
                ),
                diagnostics=diagnostics,
            )
        if supervisor is None and docker_image:
            try:
                from .core.container_supervisor import ContainerSupervisor
            except Exception as exc:  # pragma: no cover — defensive
                return _V5AgentOutcome(
                    error=f"failed to import container supervisor: {exc}",
                )
            try:
                supervisor = ContainerSupervisor(
                    image=docker_image,
                    workspace_dir=broken_dir,
                )
                supervisor.__enter__()
                supervisor_owned_locally = True
                diagnostics["docker_image"] = docker_image
                resolved = supervisor.resolved_image
                if resolved is not None:
                    diagnostics["docker_digest_source"] = resolved.source
                    diagnostics["docker_image_ref"] = resolved.image_ref
            except Exception as exc:
                return _V5AgentOutcome(
                    error=f"container supervisor failed to start: {exc}",
                    diagnostics=diagnostics,
                )

        if supervisor is None:
            logger.warning(
                "_invoke_in_container_v5_agent: no docker image / supervisor "
                "supplied — V5 will run with host bash shim (NOT real "
                "container isolation)."
            )

        # 1A: build the in-loop verifier in this runtime (None -> legacy accept).
        v5_patch_verifier = _build_v5_patch_verifier(
            workspace_dir=broken_dir,
            test_command=test_command,
            container_supervisor=supervisor,
        )
        diagnostics["patch_verifier_active"] = v5_patch_verifier is not None
        agent = InContainerAgent(
            llm_config=llm_config,
            workspace_dir=str(broken_dir),
            max_turns=int(max_turns or DEFAULT_MAX_TURNS),
            llm_caller=llm_caller,
            container_supervisor=supervisor,
            patch_verifier=v5_patch_verifier,
            patch_verifier_reject_cap=int(patch_verifier_reject_cap or 3),
        )
        try:
            summary = agent.solve_with_summary(problem_statement)
        except Exception as exc:
            return _V5AgentOutcome(
                error=f"in_container_v5 agent raised: {type(exc).__name__}: {exc}",
                diagnostics=diagnostics,
            )

        diagnostics["terminated_reason"] = summary.terminated_reason
        diagnostics["give_up_reason"] = summary.give_up_reason
        diagnostics["turn_count"] = len(summary.turns)
        diagnostics["total_elapsed_seconds"] = round(summary.total_elapsed_seconds, 4)
        diagnostics["per_turn_telemetry"] = [t.to_dict() for t in summary.turns]

        if (
            summary.terminated_reason in ("submit_patch", "submit_patch_verified")
            and summary.final_patch
        ):
            return _V5AgentOutcome(
                patch=summary.final_patch,
                diagnostics=diagnostics,
            )
        # ABSTAINED bucket — give_up / max_turns / parse_failure / llm_failure.
        diagnostics["status_marker"] = "ABSTAINED"
        return _V5AgentOutcome(diagnostics=diagnostics)
    finally:
        if supervisor_owned_locally and supervisor is not None:
            try:
                supervisor.__exit__(None, None, None)
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning("container supervisor cleanup raised: %s", exc)


def _invoke_hierarchical_v5_agent(
    *,
    broken_dir: Path,
    problem_statement: str,
    docker_image: Optional[str] = None,
    container_supervisor: Optional[Any] = None,
    llm_config: Any = None,
    llm_caller: Optional[Callable[..., Any]] = None,
    max_turns: Optional[int] = None,
    total_budget: Optional[int] = None,
    n_subtasks: Optional[int] = None,
    rebalance_strategy: str = "feedback",
    runner_override: Optional[Callable[..., Any]] = None,
) -> _V5AgentOutcome:
    """Phase 6.5: drive the planner-above-V5 agent against a sandbox.

    Mirrors the contract of :func:`_invoke_in_container_v5_agent` so the
    codegen path is shape-compatible. Returns an ``_V5AgentOutcome`` whose
    ``diagnostics`` carries the per-subtask records and the final budget
    view emitted by :class:`HierarchicalAgent`.
    """
    if runner_override is not None:
        try:
            override_outcome = runner_override(
                broken_dir,
                problem_statement,
                {
                    "docker_image": docker_image,
                    "container_supervisor": container_supervisor,
                    "llm_config": llm_config,
                    "llm_caller": llm_caller,
                    "max_turns": max_turns,
                    "total_budget": total_budget,
                    "n_subtasks": n_subtasks,
                    "rebalance_strategy": rebalance_strategy,
                },
            )
        except Exception as exc:  # pragma: no cover — defensive
            return _V5AgentOutcome(
                error=f"hierarchical_v5_runner raised: {type(exc).__name__}: {exc}",
                diagnostics={"runner_override": True},
            )
        return _coerce_runner_override(override_outcome)

    try:
        from .orchestration.budget_planner import (
            BudgetPlanner,
            TurnBudget,
        )
        from .orchestration.hierarchical_agent import (
            DEFAULT_MAX_SUBTASKS,
            HierarchicalAgent,
        )
        from .orchestrator_in_container_agent import (
            DEFAULT_MAX_TURNS,
            InContainerAgent,
        )
    except Exception as exc:  # pragma: no cover — defensive
        return _V5AgentOutcome(
            error=f"failed to import hierarchical V5 deps: {exc}",
        )

    supervisor = container_supervisor
    supervisor_owned_locally = False
    diagnostics: dict[str, Any] = {
        "broken_dir": str(broken_dir),
        "container_isolation": ("supervisor" if supervisor or docker_image else "v1_host_shim"),
    }
    try:
        if supervisor is None and docker_image:
            try:
                from .core.container_supervisor import ContainerSupervisor
            except Exception as exc:  # pragma: no cover — defensive
                return _V5AgentOutcome(
                    error=f"failed to import container supervisor: {exc}",
                )
            try:
                supervisor = ContainerSupervisor(
                    image=docker_image,
                    workspace_dir=broken_dir,
                )
                supervisor.__enter__()
                supervisor_owned_locally = True
                diagnostics["docker_image"] = docker_image
                resolved = supervisor.resolved_image
                if resolved is not None:
                    diagnostics["docker_digest_source"] = resolved.source
                    diagnostics["docker_image_ref"] = resolved.image_ref
            except Exception as exc:
                return _V5AgentOutcome(
                    error=f"container supervisor failed to start: {exc}",
                    diagnostics=diagnostics,
                )

        if supervisor is None:
            logger.warning(
                "_invoke_hierarchical_v5_agent: no docker image / supervisor "
                "supplied — V5 will run with host bash shim (NOT real "
                "container isolation)."
            )

        n_subtasks_eff = max(1, int(n_subtasks or DEFAULT_MAX_SUBTASKS))
        per_subtask_default = int(max_turns or DEFAULT_MAX_TURNS)
        total_eff = int(total_budget or per_subtask_default * n_subtasks_eff)
        if total_eff < 1:
            total_eff = per_subtask_default * n_subtasks_eff
        diagnostics["total_budget"] = total_eff
        diagnostics["n_subtasks_estimate"] = n_subtasks_eff
        diagnostics["rebalance_strategy"] = rebalance_strategy

        # The wrapped V5 agent is constructed once with a placeholder
        # max_turns; HierarchicalAgent overrides it per-subtask.
        agent = InContainerAgent(
            llm_config=llm_config,
            workspace_dir=str(broken_dir),
            max_turns=per_subtask_default,
            llm_caller=llm_caller,
            container_supervisor=supervisor,
        )
        try:
            planner = BudgetPlanner(
                total_budget=TurnBudget(total_turns=total_eff),
                n_subtasks_estimate=n_subtasks_eff,
                rebalance_strategy=rebalance_strategy,
            )
            hierarchical = HierarchicalAgent(
                planner=planner,
                in_container_agent=agent,
                max_subtasks=n_subtasks_eff,
            )
            summary = hierarchical.solve(problem_statement)
        except Exception as exc:
            return _V5AgentOutcome(
                error=f"hierarchical_v5 agent raised: {type(exc).__name__}: {exc}",
                diagnostics=diagnostics,
            )

        diagnostics["terminated_reason"] = summary.terminated_reason
        diagnostics["give_up_reason"] = summary.give_up_reason
        diagnostics["turn_count"] = len(summary.turns)
        diagnostics["total_elapsed_seconds"] = round(summary.total_elapsed_seconds, 4)
        diagnostics["subtasks"] = list(summary.subtasks)
        diagnostics["budget_view"] = dict(summary.budget)

        if (
            summary.terminated_reason in ("submit_patch", "submit_patch_verified")
            and summary.final_patch
        ):
            return _V5AgentOutcome(
                patch=summary.final_patch,
                diagnostics=diagnostics,
            )
        diagnostics["status_marker"] = "ABSTAINED"
        return _V5AgentOutcome(diagnostics=diagnostics)
    finally:
        if supervisor_owned_locally and supervisor is not None:
            try:
                supervisor.__exit__(None, None, None)
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning(
                    "hierarchical V5 container supervisor cleanup raised: %s",
                    exc,
                )


def _coerce_runner_override(value: Any) -> _V5AgentOutcome:
    """Tolerate two return shapes from a test stub: a bare patch string
    (or None) or a fully-formed _V5AgentOutcome."""
    if isinstance(value, _V5AgentOutcome):
        return value
    if value is None:
        return _V5AgentOutcome(diagnostics={"status_marker": "ABSTAINED", "runner_override": True})
    if isinstance(value, str):
        return _V5AgentOutcome(patch=value, diagnostics={"runner_override": True})
    if isinstance(value, dict) and "patch" in value:
        return _V5AgentOutcome(
            patch=value.get("patch"),
            error=value.get("error"),
            diagnostics=dict(value.get("diagnostics") or {"runner_override": True}),
        )
    return _V5AgentOutcome(
        error=f"unexpected runner_override return type: {type(value).__name__}",
    )


def _clone_repo(source: Path, dest: Path) -> None:
    if not source.exists():
        raise _ModeError(f"source repo missing: {source}")
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Try git clone first (preserves history). Fall back to copytree
    # for non-git source dirs (handy for tests + non-versioned repos).
    completed = subprocess.run(
        ["git", "clone", "--shared", "--no-hardlinks", "--quiet", str(source), str(dest)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return

    # Phase 2C 5.5: classify the git failure before silently falling
    # back to copytree. The legacy implementation would silently use
    # ``shutil.copytree`` whenever ``git clone`` returned non-zero — that
    # hid network/install/permission errors as "well, we got something
    # copied". Now we:
    #   1. log the rc + stderr (full, not truncated)
    #   2. classify the failure
    #   3. fall back to copytree only if the failure is environmental
    #      (network DNS, sandbox restrictions). Non-env failures (e.g.
    #      a permission bug, a path escape) should bubble up as a
    #      _ModeError so the caller sees the real reason.
    from .core.failure_classifier import classify_failure as _classify_clone_failure

    stderr_full = (completed.stderr or "").strip()
    classification = _classify_clone_failure(
        stderr=stderr_full,
        stdout=(completed.stdout or "").strip(),
        returncode=int(completed.returncode),
        context={"phase": "clone"},
    )
    logger.warning(
        "_clone_repo: git clone %s → %s rc=%s class=%s reason=%s; stderr=%s",
        source,
        dest,
        completed.returncode,
        classification.failure_class.value,
        classification.reason,
        stderr_full,
    )
    if not (
        classification.failure_class.is_environment
        or classification.failure_class == classification.failure_class.UNCLASSIFIED  # noqa: E721
    ):
        raise _ModeError(
            f"git clone {source} → {dest} failed (class="
            f"{classification.failure_class.value}, rc={completed.returncode}): "
            f"{stderr_full or '<no stderr>'}"
        )
    try:
        shutil.copytree(source, dest)
    except (OSError, shutil.Error) as exc:
        raise _ModeError(
            f"could not clone or copy {source} → {dest}: "
            f"git rc={completed.returncode} class={classification.failure_class.value}; "
            f"copytree exc={exc}"
        ) from exc


def _build_v5_patch_verifier(
    *,
    workspace_dir: Path,
    test_command: Optional[str],
    container_supervisor: Optional[Any] = None,
    env_overrides: Optional[dict[str, str]] = None,
    timeout_seconds: int = 600,
    max_output_bytes: int = 16_000,
) -> Optional[Callable[[str], Any]]:
    """1A: build the V5 in-loop ``patch_verifier`` or ``None`` (clean no-op gate).

    Returns ``None`` when there is no usable ``test_command`` so non-Commit0 /
    no-test paths keep the legacy immediate-accept behavior.

    The verifier RECOMPUTES the candidate's effect by running ``test_command`` in
    the agent's OWN runtime via ``_execute_in_workspace`` — the same primitive the
    agent uses for ``run_in_container`` — so dependencies/env match (a host-only
    clone would lack the Commit0 container deps and falsely reject every patch).
    The agent edits ``workspace_dir`` (``broken_dir``) in place, so that directory
    already IS the candidate; running tests there is the correct in-loop signal
    and does not mutate source (only cache artifacts). It is an ITERATE signal —
    the authoritative grade remains the downstream official audit. Failures are
    fed back (tail-biased) so the agent keeps fixing instead of the harness
    trusting a "should pass".
    """
    command = str(test_command or "").strip()
    if not command:
        return None

    # Local imports keep modes importable on hosts without the V5 build deps.
    from .orchestrator_in_container_agent import (
        PatchVerification,
        _execute_in_workspace,
        _truncate_tool_output,
    )

    workspace = str(workspace_dir)

    def _verify(patch: str) -> Any:
        try:
            result = _execute_in_workspace(
                command,
                workspace_dir=workspace,
                timeout_seconds=int(timeout_seconds),
                max_output_bytes=int(max_output_bytes),
                env_overrides=env_overrides,
                container_supervisor=container_supervisor,
            )
        except Exception as exc:  # noqa: BLE001 - verifier must never crash the loop
            return PatchVerification(
                applied=False, passed=False, failure_excerpt=f"verifier_error: {exc}"
            )
        passed = result.error is None and not result.timed_out and int(result.return_code) == 0
        if passed:
            return PatchVerification(applied=True, passed=True, failure_excerpt="")
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        excerpt, _ = _truncate_tool_output(combined, int(max_output_bytes))
        if result.timed_out:
            excerpt = f"[test command timed out after {timeout_seconds}s]\n" + excerpt
        return PatchVerification(applied=True, passed=False, failure_excerpt=excerpt.strip())

    return _verify


def _apply_patch(repo_dir: Path, patch_text: str) -> Optional[dict[str, Any]]:
    """Apply ``patch_text`` to ``repo_dir`` via ``git apply``.

    Phase 2C 2.8: returns a structured diagnostics dict so callers can
    record ``patch_apply_path`` and the captured stderrs from BOTH the
    direct attempt and the ``--3way`` fallback. None is returned for
    empty patches (no-op). Raises :class:`_ModeError` when both paths
    fail.

    Returned diagnostics shape:
        {
            "patch_apply_path": "direct" | "3way" | "failed",
            "patch_apply_stderrs": {
                "direct": str,           # full stderr, no truncation
                "3way": str | None,      # None when 3way wasn't tried
            },
            "direct_returncode": int,
            "threeway_returncode": Optional[int],
        }
    """
    if not patch_text.strip():
        return None
    apply_cmd = subprocess.run(
        ["git", "-C", str(repo_dir), "apply", "--whitespace=nowarn", "-"],
        input=patch_text,
        capture_output=True,
        text=True,
        check=False,
    )
    direct_stderr_full = apply_cmd.stderr or ""
    if apply_cmd.returncode == 0:
        return {
            "patch_apply_path": "direct",
            "patch_apply_stderrs": {
                "direct": direct_stderr_full,
                "3way": None,
            },
            "direct_returncode": int(apply_cmd.returncode),
            "threeway_returncode": None,
        }

    # Phase 2C 2.8: log full stderr (NOT truncated to 300 chars) and
    # emit a structured warning before attempting the 3way fallback.
    # The legacy code only surfaced the last 300 chars and only inside
    # the eventual exception — operators couldn't see the reason for
    # the fallback at runtime.
    logger.warning(
        "patch_apply_3way_fallback: direct git apply failed (rc=%s); "
        "attempting --3way recovery. Direct stderr: %s",
        apply_cmd.returncode,
        direct_stderr_full,
    )
    apply_3way = subprocess.run(
        ["git", "-C", str(repo_dir), "apply", "--3way", "--whitespace=nowarn", "-"],
        input=patch_text,
        capture_output=True,
        text=True,
        check=False,
    )
    threeway_stderr_full = apply_3way.stderr or ""
    if apply_3way.returncode == 0:
        return {
            "patch_apply_path": "3way",
            "patch_apply_stderrs": {
                "direct": direct_stderr_full,
                "3way": threeway_stderr_full,
            },
            "direct_returncode": int(apply_cmd.returncode),
            "threeway_returncode": int(apply_3way.returncode),
        }
    raise _ModeError(
        "git apply failed in BOTH direct and --3way modes; "
        f"direct rc={apply_cmd.returncode}, 3way rc={apply_3way.returncode}; "
        f"direct stderr={direct_stderr_full or '<empty>'}; "
        f"3way stderr={threeway_stderr_full or '<empty>'}"
    )


def _materialize_test_artifacts(sandbox: Path, artifacts: list[dict[str, Any]]) -> None:
    skipped = len(list(artifacts or [])) - len(
        safe_materialize_test_artifacts(sandbox, artifacts or [])
    )
    if skipped > 0:
        logger.warning(
            "Skipped %s unsafe or empty generated test artifact(s).",
            skipped,
        )


def _prepare_paired_sandboxes(
    *,
    repo_path: Path,
    broken_dir: Path,
    fixed_dir: Path,
    gold_patch: str,
) -> None:
    _clone_repo(repo_path, broken_dir)
    _clone_repo(repo_path, fixed_dir)
    _apply_patch(fixed_dir, gold_patch)
    # Phase 4A item 4.4: pin sandbox state for capture_oracle's validator.
    try:
        from .evaluation.oracle_capture import write_oracle_state_sentinel

        write_oracle_state_sentinel(broken_dir, "pre_fix")
        write_oracle_state_sentinel(fixed_dir, "post_fix")
    except Exception:  # pragma: no cover - defensive
        pass


def _persist_testgen_run_to_repo_memory(
    *,
    repo_path: Path,
    persist: bool,
    directory: Optional[str],
    focus_files: Optional[list[str]],
    f2p_summary: Optional[dict[str, Any]] = None,
    mutation_summary: Optional[dict[str, Any]] = None,
    coverage_gap_summary: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Phase I.7: extract cross-task testgen insights from a completed
    run and merge them into the repo's RepoMemoryStore.

    Returns the store's merge summary so callers can attach it to
    their diagnostics. Returns None when persistence is disabled or
    no insights are extracted.
    """
    if not persist:
        return None
    try:
        from .persistence import (
            extract_testgen_insights_from_run_summary,
            persist_testgen_insights_for_repo,
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("Repo memory persistence skipped: import failed (%s)", exc)
        return None
    insights = extract_testgen_insights_from_run_summary(
        focus_files=focus_files,
        f2p_summary=f2p_summary,
        mutation_summary=mutation_summary,
        coverage_gap_summary=coverage_gap_summary,
    )
    if not insights:
        return None
    try:
        return persist_testgen_insights_for_repo(
            repo_path=str(repo_path),
            insights=insights,
            directory=directory,
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("Repo memory persistence failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Mode 4 (Phase 6.1): generate_both_self_play
# ---------------------------------------------------------------------------


def run_generate_both_self_play(
    *,
    repo_path: str | Path,
    problem_statement: str,
    output_dir: str | Path,
    K: int = 4,
    M: int = 4,
    parallelism: int = 4,
    test_generator: Optional[TestGenerator] = None,
    code_generator: Optional[CodeGenerator] = None,
    language: str = "python",
    install_repo: bool = False,
    timeout_seconds: float = 300.0,
) -> ModeResult:
    """Mode 4 (Phase 6.1): adversarial test-vs-patch self-play.

    Generates K patches AND M test suites INDEPENDENTLY (no hand-off
    between them), then evaluates the K x M tournament via
    :class:`apex.capabilities.self_play.SelfPlayTournament`. The
    selected (patch, test_suite) pair is the one that maximises mutual
    confidence (the patch survives most other tests; the chosen test
    kills most other patches).

    Cost: K + M generations + K*M test executions. Default K=4, M=4
    (so 8 generations + 16 evals). Smaller K/M shrinks cost
    proportionally; the tournament still runs with K=1 or M=1, just
    degenerately.

    The K patches and M tests are produced via direct invocation of
    the ``test_generator`` and ``code_generator`` callables in
    parallel, against fresh per-rollout sandbox clones. The patches
    are seeded with the FIRST non-empty generated test suite so the
    code-generator has a concrete spec to anchor on; the seed suite
    is NOT used in the tournament — only the M independently-generated
    suites enter the K x M matrix.

    Returns a :class:`ModeResult` with:
      * ``patch`` = the selected patch
      * ``test_artifacts`` = the selected test suite
      * ``diagnostics["self_play"]`` = full tournament report
      * ``success`` = True iff the selected pair is internally
        consistent (the chosen test passes under the chosen patch).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from .capabilities.self_play import SelfPlayTournament

    repo = Path(repo_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    K = max(1, int(K))
    M = max(1, int(M))
    parallelism = max(1, int(parallelism))

    test_gen = test_generator or _default_test_generator
    code_gen = code_generator or _default_code_generator

    test_suites: list[list[dict[str, Any]]] = [[] for _ in range(M)]
    test_errors: list[Optional[str]] = [None] * M
    patch_candidates: list[Optional[str]] = [None] * K
    patch_errors: list[Optional[str]] = [None] * K

    sandboxes_root = output / "_self_play_sandboxes"
    sandboxes_root.mkdir(parents=True, exist_ok=True)

    def _gen_test_suite(
        index: int,
    ) -> tuple[int, list[dict[str, Any]], Optional[str]]:
        suite_dir = sandboxes_root / f"testgen_{index}"
        try:
            _clone_repo(repo, suite_dir)
            artifacts = test_gen(suite_dir, problem_statement) or []
            return index, list(artifacts), None
        except Exception as exc:  # pragma: no cover - defensive
            return index, [], f"{type(exc).__name__}: {exc}"

    def _gen_patch(
        index: int, seed_tests: list[dict[str, Any]]
    ) -> tuple[int, Optional[str], Optional[str]]:
        patch_dir = sandboxes_root / f"codegen_{index}"
        try:
            _clone_repo(repo, patch_dir)
            patch = code_gen(patch_dir, problem_statement, seed_tests)
            return index, patch, None
        except Exception as exc:  # pragma: no cover - defensive
            return index, None, f"{type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=parallelism) as pool:
        futures = [pool.submit(_gen_test_suite, j) for j in range(M)]
        for fut in as_completed(futures):
            j, artifacts, err = fut.result()
            test_suites[j] = artifacts
            test_errors[j] = err

    seed_suite: list[dict[str, Any]] = next((suite for suite in test_suites if suite), [])

    with ThreadPoolExecutor(max_workers=parallelism) as pool:
        futures = [pool.submit(_gen_patch, i, seed_suite) for i in range(K)]
        for fut in as_completed(futures):
            i, patch, err = fut.result()
            patch_candidates[i] = patch
            patch_errors[i] = err

    valid_patches = [
        {"index": i, "patch": p} for i, p in enumerate(patch_candidates) if p and p.strip()
    ]
    valid_tests = [{"index": j, "artifacts": s} for j, s in enumerate(test_suites) if s]

    if not valid_patches or not valid_tests:
        return ModeResult(
            mode=MODE_GENERATE_BOTH_SELF_PLAY,
            success=False,
            error=(
                "self-play could not assemble a tournament: "
                f"valid_patches={len(valid_patches)} valid_tests={len(valid_tests)}"
            ),
            diagnostics={
                "patch_errors": [e for e in patch_errors if e],
                "test_errors": [e for e in test_errors if e],
                "K_patches_requested": K,
                "M_tests_requested": M,
            },
        )

    from .evaluation import evaluate_tdd_iteration

    def _verdict(patch_candidate: dict[str, Any], test_candidate: dict[str, Any]) -> int:
        i = int(patch_candidate["index"])
        j = int(test_candidate["index"])
        verdict_dir = sandboxes_root / f"verdict_{i}_{j}"
        broken_dir = verdict_dir / "broken"
        fixed_dir = verdict_dir / "fixed"
        try:
            _clone_repo(repo, broken_dir)
            _clone_repo(repo, fixed_dir)
            _apply_patch(fixed_dir, patch_candidate["patch"])
        except _ModeError:
            return 0
        report = evaluate_tdd_iteration(
            broken_dir=broken_dir,
            fixed_dir=fixed_dir,
            test_artifacts=test_candidate["artifacts"],
            output_dir=verdict_dir / "_tdd_report",
            language=language,
            timeout_seconds=timeout_seconds,
            install_repo=install_repo,
        )
        summary = dict(report.get("summary") or {})
        return 1 if bool(summary.get("any_f2p")) else 0

    tournament = SelfPlayTournament(K_patches=K, M_tests=M, parallelism=parallelism)
    result = tournament.run(
        patch_candidates=valid_patches,
        test_candidates=valid_tests,
        verdict_fn=_verdict,
    )

    selected_patch_payload = result.selected_patch or {}
    selected_test_payload = result.selected_test or {}
    diagnostics = {
        "self_play": result.to_dict(),
        "patch_errors": [e for e in patch_errors if e],
        "test_errors": [e for e in test_errors if e],
        "K_patches_requested": K,
        "M_tests_requested": M,
        "K_patches_valid": len(valid_patches),
        "M_tests_valid": len(valid_tests),
    }
    return ModeResult(
        mode=MODE_GENERATE_BOTH_SELF_PLAY,
        success=bool(result.diagnostics.get("selection_internally_consistent")),
        test_artifacts=list(selected_test_payload.get("artifacts") or []),
        patch=selected_patch_payload.get("patch"),
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Mode 5 (Phase 6.4): testgen_with_active_learning
# ---------------------------------------------------------------------------


def run_testgen_with_active_learning(
    *,
    repo_path: str | Path,
    problem_statement: str,
    gold_patch: str,
    output_dir: str | Path,
    test_generator: Optional[TestGenerator] = None,
    targeted_test_generator: Optional[
        Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
    ] = None,
    max_iterations: int = 3,
    top_k_mutants_per_iteration: int = 8,
    language: str = "python",
    install_repo: bool = False,
    timeout_seconds: float = 300.0,
) -> ModeResult:
    """Mode 5 (Phase 6.4): mutation-targeted active learning testgen.

    Generates an initial test suite, runs mutation testing, and
    iteratively grows the suite by generating tests targeted at
    surviving mutants. The loop stops when the kill rate stabilises,
    when ``max_iterations`` is hit, or when all mutants are killed.

    Requires a ``gold_patch`` so we can build the fixed sandbox needed
    by the mutation engine. Callers without a gold patch should use
    Mode 4 (self-play) or Mode 1 (testgen_with_fix surrogate path)
    instead.

    The default ``targeted_test_generator`` reuses the supplied
    ``test_generator`` with an augmented problem statement that
    embeds the surviving mutants. Callers wanting the formal
    mutation-attack prompt template at
    ``apex/capabilities/prompts/mutation_attack.txt`` should pass an
    explicit ``targeted_test_generator``.
    """
    from .capabilities.active_learning import MutationActiveLearner
    from .evaluation.mutation_engine import (
        evaluate_mutation_score,
        generate_mutants,
        source_paths_from_patch,
    )

    repo = Path(repo_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sandboxes = output / "_active_learning_sandboxes"
    broken_dir = sandboxes / "broken"
    fixed_dir = sandboxes / "fixed"

    try:
        _prepare_paired_sandboxes(
            repo_path=repo,
            broken_dir=broken_dir,
            fixed_dir=fixed_dir,
            gold_patch=gold_patch or "",
        )
    except _ModeError as exc:
        return ModeResult(
            mode=MODE_TESTGEN_WITH_ACTIVE_LEARNING,
            success=False,
            error=str(exc),
        )

    test_gen = test_generator or _default_test_generator
    try:
        initial_artifacts = test_gen(broken_dir, problem_statement) or []
    except Exception as exc:
        return ModeResult(
            mode=MODE_TESTGEN_WITH_ACTIVE_LEARNING,
            success=False,
            error=f"test_generator raised: {type(exc).__name__}: {exc}",
        )
    if not initial_artifacts:
        return ModeResult(
            mode=MODE_TESTGEN_WITH_ACTIVE_LEARNING,
            success=False,
            error="test_generator produced no artifacts (initial pass)",
        )

    _materialize_test_artifacts(fixed_dir, initial_artifacts)

    mutated_paths = source_paths_from_patch(gold_patch, language=language) or []
    mutants: list[Any] = []
    for sp in mutated_paths:
        produced = generate_mutants(
            source_path=fixed_dir / sp,
            language=language,
        )
        # generate_mutants emits Mutants with their source_path field
        # set to the absolute path passed in. Rewrite to repo-relative
        # so the mutation runner writes them back into the right slot.
        for m in produced:
            try:
                m.source_path = sp
            except Exception:
                pass
        mutants.extend(produced)

    test_paths = [a["path"] for a in initial_artifacts if a.get("path")]
    initial_report = evaluate_mutation_score(
        fixed_dir=fixed_dir,
        mutants=mutants,
        test_paths=test_paths,
        language=language,
    )

    if targeted_test_generator is None:

        def _default_targeted(
            survivors: list[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            if not survivors:
                return []
            mutant_block = "\n".join(
                f"- {s['operator']} at {s['source_path']}:{s['line']} "
                f"(original=`{s['original_snippet']}` mutated=`{s['mutated_snippet']}`)"
                for s in survivors[:8]
            )
            augmented_problem = (
                f"{problem_statement}\n\n"
                "## Surviving mutants to attack\n"
                "Write tests that distinguish the original behaviour from "
                "each of these mutated variants:\n"
                f"{mutant_block}"
            )
            try:
                return list(test_gen(broken_dir, augmented_problem) or [])
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("default targeted_test_generator raised: %s", exc)
                return []

        targeted_test_generator = _default_targeted

    def _runner(suite: list[dict[str, Any]]) -> Any:
        _materialize_test_artifacts(fixed_dir, suite)
        suite_paths = [a["path"] for a in suite if a.get("path")]
        return evaluate_mutation_score(
            fixed_dir=fixed_dir,
            mutants=mutants,
            test_paths=suite_paths,
            language=language,
        )

    learner = MutationActiveLearner(
        max_iterations=max_iterations,
        top_k_mutants_per_iteration=top_k_mutants_per_iteration,
    )
    enhanced = learner.attack_surviving_mutants(
        test_suite=initial_artifacts,
        mutation_report=initial_report,
        targeted_test_generator=targeted_test_generator,
        mutation_runner=_runner,
    )

    diagnostics = {
        "active_learning": enhanced.to_dict(),
        "initial_kill_rate": enhanced.initial_kill_rate,
        "final_kill_rate": enhanced.final_kill_rate,
        "iterations_run": enhanced.iterations_run,
        "stop_reason": enhanced.stop_reason.value,
    }
    return ModeResult(
        mode=MODE_TESTGEN_WITH_ACTIVE_LEARNING,
        success=enhanced.final_kill_rate >= enhanced.initial_kill_rate,
        test_artifacts=list(enhanced.test_artifacts),
        mutation_summary={
            "initial_kill_rate": enhanced.initial_kill_rate,
            "final_kill_rate": enhanced.final_kill_rate,
            "final_surviving": enhanced.final_surviving,
            "iterations_run": enhanced.iterations_run,
            "stop_reason": enhanced.stop_reason.value,
        },
        diagnostics=diagnostics,
    )
