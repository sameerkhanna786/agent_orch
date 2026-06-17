"""
Prompt templates for solver agents and the selector.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from ..controller_policy import (
    EVIDENCE_MODE_EVAL_ONLY_SUITE,
    EVIDENCE_MODE_GOLD_SUITE_VISIBLE,
    EVIDENCE_MODE_NO_SUITE_VISIBLE,
    EVIDENCE_MODE_PARTIAL_SUITE_VISIBLE,
)
from ..core.config import PromptStrategy
from ..core.pytest_report_utils import protected_test_files_from_context
from ..planning.manager import IssuePlan, RolloutBrief, TestContext
from ..rollout.localizer_scope import is_repo_relative_editable_path
from ..test_portfolio import (
    extract_issue_contract_targets,
    normalize_test_generation_design_payload,
)
from .artifacts import (
    coerce_localization_artifact,
    coerce_reproduction_artifact,
    render_artifact_json,
)

SOLVER_SYSTEM_PROMPT = """\
You are an expert software engineer resolving a repository-level issue.

Work in the current workspace. Inspect, edit, run relevant validation, and keep
the patch focused. Treat existing tests as specification unless the task
explicitly permits completing placeholder test bodies.
If the work splits into a few independent threads, you may call
delegate_subtasks to launch APEX-managed child agents in isolated workspaces.
"""


REPRODUCER_SYSTEM_PROMPT = """\
You are a reproduction specialist. Write the smallest reliable reproduction that
captures the bug as a passing command once the fix is in place. Explore only as
much of the repository as needed, run the reproduction, and submit the artifact
through submit_reproduction. Treat existing visible tests as part of the
specification and avoid editing them unless the task explicitly requires it.
Use only the current workspace, visible evaluator evidence, in-repo
documentation/examples, and local runtime feedback.
Use delegate_subtasks only when the reproduction effort clearly splits into
independent lines of investigation.
"""


LOCALIZER_SYSTEM_PROMPT = """\
You are a fault-localization specialist. Use the issue, reproduction evidence,
and repository context to identify the files, symbols, and hypotheses most likely
to contain the root cause. Submit a concise ranked summary through submit_localization.
Use only the current workspace, visible evaluator evidence, in-repo
documentation/examples, and local runtime feedback.
You may call delegate_subtasks for clearly separable hypotheses or file clusters.
"""


TEST_WRITER_SYSTEM_PROMPT = """\
You are a test engineer. Build a focused test portfolio in the repository's
native framework (for example pytest, jest, vitest, go test, cargo test, or
JUnit), not one monolithic suite. Cover the bug report and failing traces first,
then add the strongest justified nearby tests: contract/API checks mined from
docs, examples, types, and existing tests; edge and negative cases; and, when
appropriate, property, metamorphic, differential, and fuzz-seed artifacts.
Existing tests define the intended behavior, so any new coverage should reinforce
that contract instead of weakening it.

When the issue centers on a named API, method, CLI, or protocol surface,
build an explicit contract matrix on that surface itself: cover the canonical
success path, an empty or missing-input boundary, a malformed or invalid-input
path, and multiplicity or ordering when the contract is collection-shaped.
The harness now scores per-axis coverage; missing an axis without an explicit
justification on the artifact lowers the per-task score. Each artifact must
declare the specific axes it covers in `contract_axes` and must not list an
axis it does not actually exercise. Alias or compatibility tests may complement
that matrix but must not replace direct coverage of the primary entry point.

CRITICAL — every test you write MUST satisfy the Fail-to-Pass (F2P) contract:
it must FAIL on the broken (un-patched) code AND PASS on the fixed (patched)
code. Tests that pass on both versions (P2P) are useless because they do not
distinguish broken behavior from fixed behavior; they will be flagged as
overfitting and lower the per-task score even when they appear in the gold
target list. Tests that fail on both (F2F) usually mean a syntax / fixture
bug — fix the test infrastructure first. Tests that pass on broken but fail
on fixed (P2F) are regressions — your assertion contradicts the intended
contract.

P2F ANTIPATTERN — common failure mode to avoid: you observe what the
worktree returns (the BROKEN behavior), then assert that as the expected
value. Example: bug report says `add(a,b)` should return `a + b`, but the
worktree's `add` actually returns `a - b`. If you write
``assert add(2, 3) == -1`` based on observed output, the test PASSES on
broken (which returns -1) and FAILS on fixed (which returns 5) — that's
a P2F regression. ALWAYS assert the EXPECTED contract from the issue /
docs / type signatures, NOT what the broken code currently returns. If
you cannot tell what the expected output is, write a property assertion
(``assert add(a, b) == add(b, a)`` for commutativity, etc.) instead of a
literal value comparison. The harness now runs a real F2P oracle (broken sandbox vs fixed
sandbox), and the rollout selector ranks candidates by
(any_f2p, mutation_score, f2p_count, f2p_rate) so a portfolio that catches
the bug AND survives mutation discrimination wins.

PROPERTY-BASED + METAMORPHIC TESTS — when the contract is shaped right
for them, populate the `properties` and `metamorphic_relations` fields on
each artifact. They exercise the contract more broadly than example-based
tests and routinely beat pure example-based suites on mutation score:
  * properties: invariants the contract must satisfy across many inputs.
    Examples: ``len(sorted(xs)) == len(xs)``, ``decode(encode(x)) == x``,
    ``add(a, b) == add(b, a)``, ``parse(x) is None or parse(x).valid``.
    Use hypothesis (Python), fast-check (JS/TS), proptest (Rust),
    quickcheck (Go), jqwik (Java) when available — fall back to a
    handful of representative inputs otherwise.
  * metamorphic_relations: transformations between inputs that the
    contract must preserve. Examples: ``f(reverse(x)) == reverse(f(x))``
    for symmetric ops, ``len(filter(p, xs)) <= len(xs)`` for predicates,
    ``cache.get(k) == cache.get(k)`` for idempotency, ``f(merge(a,b))``
    consistency. These catch bugs example-tests miss because they
    constrain RELATIONS instead of single outputs.
Skip these fields when the contract is genuinely shape-free (one-shot
constructors, pure logging, etc.) — populating them with trivia hurts.

REQUIRED — adversarial edge prediction. BEFORE producing test_artifacts,
populate the `predicted_edges` field on submit_test_suite with a structured
list of the bug's likely edge surfaces. For each predicted edge specify:
  * edge_type (one of: boundary, off_by_one, null_vs_empty, return_type,
    exception_path, ordering, encoding, concurrency, other)
  * location (file:line or symbol where the edge lives, when known)
  * rationale (why you think the gold patch likely changes this surface)
  * test_artifact_paths (list of test files in YOUR portfolio that exercise
    this edge — fill after you write the tests)

This forces structured reasoning: a test that targets a predicted edge is
much more likely to kill the mutant the gold patch produces than one that
generically asserts "function returns something". The post-iteration
feedback loop counts predicted vs exercised edges; unaddressed predictions
appear in the next iteration's prompt as gaps to fill.
For each artifact, include its path, content, strategy, justification, and
the files/tests/contracts it is grounded in. Include `contract_targets` only for
the API or entry points that artifact directly exercises, and use
`contract_axes` to declare which parts of the contract matrix it covers.
If you add tests into an existing repository test file, set
`materialization_mode` to `append`; use `replace` only when you are
intentionally returning the full rewritten file. Only mark an artifact as promotable
when it is independently justified, non-redundant, and suitable for stable
baseline-preserving validation; otherwise keep it exploratory. Submit the final
portfolio through submit_test_suite.
If delegation is enabled, decompose the work into a contract-mining pass,
multiple diverse test generators, and a separate adjudication pass that decides
which artifacts are promotable versus exploratory.
Use only the current workspace, visible evaluator evidence, in-repo
documentation/examples, and local runtime feedback.
You may call delegate_subtasks when separate test scenarios can be developed
independently and then adjudicated back into one portfolio.
"""


def gold_suite_lockdown_clause(issue_plan: Optional[IssuePlan]) -> str:
    # Gold-suite integrity is enforced by the harness, sanitizer, and final gates;
    # do not spend agent role-prompt context on benchmark policy boilerplate.
    return ""


def _incomplete_test_files(issue_plan: IssuePlan) -> list[str]:
    return [
        str(path).strip()
        for path in list(getattr(issue_plan.test_context, "incomplete_test_files", []) or [])
        if str(path).strip()
    ]


def _completion_scaffold_success_criteria(
    issue_plan: IssuePlan,
    rollout_brief: RolloutBrief,
) -> list[str]:
    criteria = list(rollout_brief.success_criteria or issue_plan.success_criteria or [])
    incomplete_tests = _incomplete_test_files(issue_plan)
    if not incomplete_tests:
        return criteria
    replacement = (
        "Complete only explicit TODO/NotImplemented test bodies in "
        "`incomplete_test_files` when needed; leave all other visible tests unchanged."
    )
    filtered: list[str] = []
    replaced = False
    for raw in criteria:
        text = str(raw or "").strip()
        lowered = text.lower()
        conflicts = (
            ("no test files" in lowered and ("modified" in lowered or "changed" in lowered))
            or "no test edits" in lowered
            or "without test edits" in lowered
            or ("do not modify" in lowered and "test" in lowered)
            or ("never modify" in lowered and "test" in lowered)
        )
        if conflicts:
            if not replaced:
                filtered.append(replacement)
                replaced = True
            continue
        filtered.append(text)
    if not replaced:
        filtered.append(replacement)
    return filtered


def _render_incomplete_test_completion_block(
    issue_plan: IssuePlan,
    *,
    concise: bool,
) -> list[str]:
    incomplete_tests = _incomplete_test_files(issue_plan)
    if not incomplete_tests:
        return []
    return [
        "",
        "# Incomplete Test Placeholder Allowlist",
        "These visible test files are the only test files with editable placeholder bodies:",
        _render_items(
            incomplete_tests,
            fallback="- none",
            concise=concise,
            limit=8,
        ),
        "If full-suite validation still fails with `NotImplementedError`, `pass`, or TODO in one of these files, treat that as an actionable contract-completion target. Replace only the placeholder body with assertions derived from the function name and adjacent completed tests.",
        "Do not edit imports, decorators, parametrization, fixtures, shared helpers, expected outputs, or non-placeholder test bodies.",
    ]


def build_stage_system_prompt(
    base_prompt: str,
    *,
    allow_delegation: bool = False,
    delegation_mode: str = "apex_tool",
    issue_plan: Optional[IssuePlan] = None,
) -> str:
    if allow_delegation:
        return base_prompt
    raw_lines = base_prompt.splitlines()
    lines: list[str] = []
    skip_continuation_line = False
    for index, line in enumerate(raw_lines):
        stripped = line.strip().lower()
        if skip_continuation_line:
            skip_continuation_line = False
            continue
        next_stripped = raw_lines[index + 1].strip().lower() if index + 1 < len(raw_lines) else ""
        if "delegate_subtasks" in stripped:
            skip_continuation_line = bool(next_stripped) and not stripped.endswith(".")
            continue
        if stripped.endswith("you may call") and next_stripped.startswith("delegate_subtasks"):
            continue
        if stripped.startswith("if delegation is enabled"):
            skip_continuation_line = bool(next_stripped) and not stripped.endswith(".")
            continue
        lines.append(line)
    return "\n".join(lines).strip()


STRATEGY_INSTRUCTIONS = {
    PromptStrategy.MINIMAL: (
        "Procedural recipe — apply IN ORDER:\n"
        "  1. Run the targeted failing tests FIRST (using the repository's test "
        "command from the Test Context) to see the exact failure messages.\n"
        "  2. Identify the smallest source-level change that would address the "
        "failure (the single function, method, or expression that is wrong — "
        "in any language).\n"
        "  3. Make ONE focused edit at that minimum scope; do not refactor, "
        "rename, or clean up adjacent code.\n"
        "  4. Re-run only the targeted failing tests to verify; if they pass, stop.\n"
        "  5. Only widen the patch if a previously-passing test regressed."
    ),
    PromptStrategy.COMPREHENSIVE: (
        "Procedural recipe — apply IN ORDER:\n"
        "  1. Read the dependency / import chain from the package's entry point "
        "down to the failing unit BEFORE editing any code. The entry point "
        "varies by language (e.g. ``__init__.py`` in Python, ``index.{js,ts}`` "
        "in Node, ``lib.rs``/``mod.rs`` in Rust, the main package file in Go, "
        "``Application``/``Module`` class in Java/Kotlin, etc.).\n"
        "  2. Identify the bug AND the nearby invariants the fix must preserve "
        "(other call sites, public API contracts, edge cases).\n"
        "  3. Make a structurally complete fix — ideally one that addresses the "
        "root cause and naturally handles related edge cases.\n"
        "  4. Validate against the broader test suite, not just the originally "
        "failing tests, to confirm no regressions."
    ),
    PromptStrategy.TEST_DRIVEN: (
        "Procedural recipe — apply IN ORDER:\n"
        "  1. Open EACH failing test file FIRST and read the assertions backwards "
        "to understand exactly what behavior is being checked.\n"
        "  2. Quote the failing assertions in your reasoning so the contract is "
        "explicit before any source change.\n"
        "  3. Implement the source change to make the asserted behavior hold.\n"
        "  4. Validate against ALL tests (not just the originally failing ones) "
        "to ensure no regression.\n"
        "  5. If a fix appears to require changing a test assertion, STOP — that "
        "is almost always a sign the source patch is wrong."
    ),
}

# Per-rollout exploration directives, indexed by the rollout's index in
# the wave. The point is to drive different *first actions* across
# rollouts that would otherwise share the same prompt — Codex CLI
# silently ignores the temperature parameter in our config (the
# Plugboard wrapper does not forward it to the model), so the rollout
# diversity machinery has to live entirely in prompt content. Without
# this directive, repeated rollouts on broad completion tasks produced
# byte-identical patches with the same residual failures
# because the model received nearly-identical context every time.
#
# Each directive is paired with a stylistic angle so the agent's first
# read targets a different region of the codebase: failing-test call
# site, dependency chain entry, or stub-marker scan. Directives are
# language-agnostic: they reference language-neutral concepts and call
# out multi-language examples where the canonical artifact differs
# (e.g. ``__init__.py`` in Python vs ``mod.rs`` in Rust). The seed
# count is chosen to align with the typical wave width. For waves
# wider than the directive list, the index wraps via modulo.
# Phase G.5: per-rollout test_writer "morphs" (e-Otter++ inspired). Each
# parallel test_writer rollout gets a different LENS on the same bug so
# the F2P-tuple selector has diverse candidates to choose from. The
# OpenHands SOTA Nov 2025 result and the e-Otter++ ICSE'26 paper both
# confirm prompt-diversity inference-time scaling is the biggest
# published delta in agentic testgen.
#
# Cost: zero new compute. Each morph adds ~100-150 tokens to the prompt
# and uses the existing parallel-rollout infrastructure (default 4
# rollouts → cycles through these 4 morphs).
_TESTGEN_ROLLOUT_MORPHS = (
    (
        "positive_path",
        "Your lens for this rollout: lead with the canonical happy-path "
        "acceptance test FIRST. Establish a passing happy-path before any "
        "boundary or negative test. Most contracts have one obviously-correct "
        "input shape — write that test first, then layer the harder cases "
        "on top.",
    ),
    (
        "boundary",
        "Your lens for this rollout: lead with BOUNDARY conditions — empty "
        "input, null/None, single-element collection, max-size input, "
        "off-by-one boundaries, default values. The bug is most likely in "
        "how the code handles values at the edge of its valid range. Write "
        "the boundary tests FIRST and only add positive-path tests once "
        "boundaries are covered.",
    ),
    (
        "negative_path",
        "Your lens for this rollout: lead with NEGATIVE / error paths — "
        "malformed input, type errors, missing required fields, exception "
        "shapes. The bug is most likely about how the code reacts to invalid "
        "input rather than valid input. Write the error-path tests FIRST "
        "and verify each raises the EXPECTED exception type with the "
        "EXPECTED message shape.",
    ),
    (
        "ordering_multiplicity",
        "Your lens for this rollout: lead with MULTI-ELEMENT and ORDERING "
        "behavior — collections with 0/1/many elements, ordering of "
        "side-effects, repeated calls with state, concurrent access if "
        "relevant. The bug is most likely about how the code handles "
        "state/order across multiple invocations. Write the multi-element "
        "and ordering tests FIRST.",
    ),
)


_ROLLOUT_EXPLORATION_DIRECTIVES = (
    (
        "Open the FIRST failing test's source and identify the call site that "
        "raises (or returns the wrong value). Edit at the smallest scope that "
        "makes that specific test pass. Resist broadening the patch until the "
        "targeted test is green."
    ),
    (
        "Trace the dependency / import chain from the package's entry point down "
        "to the failing unit (e.g. ``__init__.py`` in Python, ``index.{js,ts}`` "
        "in Node, ``lib.rs``/``mod.rs`` in Rust, the main package in Go, the "
        "module aggregator class in Java/Kotlin). Look for module-level state, "
        "missing exports, or removed names that could break loading. Edit only "
        "after you understand which dependency edge broke."
    ),
    (
        "Scan the focus files for unfilled stub markers — language varies: "
        "``raise NotImplementedError`` / ``pass  # TODO`` (Python), "
        "``throw new Error('not implemented')`` / ``// TODO`` (JS / TS), "
        '``panic!("todo")`` / ``unimplemented!()`` / ``todo!()`` (Rust), '
        '``return errors.New("not implemented")`` / ``// TODO`` (Go), '
        "``throw new UnsupportedOperationException(...)`` / ``// TODO`` "
        "(Java / Kotlin). Implement those FIRST in declaration order — the "
        "scorer typically expects every marker to be replaced with a real "
        "implementation."
    ),
    (
        "Run the repository's test-discovery command (e.g. "
        "``pytest --collect-only -q`` for Python, ``jest --listTests`` or "
        "``vitest list`` for Node, ``go test -list .`` for Go, "
        "``cargo test -- --list`` for Rust, ``mvn test -Dtest=*`` for Java) "
        "BEFORE making any edit. Compare the discovered test count against "
        "the count cited in the Issue / Test Context. If discovery "
        "undercounts, fix discovery (markers, framework config, parametrize / "
        "test-list filters) BEFORE source edits."
    ),
    (
        "Reproduce the failure in a minimal stand-alone script (or single-test "
        "invocation) first so you can iterate quickly without the full test "
        "harness. Once the minimal repro exhibits the bug, fix the underlying "
        "source until the repro passes, then re-run the originally failing "
        "test set."
    ),
)


def _rollout_index_from_brief(rollout_brief: Optional[RolloutBrief]) -> int:
    """Stable per-rollout index used to pick an exploration directive.

    Pulled out of the brief's search_policy so the directive is
    consistent across stages (reproducer / localizer / patcher /
    test-writer all see the same seed for a given rollout). Falls back
    to 0 when the policy lacks an explicit identifier.
    """

    if rollout_brief is None:
        return 0
    policy = (
        rollout_brief.search_policy
        if isinstance(getattr(rollout_brief, "search_policy", None), dict)
        else {}
    )
    for key in (
        "rollout_diversity_index",
        "rollout_index",
        "rollout_id",
    ):
        raw = policy.get(key)
        if raw is None:
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            return value
    # Fall back to a hash of the brief's title — same brief always
    # picks the same directive even when no explicit id is set.
    title = str(getattr(rollout_brief, "title", "") or "").strip()
    if title:
        return abs(hash(title)) % len(_ROLLOUT_EXPLORATION_DIRECTIVES)
    return 0


def rollout_exploration_directive(
    rollout_brief: Optional[RolloutBrief],
) -> str:
    """Return the per-rollout exploration directive string."""

    if not _ROLLOUT_EXPLORATION_DIRECTIVES:
        return ""
    index = _rollout_index_from_brief(rollout_brief) % len(_ROLLOUT_EXPLORATION_DIRECTIVES)
    return _ROLLOUT_EXPLORATION_DIRECTIVES[index]


def stage_first_action_directive(
    stage_name: str,
    rollout_brief: Optional[RolloutBrief],
) -> str:
    stage = str(stage_name or "").strip().lower()
    base_directive = rollout_exploration_directive(rollout_brief)
    if stage == "reproducer":
        return (
            "Identify the smallest repo-visible command or script that reproduces the issue. "
            "Inspect only the files needed to build that reproduction, and do not edit application "
            "or source files in this stage."
        )
    if stage == "localizer":
        return (
            "Map the likely root-cause files and symbols from the issue, reproduction, and nearby tests. "
            "This stage is read-only diagnosis: do not edit application or source files."
        )
    if stage == "test_writer":
        # Phase G.5: pick a per-rollout morph so parallel rollouts have
        # diverse lenses on the bug. The base directive stays for ALL
        # rollouts; the morph is appended.
        base = (
            "Mine the issue-declared contract and existing repo-visible evidence first, then draft the "
            "smallest high-signal synthetic portfolio that covers the required public surface. Do not "
            "modify application or source files in this stage."
        )
        if _TESTGEN_ROLLOUT_MORPHS:
            morph_index = _rollout_index_from_brief(rollout_brief) % len(_TESTGEN_ROLLOUT_MORPHS)
            morph_label, morph_text = _TESTGEN_ROLLOUT_MORPHS[morph_index]
            return f"{base}\n\nMorph ({morph_label}): {morph_text}"
        return base
    return base_directive


def permute_focus_files(
    focus_files: list[str],
    rollout_brief: Optional[RolloutBrief],
) -> list[str]:
    """Return ``focus_files`` rotated by a per-rollout offset.

    The agent typically opens files in the order presented. Rotating
    the list is enough to drive different first reads without dropping
    any file or reordering arbitrarily. Deterministic per rollout
    (same brief → same rotation) so a re-run of the same rollout
    produces the same prompt.
    """

    files = [path for path in (focus_files or []) if path]
    if len(files) <= 1:
        return files
    offset = _rollout_index_from_brief(rollout_brief) % len(files)
    if offset == 0:
        return files
    return files[offset:] + files[:offset]


def _issue_plan_regime_probability(issue_plan: IssuePlan, state: str) -> float:
    regime = getattr(issue_plan, "task_regime", None)
    probability = getattr(regime, "probability", None)
    if callable(probability):
        try:
            return float(probability(state) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    if isinstance(regime, dict):
        probabilities = regime.get("state_probabilities")
        if isinstance(probabilities, dict):
            return float(probabilities.get(state) or 0.0)
    return 0.0


def _requires_broad_completion_validation(
    issue_description: str,
    issue_plan: IssuePlan,
) -> bool:
    _ = issue_description
    features = dict(issue_plan.allocator_features or {})
    if bool(features.get("is_completion_task") or features.get("mentions_public_api")):
        return True

    if _issue_plan_regime_probability(issue_plan, "contract_gap") >= 0.5:
        return True

    if _issue_plan_regime_probability(issue_plan, "high_interface_risk") >= 0.45:
        return True

    return bool(
        issue_plan.test_context.incomplete_source_files
        or issue_plan.test_context.incomplete_test_files
    )


def _is_delegated_subtask_prompt(
    issue_plan: IssuePlan,
    rollout_brief: RolloutBrief,
) -> bool:
    planner_metadata = (
        issue_plan.planner_metadata if isinstance(issue_plan.planner_metadata, dict) else {}
    )
    if bool(planner_metadata.get("delegated_subtask")):
        return True
    search_policy = (
        rollout_brief.search_policy if isinstance(rollout_brief.search_policy, dict) else {}
    )
    return bool(
        search_policy.get("delegated_subtask") or search_policy.get("delegated_subtask_title")
    )


def _delegated_subtask_kind(
    issue_plan: IssuePlan,
    rollout_brief: RolloutBrief,
) -> str:
    planner_metadata = (
        issue_plan.planner_metadata if isinstance(issue_plan.planner_metadata, dict) else {}
    )
    kind = str(planner_metadata.get("delegated_subtask_kind") or "").strip().lower()
    if kind:
        return kind
    search_policy = (
        rollout_brief.search_policy if isinstance(rollout_brief.search_policy, dict) else {}
    )
    return str(search_policy.get("delegated_subtask_kind") or "").strip().lower()


def _delegated_subtask_scope_rules(*, validation_only: bool, bullet: bool) -> list[str]:
    prefix = "- " if bullet else ""
    if validation_only:
        return [
            prefix
            + "This is an orchestrator-scoped validation-only subtask. Reuse the assigned validation slice and report blockers; do not infer or claim new implementation ownership.",
            prefix
            + "Do not broaden into unrelated files or repository-wide validation. The parent orchestrator will handle implementation and integration work.",
            prefix
            + "If validation points to likely implementation files outside this slice, report the evidence and candidate files in followups for the parent integrator.",
            prefix
            + "Tooling enforces this read-only boundary. Repository-file writes in this subtask will be rejected.",
        ]
    return [
        prefix
        + "This is an orchestrator-scoped implementation subtask. Focus files are starting points for investigation and editing, not exclusive ownership boundaries.",
        prefix
        + "Do not broaden into an unrelated repository-wide sweep; broaden only when local evidence, imports, or targeted validation points to adjacent implementation files.",
        prefix
        + "If adjacent source edits are necessary, keep them minimal, explain the evidence in the summary or followups, and run the assigned validation slice.",
        prefix
        + "Forbidden files and protected tests remain hard boundaries; do not edit them unless explicitly allowed.",
    ]


def _render_items(
    items: list[str],
    *,
    fallback: str,
    concise: bool,
    limit: int,
) -> str:
    values = [item for item in items if item]
    if concise:
        values = values[:limit]
    return "\n".join(f"- {item}" for item in values) or fallback


def _test_context_evidence_mode(test_context: TestContext) -> str:
    evidence_policy = (
        dict(test_context.evidence_policy or {})
        if isinstance(test_context.evidence_policy, dict)
        else {}
    )
    return str(evidence_policy.get("mode") or test_context.evidence_mode or "").strip().lower()


def _split_prompt_focus_files(
    issue_plan: IssuePlan,
    rollout_brief: RolloutBrief,
) -> tuple[list[str], list[str]]:
    """Split implementation focus from read-only/context paths while preserving rollout order."""

    test_context = issue_plan.test_context
    evidence_mode = _test_context_evidence_mode(test_context)
    incomplete_test_files = list(test_context.incomplete_test_files or [])
    editable: list[str] = []
    context_only: list[str] = []
    for path in permute_focus_files(rollout_brief.focus_files, rollout_brief):
        text = str(path or "").strip().replace("\\", "/")
        if not text:
            continue
        target = (
            editable
            if is_repo_relative_editable_path(
                text,
                evidence_mode=evidence_mode,
                incomplete_test_files=incomplete_test_files,
            )
            else context_only
        )
        if text not in target:
            target.append(text)
    return editable, context_only


def _render_focus_file_sections(
    issue_plan: IssuePlan,
    rollout_brief: RolloutBrief,
    *,
    fallback: str,
    concise: bool,
    limit: int,
) -> list[str]:
    editable_focus_files, context_only_files = _split_prompt_focus_files(
        issue_plan,
        rollout_brief,
    )
    sections = [
        "# Focus Files",
        _render_items(
            editable_focus_files,
            fallback=fallback,
            concise=concise,
            limit=limit,
        ),
    ]
    if context_only_files:
        sections.extend(
            [
                "",
                "# Read-only Context Files",
                _render_items(
                    context_only_files,
                    fallback="- none",
                    concise=concise,
                    limit=limit,
                ),
            ]
        )
    return sections


def _truncate_block(text: str, *, concise: bool, max_lines: int) -> str:
    content = text.strip()
    if not concise or not content:
        return content
    lines = content.splitlines()
    if len(lines) <= max_lines:
        return content
    remaining = len(lines) - max_lines
    return "\n".join(lines[:max_lines] + [f"... ({remaining} more lines omitted)"])


def _truncate_inline(text: str, *, concise: bool, max_chars: int) -> str:
    content = re.sub(r"\s+", " ", str(text or "")).strip()
    if not concise or not content or len(content) <= max_chars:
        return content
    return content[: max(0, max_chars - 3)].rstrip(" ,;:") + "..."


def _rollout_brief_is_verifier_validity_repair(rollout_brief: RolloutBrief) -> bool:
    policy = (
        rollout_brief.search_policy
        if isinstance(getattr(rollout_brief, "search_policy", None), dict)
        else {}
    )
    if bool(policy.get("verifier_validity_repair")):
        return True
    origin = str(policy.get("origin") or "").strip().lower()
    return origin == "verifier_validity_repair"


def _render_verifier_diagnostic_location(location: Any) -> str:
    if not isinstance(location, dict):
        return ""
    path = str(location.get("path") or "").strip()
    if not path:
        return ""
    try:
        line = int(location.get("line") or 0)
        column = int(location.get("column") or 0)
    except (TypeError, ValueError):
        line = 0
        column = 0
    rendered = path
    if line > 0:
        rendered += f":{line}"
    if column > 0:
        rendered += f":{column}"
    message = _truncate_inline(location.get("message"), concise=True, max_chars=180)
    if message:
        rendered += f": {message}"
    return rendered


def _build_verifier_validity_repair_prompt(
    *,
    issue_plan: IssuePlan,
    rollout_brief: RolloutBrief,
    test_command: str | None,
    concise: bool,
) -> str:
    policy = rollout_brief.search_policy if isinstance(rollout_brief.search_policy, dict) else {}
    focus_files = list(
        dict.fromkeys(
            [
                str(item).strip()
                for item in list(policy.get("action_file_paths") or rollout_brief.focus_files or [])
                if str(item).strip()
            ]
        )
    )
    diagnostics = [
        rendered
        for rendered in (
            _render_verifier_diagnostic_location(location)
            for location in list(policy.get("verifier_diagnostic_locations") or [])
        )
        if rendered
    ]
    objective = str(policy.get("verifier_repair_objective") or "").strip()
    if not diagnostics and objective:
        diagnostics = [objective]
    elif objective and objective not in diagnostics:
        diagnostics.insert(0, objective)
    additional_diagnostics = str(policy.get("additional_validity_diagnostics") or "").strip()
    if additional_diagnostics and additional_diagnostics not in diagnostics:
        diagnostics.append(additional_diagnostics)
    source_context = str(policy.get("verifier_diagnostic_source_context") or "").strip()

    sections = [
        "# Verifier Validity Repair",
        "Repair only the hard verifier diagnostics for the current best candidate. Do not re-solve the original task.",
        "",
        "# Owned Source Files",
        _render_items(
            focus_files,
            fallback="- use only the verifier-rejected source files named by the diagnostic output",
            concise=concise,
            limit=8,
        ),
        "",
        "# Diagnostics",
        _render_items(
            diagnostics,
            fallback="- inspect the verifier output in the rollout artifacts and repair the named static validity failures",
            concise=concise,
            limit=16,
        ),
    ]
    if source_context:
        sections.extend(
            [
                "",
                "# Diagnostic Source Context",
                _truncate_block(source_context, concise=concise, max_lines=80),
            ]
        )
    if issue_plan.repo_focus_map:
        sections.extend(
            [
                "",
                "# Focus Repo Map",
                _truncate_block(issue_plan.repo_focus_map, concise=concise, max_lines=42),
            ]
        )
    if test_command:
        sections.extend(["", "# Validation Command", test_command])
    sections.extend(
        [
            "",
            "# Execution Rules",
            "- Inspect each listed diagnostic line and its local data/control flow before editing.",
            "- Apply the smallest source change that makes the named verifier diagnostic disappear.",
            "- Your diff must include a hunk adjacent to at least one listed diagnostic line when line numbers are present; unrelated edits in the same file are invalid.",
            "- Do not edit tests, generated harness helpers, expected-test-id inventories, collection configuration, or benchmark metadata.",
            "- Preserve the candidate's already passing behavior; this round is only for hard validity repair.",
            "- Run syntax/static validation for edited files and the available validation command if feasible.",
            "- Return only a JSON object summarizing the patch and tests you ran.",
        ]
    )
    return "\n".join(sections)


def _humanize_test_inventory_framework(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "go_test": "go test",
        "cargo_test": "cargo test",
        "dotnet_test": "dotnet test",
    }
    return aliases.get(token, token.replace("_", " "))


def _render_artifact_block(
    title: str,
    artifact: Any,
    *,
    concise: bool,
    max_lines: int,
) -> list[str]:
    payload = artifact
    if concise and title == "Reproduction Artifact":
        reproduction = coerce_reproduction_artifact(artifact)
        if reproduction is not None:
            payload = {
                "summary": reproduction.summary,
                "command": reproduction.command,
                "script_path": reproduction.script_path,
            }
    elif concise and title == "Localization Artifact":
        localization = coerce_localization_artifact(artifact)
        if localization is not None:
            payload = {
                "summary": localization.summary,
                "files": list(localization.files[:6]),
                "symbols": list(localization.symbols[:4]),
                "hypotheses": list(localization.hypotheses[:3]),
            }

    rendered = render_artifact_json(payload).strip()
    if not rendered:
        return []
    return [
        "",
        f"# {title}",
        "```json",
        _truncate_block(rendered, concise=concise, max_lines=max_lines),
        "```",
    ]


def _render_test_context_block(
    test_context: TestContext,
    *,
    concise: bool,
) -> list[str]:
    inventory_source = str(test_context.test_inventory_source or "").strip().lower()
    evidence_policy = (
        dict(test_context.evidence_policy or {})
        if isinstance(test_context.evidence_policy, dict)
        else {}
    )
    evidence_mode = (
        str(evidence_policy.get("mode") or test_context.evidence_mode or "").strip().lower()
    )
    visible_tests_allowed = bool(
        evidence_policy.get("visible_tests_allowed_in_prompts")
    ) or inventory_source in {
        "commit0_public_test_inventory",
        "public",
        "repo_visible",
        "visible",
    }
    expected_inventory_prompt_visible = (
        evidence_mode == EVIDENCE_MODE_GOLD_SUITE_VISIBLE and visible_tests_allowed
    )
    protected_visible_test_files = (
        protected_test_files_from_context(test_context)
        if evidence_mode == EVIDENCE_MODE_GOLD_SUITE_VISIBLE
        else []
    )
    source_contract_files = list(
        dict.fromkeys(
            list(test_context.source_focus_files or [])
            + list(test_context.incomplete_source_files or [])
            + list(test_context.terminal_source_files or [])
        )
    )
    if not (
        test_context.summary
        or test_context.command
        or test_context.failing_test_ids
        or test_context.focus_test_files
        or test_context.incomplete_test_files
        or test_context.incomplete_source_files
        or test_context.terminal_source_files
        or test_context.exception_summaries
        or test_context.expectations
        or (
            expected_inventory_prompt_visible
            and (test_context.expected_test_count or test_context.expected_test_ids)
        )
        or test_context.test_inventory_framework
        or test_context.test_inventory_language
        or test_context.test_collection_command
        or evidence_mode
        or protected_visible_test_files
        or source_contract_files
    ):
        return []

    failing_limit = 3 if concise else 8
    file_limit = 3 if concise else 6
    incomplete_limit = 2 if concise else 6
    incomplete_source_limit = 2 if concise else 6
    source_limit = 2 if concise else 4
    exception_limit = 2 if concise else 4
    expectation_limit = 3 if concise else 8
    expected_id_limit = 4 if concise else 12
    contract_file_limit = 4 if concise else 8
    lines = ["", "# Test Context"]
    if evidence_mode:
        if evidence_mode == EVIDENCE_MODE_GOLD_SUITE_VISIBLE:
            lines.extend(
                [
                    "Evidence mode: gold_suite_visible.",
                    "The visible suite is the declared target evaluator; mine it for contracts and iterate directly against it without reducing collection.",
                ]
            )
            lines.extend(
                [
                    "",
                    "Gold-suite contract:",
                    "- Visible test IDs/files are read-only evaluator evidence: inspect and run them, but do not modify them.",
                    "- Source implementation files remain editable even if their names contain test-related words; follow the source sections below rather than filename guesses.",
                ]
            )
        elif evidence_mode == EVIDENCE_MODE_PARTIAL_SUITE_VISIBLE:
            lines.extend(
                [
                    "Evidence mode: partial_suite_visible.",
                    "Visible tests are contract examples and regression checks; passing them is necessary but may not be sufficient.",
                ]
            )
        elif evidence_mode == EVIDENCE_MODE_EVAL_ONLY_SUITE:
            lines.extend(
                [
                    "Evidence mode: eval_only_suite.",
                    "Official evaluator tests/results are not rollout prompt evidence; use only agent-visible repository evidence.",
                ]
            )
        elif evidence_mode == EVIDENCE_MODE_NO_SUITE_VISIBLE:
            lines.extend(
                [
                    "Evidence mode: no_suite_visible.",
                    "Create focused repros or validation commands from the issue and repository behavior.",
                ]
            )
    if evidence_mode == EVIDENCE_MODE_GOLD_SUITE_VISIBLE and protected_visible_test_files:
        lines.extend(
            [
                "",
                "Read-only gold-suite files:",
                _render_items(
                    protected_visible_test_files,
                    fallback="- none identified",
                    concise=concise,
                    limit=contract_file_limit,
                ),
            ]
        )
    if evidence_mode == EVIDENCE_MODE_GOLD_SUITE_VISIBLE and source_contract_files:
        lines.extend(
            [
                "",
                "Source implementation files from evaluator evidence:",
                _render_items(
                    source_contract_files,
                    fallback="- none identified",
                    concise=concise,
                    limit=contract_file_limit,
                ),
            ]
        )
    if test_context.summary:
        lines.append(
            _truncate_inline(
                test_context.summary,
                concise=concise,
                max_chars=220,
            )
        )
    if test_context.command:
        lines.extend(
            [
                "",
                "Command:",
                _truncate_inline(
                    test_context.command,
                    concise=concise,
                    max_chars=220,
                ),
            ]
        )
    if (
        test_context.test_inventory_framework
        or test_context.test_inventory_language
        or test_context.test_collection_command
    ):
        inventory_details: list[str] = []
        framework_label = _humanize_test_inventory_framework(test_context.test_inventory_framework)
        if framework_label:
            inventory_details.append(f"framework={framework_label}")
        if test_context.test_inventory_language:
            inventory_details.append(
                f"language={str(test_context.test_inventory_language).strip().lower()}"
            )
        lines.extend(["", "Known test inventory:"])
        if inventory_details:
            lines.append("- " + "; ".join(inventory_details))
        if test_context.test_collection_command:
            lines.append(
                "- discovery command: "
                + _truncate_inline(
                    test_context.test_collection_command,
                    concise=concise,
                    max_chars=220,
                )
            )
    if expected_inventory_prompt_visible and (
        test_context.expected_test_count or test_context.expected_test_ids
    ):
        expected_count = int(test_context.expected_test_count or 0) or len(
            test_context.expected_test_ids or []
        )
        lines.extend(["", "Expected visible test inventory:"])
        if expected_count:
            lines.append(f"- {expected_count} pytest node ids")
        lines.append("- full inventory file: .apex_expected_test_ids.txt")
        if test_context.expected_test_ids:
            lines.append(
                _render_items(
                    test_context.expected_test_ids,
                    fallback="- none recorded",
                    concise=concise,
                    limit=expected_id_limit,
                )
            )
    if test_context.failing_test_ids:
        lines.extend(
            [
                "",
                "Currently failing visible tests:",
                _render_items(
                    test_context.failing_test_ids,
                    fallback="- none recorded",
                    concise=concise,
                    limit=failing_limit,
                ),
            ]
        )
    if test_context.focus_test_files:
        lines.extend(
            [
                "",
                "Relevant visible test files:",
                _render_items(
                    test_context.focus_test_files,
                    fallback="- none identified",
                    concise=concise,
                    limit=file_limit,
                ),
            ]
        )
    if test_context.exception_summaries:
        lines.extend(
            [
                "",
                "Direct baseline exceptions:",
                _render_items(
                    test_context.exception_summaries,
                    fallback="- none identified",
                    concise=concise,
                    limit=exception_limit,
                ),
            ]
        )
    if test_context.terminal_source_files:
        lines.extend(
            [
                "",
                "Terminal traceback source files:",
                _render_items(
                    test_context.terminal_source_files,
                    fallback="- none identified",
                    concise=concise,
                    limit=source_limit,
                ),
            ]
        )
    if test_context.incomplete_source_files:
        lines.extend(
            [
                "",
                "Visible incomplete source scaffolds:",
                _render_items(
                    test_context.incomplete_source_files,
                    fallback="- none identified",
                    concise=concise,
                    limit=incomplete_source_limit,
                ),
            ]
        )
    if test_context.incomplete_test_files:
        lines.extend(
            [
                "",
                "Visible incomplete test scaffolds:",
                _render_items(
                    test_context.incomplete_test_files,
                    fallback="- none identified",
                    concise=concise,
                    limit=incomplete_limit,
                ),
            ]
        )
    if test_context.expectations:
        lines.extend(
            [
                "",
                "Representative visible-test expectations:",
                _render_items(
                    test_context.expectations,
                    fallback="- infer expectations from nearby tests",
                    concise=concise,
                    limit=expectation_limit,
                ),
            ]
        )
    return lines


def _render_task_state_entries(
    items: list[Any],
    *,
    concise: bool,
    limit: int,
) -> list[str]:
    values = items[:limit] if concise else items[: limit * 2]
    rendered: list[str] = []
    for item in values:
        if isinstance(item, dict):
            description = (item.get("description") or item.get("summary") or "").strip()
            if not description:
                continue
            metadata: list[str] = []
            file_paths = item.get("file_paths")
            if isinstance(file_paths, list) and file_paths:
                metadata.append("files=" + ", ".join(str(path) for path in file_paths[:2]))
            test_ids = item.get("test_ids")
            if isinstance(test_ids, list) and test_ids:
                metadata.append("tests=" + ", ".join(str(test_id) for test_id in test_ids[:2]))
            symbols = item.get("symbols")
            if isinstance(symbols, list) and symbols:
                metadata.append("symbols=" + ", ".join(str(symbol) for symbol in symbols[:2]))
            outstanding_score = item.get("outstanding_score")
            if isinstance(outstanding_score, (int, float)):
                metadata.append(f"pressure={float(outstanding_score):.2f}")
            belief_score = item.get("belief_score")
            if isinstance(belief_score, (int, float)):
                metadata.append(f"belief={float(belief_score):.2f}")
            suffix = f" ({'; '.join(metadata)})" if metadata else ""
            rendered.append(description + suffix)
            continue
        text = str(item or "").strip()
        if text:
            rendered.append(text)
    return rendered


def _render_blackboard_records(
    records: list[Any],
    *,
    concise: bool,
    limit: int,
) -> list[str]:
    rendered: list[str] = []
    for item in records[:limit]:
        if not isinstance(item, dict):
            continue
        provenance = str(item.get("provenance") or "").strip().lower()
        if provenance == "evaluation_only":
            continue
        description = str(item.get("description") or "").strip()
        if not description:
            continue
        metadata: list[str] = []
        record_type = str(item.get("record_type") or "").strip()
        if record_type:
            metadata.append(record_type)
        confidence = item.get("confidence")
        if isinstance(confidence, (int, float)):
            metadata.append(f"confidence={float(confidence):.2f}")
        file_paths = item.get("file_paths")
        if isinstance(file_paths, list) and file_paths:
            metadata.append("files=" + ", ".join(str(path) for path in file_paths[:2]))
        test_ids = item.get("test_ids")
        if isinstance(test_ids, list) and test_ids:
            metadata.append("tests=" + ", ".join(str(test_id) for test_id in test_ids[:2]))
        payload = item.get("payload")
        if isinstance(payload, dict):
            command = str(payload.get("command") or "").strip()
            if command and not concise:
                metadata.append("command=" + _truncate_inline(command, concise=True, max_chars=80))
        suffix = f" ({'; '.join(metadata)})" if metadata else ""
        rendered.append(description + suffix)
    return rendered


def _render_task_state_block(
    issue_plan: IssuePlan,
    *,
    concise: bool,
    rollout_brief: RolloutBrief | None = None,
) -> list[str]:
    task_state_context = issue_plan.task_state_context or {}
    if not isinstance(task_state_context, dict) or not task_state_context:
        return []
    search_policy = rollout_brief.search_policy if rollout_brief is not None else {}
    if concise and isinstance(search_policy, dict) and search_policy.get("graph_target_id"):
        return []

    summary = str(task_state_context.get("summary") or "").strip()
    open_obligations = _render_task_state_entries(
        list(task_state_context.get("open_obligations") or []),
        concise=concise,
        limit=3,
    )
    supported_hypotheses = _render_task_state_entries(
        list(task_state_context.get("supported_hypotheses") or []),
        concise=concise,
        limit=3,
    )
    focus_files = [str(path) for path in list(task_state_context.get("focus_files") or []) if path]
    unresolved_test_ids = [
        str(test_id)
        for test_id in list(task_state_context.get("unresolved_test_ids") or [])
        if test_id
    ]
    reflection_entries: list[str] = []
    for item in list(task_state_context.get("reflection_memory") or []):
        if not isinstance(item, dict):
            text = str(item or "").strip()
            if text:
                reflection_entries.append(text)
            continue
        rendered = str(item.get("summary") or "").strip()
        if not rendered:
            continue
        metadata: list[str] = []
        file_paths = [
            str(path).strip() for path in list(item.get("file_paths") or []) if str(path).strip()
        ]
        if file_paths:
            metadata.append("files=" + ", ".join(file_paths[:2]))
        symbols = [
            str(symbol).strip() for symbol in list(item.get("symbols") or []) if str(symbol).strip()
        ]
        if symbols:
            metadata.append("symbols=" + ", ".join(symbols[:2]))
        count = item.get("count")
        if isinstance(count, int) and count > 0:
            metadata.append(f"seen={count}x")
        if metadata:
            rendered += " (" + "; ".join(metadata) + ")"
        reflection_entries.append(rendered)
    progress_ledger = (
        dict(task_state_context.get("progress_ledger") or {})
        if isinstance(task_state_context.get("progress_ledger"), dict)
        else {}
    )
    blackboard_payload = (
        dict(task_state_context.get("blackboard") or {})
        if isinstance(task_state_context.get("blackboard"), dict)
        else {}
    )
    blackboard_policy = (
        dict(blackboard_payload.get("evidence_policy") or {})
        if isinstance(blackboard_payload.get("evidence_policy"), dict)
        else {}
    )
    blackboard_records = _render_blackboard_records(
        list(blackboard_payload.get("records") or []),
        concise=concise,
        limit=4 if concise else 10,
    )
    progress_ledger_entries: list[str] = []
    progress_action = str(progress_ledger.get("next_action") or "").strip()
    if progress_action:
        progress_ledger_entries.append(f"Next action: {progress_action}")
    decision_summary = str(progress_ledger.get("decision_summary") or "").strip()
    if decision_summary:
        progress_ledger_entries.append(decision_summary)
    boundary_files = [
        str(path).strip()
        for path in list(progress_ledger.get("boundary_requested_files") or [])
        if str(path).strip()
    ]
    if boundary_files:
        progress_ledger_entries.append("Boundary files: " + ", ".join(boundary_files[:3]))
    boundary_symbols = [
        str(symbol).strip()
        for symbol in list(progress_ledger.get("boundary_interface_symbols") or [])
        if str(symbol).strip()
    ]
    if boundary_symbols:
        progress_ledger_entries.append("Interface symbols: " + ", ".join(boundary_symbols[:3]))
    if not any(
        (
            summary,
            open_obligations,
            supported_hypotheses,
            focus_files,
            unresolved_test_ids,
            reflection_entries,
            progress_ledger_entries,
            blackboard_records,
        )
    ):
        return []

    lines = ["", "# Task State"]
    if summary:
        lines.append(summary)
    if blackboard_policy:
        evidence_mode = str(blackboard_policy.get("mode") or "").strip()
        if evidence_mode:
            lines.append(f"Blackboard evidence mode: {evidence_mode}")
    if blackboard_records:
        lines.extend(
            [
                "",
                "Typed blackboard:",
                _render_items(
                    blackboard_records,
                    fallback="- no promoted facts",
                    concise=concise,
                    limit=4 if concise else 10,
                ),
            ]
        )
    if open_obligations:
        lines.extend(
            [
                "",
                "Highest-pressure unresolved obligations:",
                _render_items(
                    open_obligations,
                    fallback="- none recorded",
                    concise=concise,
                    limit=3,
                ),
            ]
        )
    if supported_hypotheses:
        lines.extend(
            [
                "",
                "Most supported current hypotheses:",
                _render_items(
                    supported_hypotheses,
                    fallback="- none recorded",
                    concise=concise,
                    limit=3,
                ),
            ]
        )
    if focus_files:
        lines.extend(
            [
                "",
                "Graph-ranked focus files:",
                _render_items(
                    focus_files,
                    fallback="- none recorded",
                    concise=concise,
                    limit=4,
                ),
            ]
        )
    if unresolved_test_ids:
        lines.extend(
            [
                "",
                "Unresolved visible tests:",
                _render_items(
                    unresolved_test_ids,
                    fallback="- none recorded",
                    concise=concise,
                    limit=3,
                ),
            ]
        )
    if reflection_entries:
        lines.extend(
            [
                "",
                "Reflective failure memory:",
                _render_items(
                    reflection_entries,
                    fallback="- none recorded",
                    concise=concise,
                    limit=3,
                ),
            ]
        )
    if progress_ledger_entries:
        lines.extend(
            [
                "",
                "Planner progress ledger:",
                _render_items(
                    progress_ledger_entries,
                    fallback="- none recorded",
                    concise=concise,
                    limit=4,
                ),
            ]
        )
    return lines


def _render_search_policy_block(
    rollout_brief: RolloutBrief,
    *,
    concise: bool,
) -> list[str]:
    policy = rollout_brief.search_policy or {}
    if not isinstance(policy, dict) or not policy:
        return []

    lines = ["", "# Search Policy"]
    mode = policy.get("mode")
    verification_focus = policy.get("verification_focus")
    variant_index = policy.get("variant_index")
    cluster_index = policy.get("cluster_index")
    if mode:
        lines.append(f"Mode: {mode}")
    if verification_focus:
        lines.append(f"Verification focus: {verification_focus}")
    if cluster_index is not None:
        lines.append(f"Cluster index: {cluster_index}")
    if isinstance(variant_index, int) and variant_index > 0:
        lines.append(f"Variant index: {variant_index}")
    if str(mode or "").strip().lower() == "agentless_pipeline":
        lines.extend(
            [
                "Pipeline: inspect failure evidence and repo-map neighbors, choose the smallest source repair, apply it, run targeted validation, then rerun a broader relevant suite if the focused check improves.",
                "Do not delegate this lane. Keep localization as a ranking prior, not an edit boundary; preserve useful progress unless validation or protected-file constraints show harm.",
            ]
        )
    if concise and len(lines) > 5:
        lines = lines[:5]
    # Decomposition module-group briefs: surface the agent's OWN disjoint file
    # partition (the otherwise-silent write-scope boundary) so it spends turns on
    # keepable files and implements its whole group breadth-first. Appended AFTER
    # the concise truncation because this is the brief's core instruction and must
    # never be cut. Layer-A general: emitted only for decomposition briefs.
    owned = [f for f in (policy.get("module_group_owned_files") or []) if f]
    if policy.get("decomposition_module_group") and owned:
        bridge = [f for f in (policy.get("module_group_bridge_files") or []) if f]
        lines.append("")
        lines.append("# Owned Module Files (implement EVERY stub in ALL of these)")
        for path in owned[:24]:
            lines.append(f"- {path}")
        if len(owned) > 24:
            lines.append(f"- ... (+{len(owned) - 24} more owned files)")
        if bridge:
            lines.append("Bridge files (read-only coordination context; do not rewrite):")
            for path in bridge[:8]:
                lines.append(f"- {path}")
        lines.append(
            "Edits outside your owned files are reverted by write-scope enforcement. "
            "Implement every NotImplementedError / empty body / TODO-pass in the owned "
            "files end-to-end and keep the test suite importable (collection intact)."
        )
    return lines


def _render_execution_decision_card(
    issue_plan: IssuePlan,
    rollout_brief: RolloutBrief,
    *,
    concise: bool,
    validation_only: bool = False,
) -> list[str]:
    """Render the compact decision-to-execution handoff for a rollout."""

    policy = rollout_brief.search_policy if isinstance(rollout_brief.search_policy, dict) else {}
    target_refs = _dedupe_nonempty(
        list(rollout_brief.focus_files or [])
        + list(policy.get("action_file_paths") or [])
        + list(policy.get("graph_target_file_paths") or [])
        + list(issue_plan.test_context.terminal_source_files or [])
    )
    evidence_refs = _dedupe_nonempty(
        list(policy.get("graph_target_test_ids") or [])
        + list(issue_plan.test_context.failing_test_ids or [])
        + list(issue_plan.test_context.focus_test_files or [])
        + list(issue_plan.test_context.exception_summaries or [])
    )
    hard_boundaries = [
        "Treat focus/localization as ranked evidence, not an edit fence; keep objective-moving changes unless validation, protected-file, or harness constraints show harm.",
        "Preserve public APIs, imports, and collected test surface while repairing source behavior.",
    ]
    if issue_plan.test_context.expected_test_count:
        hard_boundaries.append(
            f"Preserve expected test inventory coverage: {issue_plan.test_context.expected_test_count} collected tests."
        )
    if issue_plan.test_context.evidence_mode == EVIDENCE_MODE_GOLD_SUITE_VISIBLE:
        hard_boundaries.append(
            "Gold-suite visible mode: protected tests are read-only specification unless explicitly listed as incomplete placeholders."
        )
    if validation_only:
        hard_boundaries.append(
            "Validation-only handoff: gather concrete evidence and blockers; do not take ownership of implementation edits."
        )

    why_now = _decision_card_why_now(issue_plan, rollout_brief, policy)
    lines = [
        "",
        "# Execution Decision Card",
        f"Selected lane: {rollout_brief.title or 'unnamed rollout'}",
        f"Task: {rollout_brief.goal or issue_plan.summary}",
        f"Why now: {why_now}",
    ]
    if policy.get("mode"):
        lines.append(f"Decision mode: {policy.get('mode')}")
    lines.extend(
        [
            "Target refs:",
            _render_items(
                target_refs,
                fallback="- infer targets from the failure evidence and repo map",
                concise=concise,
                limit=5 if concise else 10,
            ),
            "Evidence refs:",
            _render_items(
                evidence_refs,
                fallback="- use the Test Context, reproduction handoff, and repo map",
                concise=concise,
                limit=4 if concise else 8,
            ),
            "Hard boundaries:",
            _render_items(
                hard_boundaries,
                fallback="- stay inside the task objective and repository contract",
                concise=concise,
                limit=4 if concise else 8,
            ),
            "Feedback contract: return changed_files, tests_run, residual blockers, and followups. If useful edits broaden beyond the first plan, keep the progress and explain the updated subgoal instead of discarding it as out-of-scope.",
        ]
    )
    return lines


def _decision_card_why_now(
    issue_plan: IssuePlan,
    rollout_brief: RolloutBrief,
    policy: dict[str, Any],
) -> str:
    for value in (
        policy.get("graph_target_description"),
        policy.get("graph_target_obligation_description"),
        policy.get("verifier_repair_objective"),
        rollout_brief.prompt_hint,
    ):
        text = str(value or "").strip()
        if text:
            return _collapse_whitespace(text)[:240]
    if issue_plan.test_context.exception_summaries:
        return _collapse_whitespace(issue_plan.test_context.exception_summaries[0])[:240]
    if issue_plan.test_context.failing_test_ids:
        return f"Visible failing test: {issue_plan.test_context.failing_test_ids[0]}"
    if rollout_brief.hypotheses:
        return _collapse_whitespace(rollout_brief.hypotheses[0])[:240]
    return issue_plan.summary


def _dedupe_nonempty(values: list[Any]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value or "").strip()))


def _collapse_whitespace(value: str) -> str:
    return " ".join(str(value or "").split())


def _render_frontier_target_block(
    rollout_brief: RolloutBrief,
    *,
    concise: bool,
) -> list[str]:
    policy = rollout_brief.search_policy or {}
    if not isinstance(policy, dict) or not policy:
        return []

    description = str(policy.get("graph_target_description") or "").strip()
    obligation_description = str(policy.get("graph_target_obligation_description") or "").strip()
    hypothesis_description = str(policy.get("graph_target_hypothesis_description") or "").strip()
    rationale = str(policy.get("graph_target_rationale") or "").strip()
    target_files = [str(path) for path in list(policy.get("graph_target_file_paths") or []) if path]
    target_tests = [
        str(test_id) for test_id in list(policy.get("graph_target_test_ids") or []) if test_id
    ]
    target_symbols = [
        str(symbol) for symbol in list(policy.get("graph_target_symbols") or []) if symbol
    ]
    if not any(
        (
            description,
            obligation_description,
            hypothesis_description,
            rationale,
            target_files,
            target_tests,
            target_symbols,
        )
    ):
        return []

    lines = ["", "# Frontier Target"]
    kind = str(policy.get("graph_target_kind") or "").strip()
    score = policy.get("graph_target_score")
    uncertainty = policy.get("graph_target_uncertainty")
    header_parts: list[str] = []
    if kind:
        header_parts.append(f"kind={kind}")
    if isinstance(score, (int, float)):
        header_parts.append(f"score={float(score):.2f}")
    if isinstance(uncertainty, (int, float)):
        header_parts.append(f"uncertainty={float(uncertainty):.2f}")
    if header_parts:
        lines.append("Target metadata: " + "; ".join(header_parts))
    if obligation_description:
        lines.append(
            "Behavioral obligation: "
            + _truncate_inline(
                obligation_description,
                concise=concise,
                max_chars=180,
            )
        )
    if hypothesis_description:
        lines.append(
            "Working hypothesis: "
            + _truncate_inline(
                hypothesis_description,
                concise=concise,
                max_chars=180,
            )
        )
    elif description:
        lines.append(
            "Focus: "
            + _truncate_inline(
                description,
                concise=concise,
                max_chars=180,
            )
        )
    if rationale:
        lines.append(
            "Why this target matters: "
            + _truncate_inline(
                rationale,
                concise=concise,
                max_chars=180,
            )
        )
    if target_tests:
        lines.extend(
            [
                "",
                "Prioritized validation targets:",
                _render_items(
                    target_tests,
                    fallback="- none recorded",
                    concise=concise,
                    limit=3,
                ),
            ]
        )
    if target_files:
        lines.extend(
            [
                "",
                "Target files:",
                _render_items(
                    target_files,
                    fallback="- none recorded",
                    concise=concise,
                    limit=4,
                ),
            ]
        )
    if target_symbols:
        lines.extend(
            [
                "",
                "Target symbols:",
                _render_items(
                    target_symbols,
                    fallback="- none recorded",
                    concise=concise,
                    limit=3,
                ),
            ]
        )
    return lines


def _effective_test_context(
    issue_plan: IssuePlan,
    fallback_command: str | None = None,
) -> TestContext:
    context = issue_plan.test_context
    if (
        context.summary
        or context.command
        or context.failing_test_ids
        or context.focus_test_files
        or context.expectations
    ):
        if context.command or not fallback_command:
            return context
        merged = TestContext.from_dict(context.to_dict())
        merged.command = fallback_command
        return merged
    return TestContext(command=fallback_command)


def _render_delegation_guidance(
    rollout_brief: RolloutBrief,
    *,
    delegation_mode: str = "apex_tool",
) -> list[str]:
    if not rollout_brief.delegation_enabled("patcher"):
        return []
    policy = (
        rollout_brief.delegation_policy if isinstance(rollout_brief.delegation_policy, dict) else {}
    )
    max_tasks = int(policy.get("max_tasks") or 0)
    parallelism = int(policy.get("parallelism") or 0)
    reason = str(policy.get("reason") or "").strip()
    split_confidence = policy.get("split_confidence")
    bridge_files = [
        str(item).strip() for item in list(policy.get("bridge_files") or []) if str(item).strip()
    ]
    if max_tasks <= 0 or parallelism <= 0:
        return []

    def task_id_for(task: dict[str, Any], index: int) -> str:
        raw_value = str(task.get("task_id") or task.get("title") or f"task_{index}").strip()
        normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", raw_value).strip("._-")
        return normalized or f"task_{index}"

    subtasks = [task for task in list(policy.get("subtasks") or []) if isinstance(task, dict)]

    budget_line = (
        "APEX has enabled bounded child-worker delegation for this rollout: "
        f"at most {max_tasks} subtasks with parallelism up to {parallelism}."
    )
    guidance = (
        "Use delegate_subtasks only through the orchestrator-authored subtask split below, "
        "keep ownership clear across child workers, and integrate the child summaries or "
        "patch artifacts back into the main workspace before you finish. Focus files are "
        "starting points for child investigation and editing, not exclusive source-edit "
        "boundaries."
    )
    lines = [budget_line, guidance]
    if reason:
        lines.append("Why delegation is enabled here: " + reason)
    if isinstance(split_confidence, (int, float)):
        lines.append(f"Planner split confidence: {float(split_confidence):.2f}")
    if bridge_files:
        lines.append("Boundary watchlist: " + ", ".join(bridge_files[:6]))
        lines.append(
            "If a child fix appears to require one of those files, have the child explain the "
            "evidence and keep the edit minimal; protected and forbidden files remain hard boundaries."
        )
    if len(subtasks) >= 2:
        lines.append(
            "When you call delegate_subtasks, reference only the exact Task IDs below. Do not invent or rename subtasks."
        )
        lines.append(
            "Delegation split: use the exact owned-file decomposition below rather than inventing a new subtask tree."
        )
        for index, subtask in enumerate(subtasks, start=1):
            title = str(subtask.get("title") or f"Task {index}").strip()
            task_id = task_id_for(subtask, index)
            kind = str(subtask.get("kind") or "").strip().lower()
            focus_files = [
                str(item).strip()
                for item in list(subtask.get("owned_files") or subtask.get("focus_files") or [])
                if str(item).strip()
            ]
            forbidden_files = [
                str(item).strip()
                for item in list(subtask.get("forbidden_files") or [])
                if str(item).strip()
            ]
            interface_symbols = [
                str(item).strip()
                for item in list(subtask.get("interface_symbols") or [])
                if str(item).strip()
            ]
            assumptions = [
                str(item).strip()
                for item in list(subtask.get("assumptions") or [])
                if str(item).strip()
            ]
            escalation_triggers = [
                str(item).strip()
                for item in list(subtask.get("escalation_triggers") or [])
                if str(item).strip()
            ]
            depends_on = [
                str(item).strip()
                for item in list(subtask.get("depends_on") or [])
                if str(item).strip()
            ]
            validation_targets = [
                str(item).strip()
                for item in list(subtask.get("validation_targets") or [])
                if str(item).strip()
            ]
            objective = str(subtask.get("objective") or "").strip()
            deliverable = str(subtask.get("deliverable") or "").strip()
            lines.append(f"- Task {index}: {title}")
            lines.append(f"  Task ID: {task_id}")
            if focus_files:
                label = "Context files" if kind == "validation" else "Owned files"
                lines.append(f"  {label}: " + ", ".join(focus_files))
            if forbidden_files:
                lines.append("  Forbidden files: " + ", ".join(forbidden_files[:6]))
            if interface_symbols:
                lines.append("  Interface symbols: " + ", ".join(interface_symbols[:6]))
            if objective:
                lines.append("  Objective: " + objective)
            if validation_targets:
                lines.append("  Validation targets: " + ", ".join(validation_targets))
            if deliverable:
                lines.append("  Deliverable: " + deliverable)
            if assumptions:
                lines.append("  Assumptions: " + "; ".join(assumptions[:3]))
            if escalation_triggers:
                lines.append("  Escalate if: " + "; ".join(escalation_triggers[:3]))
            if depends_on:
                lines.append("  Depends on: " + ", ".join(depends_on))
    return lines


def build_reproducer_prompt(
    issue_description: str,
    issue_plan: IssuePlan,
    rollout_brief: RolloutBrief,
    test_command: str | None,
    concise: bool = True,
) -> str:
    permuted_focus_files = permute_focus_files(rollout_brief.focus_files, rollout_brief)
    exploration_directive = stage_first_action_directive("reproducer", rollout_brief)
    sections = [
        "# Issue",
        _truncate_block(issue_description, concise=concise, max_lines=16),
        "",
        "# Primary Objective",
        issue_plan.summary,
        "",
        "# Rollout Brief",
        f"Title: {rollout_brief.title}",
        f"Goal: {rollout_brief.goal}",
        "",
        "# First-action directive (rollout-specific)",
        exploration_directive or "Use your judgment.",
        "",
        "# Focus Files",
        _render_items(
            permuted_focus_files,
            fallback="- none provided",
            concise=concise,
            limit=6,
        ),
        "",
        "# Success Criteria",
        _render_items(
            rollout_brief.success_criteria or issue_plan.success_criteria,
            fallback="- reproduce and validate the issue",
            concise=concise,
            limit=4,
        ),
        "",
        "# Focus Repo Map",
        _truncate_block(issue_plan.repo_focus_map, concise=concise, max_lines=36),
    ]
    sections.extend(_render_search_policy_block(rollout_brief, concise=concise))
    sections.extend(_render_frontier_target_block(rollout_brief, concise=concise))
    sections.extend(
        _render_test_context_block(
            issue_plan.test_context,
            concise=concise,
        )
    )
    sections.extend(
        _render_task_state_block(issue_plan, concise=concise, rollout_brief=rollout_brief)
    )
    sections.extend(
        [
            "",
            "Create or identify a targeted reproduction artifact and submit it with:",
            "- a clear summary",
            "- the command that should pass after the fix",
            "- the script path or script content if you created one",
            "- the observed output from your run",
            "- do not edit application or source files in this stage",
        ]
    )
    return "\n".join(sections)


def build_localizer_prompt(
    issue_description: str,
    issue_plan: IssuePlan,
    rollout_brief: RolloutBrief,
    reproduction_artifact: Any = None,
    reproduction_summary: str | None = None,
    concise: bool = True,
) -> str:
    permuted_focus_files = permute_focus_files(rollout_brief.focus_files, rollout_brief)
    exploration_directive = stage_first_action_directive("localizer", rollout_brief)
    sections = [
        "# Issue",
        _truncate_block(issue_description, concise=concise, max_lines=16),
        "",
        "# Primary Objective",
        issue_plan.summary,
        "",
        "# Rollout Brief",
        f"Title: {rollout_brief.title}",
        f"Goal: {rollout_brief.goal}",
        "",
        "# First-action directive (rollout-specific)",
        exploration_directive or "Inspect the top likely files first.",
        "",
        "# Focus Files",
        _render_items(
            permuted_focus_files,
            fallback="- none provided",
            concise=concise,
            limit=8,
        ),
        "",
        "# Hypotheses",
        _render_items(
            rollout_brief.hypotheses,
            fallback="- inspect the top likely files",
            concise=concise,
            limit=4,
        ),
        "",
        "# Focus Repo Map",
        _truncate_block(issue_plan.repo_focus_map, concise=concise, max_lines=40),
    ]
    sections.extend(_render_search_policy_block(rollout_brief, concise=concise))
    sections.extend(_render_frontier_target_block(rollout_brief, concise=concise))
    sections.extend(
        _render_test_context_block(
            issue_plan.test_context,
            concise=concise,
        )
    )
    sections.extend(
        _render_task_state_block(issue_plan, concise=concise, rollout_brief=rollout_brief)
    )
    handoff = reproduction_artifact if reproduction_artifact is not None else reproduction_summary
    sections.extend(
        _render_artifact_block(
            "Reproduction Artifact",
            handoff,
            concise=concise,
            max_lines=28,
        )
    )
    sections.extend(
        [
            "",
            "Return the likely root-cause locations through submit_localization.",
            "Include a concise summary, likely files, relevant symbols, and hypotheses.",
            "Keep this stage read-only: do not edit application or source files.",
        ]
    )
    return "\n".join(sections)


def build_solver_prompt(
    issue_description: str,
    issue_plan: IssuePlan,
    rollout_brief: RolloutBrief,
    strategy: PromptStrategy,
    test_command: str | None = None,
    reproduction_artifact: Any = None,
    localization_artifact: Any = None,
    reproduction_summary: str | None = None,
    localization_summary: str | None = None,
    concise: bool = True,
    compact_completion_context: bool = False,
    allow_partial_completion_roundtrip: bool = False,
    allow_delegation: bool = False,
    delegation_mode: str = "apex_tool",
    broad_revalidation_mode: str = "continuous",
) -> str:
    delegated_subtask = _is_delegated_subtask_prompt(issue_plan, rollout_brief)
    delegated_subtask_kind = _delegated_subtask_kind(issue_plan, rollout_brief)
    validation_only_delegated_subtask = (
        delegated_subtask
        and delegated_subtask_kind == "validation"
        and not list(rollout_brief.focus_files or [])
    )
    if _rollout_brief_is_verifier_validity_repair(rollout_brief):
        return _build_verifier_validity_repair_prompt(
            issue_plan=issue_plan,
            rollout_brief=rollout_brief,
            test_command=test_command,
            concise=concise,
        )
    # Whenever the planner identifies ``incomplete_test_files`` (test
    # bodies that are visibly stubbed with ``raise NotImplementedError``,
    # ``pass # TODO``, etc.), the agent must be allowed to complete
    # them — that's the literal task. The prior gate also required
    # ``not visible_test_edit_protection_enabled``, which made the
    # prompt withhold the "complete the stubs" instruction whenever
    # the benchmark protected visible tests. That made completion tasks
    # unsolvable: the agent was told the tests are read-only AND the
    # boundary enforcer reverted any edit. Both halves are now
    # corrected (the enforcer half lives in
    # ``apex/core/pytest_report_utils.py:protected_test_files_from_context``).
    allows_incomplete_test_completion = bool(issue_plan.test_context.incomplete_test_files)
    completion_like = (
        _requires_broad_completion_validation(issue_description, issue_plan)
        and not delegated_subtask
    )
    structural_unblock_round = (
        not delegated_subtask
        and bool(issue_plan.test_context.exception_summaries)
        and max(
            int(issue_plan.test_context.failing_test_count or 0),
            len(issue_plan.test_context.failing_test_ids or []),
        )
        > 0
        and int(issue_plan.test_context.passing_test_count or 0) <= 0
        and bool(
            issue_plan.test_context.terminal_source_files
            or issue_plan.test_context.source_focus_files
        )
    )
    compact_scoped_handoff = compact_completion_context or (concise and delegated_subtask)
    repro_handoff = (
        reproduction_artifact if reproduction_artifact is not None else reproduction_summary
    )
    localization_handoff = (
        localization_artifact if localization_artifact is not None else localization_summary
    )
    if compact_scoped_handoff:
        reproduction = coerce_reproduction_artifact(repro_handoff)
        localization = coerce_localization_artifact(localization_handoff)
        selected_tests = (
            list(rollout_brief.search_policy.get("graph_target_test_ids") or [])
            if isinstance(rollout_brief.search_policy, dict)
            else []
        )
        if not selected_tests:
            selected_tests = list(issue_plan.test_context.focus_test_files or [])
        if not selected_tests:
            selected_tests = list(issue_plan.test_context.failing_test_ids or [])
        selected_tests = list(dict.fromkeys(test_id for test_id in selected_tests if test_id))
        broader_visible_tests = [
            test_id
            for test_id in list(issue_plan.test_context.failing_test_ids or [])
            + list(issue_plan.test_context.focus_test_files or [])
            if test_id and test_id not in selected_tests
        ]
        broader_visible_tests = list(dict.fromkeys(broader_visible_tests))
        immediate_blocker = []
        if issue_plan.test_context.exception_summaries:
            immediate_blocker.append(issue_plan.test_context.exception_summaries[0])
        if issue_plan.test_context.terminal_source_files:
            immediate_blocker.append(
                "Traceback focus: " + ", ".join(issue_plan.test_context.terminal_source_files[:3])
            )
        if issue_plan.test_context.incomplete_source_files:
            immediate_blocker.append(
                "Nearby scaffolds: "
                + ", ".join(issue_plan.test_context.incomplete_source_files[:3])
            )
        focus_files_fallback = (
            "- no implementation files are owned in this validation-only subtask; validate the assigned slice and report blockers"
            if validation_only_delegated_subtask
            else "- infer the best implementation targets from the repo"
        )
        localization_files_fallback = (
            "- no implementation ownership was assigned in this validation-only subtask; use the validation target and reproduction handoff"
            if validation_only_delegated_subtask
            else "- infer the likely implementation files from the traceback focus"
        )
        sections = [
            "# Issue",
            _truncate_block(issue_description, concise=True, max_lines=12),
            "",
            "# Primary Objective",
            issue_plan.summary,
            *_render_execution_decision_card(
                issue_plan,
                rollout_brief,
                concise=True,
                validation_only=validation_only_delegated_subtask,
            ),
            "",
            *_render_focus_file_sections(
                issue_plan,
                rollout_brief,
                fallback=focus_files_fallback,
                concise=True,
                limit=6,
            ),
        ]
        if immediate_blocker:
            sections.extend(
                [
                    "",
                    "# Current Failure Signal"
                    if validation_only_delegated_subtask
                    else "# Immediate Blocker",
                    "\n".join(f"- {item}" for item in immediate_blocker),
                ]
            )
        if selected_tests:
            sections.extend(
                [
                    "",
                    "# Validation Target",
                    _render_items(
                        selected_tests,
                        fallback="- reuse the baseline command",
                        concise=True,
                        limit=2,
                    ),
                ]
            )
        sections.extend(_render_incomplete_test_completion_block(issue_plan, concise=True))
        if structural_unblock_round:
            sections.extend(
                [
                    "",
                    "# Structural Unblock Boundary",
                    "Start by clearing the import/collection blocker before unrelated cleanup.",
                    "- Do not run blind automated `pass`, `NotImplemented`, TODO, formatting, or scaffold-filling sweeps across the repository.",
                    "- Keep first edits on the current traceback/import frontier and its direct dependency edges.",
                    "- If broader validation exposes a connected source-contract cluster, repair that cluster even when it spans many files; edit count alone is not an out-of-scope signal.",
                    "- If broader validation exposes unrelated missing scaffolds, return the current focused fix with followups instead of mixing independent repairs.",
                ]
            )
        if broader_visible_tests:
            broader_visible_guidance = (
                "Do not stop after clearing only the validation target above. "
                "Run the broader visible failures below before finishing."
            )
            if allow_partial_completion_roundtrip:
                broader_visible_guidance = (
                    "After clearing the immediate blocker, run the broader visible failures below "
                    "once to expose the next blocker before returning."
                )
            sections.extend(
                [
                    "",
                    "# Broader Visible Suite",
                    broader_visible_guidance,
                    _render_items(
                        broader_visible_tests,
                        fallback="- reuse the broader visible test command",
                        concise=True,
                        limit=6,
                    ),
                ]
            )
        if reproduction is not None and (reproduction.command or reproduction.summary):
            reproduction_lines: list[str] = []
            if reproduction.summary:
                reproduction_lines.append(reproduction.summary)
            if reproduction.command:
                reproduction_lines.append("Command: " + reproduction.command)
            sections.extend(
                [
                    "",
                    "# Reproduction Handoff",
                    "\n".join(reproduction_lines),
                ]
            )
        if localization is not None:
            sections.extend(
                [
                    "",
                    "# Localization Handoff",
                    localization.summary
                    or "Use the localized files below as the primary implementation surface.",
                    "",
                    "Files:",
                    _render_items(
                        localization.files,
                        fallback=localization_files_fallback,
                        concise=True,
                        limit=6,
                    ),
                ]
            )
            if localization.hypotheses:
                sections.extend(
                    [
                        "",
                        "Hypotheses:",
                        _render_items(
                            localization.hypotheses,
                            fallback="- infer the missing contract from the visible tests",
                            concise=True,
                            limit=3,
                        ),
                    ]
                )
        if allow_delegation:
            sections.extend(
                [
                    "",
                    "# Delegation Plan",
                    *_render_delegation_guidance(
                        rollout_brief,
                        delegation_mode=delegation_mode,
                    ),
                ]
            )
        execution_rules = [
            "",
            "# Execution Rules",
        ]
        if validation_only_delegated_subtask:
            execution_rules.extend(
                [
                    "- Work directly in the workspace and run the assigned validation slice before responding.",
                    "- Do not edit implementation files in this subtask; use validation evidence to surface blockers and follow-up integration guidance for the parent orchestrator.",
                    "- Reuse the targeted validation command before any broader suite.",
                    "- Return only a JSON object summarizing validation outcomes, blockers, and tests you ran.",
                ]
            )
        else:
            execution_rules.extend(
                [
                    "- Work directly in the workspace and apply the fix before responding.",
                    "- Clear the direct collection or import blocker first, then broaden only if targeted tests still fail.",
                    "- Prefer implementation changes over weakening stable tests.",
                    "- Preserve the collected visible test surface; do not change parametrization inputs, helper registries, collection filters, or similar scaffolding in ways that make existing tests disappear.",
                    "- Run focused validation before any broader suite.",
                    "- Return only a JSON object summarizing the patch and tests you ran.",
                ]
            )
        sections.extend(execution_rules)
        if delegated_subtask:
            sections.extend(
                _delegated_subtask_scope_rules(
                    validation_only=validation_only_delegated_subtask,
                    bullet=True,
                )
            )
        if completion_like:
            if allow_partial_completion_roundtrip:
                sections.extend(
                    [
                        "- Do not stop after only the first previously failing test passes without rerunning the broader visible suite once.",
                        "- After the first import or collection blocker clears, rerun the visible import or failing-test command to surface the next blocker in the same package before returning.",
                        "- You may return once the immediate blocker is fixed and the broader visible suite or repository test command has been rerun to surface the next blocker. The continuation round will resume from the improved workspace.",
                        "- Preserve existing modules, imports, and public definitions unless you are replacing an obvious placeholder scaffold.",
                    ]
                )
            else:
                sections.extend(
                    [
                        "- Do not finish after only the first previously failing test passes.",
                        "- After the first import or collection blocker clears, rerun the visible import or failing-test command to surface the next blocker in the same package before stopping.",
                        "- Before returning, clear the broader visible suite or the full repository test command from the Test Context.",
                        "- Preserve existing modules, imports, and public definitions unless you are replacing an obvious placeholder scaffold.",
                    ]
                )
        if allows_incomplete_test_completion:
            sections.extend(
                [
                    "- Only the explicit placeholder TEST BODIES inside files listed in `incomplete_test_files` are editable. "
                    "Replace each `raise NotImplementedError(...)` body with a real test body derived from the "
                    "function-under-test name + the assertions in adjacent (already-completed) tests. "
                    "Do NOT skip, comment out, or weaken the expected behavior — the test name and "
                    "surrounding tests fully specify the assertion.",
                    "- In those files, do NOT edit imports, decorators, parametrization lists, helper registries, shared fixtures, top-level constants, or non-placeholder test bodies.",
                    "- Treat OTHER visible tests (those not flagged incomplete) as read-only specification. "
                    "Never rewrite assertions, expected outputs, or coverage to make failures disappear.",
                ]
            )
        elif completion_like:
            sections.extend(
                [
                    "- Treat visible tests as read-only specification for this repository-completion task. Do not modify them to make failures disappear.",
                ]
            )
        return "\n".join(sections)
    exploration_directive = rollout_exploration_directive(rollout_brief)
    sections = [
        "# Issue",
        _truncate_block(issue_description, concise=concise, max_lines=18),
        "",
        "# Primary Objective",
        issue_plan.summary,
        "",
        "# Rollout Brief",
        f"Title: {rollout_brief.title}",
        f"Goal: {rollout_brief.goal}",
        f"Hint: {rollout_brief.prompt_hint or 'Use your judgment.'}",
        *_render_execution_decision_card(
            issue_plan,
            rollout_brief,
            concise=concise,
            validation_only=validation_only_delegated_subtask,
        ),
        "",
        "# Strategy",
        STRATEGY_INSTRUCTIONS[strategy],
        "",
        # Exploration directive — forces a different first action per
        # rollout. Critical for cross-rollout diversity when the LLM
        # backend (e.g. Codex CLI via Plugboard) ignores the
        # temperature parameter and the strategy text alone is too
        # subtle to alter the model's first move.
        "# First-action directive (rollout-specific)",
        exploration_directive
        or "Use the strategy above; no specific first-move directive for this rollout.",
        "",
        *_render_focus_file_sections(
            issue_plan,
            rollout_brief,
            fallback="- none provided",
            concise=concise,
            limit=8,
        ),
        "",
        "# Hypotheses",
        _render_items(
            rollout_brief.hypotheses,
            fallback="- infer the root cause from the repo",
            concise=concise,
            limit=4,
        ),
        "",
        "# Success Criteria",
        _render_items(
            _completion_scaffold_success_criteria(issue_plan, rollout_brief),
            fallback="- resolve the issue and verify it",
            concise=concise,
            limit=4,
        ),
        "",
        "# Focus Repo Map",
        _truncate_block(issue_plan.repo_focus_map, concise=concise, max_lines=42),
    ]
    sections.extend(_render_search_policy_block(rollout_brief, concise=concise))
    sections.extend(_render_frontier_target_block(rollout_brief, concise=concise))
    sections.extend(_render_test_context_block(issue_plan.test_context, concise=concise))
    sections.extend(_render_incomplete_test_completion_block(issue_plan, concise=concise))
    sections.extend(
        _render_task_state_block(issue_plan, concise=concise, rollout_brief=rollout_brief)
    )
    if structural_unblock_round:
        sections.extend(
            [
                "",
                "# Structural Unblock Boundary",
                "Start by clearing the import/collection blocker before unrelated cleanup.",
                "- Do not run blind automated `pass`, `NotImplemented`, TODO, formatting, or scaffold-filling sweeps across the repository.",
                "- Keep first edits on the current traceback/import frontier and its direct dependency edges.",
                "- If broader validation exposes a connected source-contract cluster, repair that cluster even when it spans many files; edit count alone is not an out-of-scope signal.",
                "- If broader validation exposes unrelated missing scaffolds, return the current focused fix with followups instead of mixing independent repairs.",
            ]
        )
    sections.extend(
        _render_artifact_block(
            "Reproduction Artifact",
            repro_handoff,
            concise=concise,
            max_lines=28,
        )
    )
    sections.extend(
        _render_artifact_block(
            "Localization Artifact",
            localization_handoff,
            concise=concise,
            max_lines=28,
        )
    )
    if validation_only_delegated_subtask:
        sections.extend(
            [
                "",
                "Run the assigned validation slice in the workspace, gather concrete failure evidence, then call submit_patch.",
                "Prefer the targeted validation command from the Test Context or Reproduction Artifact. Do not broaden to repository-wide validation in this subtask.",
                "Your submit_patch payload should summarize validation outcomes, blockers, followups, tests you ran, and any changed files.",
            ]
        )
    else:
        sections.extend(
            [
                "",
                "Resolve the issue in the workspace, verify the result, then call submit_patch.",
                "Prefer focused verification first, especially the visible failing tests or most relevant test files from the Test Context. For benchmark-provided test commands, run the command directly so validation matches the scoring environment.",
                "Your submit_patch payload should summarize the fix, list tests you ran, and note changed files.",
            ]
        )
    if allow_delegation:
        sections.extend(
            [
                "",
                "# Delegation Plan",
                *_render_delegation_guidance(
                    rollout_brief,
                    delegation_mode=delegation_mode,
                ),
            ]
        )
    if delegated_subtask:
        sections.extend(
            _delegated_subtask_scope_rules(
                validation_only=validation_only_delegated_subtask,
                bullet=False,
            )
        )
    if issue_plan.test_context.exception_summaries:
        if validation_only_delegated_subtask:
            sections.extend(
                [
                    "Use the current failure signals from the Test Context as validation context. Report the blockers and candidate follow-up files instead of attempting broad implementation fixes here.",
                ]
            )
        else:
            sections.extend(
                [
                    "Treat the direct baseline exception from the Test Context as your first checkpoint. Clear that import or collection blocker before broadening the patch, then continue along the same nearby contract only if visible tests still fail.",
                ]
            )
        if completion_like:
            sections.extend(
                [
                    "For repository-completion tasks, after the first import or collection blocker clears, rerun the visible import path or failing-test command to surface the next blocker in the same package before stopping.",
                ]
            )
    if completion_like:
        # SPEED LEVER (rank-5/6, size-graduated broad re-validation).
        # ``broad_revalidation_mode`` is threaded from the orchestrator off the
        # ``_completion_loop_size_factor`` signal plus round position. The
        # default ``"continuous"`` (and every un-updated caller, every giant
        # suite, the final eligible round, and any prior-round near-pass) keeps
        # the historical "rerun the broader suite EVERY round" text
        # byte-identical. ``"convergence"`` fires ONLY for the smallest suites on
        # an early (non-final) round with no prior near-pass: the agent still
        # MUST produce a broad green confirmation run before submitting, but it
        # validates broadly on convergence / the final eligible round rather than
        # re-running the full suite on every turn once its targeted scope is
        # already green. The orchestrator's own per-round broad quick-verify is
        # the safety net between rounds, and the final-round / near-pass prompt
        # reverts to ``"continuous"`` so the broad validation that protects
        # literal final_pass_rate==1.0 is always preserved.
        _convergence_revalidation = str(broad_revalidation_mode).strip() == "convergence"
        if allow_partial_completion_roundtrip:
            if _convergence_revalidation:
                sections.extend(
                    [
                        "This is a repository-completion or public-API task. Do not stop after the first previously failing test passes without confirming the targeted scope is green.",
                        "Use this round to clear the immediate blocker and confirm the targeted scope passes; you do not need to re-run the full repository suite this round once your targeted scope is green. Return if that exposes the next nearby blocker. Later rounds will continue from the improved workspace, and the broad repository test command MUST pass on convergence (the final eligible round) before you submit — a narrow slice passing alone is not sufficient final evidence.",
                        "Preserve existing modules, imports, and public definitions unless you are replacing an obvious placeholder scaffold. Do not delete working symbols or collapse files while broadening the fix.",
                    ]
                )
            else:
                sections.extend(
                    [
                        "This is a repository-completion or public-API task. Do not stop after the first previously failing test passes without rerunning the broader visible suite once.",
                        "Use this round to clear the immediate blocker, rerun the broader repository test command from the Test Context or an equivalently broad suite once, and return if that exposes the next nearby blocker. Later rounds will continue from the improved workspace.",
                        "Preserve existing modules, imports, and public definitions unless you are replacing an obvious placeholder scaffold. Do not delete working symbols or collapse files while broadening the fix.",
                    ]
                )
        else:
            if _convergence_revalidation:
                sections.extend(
                    [
                        "This is a repository-completion or public-API task. Do not stop after the first previously failing test passes without confirming the targeted scope is green.",
                        "While iterating, it is enough to confirm the targeted scope passes; you do not need to re-run the full repository suite on every turn once that scope is green. Before you submit, the broader repository test command from the Test Context (or an equivalently broad suite that exercises alternate backends and integration paths) MUST pass — run it as your final confirmation. A narrow slice passing is not sufficient final evidence.",
                        "Preserve existing modules, imports, and public definitions unless you are replacing an obvious placeholder scaffold. Do not delete working symbols or collapse files while broadening the fix.",
                    ]
                )
            else:
                sections.extend(
                    [
                        "This is a repository-completion or public-API task. Do not stop after the first previously failing test passes.",
                        "Before finishing, rerun the broader repository test command from the Test Context, or an equivalently broad suite that exercises alternate backends and integration paths. A narrow slice passing is not sufficient evidence.",
                        "Preserve existing modules, imports, and public definitions unless you are replacing an obvious placeholder scaffold. Do not delete working symbols or collapse files while broadening the fix.",
                    ]
                )
        if _convergence_revalidation:
            sections.extend(
                [
                    "",
                    "# Convergence Budget",
                    "Aim to converge in a focused number of edit+verify cycles, like a single max-effort session: you already have the localization and the failing-test evidence, so do not re-explore the repository from scratch. Read only the files you need to fix this contract and confirm the fix.",
                    "This is advisory pacing, not a hard cap: you still MUST produce a green broad confirmation run before you submit. Spend your turns on edit-and-verify, not on repeated full-suite sweeps once your targeted scope is green.",
                ]
            )
    if allows_incomplete_test_completion:
        sections.extend(
            [
                "Treat the `incomplete_test_files` allowlist as narrow edit permission for explicit TODO/NotImplemented test bodies only. Prefer implementation fixes first, but when those placeholder bodies remain the only failing contract, complete them instead of leaving the suite red. Never rewrite assertions, expected outputs, or coverage to make failures disappear.",
            ]
        )
    elif completion_like:
        sections.extend(
            [
                "For repository-completion tasks, treat visible tests as read-only specification. Do not modify them unless a listed test file contains explicit TODO/NotImplemented scaffolding that clearly belongs to the task.",
            ]
        )
    else:
        sections.extend(
            [
                "Unless the task explicitly requires test changes, prefer implementation fixes over editing existing tests.",
            ]
        )
    if issue_plan.test_context.incomplete_source_files:
        sections.extend(
            [
                "The listed source files contain obvious implementation scaffolds. For repository-completion tasks, prefer completing the nearby contract in those files over narrowly masking a single failing reference.",
            ]
        )
    return "\n".join(sections)


def build_test_writer_prompt(
    issue_description: str,
    issue_plan: IssuePlan,
    rollout_brief: RolloutBrief,
    reproduction_artifact: Any = None,
    localization_artifact: Any = None,
    reproduction_summary: str | None = None,
    localization_summary: str | None = None,
    behavioral_obligations: Optional[list[str]] = None,
    authoritative_issue_targets: Optional[list[str]] = None,
    authoritative_required_axes: Optional[list[str]] = None,
    authoritative_test_files: Optional[list[str]] = None,
    authoritative_test_evidence_lines: Optional[list[str]] = None,
    allow_delegation: bool = False,
    concise: bool = True,
) -> str:
    permuted_focus_files = permute_focus_files(rollout_brief.focus_files, rollout_brief)
    exploration_directive = stage_first_action_directive("test_writer", rollout_brief)
    issue_contract_targets = list(
        authoritative_issue_targets or []
    ) or extract_issue_contract_targets(issue_description)
    planner_metadata = (
        issue_plan.planner_metadata if isinstance(issue_plan.planner_metadata, dict) else {}
    )
    task_state_context = (
        issue_plan.task_state_context if isinstance(issue_plan.task_state_context, dict) else {}
    )
    ledger = (
        task_state_context.get("test_generation_ledger")
        if isinstance(task_state_context.get("test_generation_ledger"), dict)
        else {}
    )
    raw_design_plan = (
        dict(ledger)
        if ledger
        else {
            "task_contract": dict(planner_metadata.get("test_generation_contract") or {}),
            "milestones": [
                dict(item)
                for item in list(planner_metadata.get("test_generation_milestones") or [])
                if isinstance(item, dict)
            ],
            "test_objectives": [
                dict(item)
                for item in list(planner_metadata.get("test_generation_objectives") or [])
                if isinstance(item, dict)
            ],
        }
    )
    design_plan = normalize_test_generation_design_payload(
        raw_design_plan,
        issue_description=issue_description,
        issue_summary=str(issue_plan.summary or "").strip(),
        success_criteria=list(rollout_brief.success_criteria or issue_plan.success_criteria),
        behavioral_obligations=list(behavioral_obligations or []),
        interface_targets=issue_contract_targets,
        required_axes=list(authoritative_required_axes or []),
    )
    task_contract = dict(design_plan.get("task_contract") or {})
    milestone_lines = [
        (
            f"{item.get('milestone_id')}: {item.get('title')} | objectives: "
            f"{', '.join(list(item.get('objective_ids') or [])) or 'none'}"
        ).strip()
        for item in list(design_plan.get("milestones") or [])
        if dict(item)
    ]
    objective_lines = [
        (
            f"{item.get('objective_id')}: {item.get('objective')} | milestone="
            f"{item.get('milestone_id')} | axes={', '.join(list(item.get('contract_axes') or [])) or 'none'}"
        ).strip()
        for item in list(design_plan.get("test_objectives") or [])
        if dict(item)
    ]
    # Phase F.2: render the prior iteration's F2P-style feedback at
    # the top of the prompt so the agent acts on it BEFORE writing the
    # next round of tests. Defensive: only renders when the engine
    # has stashed actionable feedback in planner_metadata.
    prior_feedback_section = ""
    raw_prior_feedback = planner_metadata.get("prior_iteration_f2p_feedback")
    if isinstance(raw_prior_feedback, dict) and raw_prior_feedback:
        try:
            from apex.evaluation.iteration_feedback import (
                IterationFeedback,
                render_iteration_feedback_prompt_block,
            )

            prior_feedback_obj = IterationFeedback(
                useless_p2p_tests=list(raw_prior_feedback.get("useless_p2p_tests") or []),
                likely_f2p_tests=list(raw_prior_feedback.get("likely_f2p_tests") or []),
                infrastructure_failures=list(
                    raw_prior_feedback.get("infrastructure_failures") or []
                ),
                p2p_count=int(raw_prior_feedback.get("p2p_count") or 0),
                f2p_likely_count=int(raw_prior_feedback.get("f2p_likely_count") or 0),
                infrastructure_failure_count=int(
                    raw_prior_feedback.get("infrastructure_failure_count") or 0
                ),
                iteration_index=int(raw_prior_feedback.get("iteration_index") or 0),
                failure_classification_label=str(
                    raw_prior_feedback.get("failure_classification_label") or ""
                ),
                repair_hints=[
                    dict(item)
                    for item in list(raw_prior_feedback.get("repair_hints") or [])
                    if isinstance(item, dict)
                ],
                failure_excerpts=dict(raw_prior_feedback.get("failure_excerpts") or {}),
                missing_modules=list(raw_prior_feedback.get("missing_modules") or []),
            )
            prior_feedback_section = render_iteration_feedback_prompt_block(prior_feedback_obj)
        except Exception:
            prior_feedback_section = ""

    quarantine_section = ""
    raw_quarantined = planner_metadata.get("quarantined_test_paths")
    if isinstance(raw_quarantined, dict) and raw_quarantined:
        lines = [
            "## Quarantined generated test files",
            "",
            (
                "The previous validation pass found syntax-broken generated "
                "test files. Do not re-emit these exact files unchanged; "
                "rewrite the test cleanly or choose a fresh path."
            ),
            "",
        ]
        for path, reason in list(raw_quarantined.items())[:8]:
            path_text = str(path or "").strip()
            if not path_text:
                continue
            reason_text = str(reason or "syntax_invalid").strip()
            lines.append(f"  * `{path_text}`: {reason_text[:220]}")
        quarantine_section = "\n".join(lines) + "\n"

    # Phase G.0: per-iteration axis-coverage feedback. Walks the agent's
    # own declared contract_axes; nudges next iteration to cover the
    # canonical four if any are missing. Generalizes outside benchmarks
    # because it's rooted in the agent's declared axes, not a gold
    # suite.
    axis_coverage_section = ""
    raw_axis_coverage = planner_metadata.get("prior_iteration_axis_coverage")
    if isinstance(raw_axis_coverage, dict) and raw_axis_coverage:
        try:
            from apex.evaluation.iteration_feedback import (
                AxisCoverageFeedback,
                render_axis_coverage_prompt_block,
            )

            axis_obj = AxisCoverageFeedback(
                covered_axes=list(raw_axis_coverage.get("covered_axes") or []),
                missing_axes=list(raw_axis_coverage.get("missing_axes") or []),
                artifact_count=int(raw_axis_coverage.get("artifact_count") or 0),
                iteration_index=int(raw_axis_coverage.get("iteration_index") or 0),
            )
            axis_coverage_section = render_axis_coverage_prompt_block(axis_obj)
        except Exception:
            axis_coverage_section = ""

    # Phase G.2: repo-test exemplars. Mined once per task by the
    # engine and stashed on planner_metadata; renders as a few-shot
    # section so the agent copies the project's idioms (import paths,
    # fixture style, assertion shape) instead of inventing them.
    exemplars_section = ""
    raw_exemplars = planner_metadata.get("repo_test_exemplars")
    if isinstance(raw_exemplars, list) and raw_exemplars:
        try:
            from apex.preprocessing.repo_test_exemplars import (
                TestExemplar,
                render_exemplars_prompt_block,
            )

            exemplar_objs = [
                TestExemplar(
                    path=str(item.get("path") or ""),
                    snippet=str(item.get("snippet") or ""),
                    imports=list(item.get("imports") or []),
                    target_symbols=list(item.get("target_symbols") or []),
                    score=float(item.get("score") or 0.0),
                    reason=str(item.get("reason") or ""),
                    language=str(item.get("language") or ""),
                )
                for item in raw_exemplars
                if isinstance(item, dict)
            ]
            exemplars_section = render_exemplars_prompt_block(exemplar_objs)
        except Exception:
            exemplars_section = ""

    # Phase G.9: cross-rollout testgen feedback. Mined from the shared
    # EpisodicMemoryBus, surfaces what OTHER parallel rollouts on this
    # task have discovered so the agent can converge or diverge
    # deliberately.
    cross_rollout_section = ""
    raw_cross_rollout = planner_metadata.get("cross_rollout_testgen_feedback")
    if isinstance(raw_cross_rollout, dict) and raw_cross_rollout:
        try:
            from apex.evaluation.iteration_feedback import (
                CrossRolloutFeedback,
                render_cross_rollout_prompt_block,
            )

            cross_obj = CrossRolloutFeedback(
                sibling_f2p_likely_paths=list(
                    raw_cross_rollout.get("sibling_f2p_likely_paths") or []
                ),
                sibling_axes_covered=list(raw_cross_rollout.get("sibling_axes_covered") or []),
                sibling_insensitive_paths=list(
                    raw_cross_rollout.get("sibling_insensitive_paths") or []
                ),
                sibling_count=int(raw_cross_rollout.get("sibling_count") or 0),
                iteration_index=int(raw_cross_rollout.get("iteration_index") or 0),
            )
            cross_rollout_section = render_cross_rollout_prompt_block(cross_obj)
        except Exception:
            cross_rollout_section = ""

    # Phase G.3: per-iteration unaddressed-edge feedback. When the
    # agent declared predicted_edges last iteration but didn't link
    # tests to all of them, surface the gap.
    edge_prediction_section = ""
    raw_edge_predictions = planner_metadata.get("prior_iteration_edge_predictions")
    if isinstance(raw_edge_predictions, dict) and raw_edge_predictions:
        try:
            from apex.evaluation.iteration_feedback import (
                EdgePredictionFeedback,
                render_edge_prediction_prompt_block,
            )

            edge_obj = EdgePredictionFeedback(
                predicted_count=int(raw_edge_predictions.get("predicted_count") or 0),
                exercised_count=int(raw_edge_predictions.get("exercised_count") or 0),
                unaddressed_edges=list(raw_edge_predictions.get("unaddressed_edges") or []),
                iteration_index=int(raw_edge_predictions.get("iteration_index") or 0),
            )
            edge_prediction_section = render_edge_prediction_prompt_block(edge_obj)
        except Exception:
            edge_prediction_section = ""

    # Phase G.1: per-iteration mutation-sensitivity feedback. When the
    # engine's optional in-loop mutation has stashed a sensitivity
    # report, surface it so the agent tightens loose assertions in the
    # next iteration. Empty / not-actionable reports render to "".
    mutation_sensitivity_section = ""
    raw_mutation_sensitivity = planner_metadata.get("prior_iteration_mutation_sensitivity")
    if isinstance(raw_mutation_sensitivity, dict) and raw_mutation_sensitivity:
        try:
            from apex.evaluation.iteration_feedback import (
                MutationSensitivityFeedback,
                render_mutation_sensitivity_prompt_block,
            )

            mutation_obj = MutationSensitivityFeedback(
                sensitive_count=int(raw_mutation_sensitivity.get("sensitive_count") or 0),
                insensitive_count=int(raw_mutation_sensitivity.get("insensitive_count") or 0),
                mutants_evaluated=int(raw_mutation_sensitivity.get("mutants_evaluated") or 0),
                sensitivity_score=float(raw_mutation_sensitivity.get("sensitivity_score") or 0.0),
                skip_reason=str(raw_mutation_sensitivity.get("skip_reason") or ""),
                iteration_index=int(raw_mutation_sensitivity.get("iteration_index") or 0),
                target_source_paths=list(raw_mutation_sensitivity.get("target_source_paths") or []),
                killed_mutant_signatures=list(
                    raw_mutation_sensitivity.get("killed_mutant_signatures") or []
                ),
                survived_mutant_signatures=list(
                    raw_mutation_sensitivity.get("survived_mutant_signatures") or []
                ),
            )
            mutation_sensitivity_section = render_mutation_sensitivity_prompt_block(mutation_obj)
        except Exception:
            mutation_sensitivity_section = ""

    test_quality_section = ""
    raw_test_quality = planner_metadata.get("prior_iteration_test_quality")
    if isinstance(raw_test_quality, dict) and raw_test_quality:
        try:
            from apex.evaluation.iteration_feedback import (
                GeneratedTestQualityFeedback,
                render_generated_test_quality_prompt_block,
            )

            quality_obj = GeneratedTestQualityFeedback(
                artifact_count=int(raw_test_quality.get("artifact_count") or 0),
                weak_artifact_count=int(raw_test_quality.get("weak_artifact_count") or 0),
                issue_count=int(raw_test_quality.get("issue_count") or 0),
                issue_counts={
                    str(key): int(value or 0)
                    for key, value in dict(raw_test_quality.get("issue_counts") or {}).items()
                },
                mean_assertion_effect_score=float(
                    raw_test_quality.get("mean_assertion_effect_score") or 0.0
                ),
                iteration_index=int(raw_test_quality.get("iteration_index") or 0),
            )
            test_quality_section = render_generated_test_quality_prompt_block(quality_obj)
        except Exception:
            test_quality_section = ""

    test_stability_section = ""
    raw_test_stability = planner_metadata.get("prior_iteration_test_stability")
    if isinstance(raw_test_stability, dict) and raw_test_stability:
        try:
            from apex.evaluation.iteration_feedback import (
                TestStabilityFeedback,
                render_test_stability_prompt_block,
            )

            stability_obj = TestStabilityFeedback(
                status=str(raw_test_stability.get("status") or ""),
                run_count=int(raw_test_stability.get("run_count") or 0),
                failed_run_count=int(raw_test_stability.get("failed_run_count") or 0),
                flaky_nodeids=list(raw_test_stability.get("flaky_nodeids") or []),
                iteration_index=int(raw_test_stability.get("iteration_index") or 0),
            )
            test_stability_section = render_test_stability_prompt_block(stability_obj)
        except Exception:
            test_stability_section = ""

    assertion_mutation_section = ""
    raw_assertion_mutation = planner_metadata.get("prior_iteration_assertion_mutation")
    if isinstance(raw_assertion_mutation, dict) and raw_assertion_mutation:
        try:
            from apex.evaluation.iteration_feedback import (
                AssertionMutationFeedback,
                render_assertion_mutation_prompt_block,
            )

            assertion_obj = AssertionMutationFeedback(
                status=str(raw_assertion_mutation.get("status") or ""),
                mutated_assertion_count=int(
                    raw_assertion_mutation.get("mutated_assertion_count") or 0
                ),
                survived=bool(raw_assertion_mutation.get("survived")),
                assertion_effective=bool(raw_assertion_mutation.get("assertion_effective")),
                test_paths=list(raw_assertion_mutation.get("test_paths") or []),
                iteration_index=int(raw_assertion_mutation.get("iteration_index") or 0),
            )
            assertion_mutation_section = render_assertion_mutation_prompt_block(assertion_obj)
        except Exception:
            assertion_mutation_section = ""

    # Phase I.3: per-iteration coverage-gap feedback. Tells the agent
    # WHICH lines of the focus file(s) the current portfolio does not
    # exercise — strongest possible targeting signal. Empty / skip
    # reports (no_coverage_tool, no_test_paths, etc.) render to "".
    coverage_gap_section = ""
    raw_coverage_gap = planner_metadata.get("prior_iteration_coverage_gap")
    if isinstance(raw_coverage_gap, dict) and raw_coverage_gap:
        try:
            from apex.evaluation.iteration_feedback import (
                CoverageGapFeedback,
                render_coverage_gap_prompt_block,
            )

            raw_ranges = raw_coverage_gap.get("per_file_uncovered_ranges") or {}
            normalized_ranges: dict[str, list[tuple[int, int]]] = {}
            for path, ranges in raw_ranges.items():
                normalized_ranges[str(path)] = [
                    (int(r[0]), int(r[1])) for r in ranges if len(r) >= 2
                ]
            raw_branches = raw_coverage_gap.get("per_file_missing_branches") or {}
            normalized_branches: dict[str, list[tuple[int, int]]] = {}
            for path, branches in raw_branches.items():
                normalized_branches[str(path)] = [
                    (int(branch[0]), int(branch[1])) for branch in branches if len(branch) >= 2
                ]
            coverage_obj = CoverageGapFeedback(
                target_source_paths=list(raw_coverage_gap.get("target_source_paths") or []),
                per_file_uncovered_ranges=normalized_ranges,
                per_file_missing_branches=normalized_branches,
                missing_target_source_paths=list(
                    raw_coverage_gap.get("missing_target_source_paths") or []
                ),
                per_file_total_lines=dict(raw_coverage_gap.get("per_file_total_lines") or {}),
                per_file_total_branches=dict(raw_coverage_gap.get("per_file_total_branches") or {}),
                overall_coverage_ratio=float(raw_coverage_gap.get("overall_coverage_ratio") or 0.0),
                overall_branch_coverage_ratio=float(
                    raw_coverage_gap.get("overall_branch_coverage_ratio") or 1.0
                ),
                skip_reason=str(raw_coverage_gap.get("skip_reason") or ""),
                iteration_index=int(raw_coverage_gap.get("iteration_index") or 0),
            )
            coverage_gap_section = render_coverage_gap_prompt_block(coverage_obj)
        except Exception:
            coverage_gap_section = ""

    # Phase I.7: cross-task persistent testgen insights (memory-from-
    # past-runs-on-this-repo). Surfaced from planner_metadata so the
    # benchmark / modes layer can opt in / out independently. Empty
    # list (or unset) renders to "".
    prior_testgen_memory_section = ""
    raw_prior_testgen_memory = planner_metadata.get("prior_testgen_memory_insights")
    if isinstance(raw_prior_testgen_memory, list) and raw_prior_testgen_memory:
        try:
            from apex.persistence import (
                PersistedInsight,
                render_prior_testgen_insights_prompt_block,
            )

            insight_objs = [
                PersistedInsight.from_dict(d)
                for d in raw_prior_testgen_memory
                if isinstance(d, dict)
            ]
            prior_testgen_memory_section = render_prior_testgen_insights_prompt_block(insight_objs)
        except Exception:
            prior_testgen_memory_section = ""

    sections = [
        "# Issue",
        _truncate_block(issue_description, concise=concise, max_lines=18),
        "",
    ]
    if prior_testgen_memory_section:
        sections.extend([prior_testgen_memory_section, ""])
    if prior_feedback_section:
        sections.extend([prior_feedback_section, ""])
    if quarantine_section:
        sections.extend([quarantine_section, ""])
    if axis_coverage_section:
        sections.extend([axis_coverage_section, ""])
    if mutation_sensitivity_section:
        sections.extend([mutation_sensitivity_section, ""])
    if test_quality_section:
        sections.extend([test_quality_section, ""])
    if test_stability_section:
        sections.extend([test_stability_section, ""])
    if assertion_mutation_section:
        sections.extend([assertion_mutation_section, ""])
    if coverage_gap_section:
        sections.extend([coverage_gap_section, ""])
    if exemplars_section:
        sections.extend([exemplars_section, ""])
    if edge_prediction_section:
        sections.extend([edge_prediction_section, ""])
    if cross_rollout_section:
        sections.extend([cross_rollout_section, ""])
    sections.extend(
        [
            "# Primary Objective",
            issue_plan.summary,
            "",
            "# First-action directive (rollout-specific)",
            exploration_directive or "Infer the best test targets from the repo.",
            "",
            "# Focus Files",
            _render_items(
                permuted_focus_files,
                fallback="- infer the best test targets from the repo",
                concise=concise,
                limit=8,
            ),
            "",
            "# Success Criteria",
            _render_items(
                rollout_brief.success_criteria or issue_plan.success_criteria,
                fallback="- capture the bug and expected behavior",
                concise=concise,
                limit=4,
            ),
            "",
            "# Focus Repo Map",
            _truncate_block(issue_plan.repo_focus_map, concise=concise, max_lines=42),
        ]
    )
    if task_contract:
        sections.extend(
            [
                "",
                "# Task Contract",
                "Problem Statement:",
                _truncate_block(
                    str(task_contract.get("problem_statement") or ""),
                    concise=concise,
                    max_lines=4,
                )
                or "- derive from issue and visible evidence",
                "Acceptance Requirements:",
                _render_items(
                    list(task_contract.get("acceptance_requirements") or []),
                    fallback="- derive concrete acceptance requirements",
                    concise=concise,
                    limit=8,
                ),
                "Interface Specification:",
                _render_items(
                    list(task_contract.get("interface_specification") or []),
                    fallback="- infer the named public interfaces",
                    concise=concise,
                    limit=6,
                ),
            ]
        )
    if issue_contract_targets:
        sections.extend(
            [
                "",
                "# Issue-Declared Interfaces",
                _render_items(
                    issue_contract_targets,
                    fallback="- none declared",
                    concise=concise,
                    limit=4,
                ),
            ]
        )
    if behavioral_obligations:
        sections.extend(
            [
                "",
                "# Behavioral Obligations",
                _render_items(
                    list(behavioral_obligations or []),
                    fallback="- infer the strongest authoritative obligations from the repo",
                    concise=concise,
                    limit=10,
                ),
            ]
        )
    if authoritative_required_axes:
        sections.extend(
            [
                "",
                "# Required Contract Axes",
                _render_items(
                    list(authoritative_required_axes or []),
                    fallback="- positive_path",
                    concise=concise,
                    limit=8,
                ),
            ]
        )
    if authoritative_test_files:
        sections.extend(
            [
                "",
                "# Authoritative Existing Tests",
                _render_items(
                    list(authoritative_test_files or []),
                    fallback="- infer the nearest authoritative tests from the repo",
                    concise=concise,
                    limit=6,
                ),
            ]
        )
    if authoritative_test_evidence_lines:
        sections.extend(
            [
                "",
                "# Nearby Authoritative Test Evidence",
                _render_items(
                    list(authoritative_test_evidence_lines or []),
                    fallback="- infer the strongest nearby test obligations from the repo",
                    concise=concise,
                    limit=10,
                ),
            ]
        )
    if milestone_lines:
        sections.extend(
            [
                "",
                "# Milestones",
                _render_items(
                    milestone_lines,
                    fallback="- milestone planning required",
                    concise=concise,
                    limit=6,
                ),
            ]
        )
    if objective_lines:
        sections.extend(
            [
                "",
                "# Test Objectives",
                _render_items(
                    objective_lines,
                    fallback="- derive objective-level coverage from the contract",
                    concise=concise,
                    limit=10,
                ),
            ]
        )
    sections.extend(_render_search_policy_block(rollout_brief, concise=concise))
    sections.extend(_render_frontier_target_block(rollout_brief, concise=concise))
    sections.extend(_render_test_context_block(issue_plan.test_context, concise=concise))
    sections.extend(
        _render_task_state_block(issue_plan, concise=concise, rollout_brief=rollout_brief)
    )
    repro_handoff = (
        reproduction_artifact if reproduction_artifact is not None else reproduction_summary
    )
    localization_handoff = (
        localization_artifact if localization_artifact is not None else localization_summary
    )
    sections.extend(
        _render_artifact_block(
            "Reproduction Artifact",
            repro_handoff,
            concise=concise,
            max_lines=28,
        )
    )
    sections.extend(
        _render_artifact_block(
            "Localization Artifact",
            localization_handoff,
            concise=concise,
            max_lines=28,
        )
    )
    sections.extend(
        [
            "",
            "Write a focused test portfolio in the repository's native framework and submit it with submit_test_suite.",
            "Do not return one undifferentiated suite unless the repository genuinely only supports one minimal artifact.",
            "Prioritize these buckets when justified by the repo and issue:",
            "- regression tests from the issue, reproduction artifact, and failing traces",
            "- contract or API tests mined from docs, examples, types, and existing tests",
            "- edge and negative tests",
            "- property and metamorphic tests",
            "- differential tests against a reference behavior when available",
            "- fuzz seeds for parser, serializer, or state-machine surfaces when appropriate",
            "First derive and return an explicit `task_contract` with `problem_statement`, `acceptance_requirements`, and `interface_specification`.",
            "Map that contract into explicit `milestones` and `test_objectives`. Every generated artifact must attach to one `milestone_id` and one `objective_id`.",
            "Use the five-stage tactical flow for each objective: context/hypothesis, pass-then-invert synthesis, execution-feedback refinement, mutation-driven discrimination, and dual-version verification.",
            "For each artifact, record `objective`, `acceptance_requirements`, `interface_specification`, `oracle_origin`, `pass_then_invert`, `dual_version_verified`, and `objective_status`.",
            "For each assertion-bearing artifact, explicitly state `expected_fixed_behavior`, `expected_broken_failure_mode`, `authoritative_source`, and `public_surface`. If any of those are missing or vague, the artifact is not selectable.",
            "Prefer pass-then-invert over direct failing-test synthesis when possible: first capture current passing behavior, then invert assertions into the desired failing oracle.",
            "Treat `pass_then_invert` as structured metadata, not prose. Record whether you wrote a passing precursor, how you inverted it, and what execution-feedback cleanup was needed, including an explicit `execution_feedback_summary` even when the answer is that no cleanup was needed.",
            "Do not rely on omitted IDs or inferred defaults for artifact design metadata. Emit `milestone_id`, `objective_id`, `objective`, `acceptance_requirements`, `interface_specification`, and `oracle_origin` explicitly on every assertion-bearing artifact.",
            "Use `regression_suite_summary` to describe the cumulative milestone regression suite and `minimization_summary` to explain what coverage was kept versus removed as redundant.",
            "When mutation discrimination is only partially measurable, say so explicitly in artifact metadata instead of fabricating confidence.",
            "Use `contract_hypotheses` as an explicit obligation checklist, not a brainstorm. Enumerate the authoritative behaviors the generated suite must preserve, grounded in the issue, reproduction evidence, localization evidence, and nearby visible tests.",
            "Every authoritative obligation should be covered by at least one assertion-bearing artifact in this generated portfolio.",
            "For each artifact, include path, content, strategy, materialization_mode, contract_targets, contract_axes, summary, test_descriptions, focus_files, focus_tests, contract_sources, justification, milestone_id, objective_id, and pass_then_invert metadata.",
            "Do not modify application or source files in this stage. Restrict edits to test artifacts and test-local support files only.",
            "Use contract_targets only for entry points directly exercised by that artifact; keep broader supporting citations in reference_targets.",
            "If the issue explicitly names an interface or method, at least one artifact must exercise that named surface directly; aliases or wrappers may appear only as supplemental compatibility coverage.",
            "If reproduction or localization artifacts propose a different helper, alias, or sibling API than the issue-declared surface, treat the issue-declared surface as authoritative for promotable artifacts.",
            "Start with one minimal direct regression in the nearest authoritative existing test file for that surface when one is available. Add another artifact only when it covers a distinct contract axis or an independently documented public behavior not already covered by the direct artifact.",
            "Treat nearby existing tests as authoritative evidence to mine obligations from, not as a reason to omit those obligations from the generated suite. Do not say that nearby tests already cover the runtime behavior unless you are explicitly appending that exact file and preserving the behavior there.",
            "When nearby visible tests exercise the same behavior through both a higher-level public surface and a lower-level helper or category, keep the higher-level public surface as the promotable contract and treat helper-only coverage as supplemental.",
            "Use the nearest existing repository tests as executable templates: copy their import style, fixtures, setup hooks, mocks, test registration, async wrappers, and runner conventions before inventing any standalone harness.",
            "For the direct issue-surface artifact, include one minimal canonical happy-path example before broader edge or malformed cases.",
            "When both positive-path and multi-ordering coverage matter, keep them as separate examples: first one single canonical success, then one ordered multi-value case.",
            "Keep the named issue surface as the assertion surface for each required contract axis; do not satisfy missing, malformed, or ordering coverage only through an alias, helper, or sibling API.",
            "When a helper, wrapper, or module option is involved, include at least one observable behavior assertion at the public surface, not only plumbing or forwarding assertions.",
            "When the same option, auth source, flag, or behavior appears on multiple named public surfaces in the task evidence, give each surface its own direct assertion-bearing artifact instead of collapsing coverage into only a shared lower-level helper or request-stack test.",
            "When nearby authoritative tests exercise both a public wrapper, context manager, or entry-point surface and a lower-level helper or filter in the same module, keep the public wrapper or entry-point as the end-to-end assertion surface for the required behavior; helper or filter checks are supplemental only.",
            "When authoritative evidence shows that a concrete input source, auth source, option toggle, or default-vs-opt-out path drives the public behavior, encode that input on the same artifact that asserts the visible result.",
            "Do not replace concrete public-outcome checks with generic passthrough assertions such as `assert result == (...)` or placeholder mocked content while separately checking forwarded kwargs.",
            "When the same documented option or auth-source behavior appears on multiple public surfaces, give each surfaced API its own observable assertion-bearing case instead of leaving any sibling surface as a forwarding-only wrapper test.",
            "Preserve concrete user-visible semantics from nearby visible tests, including focus movement, caret or selection behavior, positional rendering, ignored invalid input, and no-op boundaries, rather than replacing them with broader but behaviorally different assertions.",
            "When the task evidence describes both unchanged default behavior and an explicit opt-out or override, cover both as separate assertion-bearing cases.",
            "Do not invert a documented default or legacy behavior just because the patch introduces a new knob or helper. Follow the authoritative visible tests, docs, types, and reproduction evidence when they constrain the default path.",
            "When the issue, reproduction, docs, or visible tests show a concrete input shape, include one exact-shape regression on that surface before adding generalized variants.",
            "When nearby visible tests enumerate exact literals or parametrized value matrices on the same surface, preserve those concrete examples instead of collapsing them into one broader generalized case.",
            "For a documented leaf-field contract, make the canonical positive example use only the minimal required container and leaf fields. Do not add discriminator or sibling fields such as `type`, `kind`, `tag`, `variant`, or `mode` unless authoritative existing tests, docs, or types for that same surface explicitly require them.",
            "For malformed coverage on a documented field-path contract, start by removing or corrupting the documented required fields themselves before inventing extra enum tags, helper-only sentinels, or unrelated object-shape failures.",
            "For malformed coverage on a documented field-path contract, prefer a small negative-shape matrix: one case where the expected container key is absent, one case where an unexpected sibling key replaces it, and one case where the container exists but the documented leaf field is missing or empty.",
            "When the public contract is described through a field path, return shape, or visible assertion surface, encode that rule directly. Do not require extra sibling keys, helper-only sentinel fields, or internal round-trip invariants unless docs, types, or existing tests independently justify them.",
            "If an artifact extends an existing repository test file, set materialization_mode to append. Use replace only for full-file rewrites or brand-new generated test files.",
            "Each broadened malformed variant, fixture-shape normalization, alias claim, or sibling API claim must cite an independent source in contract_sources; otherwise keep it exploratory rather than promotable.",
            "Do not turn inferred implementation details, alias parity, or helper coupling into strong contract assertions unless docs, types, existing tests, or the reproduction evidence independently justify them.",
            "Only mark an artifact promotable when it is independently justified, non-redundant, and likely to survive baseline-preserving reruns. Otherwise keep it exploratory.",
        ]
    )
    if allow_delegation:
        sections.append(
            "If delegation is enabled, use one contract-miner thread, multiple diverse generators from different vendors when available, and one adjudication thread before finalizing the portfolio."
        )
    return "\n".join(sections)


def build_selector_prompt(
    issue_description: str,
    patches: list[dict[str, Any]],
    concise: bool = True,
) -> str:
    sections = [
        "# Issue",
        _truncate_block(issue_description, concise=concise, max_lines=18),
        "",
        "# Candidate Patches",
        "Choose the patch most likely to resolve the issue correctly.",
        "",
    ]

    for index, patch in enumerate(patches, start=1):
        sections.extend(
            [
                f"## Patch {index}",
                f"Summary: {patch.get('summary') or 'n/a'}",
                f"Changed files: {', '.join(patch.get('changed_files', [])) or 'n/a'}",
                f"Verification score: {patch.get('verification_score', 0):.2f}",
                f"Cross-validation score: {patch.get('cross_validation_score', 0):.2f}",
                f"Cluster size: {patch.get('cluster_size', 1)}",
                "```diff",
                _truncate_block(str(patch.get("diff") or ""), concise=concise, max_lines=120),
                "```",
                "",
            ]
        )

    sections.extend(
        [
            "Respond with only the number of the best patch.",
        ]
    )
    return "\n".join(sections)
