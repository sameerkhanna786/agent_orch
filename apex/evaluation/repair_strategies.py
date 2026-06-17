"""Diversified, scope-shrinking repair strategies for generated tests."""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .oracle_repair import repair_assertions_with_captured_oracles
from .test_minimizer import drop_tests_from_artifact_with_report

logger = logging.getLogger(__name__)


# Caller-provided LLM repair function. Takes (prompt, schema_or_None) and
# returns a dict-shaped response with at least ``tool`` (= "repair" |
# "give_up") and either ``test_source`` (when repairing) or ``reason``.
# Returning ``None`` is treated as "LLM unavailable" and the test is
# dropped without further repair attempts.
TraceRepairLLMCaller = Callable[[str, dict[str, Any]], Optional[dict[str, Any]]]


REPAIR_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "tool": {"type": "string", "enum": ["repair", "give_up"]},
        "test_source": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["tool"],
}


@dataclass(frozen=True)
class RepairStrategyResult:
    status: str
    artifact_text: str
    strategy: str
    changed: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_repair_strategy(
    artifact_text: str,
    diagnostic: dict[str, Any],
    *,
    attempt: int,
    workdir: Path | None = None,
    docker_runner: Any | None = None,
) -> RepairStrategyResult:
    attempt = max(0, int(attempt or 0))
    if attempt <= 0:
        return RepairStrategyResult(
            status="noop",
            artifact_text=artifact_text,
            strategy="regenerate_full",
        )
    if attempt == 1:
        failing = _failing_names(diagnostic)
        repaired, dropped = drop_tests_from_artifact_with_report(
            artifact_text,
            failing,
            keep_minimum=1,
        )
        return RepairStrategyResult(
            status="repaired" if dropped else "unchanged",
            artifact_text=repaired,
            strategy="drop_failing_tests",
            changed=bool(dropped),
            diagnostics={"dropped_tests": dropped},
        )
    if attempt == 2:
        # Execution-grounded oracle repair (W4). Falls through to the next
        # strategy when no workdir is available or no candidate assertions
        # were rewritten.
        if workdir is not None:
            failing = _failing_names(diagnostic)
            outcome = repair_assertions_with_captured_oracles(
                artifact_text,
                workdir=workdir,
                failing_test_names=failing,
                docker_runner=docker_runner,
            )
            if outcome.changed:
                return RepairStrategyResult(
                    status="repaired",
                    artifact_text=outcome.artifact_text,
                    strategy="execution_grounded_oracle",
                    changed=True,
                    diagnostics={
                        "rewritten_count": outcome.rewritten_count,
                        "skipped_count": outcome.skipped_count,
                        "per_assertion": outcome.diagnostics,
                    },
                )
        return _transform_assertions(
            artifact_text,
            strategy="simplify_oracle_to_repr",
            transformer=_ReprEqualityTransformer(),
        )
    if attempt == 3:
        return _transform_assertions(
            artifact_text,
            strategy="drop_assertion_keep_call",
            transformer=_DropAssertionKeepCallTransformer(),
        )
    failing = _failing_names(diagnostic)
    repaired, dropped = drop_tests_from_artifact_with_report(
        artifact_text,
        failing,
        keep_minimum=0,
    )
    return RepairStrategyResult(
        status="repaired" if dropped else "unchanged",
        artifact_text=repaired,
        strategy="drop_test",
        changed=bool(dropped),
        diagnostics={"dropped_tests": dropped},
    )


def strategy_name_for_attempt(attempt: int) -> str:
    return [
        "regenerate_full",
        "drop_failing_tests",
        "execution_grounded_oracle",
        "drop_assertion_keep_call",
        "drop_test",
    ][min(max(0, int(attempt or 0)), 4)]


def _transform_assertions(
    source: str,
    *,
    strategy: str,
    transformer: ast.NodeTransformer,
) -> RepairStrategyResult:
    try:
        tree = ast.parse(source or "")
    except SyntaxError as exc:
        return RepairStrategyResult(
            status="unchanged",
            artifact_text=source,
            strategy=strategy,
            diagnostics={"error": str(exc)},
        )
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    try:
        rendered = ast.unparse(new_tree).strip() + "\n"
    except Exception as exc:
        return RepairStrategyResult(
            status="unchanged",
            artifact_text=source,
            strategy=strategy,
            diagnostics={"error": f"unparse_failed: {type(exc).__name__}"},
        )
    # Strict W3 gate after every artifact mutation: parse + compile.
    # If the transformer left the source in an unbuildable shape we
    # revert rather than ship a broken candidate.
    from .final_acceptance_gate import strict_syntax_check

    syntax_ok, syntax_err = strict_syntax_check(rendered)
    if not syntax_ok:
        return RepairStrategyResult(
            status="unchanged",
            artifact_text=source,
            strategy=strategy,
            diagnostics={"error": f"strict_syntax_check failed: {syntax_err}"},
        )
    changed = rendered != str(source or "")
    return RepairStrategyResult(
        status="repaired" if changed else "unchanged",
        artifact_text=rendered,
        strategy=strategy,
        changed=changed,
    )


def _failing_names(diagnostic: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for key in ("failing_tests", "errored_tests", "failing_test_names", "errored_test_names"):
        for item in list((diagnostic or {}).get(key) or []):
            text = str(item)
            if "::" in text:
                text = text.rsplit("::", 1)[-1]
            text = text.split("[", 1)[0]
            if text.startswith("test_"):
                names.add(text)
    per_test = dict((diagnostic or {}).get("per_test_status") or {})
    for nodeid, status in per_test.items():
        if str(status).lower() in {"fail", "error"}:
            text = str(nodeid).rsplit("::", 1)[-1].split("[", 1)[0]
            if text.startswith("test_"):
                names.add(text)
    return names


class _ReprEqualityTransformer(ast.NodeTransformer):
    def visit_Assert(self, node: ast.Assert) -> ast.AST:
        node = self.generic_visit(node)
        if (
            isinstance(node, ast.Assert)
            and isinstance(node.test, ast.Compare)
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Eq)
            and len(node.test.comparators) == 1
        ):
            left = ast.Call(
                func=ast.Name(id="repr", ctx=ast.Load()), args=[node.test.left], keywords=[]
            )
            right = ast.Call(
                func=ast.Name(id="repr", ctx=ast.Load()),
                args=[node.test.comparators[0]],
                keywords=[],
            )
            node.test = ast.Compare(left=left, ops=[ast.Eq()], comparators=[right])
        return node


class _DropAssertionKeepCallTransformer(ast.NodeTransformer):
    def visit_Assert(self, node: ast.Assert) -> ast.AST | list[ast.AST]:
        if isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Call):
            return ast.Expr(value=node.test.left)
        if isinstance(node.test, ast.Call):
            return ast.Expr(value=node.test)
        return ast.Pass()


# ---------------------------------------------------------------------------
# Execution-trace-driven LLM repair (P0 step 2 of TESTGEN_QUALITY_PLAN)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceRepairAttemptResult:
    """One per-test repair attempt outcome."""

    test_name: str
    status: str  # "repaired" | "give_up" | "llm_unavailable" | "swap_failed" | "skipped"
    reason: str = ""
    new_source: str = ""
    error_kind: str = ""
    attempts_consumed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# P2 step 9: per-error-kind LLM call budget. Budget 0 ⇒ skip the LLM
# entirely (the trace tells us we can't repair this kind of failure).
# Budget N ⇒ allow up to N LLM-call attempts across gate iterations.
DEFAULT_ERROR_KIND_BUDGETS: dict[str, int] = {
    "AssertionError": 2,
    "AttributeError": 1,
    "NameError": 1,
    "TypeError": 1,
    "ValueError": 1,
    "KeyError": 1,
    "IndexError": 1,
    "RuntimeError": 1,
    "SyntaxError": 0,
    "IndentationError": 0,
    "ImportError": 0,
    "ModuleNotFoundError": 0,
    "unknown": 1,
}


_ERROR_KIND_RE = re.compile(
    r"^E\s+([A-Za-z_][A-Za-z0-9_.]*Error|[A-Za-z_][A-Za-z0-9_.]*Exception):"
)


def _classify_error_kind(trace_text: str) -> str:
    """Classify the trailing pytest trace into a Python exception class name.

    Looks for the canonical ``E   <ErrorType>: ...`` line that pytest emits
    in ``--tb=short`` mode. Returns the bare class name (not the dotted
    path) so the budget lookup is stable. Falls back to ``"unknown"`` when
    no such marker is found — the caller assigns the unknown-default budget.
    """

    if not trace_text:
        return "unknown"
    for line in trace_text.splitlines():
        match = _ERROR_KIND_RE.match(line.strip())
        if match:
            kind = match.group(1)
            if "." in kind:
                kind = kind.rsplit(".", 1)[-1]
            return kind
    return "unknown"


@dataclass(frozen=True)
class TraceRepairResult:
    """Aggregate outcome of repair_failing_tests_with_trace."""

    artifact_text: str
    attempts: list[TraceRepairAttemptResult] = field(default_factory=list)
    repaired_count: int = 0
    skipped_count: int = 0

    @property
    def changed(self) -> bool:
        return self.repaired_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_text": self.artifact_text,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "repaired_count": self.repaired_count,
            "skipped_count": self.skipped_count,
            "changed": self.changed,
        }


def repair_failing_tests_with_trace(
    *,
    artifact_text: str,
    failing_test_traces: dict[str, str],
    focal_source: str = "",
    focal_module_path: str = "",
    llm_caller: Optional[TraceRepairLLMCaller],
    max_repairs_per_call: int = 5,
    kind_attempt_counts: Optional[dict[str, int]] = None,
    error_kind_budgets: Optional[dict[str, int]] = None,
) -> TraceRepairResult:
    """For each failing test, send the trace + test source + focal source
    to the LLM and ask for a corrected test. Swap successful repairs
    back into the artifact (preserve test name / decorators).

    Sits on top of the in-process per-test acceptance gate (P0 step 1).
    The gate calls this BEFORE the AST drop step so brittle tests get
    one repair attempt before being abandoned. Tests that the LLM
    refuses to repair (or whose swap fails) get dropped by the caller.

    Args:
        artifact_text: full candidate test file source.
        failing_test_traces: ``{bare_test_name: pytest_trace}``. Produced
            by ``final_acceptance_gate.extract_pytest_traces``.
        focal_source: optional focal-module source (provides repair
            context). When empty, the LLM repairs from the trace alone.
        focal_module_path: optional focal-module file path (cosmetic; for
            the prompt context).
        llm_caller: pluggable LLM bridge. ``None`` short-circuits — the
            function returns the artifact unchanged. Production wires
            this to ``CLIModelClient.run_structured_prompt``.
        max_repairs_per_call: cap on per-invocation LLM calls. Protects
            against runaway when many tests fail simultaneously. Tests
            beyond the cap are reported as ``status="skipped"``.

    Returns:
        ``TraceRepairResult`` with the (possibly partially-) repaired
        artifact + per-attempt diagnostics.
    """

    if llm_caller is None or not failing_test_traces:
        return TraceRepairResult(artifact_text=artifact_text)

    try:
        tree = ast.parse(artifact_text or "")
    except SyntaxError:
        # Can't surgically repair if we can't parse. Caller's drop-step
        # is the right path here; this function is a no-op.
        return TraceRepairResult(artifact_text=artifact_text)

    test_node_index = _index_test_nodes(tree)
    attempts: list[TraceRepairAttemptResult] = []
    repaired_count = 0
    skipped_count = 0
    cap = max(1, int(max_repairs_per_call))
    invoked = 0

    # P2 step 9: per-error-kind budget. The caller may pass a mutable
    # ``kind_attempt_counts`` dict that we mutate in place across
    # iterations of the gate's outer loop.
    budgets = dict(DEFAULT_ERROR_KIND_BUDGETS)
    if error_kind_budgets:
        budgets.update(error_kind_budgets)
    counts = kind_attempt_counts if kind_attempt_counts is not None else {}

    # Iterate in deterministic (alphabetical) order so retries land
    # consistently and tests are easy to reproduce.
    for test_name in sorted(failing_test_traces):
        if invoked >= cap:
            attempts.append(
                TraceRepairAttemptResult(
                    test_name=test_name,
                    status="skipped",
                    reason=f"per-call repair cap of {cap} reached",
                )
            )
            skipped_count += 1
            continue

        # P2 step 9 budget gate: classify error kind, look up budget,
        # short-circuit when this test has used or exceeded the budget.
        error_kind = _classify_error_kind(failing_test_traces[test_name])
        budget = budgets.get(error_kind, budgets.get("unknown", 1))
        attempts_so_far = int(counts.get(test_name, 0))
        if budget <= 0 or attempts_so_far >= budget:
            attempts.append(
                TraceRepairAttemptResult(
                    test_name=test_name,
                    status="give_up",
                    reason=(
                        f"error_kind {error_kind!r} exhausted "
                        f"budget={budget} (used {attempts_so_far})"
                    ),
                    error_kind=error_kind,
                    attempts_consumed=attempts_so_far,
                )
            )
            skipped_count += 1
            continue

        original_node = test_node_index.get(test_name)
        if original_node is None:
            attempts.append(
                TraceRepairAttemptResult(
                    test_name=test_name,
                    status="skipped",
                    reason="test function not found in artifact AST",
                    error_kind=error_kind,
                    attempts_consumed=attempts_so_far,
                )
            )
            skipped_count += 1
            continue
        try:
            original_source = ast.unparse(original_node)
        except Exception as exc:  # pragma: no cover - defensive
            attempts.append(
                TraceRepairAttemptResult(
                    test_name=test_name,
                    status="skipped",
                    reason=f"ast.unparse failed: {type(exc).__name__}",
                    error_kind=error_kind,
                    attempts_consumed=attempts_so_far,
                )
            )
            skipped_count += 1
            continue

        prompt = _build_trace_repair_prompt(
            test_name=test_name,
            test_source=original_source,
            trace=failing_test_traces[test_name],
            focal_source=focal_source,
            focal_module_path=focal_module_path,
        )
        invoked += 1
        # Tag the attempt now (we're committing to the LLM call).
        counts[test_name] = attempts_so_far + 1
        new_attempts_consumed = counts[test_name]
        try:
            response = llm_caller(prompt, REPAIR_RESPONSE_SCHEMA)
        except Exception as exc:  # pragma: no cover - LLM-side errors are diagnostic
            logger.warning("trace repair LLM call failed for %s: %s", test_name, exc)
            attempts.append(
                TraceRepairAttemptResult(
                    test_name=test_name,
                    status="llm_unavailable",
                    reason=f"{type(exc).__name__}: {exc}",
                    error_kind=error_kind,
                    attempts_consumed=new_attempts_consumed,
                )
            )
            skipped_count += 1
            continue
        if not isinstance(response, dict):
            attempts.append(
                TraceRepairAttemptResult(
                    test_name=test_name,
                    status="llm_unavailable",
                    reason="non-dict response",
                    error_kind=error_kind,
                    attempts_consumed=new_attempts_consumed,
                )
            )
            skipped_count += 1
            continue
        tool = str(response.get("tool") or "").strip().lower()
        if tool == "give_up":
            attempts.append(
                TraceRepairAttemptResult(
                    test_name=test_name,
                    status="give_up",
                    reason=str(response.get("reason") or "")[:500],
                    error_kind=error_kind,
                    attempts_consumed=new_attempts_consumed,
                )
            )
            skipped_count += 1
            continue
        if tool != "repair":
            attempts.append(
                TraceRepairAttemptResult(
                    test_name=test_name,
                    status="llm_unavailable",
                    reason=f"unknown tool: {tool!r}",
                    error_kind=error_kind,
                    attempts_consumed=new_attempts_consumed,
                )
            )
            skipped_count += 1
            continue
        new_source = str(response.get("test_source") or "").strip()
        if not new_source:
            attempts.append(
                TraceRepairAttemptResult(
                    test_name=test_name,
                    status="give_up",
                    reason="empty test_source",
                    error_kind=error_kind,
                    attempts_consumed=new_attempts_consumed,
                )
            )
            skipped_count += 1
            continue
        # Validate the new source: must parse, must define a function
        # with the same name as the original. Anything else gets
        # rejected so we never land malformed code in the artifact.
        new_node = _parse_single_test_function(new_source, expected_name=test_name)
        if new_node is None:
            attempts.append(
                TraceRepairAttemptResult(
                    test_name=test_name,
                    status="swap_failed",
                    reason="repair source did not parse to a single test function",
                    new_source=new_source[:1000],
                    error_kind=error_kind,
                    attempts_consumed=new_attempts_consumed,
                )
            )
            skipped_count += 1
            continue
        # Swap into the AST and rebuild the artifact text.
        artifact_text = _swap_test_node(
            original_artifact=artifact_text,
            old_node=original_node,
            new_node=new_node,
        )
        if artifact_text is None:
            attempts.append(
                TraceRepairAttemptResult(
                    test_name=test_name,
                    status="swap_failed",
                    reason="ast.unparse round-trip rejected",
                    new_source=new_source[:1000],
                    error_kind=error_kind,
                    attempts_consumed=new_attempts_consumed,
                )
            )
            skipped_count += 1
            continue
        # Re-parse so subsequent test_node_index lookups see fresh nodes.
        try:
            tree = ast.parse(artifact_text)
            test_node_index = _index_test_nodes(tree)
        except SyntaxError as exc:
            attempts.append(
                TraceRepairAttemptResult(
                    test_name=test_name,
                    status="swap_failed",
                    reason=f"swap broke artifact: {exc.msg}",
                    error_kind=error_kind,
                    attempts_consumed=new_attempts_consumed,
                )
            )
            skipped_count += 1
            continue
        attempts.append(
            TraceRepairAttemptResult(
                test_name=test_name,
                status="repaired",
                reason=str(response.get("reason") or "")[:500],
                new_source=new_source[:1000],
                error_kind=error_kind,
                attempts_consumed=new_attempts_consumed,
            )
        )
        repaired_count += 1

    return TraceRepairResult(
        artifact_text=artifact_text,
        attempts=attempts,
        repaired_count=repaired_count,
        skipped_count=skipped_count,
    )


def _index_test_nodes(tree: ast.AST) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Map ``test_*`` function names → their AST nodes (top-level only)."""

    out: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in getattr(tree, "body", []) or []:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test_"
        ):
            out[node.name] = node
    return out


def _parse_single_test_function(
    source: str,
    *,
    expected_name: str,
) -> Optional[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Parse ``source`` and return the single test function it defines,
    or ``None`` if the source isn't well-formed for a swap."""

    text = (source or "").strip()
    if not text:
        return None
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    funcs = [
        node
        for node in (tree.body or [])
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(funcs) != 1:
        return None
    func = funcs[0]
    if func.name != expected_name:
        return None
    return func


def _swap_test_node(
    *,
    original_artifact: str,
    old_node: ast.FunctionDef | ast.AsyncFunctionDef,
    new_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Optional[str]:
    """Replace ``old_node`` with ``new_node`` in the artifact AST and
    re-render. Returns the new source, or ``None`` if round-trip fails.
    """

    try:
        tree = ast.parse(original_artifact)
    except SyntaxError:
        return None
    swapped = False
    for index, node in enumerate(list(tree.body or [])):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == old_node.name:
            tree.body[index] = new_node
            swapped = True
            break
    if not swapped:
        return None
    ast.fix_missing_locations(tree)
    try:
        rendered = ast.unparse(tree).strip() + "\n"
    except Exception:
        return None
    # Strict W3 gate after the swap: parse + compile. The trace-repair
    # LLM CAN return source that parses on its own but breaks the larger
    # artifact (e.g. duplicate def, async-await scope mismatch). Run the
    # full gate before accepting the swap.
    from .final_acceptance_gate import strict_syntax_check

    syntax_ok, _ = strict_syntax_check(rendered)
    if not syntax_ok:
        return None
    return rendered


def _build_trace_repair_prompt(
    *,
    test_name: str,
    test_source: str,
    trace: str,
    focal_source: str,
    focal_module_path: str,
) -> str:
    """Construct the per-test repair prompt. Concise + fact-dense; the
    LLM's job is to fix one test given exact context."""

    parts = [
        "You are a senior test engineer. One pytest test in a file failed.",
        "Your job: produce a corrected version of JUST that test, or"
        " emit ``give_up`` if the trace shows the test is unfixable",
        "(e.g. tests an API that doesn't exist; depends on missing infra).",
        "",
        "Output JSON matching the schema. Fields:",
        '  - ``tool``: "repair" or "give_up"',
        "  - ``test_source``: the corrected single-function definition"
        " (only when tool=repair). Must be valid Python; must define exactly",
        f"    one function named ``{test_name}``; must not introduce new imports.",
        "  - ``reason``: brief justification (any tool).",
        "",
        f"Test name: {test_name}",
    ]
    if focal_module_path:
        parts.append(f"Focal module path: {focal_module_path}")
    if focal_source:
        # Cap focal source so the prompt stays bounded; the LLM has the
        # focal-module-context elsewhere if it really needs more.
        focal_excerpt = focal_source[:6000]
        parts.extend(
            [
                "Focal source (first 6000 chars):",
                "```python",
                focal_excerpt,
                "```",
            ]
        )
    parts.extend(
        [
            "",
            "Original test source:",
            "```python",
            test_source,
            "```",
            "",
            "Pytest failure trace:",
            "```",
            (trace or "")[:4000],
            "```",
            "",
            "Repair guidance:",
            "  - Prefer correcting the assertion to the actually-observed value",
            "    over deleting the assertion.",
            "  - Prefer using the focal API as documented in the focal source",
            "    over inventing helpers.",
            "  - If the test relies on something that doesn't exist (the trace",
            "    has NameError / AttributeError on a fabricated symbol),",
            "    emit ``give_up``.",
        ]
    )
    return "\n".join(parts)
