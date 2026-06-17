"""Execution-grounded oracle repair for failing generated tests.

Walks a generated test artifact, finds ``assert <call> == <literal>`` style
assertions, executes each call in the project workdir to capture the actual
return value, and rewrites the assertion's literal with the captured value.
This converts the dominant May-5 failure class (45/118 unfiltered fails were
``AssertionError`` from speculative oracles) into passing tests whose
oracles match observed behavior.

The repair is deterministic: no LLM call. It is also conservative: we only
rewrite when

  - the call site has no free variables we cannot resolve,
  - the call returns a value that is JSON/repr serializable,
  - the captured value is the same on two consecutive runs (deterministic).

Anything else falls through unchanged so the next repair strategy can act.

The assertion shape is chosen by ``mutation_targeting.choose_assertion_shape``
so floats are compared via ``pytest.approx`` and ndarrays via ``assert_allclose``,
preserving mutation-killing power while avoiding floating-point brittleness.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .mutation_targeting import choose_assertion_shape


@dataclass(frozen=True)
class OracleRepairOutcome:
    status: str
    artifact_text: str
    rewritten_count: int = 0
    skipped_count: int = 0
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.status == "rewritten" and self.rewritten_count > 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def repair_assertions_with_captured_oracles(
    source: str,
    *,
    workdir: Path,
    failing_test_names: Iterable[str] = (),
    timeout: float = 10.0,
    python_executable: str | None = None,
    env: dict[str, str] | None = None,
    docker_runner: Optional[Any] = None,
    style: Any | None = None,
) -> OracleRepairOutcome:
    """Replace literal-equality assertions in failing tests with captured values.

    When ``docker_runner`` is supplied, all candidate captures for this
    artifact are batched into a SINGLE driver and executed inside the
    project's docker container (via ``docker_runner(driver_source)``). This
    is the path used in production for projects whose deps aren't in the
    apex venv (Django/sympy/Flask). The local subprocess fallback (one call
    per assertion) stays as the no-docker default.
    """

    failing = {
        str(name).rsplit("::", 1)[-1].split("[", 1)[0]
        for name in (failing_test_names or [])
        if str(name).startswith("test_") or "::test_" in str(name)
    }
    try:
        tree = ast.parse(source or "")
    except SyntaxError as exc:
        return OracleRepairOutcome(
            status="parse_error",
            artifact_text=source,
            diagnostics=[{"error": f"SyntaxError: {exc}"}],
        )

    targets = _candidate_targets(tree, failing_function_names=failing)
    if not targets:
        return OracleRepairOutcome(
            status="no_candidates",
            artifact_text=source,
        )

    captures = _capture_targets_in_batch(
        targets,
        workdir=workdir,
        timeout=timeout,
        python_executable=python_executable,
        env=env,
        docker_runner=docker_runner,
    )

    rewritten = 0
    skipped = 0
    diagnostics: list[dict[str, Any]] = []
    needs_pytest_approx = False
    needs_numpy_testing = False
    for target, capture in zip(targets, captures):
        diag = {
            "function": target.function_name,
            "call": target.call_source,
            "kind": capture.get("kind"),
        }
        if capture.get("kind") != "value":
            skipped += 1
            diag["reason"] = capture.get("error") or capture.get("kind")
            diagnostics.append(diag)
            continue
        captured_repr = str(capture.get("repr_text") or "")
        captured_value = capture.get("value")
        captured_type = str(capture.get("result_type") or "")
        if not captured_repr or captured_repr == target.original_literal_repr:
            skipped += 1
            diag["reason"] = "captured value matches original literal"
            diagnostics.append(diag)
            continue
        # W9 mutation-targeting: pick the strongest safe assertion shape.
        shape = choose_assertion_shape(captured_value, result_type=captured_type)
        if shape.name == "pytest_approx" and isinstance(captured_value, float):
            if _style_allows_helper(style, "pytest.approx"):
                target.replace_with_pytest_approx(captured_repr)
                needs_pytest_approx = True
            else:
                target.replace_with_math_isclose(captured_repr)
        elif shape.name == "numpy_assert_allclose" and captured_value is not None:
            if _style_allows_helper(style, "numpy.testing.assert_allclose"):
                target.replace_with_numpy_assert_allclose(captured_repr)
                needs_numpy_testing = True
            else:
                target.replace_with_tolist_equality(captured_repr)
        else:
            target.replace_literal_with(captured_repr)
        rewritten += 1
        diag["captured_repr"] = captured_repr
        diag["assertion_shape"] = shape.name
        diagnostics.append(diag)

    if rewritten == 0:
        return OracleRepairOutcome(
            status="no_changes",
            artifact_text=source,
            skipped_count=skipped,
            diagnostics=diagnostics,
        )

    _ensure_required_imports(
        tree,
        needs_pytest=needs_pytest_approx,
        needs_numpy=needs_numpy_testing,
    )
    new_text = ast.unparse(tree).strip() + "\n"
    policy_violations = _policy_violations(new_text, style)
    if policy_violations:
        diagnostics.append(
            {
                "stage": "oracle_repair_post_policy",
                "status": "policy_violation",
                "forbidden_imports": policy_violations,
            }
        )
        return OracleRepairOutcome(
            status="no_changes",
            artifact_text=source,
            skipped_count=skipped + rewritten,
            diagnostics=diagnostics,
        )
    # Strict W3 gate after every artifact mutation. If the rewritten
    # artifact failed to byte-compile we fall back to the original
    # source rather than ship something that will SyntaxError later.
    from .final_acceptance_gate import strict_syntax_check

    syntax_ok, syntax_err = strict_syntax_check(new_text)
    if not syntax_ok:
        diagnostics.append(
            {
                "stage": "oracle_repair_post_unparse",
                "status": "syntax_error",
                "error": syntax_err or "strict_syntax_check failed",
            }
        )
        return OracleRepairOutcome(
            status="no_changes",
            artifact_text=source,
            skipped_count=skipped + rewritten,
            diagnostics=diagnostics,
        )
    return OracleRepairOutcome(
        status="rewritten",
        artifact_text=new_text,
        rewritten_count=rewritten,
        skipped_count=skipped,
        diagnostics=diagnostics,
    )


def _ensure_required_imports(
    tree: ast.Module,
    *,
    needs_pytest: bool,
    needs_numpy: bool,
) -> None:
    """Insert ``import pytest`` / ``import numpy`` if rewrites added refs."""

    if not (needs_pytest or needs_numpy):
        return
    existing: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                existing.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(stmt, ast.ImportFrom):
            for alias in stmt.names:
                existing.add(alias.asname or alias.name)
    insertions: list[ast.stmt] = []
    if needs_pytest and "pytest" not in existing:
        insertions.append(ast.Import(names=[ast.alias(name="pytest")]))
    if needs_numpy and "numpy" not in existing and "np" not in existing:
        insertions.append(ast.Import(names=[ast.alias(name="numpy")]))
    if insertions:
        # Insert after any existing future imports / docstring; for simplicity
        # we put them at position 0 and rely on Python's import flexibility.
        tree.body[0:0] = insertions
        ast.fix_missing_locations(tree)


def _style_allows_helper(style: Any | None, helper: str) -> bool:
    if style is None:
        # Preserve legacy behavior for call sites that have not yet plumbed
        # runner profiles through repair.
        return helper.startswith(("pytest.", "numpy."))
    runner = str(getattr(style, "runner", "") or "").lower()
    if not runner:
        return helper.startswith(("pytest.", "numpy."))
    try:
        from .test_style import runner_profile_for_style

        return runner_profile_for_style(style).allows_helper(helper)
    except Exception:
        if helper.startswith(("pytest.", "numpy.")):
            return runner in {"", "pytest"}
        return True


def _policy_violations(source: str, style: Any | None) -> list[str]:
    if style is None:
        return []
    try:
        from .test_style import imports_forbidden_by_style

        return imports_forbidden_by_style(source, style)
    except Exception:
        return []


@dataclass
class _RewriteTarget:
    function_name: str
    call_source: str
    preamble_source: str
    original_literal_repr: str
    _assert_node: ast.Assert
    _comparator_index: int

    def replace_literal_with(self, captured_repr: str) -> None:
        try:
            new_node = ast.parse(captured_repr, mode="eval").body
        except SyntaxError:
            new_node = ast.Constant(value=captured_repr)
        compare = self._assert_node.test
        if isinstance(compare, ast.Compare):
            compare.comparators[self._comparator_index] = new_node

    def replace_with_pytest_approx(self, captured_repr: str, *, rel: float = 1e-6) -> None:
        """Wrap the captured value in ``pytest.approx(..., rel=...)``."""
        try:
            inner = ast.parse(captured_repr, mode="eval").body
        except SyntaxError:
            inner = ast.Constant(value=captured_repr)
        approx_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="pytest", ctx=ast.Load()),
                attr="approx",
                ctx=ast.Load(),
            ),
            args=[inner],
            keywords=[ast.keyword(arg="rel", value=ast.Constant(value=rel))],
        )
        compare = self._assert_node.test
        if isinstance(compare, ast.Compare):
            compare.comparators[self._comparator_index] = approx_call

    def replace_with_numpy_assert_allclose(self, captured_repr: str) -> None:
        """Replace the assert with ``numpy.testing.assert_allclose(call, value)``."""
        try:
            captured = ast.parse(captured_repr, mode="eval").body
        except SyntaxError:
            captured = ast.Constant(value=captured_repr)
        try:
            call_expr = ast.parse(self.call_source, mode="eval").body
        except SyntaxError:
            return
        new_call = ast.Call(
            func=ast.Attribute(
                value=ast.Attribute(
                    value=ast.Name(id="numpy", ctx=ast.Load()),
                    attr="testing",
                    ctx=ast.Load(),
                ),
                attr="assert_allclose",
                ctx=ast.Load(),
            ),
            args=[call_expr, captured],
            keywords=[],
        )
        # Replace the Assert node's body with an Expr wrapping the call.
        self._assert_node.test = ast.Constant(value=True)  # neutralize old assert
        # We can't easily replace the Assert in the parent body from here;
        # the caller's ast.unparse will still emit a no-op `assert True` plus
        # the original failing assertion. To avoid that, we mutate the assert
        # itself into a Compare equivalent by wrapping the comparison so the
        # whole assertion becomes `assert numpy.testing.assert_allclose(...)
        # is None` which is True when the comparison passes and raises
        # otherwise.
        self._assert_node.test = ast.Compare(
            left=new_call,
            ops=[ast.Is()],
            comparators=[ast.Constant(value=None)],
        )

    def replace_with_math_isclose(self, captured_repr: str, *, rel: float = 1e-6) -> None:
        try:
            captured = ast.parse(captured_repr, mode="eval").body
        except SyntaxError:
            captured = ast.Constant(value=captured_repr)
        try:
            call_expr = ast.parse(self.call_source, mode="eval").body
        except SyntaxError:
            return
        self._assert_node.test = ast.Call(
            func=ast.Attribute(
                value=ast.Call(
                    func=ast.Name(id="__import__", ctx=ast.Load()),
                    args=[ast.Constant(value="math")],
                    keywords=[],
                ),
                attr="isclose",
                ctx=ast.Load(),
            ),
            args=[call_expr, captured],
            keywords=[ast.keyword(arg="rel_tol", value=ast.Constant(value=rel))],
        )

    def replace_with_tolist_equality(self, captured_repr: str) -> None:
        try:
            captured = ast.parse(captured_repr, mode="eval").body
            call_expr = ast.parse(self.call_source, mode="eval").body
        except SyntaxError:
            return
        tolist_call = ast.Call(
            func=ast.Attribute(value=call_expr, attr="tolist", ctx=ast.Load()),
            args=[],
            keywords=[],
        )
        self._assert_node.test = ast.Compare(
            left=tolist_call,
            ops=[ast.Eq()],
            comparators=[captured],
        )


def _candidate_targets(
    tree: ast.Module,
    *,
    failing_function_names: set[str],
) -> list[_RewriteTarget]:
    targets: list[_RewriteTarget] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test_"):
            continue
        if failing_function_names and node.name not in failing_function_names:
            continue
        preamble = _module_level_preamble(tree, exclude=node)
        for stmt_index, stmt in enumerate(node.body):
            if not isinstance(stmt, ast.Assert):
                continue
            test = stmt.test
            if not isinstance(test, ast.Compare) or len(test.ops) != 1:
                continue
            if not isinstance(test.ops[0], ast.Eq) or len(test.comparators) != 1:
                continue
            left, right = test.left, test.comparators[0]
            call_node = None
            literal_node = None
            comp_index = 0
            if isinstance(left, ast.Call) and _is_literal(right):
                call_node, literal_node = left, right
                comp_index = 0
                # The comparator is right; we want to replace right.
                # ast.Compare stores comparators on the right side;
                # left is the leftmost expression, comparators is a list.
                # When the call is on the left and the literal is on the
                # right, we replace comparators[0]. Same index value.
            elif isinstance(right, ast.Call) and _is_literal(left):
                # We must put the call on left and the literal on right,
                # then rewrite the comparator. Easier: swap the roles.
                test.left, test.comparators[0] = right, left
                call_node, literal_node = right, left
                comp_index = 0
            else:
                continue
            try:
                call_source = ast.unparse(call_node)
                literal_repr = ast.unparse(literal_node)
            except Exception:
                continue
            preamble_source = "\n".join(preamble) + "\n" if preamble else ""
            targets.append(
                _RewriteTarget(
                    function_name=node.name,
                    call_source=call_source,
                    preamble_source=preamble_source,
                    original_literal_repr=literal_repr,
                    _assert_node=stmt,
                    _comparator_index=comp_index,
                )
            )
    return targets


def _is_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_literal(node.operand)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_is_literal(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            (key is None or _is_literal(key)) and _is_literal(value)
            for key, value in zip(node.keys, node.values)
        )
    return False


def _module_level_preamble(
    tree: ast.Module,
    *,
    exclude: ast.AST,
) -> list[str]:
    """Return module-level imports and assignments, excluding the test function."""

    chunks: list[str] = []
    for stmt in tree.body:
        if stmt is exclude:
            continue
        if isinstance(
            stmt,
            (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.ClassDef),
        ):
            try:
                chunks.append(ast.unparse(stmt))
            except Exception:
                continue
    return chunks


def _capture_targets_in_batch(
    targets: list["_RewriteTarget"],
    *,
    workdir: Path,
    timeout: float,
    python_executable: str | None,
    env: dict[str, str] | None,
    docker_runner: Optional[Any],
) -> list[dict[str, Any]]:
    """Capture every target's call. With docker_runner, batch into one call."""

    if docker_runner is None:
        return [
            _capture_call_text(
                t.call_source,
                t.preamble_source,
                workdir=workdir,
                timeout=timeout,
                python_executable=python_executable,
                env=env,
            )
            for t in targets
        ]
    # Docker path: batch all calls into one container invocation.
    driver = _build_batched_driver(targets)
    result = docker_runner(driver)
    stdout = getattr(result, "stdout", "") or ""
    if not stdout.strip():
        diag = {"kind": "harness_error", "error": "docker stdout empty"}
        return [diag for _ in targets]
    try:
        payload = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError as exc:
        diag = {"kind": "harness_error", "error": f"json decode: {exc}"}
        return [diag for _ in targets]
    captures = list(payload.get("captures") or [])
    # Pad/truncate to match target count
    while len(captures) < len(targets):
        captures.append({"kind": "harness_error", "error": "missing capture row"})
    return captures[: len(targets)]


def _build_batched_driver(targets: list["_RewriteTarget"]) -> str:
    """Render a single Python driver that captures every target call and
    emits one JSON line on stdout."""

    # Use the first target's preamble (they share module scope by construction).
    preamble = targets[0].preamble_source if targets else ""
    parts: list[str] = [
        "import json",
        preamble.rstrip(),
        "",
        "def __apex_encode(__value):",
        "    __payload = {'kind': 'value', 'repr_text': repr(__value),",
        "                 'result_type': type(__value).__name__,",
        "                 'result_module': type(__value).__module__}",
        "    if isinstance(__value, (int, float, str, bool, type(None))):",
        "        __payload['value'] = __value",
        "    elif isinstance(__value, (list, tuple)):",
        "        try:",
        "            json.dumps(list(__value))",
        "            __payload['value'] = list(__value)",
        "        except (TypeError, ValueError):",
        "            pass",
        "    elif isinstance(__value, dict):",
        "        try:",
        "            json.dumps(__value)",
        "            __payload['value'] = __value",
        "        except (TypeError, ValueError):",
        "            pass",
        "    elif type(__value).__module__.startswith('numpy'):",
        "        __payload['result_type'] = 'ndarray'",
        "    return __payload",
        "",
        "def __apex_invoke(__call_text):",
        "    try:",
        "        __value = eval(compile(__call_text, '<apex>', 'eval'), globals())",
        "    except BaseException as exc:",
        "        return {'kind': 'exception', 'exc_type': type(exc).__name__, 'exc_message': str(exc)}",
        "    try:",
        "        return __apex_encode(__value)",
        "    except BaseException as exc:",
        "        return {'kind': 'unrepresentable', 'error': str(exc)}",
        "",
        "__apex_calls = " + json.dumps([t.call_source for t in targets]),
        "__apex_first = [__apex_invoke(__c) for __c in __apex_calls]",
        "__apex_second = [__apex_invoke(__c) for __c in __apex_calls]",
        "",
        "__apex_captures = []",
        "for __a, __b in zip(__apex_first, __apex_second):",
        "    if __a == __b and __a.get('kind') == 'value':",
        "        __apex_captures.append(__a)",
        "    elif __a.get('kind') == 'value' and __b.get('kind') == 'value':",
        "        __apex_captures.append({'kind': 'non_deterministic'})",
        "    else:",
        "        __apex_captures.append(__a)",
        "",
        "print(json.dumps({'captures': __apex_captures}))",
    ]
    return "\n".join(parts) + "\n"


def _capture_call_text(
    call_source: str,
    preamble_source: str,
    *,
    workdir: Path,
    timeout: float,
    python_executable: str | None,
    env: dict[str, str] | None,
) -> dict[str, Any]:
    """Run ``preamble; print(repr(call))`` twice; require deterministic output."""

    executable = python_executable or sys.executable
    driver = (
        "import json, sys\n"
        f"{preamble_source}"
        "def __apex_encode(__value):\n"
        "    __payload = {'kind': 'value', 'repr_text': repr(__value),\n"
        "                 'result_type': type(__value).__name__,\n"
        "                 'result_module': type(__value).__module__}\n"
        "    if isinstance(__value, (int, float, str, bool, type(None))):\n"
        "        __payload['value'] = __value\n"
        "    elif isinstance(__value, (list, tuple)):\n"
        "        try:\n"
        "            json.dumps(list(__value))\n"
        "            __payload['value'] = list(__value)\n"
        "        except (TypeError, ValueError):\n"
        "            pass\n"
        "    elif isinstance(__value, dict):\n"
        "        try:\n"
        "            json.dumps(__value)\n"
        "            __payload['value'] = __value\n"
        "        except (TypeError, ValueError):\n"
        "            pass\n"
        "    elif type(__value).__module__.startswith('numpy'):\n"
        "        __payload['result_type'] = 'ndarray'\n"
        "    return __payload\n"
        "def __apex_invoke():\n"
        "    try:\n"
        f"        __value = ({call_source})\n"
        "    except BaseException as exc:\n"
        "        return {'kind': 'exception', 'exc_type': type(exc).__name__, 'exc_message': str(exc)}\n"
        "    try:\n"
        "        return __apex_encode(__value)\n"
        "    except BaseException as exc:\n"
        "        return {'kind': 'unrepresentable', 'error': str(exc)}\n"
        "first = __apex_invoke()\n"
        "second = __apex_invoke()\n"
        "print(json.dumps({'first': first, 'second': second}))\n"
    )
    with tempfile.TemporaryDirectory(prefix="apex_oracle_repair_") as tmp:
        driver_path = Path(tmp) / "driver.py"
        driver_path.write_text(driver, encoding="utf-8")
        run_env = dict(os.environ)
        if env:
            run_env.update({str(k): str(v) for k, v in env.items()})
        try:
            completed = subprocess.run(
                [executable, str(driver_path)],
                cwd=str(workdir),
                env=run_env,
                capture_output=True,
                text=True,
                timeout=float(timeout),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"kind": "harness_error", "error": "capture timed out"}
        except OSError as exc:
            return {"kind": "harness_error", "error": f"{type(exc).__name__}: {exc}"}
    if completed.returncode != 0:
        return {
            "kind": "harness_error",
            "error": (completed.stderr or completed.stdout or "capture failed")[-2000:],
        }
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"kind": "harness_error", "error": "non-json driver output"}
    first = dict(payload.get("first") or {})
    second = dict(payload.get("second") or {})
    if first.get("kind") != "value":
        return first
    if first.get("repr_text") != second.get("repr_text"):
        return {"kind": "non_deterministic"}
    return first


def replace_failing_assertions_with_captured_values(
    artifacts: list[dict[str, Any]],
    *,
    workdir: Path,
    failing_test_names: Iterable[str] = (),
    timeout: float = 10.0,
    python_executable: str | None = None,
    env: dict[str, str] | None = None,
    docker_runner: Optional[Any] = None,
    style: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply oracle repair to every artifact in the list."""

    repaired: list[dict[str, Any]] = []
    per_artifact: list[dict[str, Any]] = []
    total_rewrites = 0
    for original in artifacts:
        if not isinstance(original, dict):
            repaired.append(original)
            continue
        text = str(original.get("content") or "")
        outcome = repair_assertions_with_captured_oracles(
            text,
            workdir=workdir,
            failing_test_names=failing_test_names,
            timeout=timeout,
            python_executable=python_executable,
            env=env,
            docker_runner=docker_runner,
            style=style,
        )
        per_artifact.append(
            {
                "path": original.get("path"),
                "status": outcome.status,
                "rewritten_count": outcome.rewritten_count,
                "skipped_count": outcome.skipped_count,
                "diagnostics": outcome.diagnostics,
            }
        )
        if outcome.changed:
            updated = dict(original)
            updated["content"] = outcome.artifact_text
            repaired.append(updated)
            total_rewrites += outcome.rewritten_count
        else:
            repaired.append(dict(original))
    return (
        repaired,
        {
            "status": "ok",
            "rewritten_count": total_rewrites,
            "per_artifact": per_artifact,
        },
    )
