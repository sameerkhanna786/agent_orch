"""Determinism + capability sandbox for generated orchestration code (plan §7.3).

The generated orchestrator is *cooperative* model output (not adversarial), so
this is a SOUNDNESS / REPLAYABILITY boundary, not a hard security boundary: it
guarantees the script (a) is deterministic control flow (no clock/RNG/network, so
replay over the frozen script is faithful) and (b) can only reach the curated
``ctx`` API (no filesystem, subprocess, imports, or accept-setter).  Honest
caveat: AST-allowlist + restricted builtins is robust for our own planner's
output but is not a jail against deliberately malicious code.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Optional

from ..errors import FailLoud


# Modules / names a deterministic, capability-restricted orchestrator must not use.
_FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "compile", "open", "__import__", "input", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "memoryview", "breakpoint", "exit", "quit",
    "os", "sys", "subprocess", "socket", "time", "random", "datetime", "threading",
    "multiprocessing", "pathlib", "shutil", "importlib", "ctypes", "pickle", "marshal",
    "builtins", "__builtins__", "help",
})

# Cardinal Contract, enforced structurally (Backbone 2.2): authored code may READ a
# candidate's execution keys but must never ASSIGN them — acceptance is engine-owned and
# execution-grounded. A pattern/orchestrator steers via ctx.* (set_soft/refute/solve),
# never `c.accepted = True`. (set_soft/refute live host-side in select.py, not linted.)
_PROTECTED_ATTRS = frozenset({
    "accepted", "combined_score", "public_signal_score", "verification_score",
    "critic_score", "size",
})

# Safe builtins exposed to generated code (pure, deterministic).
SAFE_BUILTINS: dict[str, Any] = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in (
        # iteration / collections
        "len", "range", "enumerate", "zip", "map", "filter", "sorted", "reversed",
        "sum", "min", "max", "abs", "round", "all", "any", "next", "iter", "slice",
        "list", "dict", "set", "tuple", "frozenset", "bytes", "bytearray",
        # scalars / conversions
        "str", "int", "float", "bool", "complex", "ord", "chr", "bin", "hex", "oct",
        "ascii", "repr", "format", "divmod", "pow", "hash",
        # introspection-lite (dunder ATTRIBUTE access is still blocked by the lint,
        # so these can't be used to escape — they just stop normal idioms NameError-ing)
        "type", "isinstance", "issubclass", "hasattr", "callable", "object",
        "print",  # harmless (stdout); nudge toward ctx.log but don't crash if used
        # exceptions authored code may raise/catch
        "Exception", "ValueError", "KeyError", "IndexError", "TypeError",
        "RuntimeError", "StopIteration", "AttributeError", "NotImplementedError",
        "ZeroDivisionError", "ArithmeticError", "AssertionError",
    )
}


@dataclass
class LintResult:
    ok: bool
    violations: list[str] = field(default_factory=list)


def lint_source(source: str) -> LintResult:
    """Reject imports, forbidden names/modules, dunder access, and non-deterministic
    constructs.  Returns a LintResult (never raises on a lint *finding*)."""
    violations: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return LintResult(False, [f"syntax error: {exc}"])

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            violations.append(f"import is forbidden (line {node.lineno})")
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            violations.append(f"forbidden name '{node.id}' (line {node.lineno})")
        elif isinstance(node, ast.Attribute) and isinstance(node.attr, str) and node.attr.startswith("__"):
            violations.append(f"dunder attribute access '{node.attr}' (line {node.lineno})")
        elif (isinstance(node, ast.Attribute)
              and isinstance(getattr(node, "ctx", None), (ast.Store, ast.Del))
              and node.attr in _PROTECTED_ATTRS):
            # ANY Store/Del binding of an execution-gate attribute, at ANY nesting depth:
            # direct (`c.accepted = True`), tuple/list/starred unpack (`c.accepted, _ = ...`),
            # for-target (`for c.accepted in ...`), or comprehension target. The Cardinal
            # Contract: acceptance is EARNED via execution, never assigned by authored code.
            violations.append(f"cannot bind execution-gate attribute '{node.attr}' (line {node.lineno})")
        elif (isinstance(node, ast.Subscript)
              and isinstance(getattr(node, "ctx", None), (ast.Store, ast.Del))
              and isinstance(getattr(node, "slice", None), ast.Constant)
              and node.slice.value in _PROTECTED_ATTRS):
            # same guard for the dict-style write `c["accepted"] = True` (Candidate.__setitem__
            # does not exist, but block it structurally regardless, at any nesting depth).
            violations.append(f"cannot bind execution-gate key '{node.slice.value}' (line {node.lineno})")
        elif isinstance(node, (ast.With, ast.AsyncWith, ast.AsyncFor, ast.AsyncFunctionDef, ast.Await)):
            violations.append(f"async/with construct forbidden (line {node.lineno})")
        elif isinstance(node, ast.Lambda):
            pass  # lambdas allowed (needed for thunks)

    # must define orchestrate(ctx)
    has_orch = any(
        isinstance(n, ast.FunctionDef) and n.name == "orchestrate" and len(n.args.args) >= 1
        for n in tree.body
    )
    if not has_orch:
        violations.append("must define a top-level function orchestrate(ctx)")
    return LintResult(not violations, violations)


def run_orchestration(source: str, ctx: Any, *, lint: bool = True):
    """Execute a frozen orchestration script in the restricted namespace and call
    ``orchestrate(ctx)``.  Raises FailLoud if the source fails the lint (the
    caller is expected to fall open to verified best-of-N)."""
    if lint:
        result = lint_source(source)
        if not result.ok:
            raise FailLoud("generated orchestration failed lint: " + "; ".join(result.violations))
    glb: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
    try:
        code = compile(source, "<generated-orchestrator>", "exec")
    except SyntaxError as exc:
        raise FailLoud(f"generated orchestration does not compile: {exc}")
    exec(code, glb, glb)  # noqa: S102 - restricted namespace; cooperative model output
    fn = glb.get("orchestrate")
    if not callable(fn):
        raise FailLoud("generated orchestration did not define orchestrate(ctx)")
    return fn(ctx)


def extract_code(text: str) -> str:
    """Pull a python code block out of a model's markdown reply (best-effort)."""
    if "```" not in text:
        return text.strip()
    parts = text.split("```")
    # find the block whose fence is python/py or that contains def orchestrate
    for i in range(1, len(parts), 2):
        block = parts[i]
        first_nl = block.find("\n")
        lang = block[:first_nl].strip().lower() if first_nl != -1 else ""
        body = block[first_nl + 1:] if first_nl != -1 else block
        if lang in ("python", "py", "") and "orchestrate" in body:
            return body.strip()
    # fallback: first fenced block
    return parts[1].split("\n", 1)[-1].strip() if len(parts) > 1 else text.strip()
