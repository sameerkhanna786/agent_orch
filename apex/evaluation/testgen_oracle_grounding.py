"""Benchmark-agnostic oracle grounding for test generation."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .oracle_capture import (
    CallSpec,
    CaptureResult,
    capture_oracle,
    capture_oracle_with_runner,
    synthesize_assertion,
)


@dataclass(frozen=True)
class OracleGroundingReport:
    status: str
    call_specs: list[CallSpec] = field(default_factory=list)
    captures: list[CaptureResult] = field(default_factory=list)
    synthesized_assertions: list[str] = field(default_factory=list)
    oracle_fingerprints: list[dict[str, Any]] = field(default_factory=list)
    unsupported_call_specs: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""

    @property
    def grounded_count(self) -> int:
        return sum(1 for capture in self.captures if capture.kind in {"value", "repr", "exception"})

    @property
    def score(self) -> float:
        if not self.call_specs:
            return 0.0
        return min(1.0, self.grounded_count / max(len(self.call_specs), 1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "call_specs": [spec.to_dict() for spec in self.call_specs],
            "captures": [capture.to_dict() for capture in self.captures],
            "synthesized_assertions": list(self.synthesized_assertions),
            "oracle_fingerprints": list(self.oracle_fingerprints),
            "__apex_oracle_fingerprints__": list(self.oracle_fingerprints),
            "unsupported_call_specs": list(self.unsupported_call_specs),
            "grounded_count": self.grounded_count,
            "score": round(self.score, 4),
            "error": self.error,
        }


def ground_oracles_for_testgen(
    *,
    focal_source: str,
    focal_path: str,
    existing_test_source: str = "",
    workdir: Path | None = None,
    style: Any = None,
    language: str = "python",
    max_specs: int = 5,
    timeout_seconds: float = 5.0,
    python_executable: str | None = None,
    target_runner: Any | None = None,
    expected_state: str = "post_fix",
) -> OracleGroundingReport:
    """Derive executable call specs and capture their observed behavior.

    This does not assume a benchmark. It works from the common testgen task
    fields: focal source/path, optional existing tests, and a workdir.

    Phase 4A item 4.4 — ``expected_state`` is forwarded to
    :func:`oracle_capture.capture_oracle` so the workdir/oracle-state
    assertion fires before any call spec is invoked. Defaults to
    ``"post_fix"`` because the standard TestGenEval workdir contract
    is the gold-fixed repo. Callers running against a broken sandbox
    must pass ``"pre_fix"`` explicitly.
    """

    if (language or "python").lower() not in {"python", "py", "python3"}:
        return OracleGroundingReport(status=f"unsupported_language:{language or 'unknown'}")
    call_specs, unsupported = derive_call_specs(
        focal_source=focal_source,
        focal_path=focal_path,
        existing_test_source=existing_test_source,
        max_specs=max_specs,
    )
    if not call_specs:
        return OracleGroundingReport(status="no_call_specs", unsupported_call_specs=unsupported)
    if workdir is None:
        return OracleGroundingReport(
            status="no_workdir",
            call_specs=call_specs,
            unsupported_call_specs=unsupported,
        )
    captures: list[CaptureResult] = []
    assertions: list[str] = []
    fingerprints: list[dict[str, Any]] = []
    for spec in call_specs:
        if target_runner is not None:
            capture = capture_oracle_with_runner(spec, runner=target_runner)
        else:
            capture = capture_oracle(
                spec,
                workdir=Path(workdir),
                expected_state=expected_state,  # type: ignore[arg-type]
                timeout=timeout_seconds,
                python_executable=python_executable,
            )
        captures.append(capture)
        fingerprint = _fingerprint_capture(capture)
        if fingerprint:
            fingerprints.append(fingerprint)
        assertion = synthesize_assertion(capture, style=style or object())
        if assertion.strip():
            assertions.append(assertion.strip())
    status = (
        "ok"
        if any(c.kind in {"value", "repr", "exception"} for c in captures)
        else "no_grounded_captures"
    )
    return OracleGroundingReport(
        status=status,
        call_specs=call_specs,
        captures=captures,
        synthesized_assertions=assertions,
        oracle_fingerprints=fingerprints,
        unsupported_call_specs=unsupported,
    )


def derive_call_specs(
    *,
    focal_source: str,
    focal_path: str,
    existing_test_source: str = "",
    max_specs: int = 5,
) -> tuple[list[CallSpec], list[dict[str, Any]]]:
    module = _module_name_from_path(focal_path)
    public_names = _public_callable_names(focal_source)
    specs: list[CallSpec] = []
    unsupported: list[dict[str, Any]] = []
    for spec in _literal_call_specs_from_existing_tests(
        existing_test_source=existing_test_source,
        module=module,
        public_names=set(public_names),
    ):
        _append_unique_spec(specs, spec, max_specs=max_specs)
    try:
        tree = ast.parse(focal_source or "")
    except SyntaxError as exc:
        unsupported.append({"reason": "focal_parse_error", "error": str(exc)})
        return specs, unsupported
    for node in tree.body:
        if len(specs) >= max_specs:
            break
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            args, kwargs, reason = _args_from_defaults(node)
            if reason:
                unsupported.append({"qualname": node.name, "reason": reason})
                continue
            _append_unique_spec(
                specs,
                CallSpec(module=module, qualname=node.name, args=args, kwargs=kwargs),
                max_specs=max_specs,
            )
        elif isinstance(node, ast.ClassDef):
            # P1 step 6: walk class methods, synthesize a receiver, and
            # emit method-bound call specs the driver can execute.
            if node.name.startswith("_"):
                continue
            recv_args, recv_kwargs, recv_reason = _synthesize_receiver(node)
            if recv_reason:
                unsupported.append({"qualname": node.name, "reason": recv_reason})
                continue
            for method_node in _iter_class_method_nodes(node):
                if len(specs) >= max_specs:
                    break
                if method_node.name.startswith("_"):
                    continue
                method_args, method_kwargs, method_reason = _args_from_defaults_for_method(
                    method_node
                )
                if method_reason:
                    unsupported.append(
                        {
                            "qualname": f"{node.name}.{method_node.name}",
                            "reason": method_reason,
                        }
                    )
                    continue
                _append_unique_spec(
                    specs,
                    CallSpec(
                        module=module,
                        qualname=f"{node.name}.{method_node.name}",
                        args=method_args,
                        kwargs=method_kwargs,
                        receiver_qualname=node.name,
                        receiver_args=recv_args,
                        receiver_kwargs=recv_kwargs,
                    ),
                    max_specs=max_specs,
                )
    return specs, unsupported


def render_oracle_grounding_block(report: OracleGroundingReport | dict[str, Any]) -> str:
    payload = report.to_dict() if isinstance(report, OracleGroundingReport) else dict(report or {})
    captures = [item for item in list(payload.get("captures") or []) if isinstance(item, dict)]
    assertions = [
        str(item).strip()
        for item in list(payload.get("synthesized_assertions") or [])
        if str(item).strip()
    ]
    if not captures and not assertions:
        return ""
    lines = [
        "## Execution-grounded oracle observations",
        "Use these observed behaviors as oracle evidence. Do not invent expected return values beyond this evidence.",
    ]
    for capture in captures[:8]:
        call = dict(capture.get("call_spec") or {}).get("call_source") or _render_spec_call(
            dict(capture.get("call_spec") or {})
        )
        kind = str(capture.get("kind") or "")
        if kind == "value":
            lines.append(
                f"- `{call}` returned `{capture.get('repr_text') or repr(capture.get('value'))}`"
            )
        elif kind == "repr":
            lines.append(f"- `{call}` has repr `{capture.get('repr_text')}`")
        elif kind == "exception":
            lines.append(
                f"- `{call}` raised `{capture.get('exc_type')}`"
                + (f": {capture.get('exc_message')}" if capture.get("exc_message") else "")
            )
        elif kind == "non_deterministic":
            lines.append(
                f"- `{call}` was non-deterministic; use property assertions, not exact values."
            )
    if assertions:
        lines.extend(["", "Seed assertion shapes:", "```python"])
        lines.extend(assertions[:6])
        lines.append("```")
    return "\n".join(lines)


def _literal_call_specs_from_existing_tests(
    *,
    existing_test_source: str,
    module: str,
    public_names: set[str],
) -> list[CallSpec]:
    if not existing_test_source.strip() or not public_names:
        return []
    try:
        tree = ast.parse(existing_test_source)
    except SyntaxError:
        return []
    specs: list[CallSpec] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        qualname = _call_name(node.func)
        leaf = qualname.rsplit(".", 1)[-1]
        if leaf not in public_names:
            continue
        parsed = _literal_args(node)
        if parsed is None:
            continue
        args, kwargs = parsed
        specs.append(
            CallSpec(
                module=module,
                qualname=leaf,
                args=args,
                kwargs=kwargs,
                call_source=f"{leaf}({', '.join([repr(a) for a in args])})" if not kwargs else "",
            )
        )
    return specs


def _args_from_defaults(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[list[Any], dict[str, Any], str]:
    positional = list(node.args.posonlyargs or []) + list(node.args.args or [])
    if positional and positional[0].arg in {"self", "cls"}:
        return [], {}, "method_requires_receiver"
    required_count = len(positional) - len(node.args.defaults or [])
    if required_count > 0:
        return [], {}, "required_args_without_literals"
    args: list[Any] = []
    if node.args.defaults:
        for default in node.args.defaults:
            try:
                args.append(ast.literal_eval(default))
            except (ValueError, SyntaxError):
                return [], {}, "non_literal_default"
    kwargs: dict[str, Any] = {}
    for kwarg, default in zip(node.args.kwonlyargs or [], node.args.kw_defaults or []):
        if default is None:
            return [], {}, "required_keyword_only_arg"
        try:
            kwargs[kwarg.arg] = ast.literal_eval(default)
        except (ValueError, SyntaxError):
            return [], {}, "non_literal_default"
    return args, kwargs, ""


def _args_from_defaults_for_method(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[list[Any], dict[str, Any], str]:
    """Like ``_args_from_defaults`` but skips the leading ``self``/``cls`` arg.

    Returns the positional + keyword defaults for the method body itself
    so the in-process driver can call ``instance.method(*args, **kwargs)``.
    Mirrors the same conservative literal-only stance — non-literal
    defaults or missing required args produce a no-op result with a
    machine-readable reason for telemetry.
    """

    positional = list(node.args.posonlyargs or []) + list(node.args.args or [])
    if not positional or positional[0].arg not in {"self", "cls"}:
        # Static methods or oddly-shaped defs: route through the regular path.
        return _args_from_defaults(node)
    bound_positional = positional[1:]  # drop self/cls
    required_count = len(bound_positional) - len(node.args.defaults or [])
    if required_count > 0:
        return [], {}, "required_args_without_literals"
    args: list[Any] = []
    if node.args.defaults:
        for default in node.args.defaults:
            try:
                args.append(ast.literal_eval(default))
            except (ValueError, SyntaxError):
                return [], {}, "non_literal_default"
    kwargs: dict[str, Any] = {}
    for kwarg, default in zip(node.args.kwonlyargs or [], node.args.kw_defaults or []):
        if default is None:
            return [], {}, "required_keyword_only_arg"
        try:
            kwargs[kwarg.arg] = ast.literal_eval(default)
        except (ValueError, SyntaxError):
            return [], {}, "non_literal_default"
    return args, kwargs, ""


def _iter_class_method_nodes(
    cls_node: ast.ClassDef,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Yield function/async-function defs directly inside *cls_node*.

    Skips dunders and private methods at the call site (the caller
    filters), but ``__init__`` itself is excluded here because the
    receiver-synthesis path uses it separately and we don't want it as a
    standalone capture target.
    """

    out: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for child in cls_node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if child.name == "__init__":
                continue
            out.append(child)
    return out


def _synthesize_receiver(
    cls_node: ast.ClassDef,
) -> tuple[list[Any], dict[str, Any], str]:
    """Try to build args for ``ClassName(*args, **kwargs)`` from the focal AST.

    Strategy (conservative, no imports of the focal module):

    1. If there's no ``__init__`` (or only the implicit ``object.__init__``):
       receiver is ``cls()``.
    2. If ``__init__`` exists with literal defaults for all non-``self``
       args: use those defaults.
    3. If ``__init__`` requires args we can't deduce from literals: return
       ``(_, _, "receiver_requires_args")`` so the caller records it as
       an unsupported class rather than synthesizing a wrong receiver.

    Bias is hard toward false-negatives. We never guess ``0``, ``""``,
    ``None`` for an annotated-but-required arg — the call would silently
    succeed with the wrong shape and the captured oracle would be a lie.
    Future iterations can layer in annotation-driven defaults; for now
    we keep the contract tight.
    """

    init_node: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for child in cls_node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == "__init__":
            init_node = child
            break

    # Skip classes that explicitly inherit from things we can't replicate
    # (e.g. ``Protocol``, ``ABC``, ``Generic[T]``). The receiver build
    # would either fail at runtime or produce a non-instantiable proxy.
    for base in cls_node.bases:
        base_name = ""
        if isinstance(base, ast.Name):
            base_name = base.id
        elif isinstance(base, ast.Attribute):
            base_name = base.attr
        elif isinstance(base, ast.Subscript) and isinstance(base.value, ast.Name):
            base_name = base.value.id
        if base_name in {"Protocol", "ABC", "ABCMeta"}:
            return [], {}, "receiver_abstract_or_protocol"

    if init_node is None:
        return [], {}, ""

    args, kwargs, reason = _args_from_defaults_for_method(init_node)
    if reason == "required_args_without_literals":
        return [], {}, "receiver_requires_args"
    if reason == "non_literal_default":
        return [], {}, "receiver_non_literal_default"
    if reason == "required_keyword_only_arg":
        return [], {}, "receiver_required_kwonly"
    return args, kwargs, reason


def _literal_args(node: ast.Call) -> tuple[list[Any], dict[str, Any]] | None:
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    try:
        for arg in node.args:
            args.append(ast.literal_eval(arg))
        for keyword in node.keywords:
            if not keyword.arg:
                return None
            kwargs[keyword.arg] = ast.literal_eval(keyword.value)
    except (ValueError, SyntaxError):
        return None
    return args, kwargs


def _public_callable_names(source: str) -> list[str]:
    try:
        tree = ast.parse(source or "")
    except SyntaxError:
        return []
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith(
            "_"
        ):
            names.append(node.name)
    return names


def _append_unique_spec(specs: list[CallSpec], spec: CallSpec, *, max_specs: int) -> None:
    key = (spec.module, spec.qualname, repr(spec.args), repr(sorted(spec.kwargs.items())))
    existing = {
        (item.module, item.qualname, repr(item.args), repr(sorted(item.kwargs.items())))
        for item in specs
    }
    if key in existing or len(specs) >= max_specs:
        return
    specs.append(spec)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _module_name_from_path(path: str) -> str:
    raw = str(path or "").replace("\\", "/")
    if raw.endswith(".py"):
        raw = raw[:-3]
    parts = [part for part in raw.split("/") if part and part != "__init__"]
    return ".".join(parts)


def _render_spec_call(spec: dict[str, Any]) -> str:
    args = [repr(item) for item in list(spec.get("args") or [])]
    args.extend(f"{key}={value!r}" for key, value in dict(spec.get("kwargs") or {}).items())
    return f"{spec.get('qualname') or '<call>'}({', '.join(args)})"


def _fingerprint_capture(capture: CaptureResult) -> dict[str, Any]:
    if capture.kind == "value":
        return {
            "call_spec_id": _call_spec_id(capture.call_spec),
            "assertion_op": "==",
            "rhs_literal_shape": _value_shape(capture.value),
            "rhs_literal_repr": repr(capture.value),
            "capture_kind": capture.kind,
            "result_type": capture.result_type,
        }
    if capture.kind == "repr":
        return {
            "call_spec_id": _call_spec_id(capture.call_spec),
            "assertion_op": "==",
            "rhs_literal_shape": "str",
            "rhs_literal_repr": repr(capture.repr_text),
            "capture_kind": capture.kind,
            "result_type": capture.result_type,
        }
    if capture.kind == "exception":
        return {
            "call_spec_id": _call_spec_id(capture.call_spec),
            "assertion_op": "raises",
            "rhs_literal_shape": "exception",
            "rhs_literal_repr": repr(capture.exc_type),
            "capture_kind": capture.kind,
        }
    return {}


def _call_spec_id(spec: CallSpec) -> str:
    return "|".join(
        [
            spec.module,
            spec.qualname,
            repr(spec.args),
            repr(sorted(spec.kwargs.items())),
            spec.receiver_qualname,
            repr(spec.receiver_args),
            repr(sorted(spec.receiver_kwargs.items())),
        ]
    )


def _value_shape(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return f"list:{len(value)}"
    if isinstance(value, tuple):
        return f"tuple:{len(value)}"
    if isinstance(value, set):
        return f"set:{len(value)}"
    if isinstance(value, dict):
        return f"dict:{len(value)}"
    return type(value).__name__
