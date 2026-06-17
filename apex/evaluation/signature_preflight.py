"""Signature preflight for generated Python call sites."""

from __future__ import annotations

import ast
import difflib
import inspect
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignaturePreflightResult:
    status: str
    diagnostics: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def preflight_signatures(
    source: str,
    signatures: dict[str, str],
) -> SignaturePreflightResult:
    try:
        tree = ast.parse(source or "")
    except SyntaxError as exc:
        return SignaturePreflightResult(status="fail", diagnostics=[str(exc)])
    parsed = {
        name: signature for name, raw in signatures.items() if (signature := _parse_signature(raw))
    }
    diagnostics: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if not name:
            continue
        signature = parsed.get(name) or parsed.get(name.rsplit(".", 1)[-1])
        if signature is None:
            continue
        args = [object()] * len(node.args)
        kwargs = {kw.arg: object() for kw in node.keywords if kw.arg is not None}
        try:
            signature.bind(*args, **kwargs)
        except TypeError as exc:
            diagnostics.append(_render_signature_error(name, signature, str(exc), kwargs))
    return SignaturePreflightResult(
        status="pass" if not diagnostics else "fail",
        diagnostics=diagnostics,
    )


def signatures_from_api_probe(api_probe: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for symbol in list(getattr(api_probe, "symbols", []) or []):
        name = str(getattr(symbol, "name", "") or "")
        signature = str(getattr(symbol, "signature", "") or "")
        if name and "(" in signature:
            out[name] = signature
            out[name.rsplit(".", 1)[-1]] = signature
    return out


def _parse_signature(raw: str) -> inspect.Signature | None:
    text = str(raw or "").strip()
    if not text:
        return None
    params = text[text.find("(") :] if "(" in text else text
    if "->" in params:
        params = params.split("->", 1)[0].rstrip()
    try:
        module = ast.parse(f"def __apex_sig__{params}:\n    pass\n")
        function = module.body[0]
        if not isinstance(function, ast.FunctionDef):
            return None
        return _signature_from_ast_arguments(function.args)
    except Exception as exc:
        # Audit H12: log so a malformed signature string surfaces in the
        # debug log instead of silently disabling preflight for the
        # whole task. ``None`` is still returned so callers behave the
        # same on the happy path.
        logger.debug(
            "signature_preflight._parse_signature: failed to parse %r "
            "(%s: %s); preflight will skip this call site",
            text[:200],
            type(exc).__name__,
            exc,
        )
        return None


def _signature_from_ast_arguments(args: ast.arguments) -> inspect.Signature:
    """Build an inspect.Signature from AST-only parameter metadata.

    This intentionally ignores annotation/default values instead of evaluating
    them. Bind validation only needs parameter names, order, and kind.
    """

    parameters: list[inspect.Parameter] = []
    posonly = list(args.posonlyargs or [])
    positional = list(args.args or [])
    default_offset = len(posonly) + len(positional) - len(args.defaults or [])
    for index, arg in enumerate(posonly + positional):
        default = inspect.Parameter.empty if index < default_offset else None
        kind = (
            inspect.Parameter.POSITIONAL_ONLY
            if index < len(posonly)
            else inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        parameters.append(inspect.Parameter(arg.arg, kind=kind, default=default))
    if args.vararg is not None:
        parameters.append(inspect.Parameter(args.vararg.arg, inspect.Parameter.VAR_POSITIONAL))
    for kwarg, default_node in zip(args.kwonlyargs or [], args.kw_defaults or []):
        parameters.append(
            inspect.Parameter(
                kwarg.arg,
                inspect.Parameter.KEYWORD_ONLY,
                default=inspect.Parameter.empty if default_node is None else None,
            )
        )
    if args.kwarg is not None:
        parameters.append(inspect.Parameter(args.kwarg.arg, inspect.Parameter.VAR_KEYWORD))
    return inspect.Signature(parameters)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _render_signature_error(
    name: str,
    signature: inspect.Signature,
    error: str,
    kwargs: dict[str, Any],
) -> str:
    allowed = [
        param.name
        for param in signature.parameters.values()
        if param.kind
        in {
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    ]
    unknown = [key for key in kwargs if key not in allowed]
    if unknown and allowed:
        suggestion = difflib.get_close_matches(unknown[0], allowed, n=1, cutoff=0.0)
        if suggestion:
            return f"{name}: {error}; closest keyword: {suggestion[0]}"
    return f"{name}: {error}"
