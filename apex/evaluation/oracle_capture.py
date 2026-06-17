"""Execution-grounded oracle capture for generated test assertions."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from .mutation_targeting import choose_assertion_shape

logger = logging.getLogger(__name__)


# Phase 4A item 4.4: ``capture_oracle`` now requires the caller to
# declare which workdir state ("pre_fix" / "post_fix") the oracle is
# being captured against. The harness verifies the workdir actually
# matches the expected state via a sentinel run before invoking the
# call spec — silently capturing an oracle in the wrong state was the
# structural cause of W4 disagreements that the V4 audit flagged.
ExpectedOracleState = Literal["pre_fix", "post_fix"]


class OracleStateMismatchError(RuntimeError):
    """Raised when the workdir state does not match the declared
    ``expected_state`` for an oracle capture.

    Aborting on mismatch (rather than silently proceeding) prevents the
    common W4 footgun: capturing assertion oracles against a sandbox
    that was meant to be the broken pre-fix repo (so the captured
    values reflect buggy behavior) or vice versa. The two oracles are
    NOT interchangeable — passing one as the other corrupts every
    assertion the caller subsequently rewrites.
    """


# Validator signature: takes the workdir path, returns the detected
# state ("pre_fix" / "post_fix") or None when the state can't be
# inferred (in which case the caller's declaration is trusted).
from typing import Callable as _Callable  # noqa: E402

OracleStateValidator = _Callable[[Path], "ExpectedOracleState | None"]


_ORACLE_STATE_SENTINEL_NAME = ".apex_oracle_state"


def _default_state_validator(workdir: Path) -> "ExpectedOracleState | None":
    """Inspect ``workdir / .apex_oracle_state`` for a state sentinel.

    The sentinel is a one-line text file containing ``pre_fix`` or
    ``post_fix``. Production callers writing the file at sandbox setup
    time get the assertion for free; callers that don't write the
    sentinel get the legacy "trust the caller" behavior (validator
    returns None). Sentinels with other content are ignored with a
    log warning so a malformed file doesn't accidentally lock callers
    out — the only states that are *enforced* are the two declared
    above.
    """

    sentinel = workdir / _ORACLE_STATE_SENTINEL_NAME
    if not sentinel.is_file():
        return None
    try:
        text = sentinel.read_text(encoding="utf-8").strip().lower()
    except OSError as exc:  # pragma: no cover - defensive
        logger.warning(
            "oracle_state sentinel %s unreadable (%s); proceeding without state assertion",
            sentinel,
            exc,
        )
        return None
    if text in {"pre_fix", "pre-fix", "broken"}:
        return "pre_fix"
    if text in {"post_fix", "post-fix", "fixed"}:
        return "post_fix"
    logger.warning(
        "oracle_state sentinel %s contains unrecognized state %r; ignoring",
        sentinel,
        text,
    )
    return None


def write_oracle_state_sentinel(workdir: Path, state: ExpectedOracleState) -> None:
    """Write ``.apex_oracle_state`` so :func:`_default_state_validator`
    can confirm the workdir matches a caller's declared state.

    Production sandbox setup (e.g. ``_prepare_paired_sandboxes``)
    should call this immediately after creating each sandbox so that
    every later ``capture_oracle`` invocation against that workdir
    is guarded by the validator without the caller having to thread
    its own check through.
    """

    if state not in {"pre_fix", "post_fix"}:
        raise ValueError(f"state must be 'pre_fix' or 'post_fix'; got {state!r}")
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / _ORACLE_STATE_SENTINEL_NAME).write_text(state, encoding="utf-8")


@dataclass(frozen=True)
class CallSpec:
    module: str
    qualname: str
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    call_source: str = ""
    # When ``receiver_qualname`` is set, the spec describes an instance
    # method call. The driver constructs ``cls = module.<receiver_qualname>``,
    # builds an instance via ``cls(*receiver_args, **receiver_kwargs)``,
    # then calls the method whose name is ``qualname.split('.')[-1]`` on
    # the instance with ``args``/``kwargs``. This closes the W4 gap for
    # methods that need ``self`` (P1 step 6).
    receiver_qualname: str = ""
    receiver_args: list[Any] = field(default_factory=list)
    receiver_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render_call(self) -> str:
        if self.call_source:
            return self.call_source
        args = [repr(item) for item in self.args]
        args.extend(f"{key}={value!r}" for key, value in self.kwargs.items())
        if self.receiver_qualname:
            recv_args = [repr(item) for item in self.receiver_args]
            recv_args.extend(f"{key}={value!r}" for key, value in self.receiver_kwargs.items())
            method_name = self.qualname.rsplit(".", 1)[-1]
            return (
                f"{self.receiver_qualname}({', '.join(recv_args)}).{method_name}({', '.join(args)})"
            )
        return f"{self.qualname}({', '.join(args)})"


@dataclass(frozen=True)
class CaptureResult:
    kind: str
    call_spec: CallSpec
    value: Any = None
    repr_text: str = ""
    result_type: str = ""
    exc_type: str = ""
    exc_message: str = ""
    error: str = ""
    oracle_origin: str = "executed"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["call_spec"] = self.call_spec.to_dict()
        return payload


def capture_oracle(
    call_spec: CallSpec,
    *,
    workdir: Path,
    expected_state: ExpectedOracleState,
    env: dict[str, str] | None = None,
    timeout: float = 10.0,
    python_executable: str | None = None,
    state_validator: "OracleStateValidator | None" = None,
) -> CaptureResult:
    """Execute a call in ``workdir`` and return the observed outcome.

    Phase 4A item 4.4 — ``expected_state`` is required. Callers must
    declare whether ``workdir`` is meant to be a pre-fix (broken) or
    post-fix (gold-fixed) repo. Before running ``call_spec`` the
    harness invokes ``state_validator`` (defaults to
    :func:`_default_state_validator`, which inspects ``.apex_oracle_state``
    sentinels) to confirm the workdir matches the declared state. On
    mismatch we raise :class:`OracleStateMismatchError` — silently
    capturing in the wrong state would corrupt every assertion the
    caller subsequently rewrites (broken-state oracles are wrong by
    construction; mistakenly capturing in fixed-state when the test
    expects broken behavior is just as wrong).

    The validator is pluggable so production callers can probe a
    project-specific sentinel (e.g. running ``task.test_command`` and
    checking which tests pass) without dragging that knowledge into
    this module. For unit tests, the default validator's sentinel-file
    contract is enough to demonstrate the gate fires.
    """

    if expected_state not in {"pre_fix", "post_fix"}:
        raise ValueError(f"expected_state must be 'pre_fix' or 'post_fix'; got {expected_state!r}")
    workdir_path = Path(workdir)
    validator = state_validator or _default_state_validator
    actual_state = validator(workdir_path)
    if actual_state is not None and actual_state != expected_state:
        raise OracleStateMismatchError(
            f"workdir {workdir_path} is in state {actual_state!r} but "
            f"caller declared expected_state={expected_state!r}; aborting "
            "to avoid silently capturing the wrong oracle"
        )
    executable = python_executable or sys.executable
    with tempfile.TemporaryDirectory(prefix="apex_oracle_capture_") as tmp:
        spec_path = Path(tmp) / "call_spec.json"
        driver_path = Path(tmp) / "driver.py"
        spec_path.write_text(json.dumps(call_spec.to_dict()), encoding="utf-8")
        driver_path.write_text(_DRIVER_SOURCE, encoding="utf-8")
        run_env = dict(os.environ)
        if env:
            run_env.update({str(k): str(v) for k, v in env.items()})
        try:
            completed = subprocess.run(
                [executable, str(driver_path), str(spec_path)],
                cwd=str(workdir),
                env=run_env,
                capture_output=True,
                text=True,
                timeout=float(timeout),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return CaptureResult(
                kind="harness_error",
                call_spec=call_spec,
                error="capture timed out",
            )
        except OSError as exc:
            return CaptureResult(
                kind="harness_error",
                call_spec=call_spec,
                error=f"{type(exc).__name__}: {exc}",
            )
    if completed.returncode != 0:
        return CaptureResult(
            kind="harness_error",
            call_spec=call_spec,
            error=(completed.stderr or completed.stdout or "capture failed")[-4000:],
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return CaptureResult(
            kind="harness_error",
            call_spec=call_spec,
            error=(completed.stdout or completed.stderr or "invalid capture payload")[-4000:],
        )
    first = dict(payload.get("first") or {})
    second = dict(payload.get("second") or {})
    if _captures_differ(first, second):
        return CaptureResult(kind="non_deterministic", call_spec=call_spec)
    return _capture_result_from_payload(first, call_spec)


def capture_oracle_with_runner(
    call_spec: CallSpec,
    *,
    runner: Any,
) -> CaptureResult:
    """Execute a call through a caller-supplied target-environment runner."""

    driver = _DRIVER_SOURCE.replace(
        'spec = json.loads(open(sys.argv[1], encoding="utf-8").read())',
        f"spec = {call_spec.to_dict()!r}",
    )
    try:
        completed = runner(driver)
    except Exception as exc:  # pragma: no cover - defensive runner boundary
        return CaptureResult(
            kind="harness_error",
            call_spec=call_spec,
            error=f"{type(exc).__name__}: {exc}",
        )
    returncode = int(getattr(completed, "returncode", 1) or 0)
    stdout = str(getattr(completed, "stdout", "") or "")
    stderr = str(getattr(completed, "stderr", "") or "")
    if returncode != 0:
        return CaptureResult(
            kind="harness_error",
            call_spec=call_spec,
            error=(stderr or stdout or "capture failed")[-4000:],
        )
    try:
        payload = json.loads(stdout.splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return CaptureResult(
            kind="harness_error",
            call_spec=call_spec,
            error=(stdout or stderr or "invalid capture payload")[-4000:],
        )
    first = dict(payload.get("first") or {})
    second = dict(payload.get("second") or {})
    if _captures_differ(first, second):
        return CaptureResult(kind="non_deterministic", call_spec=call_spec)
    return _capture_result_from_payload(first, call_spec)


def summarize_captures_for_diagnostics(
    captures: "Iterable[CaptureResult | dict[str, Any]]",
) -> dict[str, Any]:
    """Convert W4 captured values into the ledger-friendly ``{repr_key: value}`` dict.

    This is the canonical shape consumed by the V5 anti-hack ledger: each key
    is a string that, if it appears as a substring of a generated assertion,
    means the LLM grounded that assertion in observed runtime evidence rather
    than fabricating it. Both ``CaptureResult`` instances and pre-serialized
    dicts (``CaptureResult.to_dict()`` payloads) are accepted so callers can
    feed in either runtime captures or already-persisted diagnostics.

    The mapping mirrors what ``_v5_captured_oracle_values`` does on the
    consumer side:
      * ``repr_text`` (e.g. ``"[1, 2, 3]"``) → used verbatim.
      * ``exc_type`` / ``exc_message`` for raised-exception captures.
      * ``value`` is registered under its ``repr(...)`` key, plus its
        ``str(...)`` key when scalar so assertions like ``assert x == 7``
        match without needing the literal ``7``-quoted form.
    """

    values: dict[str, Any] = {}
    for capture in captures or []:
        if isinstance(capture, CaptureResult):
            payload = capture.to_dict()
        elif isinstance(capture, dict):
            payload = dict(capture)
        else:
            continue
        for key in ("repr_text", "exc_type", "exc_message"):
            value = payload.get(key)
            if value:
                values[str(value)] = value
        if "value" in payload:
            value = payload.get("value")
            try:
                values[repr(value)] = value
            except Exception:
                # repr() on exotic objects can raise; skip rather than break.
                pass
            if isinstance(value, (str, int, float, bool)):
                values[str(value)] = value
    return values


def synthesize_assertion(
    capture: CaptureResult,
    *,
    style: Any,
    tolerance: float | None = None,
) -> str:
    """Render assertion source for an executed capture."""

    if capture.kind == "non_deterministic":
        return ""
    call = capture.call_spec.render_call()
    runner = _runner_name(style)
    assertion_style = str(getattr(style, "assertion_style", "") or "").lower()
    if capture.kind == "exception":
        if "unittest" in assertion_style or "self.assert" in assertion_style:
            return f"with self.assertRaises({capture.exc_type}):\n    {call}\n"
        if runner == "sympy-bin-test":
            return f"raises({capture.exc_type}, lambda: {call})\n"
        if _style_allows_helper(style, "pytest.raises"):
            return f"with pytest.raises({capture.exc_type}):\n    {call}\n"
        return f"try:\n    {call}\nexcept {capture.exc_type}:\n    pass\nelse:\n    assert False\n"
    if capture.result_type == "ndarray" and capture.value is not None:
        if _style_allows_helper(style, "numpy.testing.assert_allclose"):
            return f"result = {call}\nnumpy.testing.assert_allclose(result, {capture.value!r})\n"
        return f"result = {call}\nassert getattr(result, 'tolist', lambda: result)() == {capture.value!r}\n"
    if capture.kind == "value":
        shape = choose_assertion_shape(capture.value, result_type=capture.result_type)
        if shape.name == "pytest_approx":
            tol = tolerance if tolerance is not None else 1e-12
            if _style_allows_helper(style, "pytest.approx"):
                return f"result = {call}\nassert result == pytest.approx({capture.value!r}, rel={tol!r})\n"
            if "self.assert" in assertion_style:
                return f"result = {call}\nself.assertAlmostEqual(result, {capture.value!r})\n"
            return (
                f"result = {call}\n"
                f"assert __import__('math').isclose(result, {capture.value!r}, rel_tol={tol!r})\n"
            )
        return f"result = {call}\nassert result == {capture.value!r}\n"
    if capture.kind == "repr":
        return f"result = {call}\nassert repr(result) == {capture.repr_text!r}\n"
    return ""


def _runner_name(style: Any) -> str:
    return str(getattr(style, "runner", "") or "").lower()


def _style_allows_helper(style: Any, helper: str) -> bool:
    if not _runner_name(style):
        return helper.startswith(("pytest.", "numpy."))
    try:
        from .test_style import runner_profile_for_style

        return runner_profile_for_style(style).allows_helper(helper)
    except Exception:
        runner = _runner_name(style)
        if helper.startswith(("pytest.", "numpy.")):
            return runner in {"", "pytest"}
        return True


def _capture_result_from_payload(payload: dict[str, Any], call_spec: CallSpec) -> CaptureResult:
    kind = str(payload.get("kind") or "")
    if kind == "exception":
        return CaptureResult(
            kind="exception",
            call_spec=call_spec,
            exc_type=str(payload.get("exc_type") or "Exception"),
            exc_message=str(payload.get("exc_message") or ""),
        )
    if kind == "value":
        return CaptureResult(
            kind="value",
            call_spec=call_spec,
            value=payload.get("value"),
            result_type=str(payload.get("result_type") or ""),
            repr_text=str(payload.get("repr_text") or ""),
        )
    if kind == "repr":
        return CaptureResult(
            kind="repr",
            call_spec=call_spec,
            value=payload.get("value"),
            result_type=str(payload.get("result_type") or ""),
            repr_text=str(payload.get("repr_text") or ""),
        )
    return CaptureResult(
        kind="harness_error",
        call_spec=call_spec,
        error=str(payload.get("error") or "unknown capture kind"),
    )


def _captures_differ(first: dict[str, Any], second: dict[str, Any]) -> bool:
    stable_keys = ("kind", "value", "repr_text", "result_type", "exc_type", "exc_message")
    return {key: first.get(key) for key in stable_keys} != {
        key: second.get(key) for key in stable_keys
    }


_DRIVER_SOURCE = r"""
import builtins
import importlib
import json
import os
import sys

sys.path.insert(0, os.getcwd())


def resolve(module_name, qualname):
    root = builtins if module_name == "builtins" else importlib.import_module(module_name)
    obj = root
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def encode_result(value):
    result_type = type(value).__name__
    module = type(value).__module__
    if module.startswith("numpy") and hasattr(value, "tolist"):
        return {
            "kind": "repr",
            "value": value.tolist(),
            "repr_text": repr(value),
            "result_type": "ndarray",
        }
    try:
        json.dumps(value)
    except TypeError:
        return {
            "kind": "repr",
            "repr_text": repr(value),
            "result_type": result_type,
        }
    return {
        "kind": "value",
        "value": value,
        "repr_text": repr(value),
        "result_type": result_type,
    }


def invoke(spec):
    try:
        receiver_qualname = spec.get("receiver_qualname") or ""
        if receiver_qualname:
            cls = resolve(spec["module"], receiver_qualname)
            instance = cls(
                *(spec.get("receiver_args") or []),
                **(spec.get("receiver_kwargs") or {}),
            )
            method_name = spec["qualname"].rsplit(".", 1)[-1]
            method = getattr(instance, method_name)
            return encode_result(
                method(
                    *(spec.get("args") or []),
                    **(spec.get("kwargs") or {}),
                )
            )
        func = resolve(spec["module"], spec["qualname"])
        return encode_result(func(*(spec.get("args") or []), **(spec.get("kwargs") or {})))
    except BaseException as exc:
        return {
            "kind": "exception",
            "exc_type": type(exc).__name__,
            "exc_message": str(exc),
        }


spec = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(json.dumps({"first": invoke(spec), "second": invoke(spec)}))
"""
