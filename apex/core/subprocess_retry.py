"""Subprocess retry helper with failure classification.

This module exists so that benchmark runners can transparently retry
subprocess invocations whose failures are *environment* (network, install,
timeout, resource) or *harness bug* in nature, while leaving real APEX
miss / unclassified failures untouched.

Until Phase 1, every benchmark runner ran subprocesses exactly once and
silently rolled "the upstream harness crashed" / "the package index 502'd"
into "APEX missed the patch". That inflates APEX's published miss rate
and hides infra regressions. The fix:

* Classify each failure with :func:`apex.core.failure_classifier.classify_failure`.
* Retry only on a configurable set of retryable classes (env_* + harness_bug
  by default).
* Use exponential backoff (1s, 2s, 4s, ...).
* Never retry on :data:`FailureClass.APEX_MISS` -- a real test failure should
  surface immediately.

This module is deliberately self-contained: it has no dependency on any
benchmark-specific code so it can be reused by Commit0 / TestGenEval /
SWT-Bench wirings.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from apex.core.failure_classifier import (
    ClassificationResult,
    FailureClass,
    classify_failure,
)

logger = logging.getLogger("apex.subprocess_retry")


# Default retryable classes: anything that is plausibly transient.
# APEX_MISS is intentionally excluded -- real assertion failures must not be
# masked by retry. UNCLASSIFIED is also excluded by default; callers that
# want lenient retry can opt-in explicitly.
DEFAULT_RETRY_ON: frozenset[FailureClass] = frozenset(
    {
        FailureClass.ENV_NETWORK,
        FailureClass.ENV_INSTALL,
        FailureClass.ENV_TIMEOUT,
        FailureClass.ENV_RESOURCE,
        FailureClass.HARNESS_BUG,
    }
)


@dataclass
class AttemptRecord:
    """Per-attempt diagnostic record."""

    attempt: int  # 1-indexed
    returncode: int
    classification: Optional[ClassificationResult] = None
    duration_seconds: float = 0.0
    timed_out: bool = False
    spawn_error: Optional[str] = None


@dataclass
class RetryDiagnostics:
    """Aggregate diagnostics from a :func:`run_with_classification` call."""

    attempts: list[AttemptRecord] = field(default_factory=list)
    final_classification: Optional[ClassificationResult] = None
    succeeded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempts": [
                {
                    "attempt": a.attempt,
                    "returncode": a.returncode,
                    "duration_seconds": round(a.duration_seconds, 4),
                    "timed_out": a.timed_out,
                    "spawn_error": a.spawn_error,
                    "classification": (a.classification.to_dict() if a.classification else None),
                }
                for a in self.attempts
            ],
            "final_classification": (
                self.final_classification.to_dict() if self.final_classification else None
            ),
            "succeeded": self.succeeded,
            "num_attempts": len(self.attempts),
        }


def _exponential_backoff_seconds(attempt: int, base: float = 1.0) -> float:
    """Return the wait before the *next* attempt after attempt *attempt* (1-indexed).

    Sequence: 1s, 2s, 4s, 8s, ...
    """
    return float(base) * (2 ** max(0, attempt - 1))


def run_with_classification(
    cmd: list[str] | str,
    *,
    max_attempts: int = 3,
    backoff: str = "exponential",
    retry_on: Iterable[FailureClass] = DEFAULT_RETRY_ON,
    classifier_context: Optional[dict[str, Any]] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    diagnostics_sink: Optional[RetryDiagnostics] = None,
    **subprocess_kwargs: Any,
) -> subprocess.CompletedProcess:
    """Run ``cmd`` via ``subprocess.run`` with classification-driven retry.

    Args:
        cmd: Command (list or string) passed to ``subprocess.run``.
        max_attempts: Total attempt budget. ``1`` disables retry.
        backoff: ``"exponential"`` (default; 1s/2s/4s/...) or ``"none"``.
        retry_on: Set of :class:`FailureClass` values that trigger a retry.
            Defaults to all env_* + HARNESS_BUG.
        classifier_context: Forwarded to ``classify_failure(context=...)``.
            Use this to pass ``{"phase": "test_execution"}`` etc.
        sleep_fn: Override for ``time.sleep``; tests inject a no-op.
        diagnostics_sink: Optional :class:`RetryDiagnostics` instance to
            populate. A fresh one is allocated when omitted.
        **subprocess_kwargs: Forwarded verbatim to ``subprocess.run``.
            ``capture_output`` / ``text`` default to True so we can
            classify stderr; pass them explicitly to override.

    Returns:
        The final ``CompletedProcess`` (success or last failure). Never
        raises ``subprocess.TimeoutExpired`` -- timeouts are converted to a
        synthetic ``CompletedProcess(returncode=124)`` so callers can
        inspect them uniformly.

    Notes:
        * ``UNCLASSIFIED`` failures are never retried by default. If you
          want to retry on unknown failures, include ``FailureClass.UNCLASSIFIED``
          in ``retry_on``.
        * ``APEX_MISS`` is ALWAYS excluded from retry, even if the caller
          adds it to ``retry_on``. This is deliberate: a real test failure
          must surface immediately.
    """
    diag = diagnostics_sink if diagnostics_sink is not None else RetryDiagnostics()
    retry_set = frozenset(retry_on) - {FailureClass.APEX_MISS}

    # Default to capturing output so we can classify stderr. Callers that
    # want streaming output should pass capture_output=False explicitly.
    subprocess_kwargs.setdefault("capture_output", True)
    subprocess_kwargs.setdefault("text", True)
    subprocess_kwargs.setdefault("check", False)

    attempts = max(1, int(max_attempts))
    last_completed: Optional[subprocess.CompletedProcess] = None
    last_classification: Optional[ClassificationResult] = None

    for attempt_idx in range(1, attempts + 1):
        started = time.time()
        timed_out = False
        spawn_error: Optional[str] = None
        try:
            completed = subprocess.run(cmd, **subprocess_kwargs)
        except subprocess.TimeoutExpired as exc:
            # Synthesize a CompletedProcess so callers can inspect it.
            timed_out = True
            stdout = (
                exc.stdout.decode("utf-8", errors="replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )
            stderr = (
                exc.stderr.decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "")
            )
            completed = subprocess.CompletedProcess(
                args=cmd,
                returncode=124,
                stdout=str(stdout),
                stderr=str(stderr) + f"\nsubprocess.TimeoutExpired after {exc.timeout}s\n",
            )
        except (FileNotFoundError, PermissionError, OSError) as exc:
            # Spawn failure. Treat as HARNESS_BUG so we record the attempt
            # but do NOT retry (no amount of retrying will materialize the
            # binary). Caller can still inspect via the returned record.
            spawn_error = f"{type(exc).__name__}: {exc}"
            completed = subprocess.CompletedProcess(
                args=cmd,
                returncode=127,
                stdout="",
                stderr=spawn_error,
            )

        duration = time.time() - started

        if completed.returncode == 0 and not timed_out:
            diag.attempts.append(
                AttemptRecord(
                    attempt=attempt_idx,
                    returncode=completed.returncode,
                    classification=None,
                    duration_seconds=duration,
                    timed_out=False,
                    spawn_error=None,
                )
            )
            diag.succeeded = True
            return completed

        # Classify the failure.
        ctx = dict(classifier_context or {})
        if timed_out:
            ctx["timed_out"] = True
        classification = classify_failure(
            stderr=str(getattr(completed, "stderr", "") or ""),
            stdout=str(getattr(completed, "stdout", "") or ""),
            returncode=int(completed.returncode or 0),
            context=ctx,
        )
        last_completed = completed
        last_classification = classification
        diag.attempts.append(
            AttemptRecord(
                attempt=attempt_idx,
                returncode=int(completed.returncode or 0),
                classification=classification,
                duration_seconds=duration,
                timed_out=timed_out,
                spawn_error=spawn_error,
            )
        )

        # Decide retry.
        if attempt_idx >= attempts:
            break
        if spawn_error is not None:
            # Spawn failures are not transient.
            break
        if classification.failure_class not in retry_set:
            logger.debug(
                "subprocess_retry: not retrying failure_class=%s (retry_set=%s) on attempt %d",
                classification.failure_class.value,
                sorted(c.value for c in retry_set),
                attempt_idx,
            )
            break

        # Retry. Sleep with backoff.
        if backoff == "none":
            wait = 0.0
        else:
            wait = _exponential_backoff_seconds(attempt_idx)
        logger.warning(
            "subprocess_retry: attempt %d/%d failed with %s; retrying in %.1fs",
            attempt_idx,
            attempts,
            classification.failure_class.value,
            wait,
        )
        if wait > 0:
            sleep_fn(wait)

    diag.final_classification = last_classification
    diag.succeeded = False
    assert last_completed is not None
    return last_completed


__all__ = [
    "AttemptRecord",
    "DEFAULT_RETRY_ON",
    "RetryDiagnostics",
    "run_with_classification",
]
