"""
Availability-aware LLM routing helpers.

These utilities keep short-lived failure memory for backend-level CLI/session
and infrastructure errors so Apex can reroute to another configured backend
instead of repeatedly rediscovering the same failure inside every
planner/rollout.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Iterable, Optional

from .cli_backend import cli_backend_is_healthy, cli_backend_unavailable_reason
from .config import LLMConfig

logger = logging.getLogger("apex.llm_routing")

_RECORDED_BACKEND_FAILURES_LOCK = threading.Lock()
_RECORDED_BACKEND_FAILURES: dict[tuple[str, str, str, str], str] = {}
_BACKEND_FAILURE_MARKERS: tuple[str, ...] = (
    "invalid_api_key",
    "incorrect api key",
    "authentication_error",
    "authentication error",
    "unauthorized",
    "error code: 401",
    '"status": 401',
    "'status': 401",
    "http/1.1 401",
    "openai python sdk could not be imported",
    "does not expose the openai client",
    "detected unsettled top-level await",
    "top-level await",
    "yoga-layout",
    "err_require_esm",
    "esm module",
    "did not respond to '--help'",
    "did not respond to '--version'",
    "failed its startup probe",
    "is not installed",
    "target-container auth is not configured",
    "target cli auth missing",
    "provider error (http 400)",
    "function_call' was provided without its required 'reasoning'",
)
_CALL_FAILOVER_MARKERS: tuple[str, ...] = (
    "exited during startup without producing structured output",
    "exited during bootstrap/setup without producing structured output",
    "without terminal result or workspace changes",
    "content_free_cli_exit",
    # A no-output stall proves this invocation made no progress; concurrent
    # invocations of the same backend may still be healthy.
    "cli backend stalled after",
    # Provider transport failures are already retried inside the CLI backend.
    # If those retries are exhausted, reroute the current stage instead of
    # burning the rollout as an implementation failure.
    "api error: unable to connect to api",
    "connection reset by peer",
    "econnreset",
    "error from external service",
    "error sending request for url",
    "error code: 429",
    "error code: 529",
    "http 429",
    "http 529",
    "insufficient_quota",
    "overloaded",
    "quota exceeded",
    "rate limit",
    "rate_limit_error",
    "request was throttled",
    "socket hang up",
    "socket connection was closed unexpectedly",
    '"status": 429',
    '"status": 529',
    "'status': 429",
    "'status': 529",
    "stream disconnected before completion",
    "tls handshake timeout",
    "too many requests",
)


def llm_backend_fingerprint(config: Any) -> tuple[str, str, str, str]:
    """Return a stable identity tuple for one configured backend."""

    backend = getattr(getattr(config, "backend", None), "value", getattr(config, "backend", ""))
    model = getattr(config, "model", "")
    api_key_env = (
        "" if bool(getattr(config, "is_cli_backend", False)) else getattr(config, "api_key_env", "")
    )
    cli_command = getattr(config, "resolved_cli_command", "")
    return (
        str(backend or "").strip().lower(),
        str(model or "").strip().lower(),
        str(api_key_env or "").strip(),
        str(cli_command or "").strip().lower(),
    )


def reset_recorded_llm_backend_failures() -> None:
    """Clear the per-run recorded backend failure cache."""

    with _RECORDED_BACKEND_FAILURES_LOCK:
        _RECORDED_BACKEND_FAILURES.clear()


def recorded_llm_backend_failure_reason(config: Any) -> str:
    """Return the cached backend-level failure reason for ``config`` if present."""

    fingerprint = llm_backend_fingerprint(config)
    with _RECORDED_BACKEND_FAILURES_LOCK:
        return str(_RECORDED_BACKEND_FAILURES.get(fingerprint) or "")


def clear_recorded_llm_backend_failure(config: Any) -> None:
    """Clear any cached backend failure for ``config``."""

    fingerprint = llm_backend_fingerprint(config)
    with _RECORDED_BACKEND_FAILURES_LOCK:
        _RECORDED_BACKEND_FAILURES.pop(fingerprint, None)


def _failure_fragments(exc_or_reason: Any) -> list[str]:
    text = str(exc_or_reason or "").strip()
    fragments: list[str] = [text] if text else []
    backend_anomaly = getattr(exc_or_reason, "backend_anomaly", None)
    if isinstance(backend_anomaly, dict):
        for key in ("reason", "kind", "terminal_state"):
            value = str(backend_anomaly.get(key) or "").strip()
            if value:
                fragments.append(value)
    if isinstance(exc_or_reason, dict):
        for key in ("reason", "recorded_backend_failure_reason", "kind", "terminal_state"):
            value = str(exc_or_reason.get(key) or "").strip()
            if value:
                fragments.append(value)
    return fragments


def _first_matching_failure_fragment(
    fragments: list[str],
    markers: tuple[str, ...],
) -> str:
    if not fragments:
        return ""
    lowered = "\n".join(fragments).lower()
    if any(marker in lowered for marker in markers):
        for fragment in fragments:
            fragment_lowered = fragment.lower()
            if any(marker in fragment_lowered for marker in markers):
                return fragment
        return fragments[0]
    return ""


def classify_llm_backend_failure(exc_or_reason: Any) -> str:
    """Return a cached-failure reason for backend-level infra/auth errors only."""

    return _first_matching_failure_fragment(
        _failure_fragments(exc_or_reason),
        _BACKEND_FAILURE_MARKERS,
    )


def classify_llm_call_failover_failure(exc_or_reason: Any) -> str:
    """Return a reason for invocation-local failures that deserve route failover.

    These failures are severe for the current stage, but they do not prove the
    backend is unavailable globally. The caller may retry a different configured
    route without writing to the per-run backend-unavailable cache.
    """

    return _first_matching_failure_fragment(
        _failure_fragments(exc_or_reason),
        _CALL_FAILOVER_MARKERS,
    )


def record_llm_backend_failure(config: Any, exc_or_reason: Any) -> str:
    """Persist a backend-level failure so later routing can avoid it."""

    reason = classify_llm_backend_failure(exc_or_reason)
    if not reason:
        return ""
    fingerprint = llm_backend_fingerprint(config)
    with _RECORDED_BACKEND_FAILURES_LOCK:
        _RECORDED_BACKEND_FAILURES[fingerprint] = reason
    logger.warning(
        "Recorded backend failure for %s/%s: %s",
        fingerprint[0] or "unknown",
        fingerprint[1] or "unknown",
        reason,
    )
    return reason


def llm_backend_unavailable_reason(
    config: Any,
    *,
    refresh: bool = False,
) -> str:
    """Return a human-readable reason why ``config`` should be treated as unavailable."""

    cached = recorded_llm_backend_failure_reason(config)
    if cached:
        return cached
    if bool(getattr(config, "is_cli_backend", False)):
        if cli_backend_is_healthy(config, refresh=refresh):
            return ""
        return cli_backend_unavailable_reason(config, refresh=refresh)
    if not bool(getattr(config, "has_api_key", False)):
        backend = str(
            getattr(getattr(config, "backend", None), "value", getattr(config, "backend", "")) or ""
        ).strip()
        model = str(getattr(config, "model", "") or "").strip()
        target = f"backend '{backend}'" if backend else "the selected backend"
        if model:
            target = f"{target} for model '{model}'"
        return (
            f"Non-CLI {target} is unavailable in Apex's CLI-session execution mode. "
            "Use a configured agentic CLI backend such as codex_cli, claude_cli, "
            "gemini_cli, opencode_cli, or metacode_cli."
        )
    return ""


def llm_backend_is_available(
    config: Any,
    *,
    refresh: bool = False,
) -> bool:
    """Return whether ``config`` should be considered usable right now."""

    return not bool(llm_backend_unavailable_reason(config, refresh=refresh))


def _candidate_failover_rank(
    requested: Any,
    candidate: Any,
    *,
    index: int,
) -> tuple[int, int, int, int, int]:
    requested_model = str(getattr(requested, "model", "") or "").strip().lower()
    candidate_model = str(getattr(candidate, "model", "") or "").strip().lower()
    requested_backend = (
        str(
            getattr(getattr(requested, "backend", None), "value", getattr(requested, "backend", ""))
            or ""
        )
        .strip()
        .lower()
    )
    candidate_backend = (
        str(
            getattr(getattr(candidate, "backend", None), "value", getattr(candidate, "backend", ""))
            or ""
        )
        .strip()
        .lower()
    )
    requested_is_cli = bool(getattr(requested, "is_cli_backend", False))
    candidate_is_cli = bool(getattr(candidate, "is_cli_backend", False))
    same_model = candidate_model == requested_model
    same_backend = candidate_backend == requested_backend
    same_model_cli_failover = same_model and candidate_is_cli and not requested_is_cli
    return (
        1 if same_model else 0,
        1 if same_model_cli_failover else 0,
        1 if same_backend else 0,
        1 if candidate_is_cli else 0,
        -int(index),
    )


def resolve_available_llm_config(
    requested: LLMConfig,
    candidates: Iterable[LLMConfig],
    *,
    exclude_fingerprints: Optional[set[tuple[str, str, str, str]]] = None,
    purpose: str = "execution",
) -> tuple[LLMConfig, dict[str, Any]]:
    """Resolve ``requested`` to the best currently available configured backend."""

    excluded = set(exclude_fingerprints or set())
    requested_fingerprint = llm_backend_fingerprint(requested)
    requested_reason = (
        "" if requested_fingerprint in excluded else llm_backend_unavailable_reason(requested)
    )
    if requested_fingerprint not in excluded and not requested_reason:
        return requested, {
            "purpose": purpose,
            "fallback_applied": False,
            "fallback_kind": "",
            "requested_unavailable_reason": "",
            "requested_backend": str(getattr(requested.backend, "value", requested.backend) or ""),
            "requested_model": str(requested.model or ""),
            "resolved_backend": str(getattr(requested.backend, "value", requested.backend) or ""),
            "resolved_model": str(requested.model or ""),
            "resolved_fingerprint": requested_fingerprint,
        }

    best_candidate: Optional[LLMConfig] = None
    best_rank: Optional[tuple[int, int, int, int, int]] = None
    for index, candidate in enumerate(list(candidates or [])):
        fingerprint = llm_backend_fingerprint(candidate)
        if fingerprint in excluded:
            continue
        if fingerprint == requested_fingerprint and requested_reason:
            continue
        if llm_backend_unavailable_reason(candidate):
            continue
        rank = _candidate_failover_rank(requested, candidate, index=index)
        if best_rank is None or rank > best_rank:
            best_candidate = candidate
            best_rank = rank

    if best_candidate is None:
        return requested, {
            "purpose": purpose,
            "fallback_applied": False,
            "fallback_kind": "",
            "requested_unavailable_reason": requested_reason,
            "requested_backend": str(getattr(requested.backend, "value", requested.backend) or ""),
            "requested_model": str(requested.model or ""),
            "resolved_backend": str(getattr(requested.backend, "value", requested.backend) or ""),
            "resolved_model": str(requested.model or ""),
            "resolved_fingerprint": requested_fingerprint,
        }

    fallback_kind = "healthy_backend_substitution"
    if (
        str(best_candidate.model or "").strip().lower()
        == str(requested.model or "").strip().lower()
    ):
        fallback_kind = "same_model_backend_failover"
        if bool(getattr(best_candidate, "is_cli_backend", False)) and not bool(
            getattr(requested, "is_cli_backend", False)
        ):
            fallback_kind = "same_model_cli_failover"

    return best_candidate, {
        "purpose": purpose,
        "fallback_applied": True,
        "fallback_kind": fallback_kind,
        "requested_unavailable_reason": requested_reason,
        "requested_backend": str(getattr(requested.backend, "value", requested.backend) or ""),
        "requested_model": str(requested.model or ""),
        "resolved_backend": str(
            getattr(best_candidate.backend, "value", best_candidate.backend) or ""
        ),
        "resolved_model": str(best_candidate.model or ""),
        "resolved_fingerprint": llm_backend_fingerprint(best_candidate),
    }
