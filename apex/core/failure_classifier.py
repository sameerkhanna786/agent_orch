"""Core, phase-aware failure classifier for APEX runs.

The point of this module is small but load-bearing: separate failures the
LLM/agent caused (``APEX_MISS``) from failures the *environment* caused
(network/install/timeout/resource), and from failures that come from a
crashing harness (``HARNESS_BUG``). Today, every one of those classes
silently rolls up into "the model wrote a bad patch", which inflates
APEX's published miss rate and hides infra regressions.

This module deliberately uses a coarser taxonomy than
``apex/evaluation/failure_classifier.py`` (which is testgen-validation
specific). The two coexist on purpose:

* ``apex.evaluation.failure_classifier.FailureClass`` — fine-grained
  bucketing for the testgen validation pipeline (``APEX_SYNTAX``,
  ``APEX_SPLICE_COLLISION``, ``ENV_DB_MISSING``, ...).
* ``apex.core.failure_classifier.FailureClass`` — *core* taxonomy used
  across the whole orchestrator. Six buckets:

  - ``APEX_MISS`` — the LLM/agent produced a wrong answer.
  - ``ENV_NETWORK`` — clone failure, package index unreachable, DNS, etc.
  - ``ENV_INSTALL`` — pip/apt install crash, missing system dep at install.
  - ``ENV_TIMEOUT`` — wall-clock timeout (vs. the agent's own turn timeout).
  - ``ENV_RESOURCE`` — OOM, disk full, container OOM-killed.
  - ``HARNESS_BUG`` — upstream test runner / scoring crashed before the
    agent had a chance to be scored on its output.
  - ``UNCLASSIFIED`` — couldn't decide; default to charging APEX is the
    *caller's* policy, not this module's.

Pattern library is harvested from the two existing per-benchmark
classifiers so we stay bug-for-bug compatible with what we already do:

* ``apex/evaluation/commit0_benchmark.py::_baseline_signals_host_env_failure``
* ``apex/evaluation/commit0_benchmark.py::_classify_prepare_error``
* ``apex/evaluation/testgeneval_benchmark.py::_should_retry_env_failure``
  / ``_is_env_or_harness_failure``

Phase-aware disambiguation:

The classic ambiguous case is ``ModuleNotFoundError`` — during
``pre_install`` it's an ENV failure (the repo's bootstrap script can't
find a system dep); during test execution after a clean install it's an
APEX miss (the agent's patch broke an import). Callers pass
``context={"phase": ...}`` to disambiguate. Recognised phases:

  ``"clone"`` | ``"pre_install"`` | ``"install"`` | ``"baseline"`` |
  ``"test_execution"`` | ``"scoring"``
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional


class FailureClass(str, Enum):
    """Coarse, orchestrator-wide failure taxonomy. See module docstring."""

    APEX_MISS = "apex_miss"
    ENV_NETWORK = "env_network"
    ENV_INSTALL = "env_install"
    ENV_TIMEOUT = "env_timeout"
    ENV_RESOURCE = "env_resource"
    # Target-runtime / control-plane / agent-auth failures: the container was
    # reaped or unreachable, the OCI exec failed, or the agentic CLI backend was
    # not authenticated. These are infra facts, not a wrong patch, so they must
    # not be charged to the APEX miss rate. (Benchmark-agnostic: any run that
    # executes inside a container or drives an agentic CLI can hit these.)
    ENV_RUNTIME = "env_runtime"
    # NDFF: a genuinely non-deterministic / flaky failure of the scoring oracle
    # itself — a teardown/finalizer leak (Twisted DirtyReactor, unclosed asyncio
    # loop, atexit/threadpool), an order-dependent test, or a test whose new
    # failure does not intersect the candidate's changed lines (DeFlaker). These
    # must never charge an APEX miss: a flaky gold test failing is not a wrong
    # patch. Retryable like an env failure (rerun/quarantine resolves it).
    NON_DETERMINISTIC = "non_deterministic"
    HARNESS_BUG = "harness_bug"
    UNCLASSIFIED = "unclassified"

    @property
    def is_environment(self) -> bool:
        """True if this class represents an environment/infra failure.

        Callers use this to decide between "retry on a fresh container"
        (env) vs. "score this as an APEX miss" (apex). Harness bugs are
        *not* environmental in the retry sense — a clean container won't
        fix a broken upstream report generator.
        """
        return self in {
            FailureClass.ENV_NETWORK,
            FailureClass.ENV_INSTALL,
            FailureClass.ENV_TIMEOUT,
            FailureClass.ENV_RESOURCE,
            FailureClass.ENV_RUNTIME,
            FailureClass.NON_DETERMINISTIC,
        }

    @property
    def is_nondeterministic(self) -> bool:
        """True if this class is a flaky/non-deterministic failure of the oracle
        rather than a deterministic env/infra or candidate-quality failure."""
        return self is FailureClass.NON_DETERMINISTIC

    @property
    def charges_apex(self) -> bool:
        """True if this class should count against the APEX miss rate.

        Anything env_* or harness_bug is excluded from the APEX
        denominator by default. Unclassified is included — better to
        over-count and investigate later than to silently exempt.
        """
        return self in {FailureClass.APEX_MISS, FailureClass.UNCLASSIFIED}


@dataclass(frozen=True)
class ClassificationResult:
    """Verdict for a single failure event.

    Attributes:
        failure_class: The taxonomy bucket.
        confidence: 0.0–1.0; downstream code can use a threshold to
            decide whether to trust an automatic classification or fall
            back to manual review.
        matched_pattern: The substring / regex that decided the verdict,
            truncated to 240 chars. ``None`` for default fall-throughs.
        reason: Human-readable explanation, suitable for logging.
    """

    failure_class: FailureClass
    confidence: float = 0.0
    matched_pattern: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_class": self.failure_class.value,
            "confidence": round(float(self.confidence), 4),
            "matched_pattern": self.matched_pattern,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Pattern library
# ---------------------------------------------------------------------------
#
# Patterns are stored as lowercase substring or compiled regex matchers so we
# can answer "did this stderr say X" once on a single lowercased copy of the
# input. Order matters in ``classify``: more specific patterns are checked
# first so we don't, say, flag "TimeoutError" inside a cleanup teardown as
# the *primary* failure.

_NETWORK_SUBSTRINGS: tuple[str, ...] = (
    # Harvested from commit0_benchmark._classify_prepare_error
    "could not resolve host",
    "connection reset",
    "connection refused",
    "operation timed out",  # network-level, distinct from wall-clock
    "early eof",
    "rpc failed",
    "ssl_read",
    "proxy connect",
    # General curl/network signatures
    "the requested url returned error: 5",  # 5xx from package index
    "temporary failure in name resolution",
    "name or service not known",
    "network is unreachable",
    "no route to host",
    "dns resolution failed",
    "failed to connect to",
    # git clone specific (commit0_benchmark)
    "fatal: unable to access",
    "remote end hung up unexpectedly",
)

# git clone failures need both "git clone" AND a network signature in the
# original classifier. We keep that same conjunction here to avoid flagging
# "git clone succeeded" in passing log lines.
_NETWORK_GIT_CLONE_REQUIRED = "git clone"

_INSTALL_SUBSTRINGS: tuple[str, ...] = (
    # Harvested from commit0_benchmark._classify_prepare_error
    "build backend returned an error",
    "pep 517",
    "setuptools.build_meta",
    "metadata-generation-failed",
    "pre_install",
    "pre-install",
    # pip / apt classics
    "error: failed building wheel",
    "error: could not build wheels",
    "error: subprocess-exited-with-error",
    "error: command 'gcc' failed",
    "error: command 'cc' failed",
    "fatal error: python.h: no such file",
    "unable to locate package",
    "e: package",
    "command not found",
    "no such file or directory: 'apt-get'",
    "dpkg: error processing",
)

# pip-style "ERROR: ... pip install" pattern lives in regex form because
# the commit0 trace surfaces lines like
#     ERROR: Could not install packages due to an OSError: ...
# without the canonical "pip install failed" string.
_INSTALL_REGEXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\berror:\s+could not install packages\b", re.IGNORECASE),
    re.compile(r"\bpip\b[^\n]*\binstall\b[^\n]*\b(failed|error)\b", re.IGNORECASE),
)

_TIMEOUT_SUBSTRINGS: tuple[str, ...] = (
    # Harvested from commit0_benchmark._classify_prepare_error
    "timed out after",
    "timeoutexpired",
    # General wall-clock signatures
    "wall-clock timeout",
    "killed by signal 9 after",
    "deadline exceeded",
    "command timed out",
)

_TIMEOUT_REGEXES: tuple[re.Pattern[str], ...] = (
    # ``subprocess.TimeoutExpired: Command 'pytest' timed out after 1800 seconds``
    re.compile(r"\bsubprocess\.timeoutexpired\b", re.IGNORECASE),
    # bash 124 sentinel (``timeout`` coreutil) — only if returncode also matches.
    # The returncode check is in ``classify``; the regex catches the textual form.
    re.compile(r"\btimeout:\s+sending\s+signal\b", re.IGNORECASE),
)

_RESOURCE_SUBSTRINGS: tuple[str, ...] = (
    # Harvested from evaluation/failure_classifier
    "no space left on device",
    # OOM signatures
    "oom-killed",
    "oom_killed",
    "killed",  # ambiguous; the regex below scopes it to "oom" context
    "memoryerror",
    "cannot allocate memory",
    "out of memory",
    "container_oom",
    "exit code 137",  # OOM-killed sentinel from runc
    "disk quota exceeded",
)

_RESOURCE_REGEXES: tuple[re.Pattern[str], ...] = (
    # ``Killed`` alone is too noisy; require an OOM context.
    re.compile(r"\bkilled\b[^\n]{0,80}\b(oom|memory|out\s+of\s+memory)\b", re.IGNORECASE),
    re.compile(r"\b(oom[\-_]?killed|oomkilled)\b", re.IGNORECASE),
)

# Target-runtime control-plane and agentic-CLI auth signatures. These are
# unambiguous infra facts (the container/daemon failed, or the CLI was never
# logged in), so they fire regardless of phase and route to ENV_RUNTIME rather
# than being charged to APEX. Kept general: no benchmark/repo/vendor specifics
# beyond the docker/OCI and CLI-login wording the providers actually emit.
_RUNTIME_SUBSTRINGS: tuple[str, ...] = (
    # Docker / OCI control-plane loss (container reaped mid-run, daemon down).
    "error response from daemon",
    "no such container",
    "cannot connect to the docker daemon",
    "is the docker daemon running",
    "oci runtime exec failed",
    "oci runtime create failed",
    "containerd: ",
    # Agentic-CLI backend not authenticated in the target runtime.
    "not logged in",
    "please run /login",
    "please run `/login`",
    "target-container auth probe failed",
    "please set an auth method",
    "invalid api key",
    "authentication_error",
    "401 unauthorized",
)

_HARNESS_BUG_SUBSTRINGS: tuple[str, ...] = (
    # The canonical commit0 generate_report.py crash referenced in the spec.
    "keyerror: 'baseline_covs'",
    'keyerror: "baseline_covs"',
    "generate_report.py",
    "run_evaluation.py",
    "_apex_run_expected_ids.py",
    "_apex_expected_ids_filter.py",
    "module 'pytest' has no attribute 'stashkey'",
    'module "pytest" has no attribute "stashkey"',
    # pytest-internal traceback (NOT a test failure)
    "internal error",
    "internalerror",
    "pytest: error:",
    "pytest-internalerror",
    # Harness-side asserts
    "assertionerror in harness",
)

_HARNESS_BUG_REGEXES: tuple[re.Pattern[str], ...] = (
    # ``KeyError(<anything>baseline_covs<anything>)`` — defensive for quote style
    re.compile(r"keyerror[\s:\(]+['\"]?[^'\"]*baseline_covs", re.IGNORECASE),
    # pytest's own "INTERNALERROR>" prefix on an exception line
    re.compile(r"^\s*internalerror>", re.IGNORECASE | re.MULTILINE),
)

# Patterns that, in ``test_execution`` phase only, indicate that the LLM's
# patch broke something. These same patterns in ``pre_install`` / ``install``
# phase indicate an env problem instead.
_AMBIGUOUS_PHASE_DEPENDENT_SUBSTRINGS: tuple[str, ...] = (
    "modulenotfounderror",
    "no module named",
    "importerror",
    "cannot import name",
)

# These are unambiguous APEX miss signatures regardless of phase, harvested
# from evaluation/failure_classifier.py and quick_verify_failure_classifier.
_APEX_MISS_SUBSTRINGS: tuple[str, ...] = (
    "syntaxerror",
    "indentationerror",
    "taberror",
    "unterminated string literal",
    "unexpected indent",
    "assertionerror",
    "did not raise",
    "wrong exception",
)


@dataclass
class FailureClassifier:
    """Phase-aware classifier producing a single ``ClassificationResult``.

    Stateless; safe to instantiate once and reuse. Field defaults exist
    only so callers can override pattern lists for tests if needed.
    """

    network_substrings: tuple[str, ...] = field(default=_NETWORK_SUBSTRINGS)
    install_substrings: tuple[str, ...] = field(default=_INSTALL_SUBSTRINGS)
    install_regexes: tuple[re.Pattern[str], ...] = field(default=_INSTALL_REGEXES)
    timeout_substrings: tuple[str, ...] = field(default=_TIMEOUT_SUBSTRINGS)
    timeout_regexes: tuple[re.Pattern[str], ...] = field(default=_TIMEOUT_REGEXES)
    resource_substrings: tuple[str, ...] = field(default=_RESOURCE_SUBSTRINGS)
    resource_regexes: tuple[re.Pattern[str], ...] = field(default=_RESOURCE_REGEXES)
    runtime_substrings: tuple[str, ...] = field(default=_RUNTIME_SUBSTRINGS)
    harness_substrings: tuple[str, ...] = field(default=_HARNESS_BUG_SUBSTRINGS)
    harness_regexes: tuple[re.Pattern[str], ...] = field(default=_HARNESS_BUG_REGEXES)
    apex_miss_substrings: tuple[str, ...] = field(default=_APEX_MISS_SUBSTRINGS)
    ambiguous_substrings: tuple[str, ...] = field(default=_AMBIGUOUS_PHASE_DEPENDENT_SUBSTRINGS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify(
        self,
        stderr: str,
        stdout: str = "",
        returncode: int = 1,
        context: Optional[dict[str, Any]] = None,
    ) -> ClassificationResult:
        """Classify a single failure event.

        Args:
            stderr: Captured stderr from the failing process. May be empty.
            stdout: Captured stdout. Sometimes Python tracebacks land here.
            returncode: Process exit code. ``0`` is treated as
                ``UNCLASSIFIED`` regardless of stderr text — a successful
                process is not a failure to classify.
            context: Optional disambiguation hints. Recognised keys:
                * ``phase``: one of ``"clone"``, ``"pre_install"``,
                  ``"install"``, ``"baseline"``, ``"test_execution"``,
                  ``"scoring"``. Drives ambiguous-pattern routing.
                * ``timed_out``: bool; if True, classification short-
                  circuits to ``ENV_TIMEOUT`` regardless of stderr.

        Returns:
            ``ClassificationResult``. Always returns; never raises.
        """
        ctx = dict(context or {})
        phase = str(ctx.get("phase") or "").strip().lower()
        # An honestly-zero returncode is not a failure.
        if int(returncode or 0) == 0 and not bool(ctx.get("timed_out")):
            return ClassificationResult(
                failure_class=FailureClass.UNCLASSIFIED,
                confidence=1.0,
                matched_pattern=None,
                reason="returncode=0; no failure to classify",
            )

        # ``timeout`` coreutil sentinel + caller-asserted timeouts are
        # the highest-confidence env signal we have.
        if int(returncode or 0) == 124 or bool(ctx.get("timed_out")):
            return ClassificationResult(
                failure_class=FailureClass.ENV_TIMEOUT,
                confidence=1.0,
                matched_pattern=(
                    "context.timed_out=true" if ctx.get("timed_out") else "returncode=124"
                ),
                reason="caller-asserted or coreutil-timeout sentinel",
            )

        merged = (str(stderr or "") + "\n" + str(stdout or "")).lower()

        # --------------------------------------------------------------
        # Phase 0: ENV_RUNTIME. Docker/OCI control-plane loss and agentic-CLI
        # auth failures are unambiguous infra facts and can co-occur with
        # downstream collection/coverage noise that would otherwise look like an
        # APEX miss, so they are resolved first. A reaped container or an
        # unauthenticated backend is never a wrong patch.
        # --------------------------------------------------------------
        match = _first_substring_match(merged, self.runtime_substrings)
        if match is not None:
            return ClassificationResult(
                failure_class=FailureClass.ENV_RUNTIME,
                confidence=0.9,
                matched_pattern=_truncate(match),
                reason="target-runtime/control-plane or agent-auth failure",
            )

        # --------------------------------------------------------------
        # Phase 1: HARNESS_BUG. We check this first because a harness
        # crash often dumps both APEX_MISS-looking and ENV-looking
        # tracebacks in a single payload, but the *primary* failure is
        # the harness having keeled over.
        # --------------------------------------------------------------
        match = _first_substring_match(merged, self.harness_substrings)
        if match is not None:
            return ClassificationResult(
                failure_class=FailureClass.HARNESS_BUG,
                confidence=0.9,
                matched_pattern=_truncate(match),
                reason="upstream harness/runner crashed before scoring APEX",
            )
        regex_match = _first_regex_match(merged, self.harness_regexes)
        if regex_match is not None:
            return ClassificationResult(
                failure_class=FailureClass.HARNESS_BUG,
                confidence=0.9,
                matched_pattern=_truncate(regex_match.group(0)),
                reason="upstream harness/runner crashed before scoring APEX",
            )

        # --------------------------------------------------------------
        # Phase 2: ENV_TIMEOUT (textual). Returncode 124 already handled
        # above; this catches "subprocess.TimeoutExpired" tracebacks
        # surfaced by the prepare-step exception classifier.
        # --------------------------------------------------------------
        match = _first_substring_match(merged, self.timeout_substrings)
        if match is not None:
            return ClassificationResult(
                failure_class=FailureClass.ENV_TIMEOUT,
                confidence=0.85,
                matched_pattern=_truncate(match),
                reason="wall-clock timeout signature in process output",
            )
        regex_match = _first_regex_match(merged, self.timeout_regexes)
        if regex_match is not None:
            return ClassificationResult(
                failure_class=FailureClass.ENV_TIMEOUT,
                confidence=0.85,
                matched_pattern=_truncate(regex_match.group(0)),
                reason="wall-clock timeout signature in process output",
            )

        # --------------------------------------------------------------
        # Phase 3: ENV_RESOURCE. OOM-killed containers exit 137; the
        # textual signature is preferred but we honour the exit code as
        # a soft signal.
        # --------------------------------------------------------------
        regex_match = _first_regex_match(merged, self.resource_regexes)
        if regex_match is not None:
            return ClassificationResult(
                failure_class=FailureClass.ENV_RESOURCE,
                confidence=0.85,
                matched_pattern=_truncate(regex_match.group(0)),
                reason="OOM / disk-full / resource exhaustion signature",
            )
        # Substring match. We deliberately exclude bare "killed" because
        # it's far too noisy (test names, log lines etc.). The regex
        # above handles the OOM-scoped "killed" case.
        for substring in self.resource_substrings:
            if substring == "killed":
                continue
            if substring in merged:
                return ClassificationResult(
                    failure_class=FailureClass.ENV_RESOURCE,
                    confidence=0.85,
                    matched_pattern=_truncate(substring),
                    reason="OOM / disk-full / resource exhaustion signature",
                )
        if int(returncode or 0) == 137:
            return ClassificationResult(
                failure_class=FailureClass.ENV_RESOURCE,
                confidence=0.7,
                matched_pattern="returncode=137",
                reason="exit code 137 typically indicates OOM-kill",
            )

        # --------------------------------------------------------------
        # Phase 4: ENV_NETWORK. Most network signatures are unambiguous,
        # but the original commit0 classifier conjuncts "git clone" with
        # the network token to avoid flagging downloads in benign log
        # lines. We honour that conjunction when the phase is "clone".
        # --------------------------------------------------------------
        match = _first_substring_match(merged, self.network_substrings)
        if match is not None:
            requires_clone_conjunction = phase == "clone" or (_NETWORK_GIT_CLONE_REQUIRED in merged)
            if requires_clone_conjunction or _looks_unambiguously_network(match):
                return ClassificationResult(
                    failure_class=FailureClass.ENV_NETWORK,
                    confidence=0.9,
                    matched_pattern=_truncate(match),
                    reason="network unreachable / DNS / clone failure",
                )

        # --------------------------------------------------------------
        # Phase 5: ENV_INSTALL. Build-backend / setup.py errors land
        # here. We check this BEFORE the ambiguous ModuleNotFoundError
        # branch because PEP 517 failures often *also* contain an
        # ImportError tail — but the install was the proximate cause.
        # --------------------------------------------------------------
        match = _first_substring_match(merged, self.install_substrings)
        if match is not None:
            return ClassificationResult(
                failure_class=FailureClass.ENV_INSTALL,
                confidence=0.85,
                matched_pattern=_truncate(match),
                reason="package install / build-backend failure",
            )
        regex_match = _first_regex_match(merged, self.install_regexes)
        if regex_match is not None:
            return ClassificationResult(
                failure_class=FailureClass.ENV_INSTALL,
                confidence=0.85,
                matched_pattern=_truncate(regex_match.group(0)),
                reason="package install / build-backend failure",
            )

        # --------------------------------------------------------------
        # Phase 6: phase-dependent ambiguity (the user-spec'd
        # ModuleNotFoundError disambiguation).
        # --------------------------------------------------------------
        match = _first_substring_match(merged, self.ambiguous_substrings)
        if match is not None:
            if phase in {"pre_install", "install", "clone", "baseline"}:
                return ClassificationResult(
                    failure_class=FailureClass.ENV_INSTALL,
                    confidence=0.75,
                    matched_pattern=_truncate(match),
                    reason=(
                        f"import-failure during phase={phase!r} — install "
                        "didn't provide a required module"
                    ),
                )
            if phase in {"test_execution", "scoring", ""}:
                # The empty-phase case defaults to APEX_MISS because the
                # historical commit0 path treats unscoped import failures
                # as patch problems. Callers should pass phase explicitly
                # when they can.
                return ClassificationResult(
                    failure_class=FailureClass.APEX_MISS,
                    confidence=0.75 if phase else 0.55,
                    matched_pattern=_truncate(match),
                    reason=(
                        f"import-failure during phase={phase or 'unknown'!r} — "
                        "patch likely broke an import or removed a symbol"
                    ),
                )

        # --------------------------------------------------------------
        # Phase 7: unambiguous APEX miss signatures.
        # --------------------------------------------------------------
        match = _first_substring_match(merged, self.apex_miss_substrings)
        if match is not None:
            return ClassificationResult(
                failure_class=FailureClass.APEX_MISS,
                confidence=0.9,
                matched_pattern=_truncate(match),
                reason="syntax / assertion / wrong-exception in patched code",
            )

        # --------------------------------------------------------------
        # Phase 8: nothing matched. Default to UNCLASSIFIED so callers
        # can apply their own policy (commit0 currently charges to APEX;
        # testgen retries the env once first).
        # --------------------------------------------------------------
        return ClassificationResult(
            failure_class=FailureClass.UNCLASSIFIED,
            confidence=0.2,
            matched_pattern=None,
            reason=f"no recognised pattern; returncode={returncode}",
        )


# ---------------------------------------------------------------------------
# Convenience top-level function
# ---------------------------------------------------------------------------

_DEFAULT_CLASSIFIER = FailureClassifier()


def classify_failure(
    stderr: str,
    stdout: str = "",
    returncode: int = 1,
    context: Optional[dict[str, Any]] = None,
) -> ClassificationResult:
    """Module-level convenience: ``FailureClassifier().classify(...)``."""
    return _DEFAULT_CLASSIFIER.classify(
        stderr=stderr,
        stdout=stdout,
        returncode=returncode,
        context=context,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_substring_match(haystack: str, needles: Iterable[str]) -> Optional[str]:
    for needle in needles:
        if needle and needle in haystack:
            return needle
    return None


def _first_regex_match(
    haystack: str,
    patterns: Iterable[re.Pattern[str]],
) -> Optional[re.Match[str]]:
    for pattern in patterns:
        match = pattern.search(haystack)
        if match is not None:
            return match
    return None


def _truncate(text: str, limit: int = 240) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _looks_unambiguously_network(needle: str) -> bool:
    """True if a network substring is specific enough to fire without
    a "git clone" conjunction.

    Things like "could not resolve host" / "temporary failure in name
    resolution" / "no route to host" are never ambiguous. Things like
    "operation timed out" CAN appear in non-network contexts, so we
    require the clone conjunction for them.
    """
    safe = {
        "could not resolve host",
        "temporary failure in name resolution",
        "name or service not known",
        "network is unreachable",
        "no route to host",
        "dns resolution failed",
        "fatal: unable to access",
        "remote end hung up unexpectedly",
        "ssl_read",
        "proxy connect",
        "the requested url returned error: 5",
    }
    return needle in safe


__all__ = [
    "FailureClass",
    "ClassificationResult",
    "FailureClassifier",
    "classify_failure",
]
