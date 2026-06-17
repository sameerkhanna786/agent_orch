"""Classify quick-verification test-run failures into actionable buckets.

Today, when a rollout's quick-verification pytest run fails, the agent
receives the entire (often multi-MB) traceback dump and burns iterations
trying to fix the wrong layer of problem. The Apr 27 validate run showed
the same `unexpected indent` syntax error reported through 8 successive
verifier passes — the orchestrator had no way to tell the agent "this
is a syntax error, fix it, the test logic is fine."

This module looks at the merged stdout/stderr from a pytest invocation
plus its returncode and reports a single failure class:

    "ok"          : returncode == 0 with no FAILED/ERROR markers
    "timeout"     : returncode == 124 (the verifier's timeout sentinel)
    "syntax"      : SyntaxError / IndentationError / TabError. Includes a
                    (file, line, excerpt) triple the agent can act on.
    "env"         : ModuleNotFoundError / dependency / pip / command-not-found.
    "collection"  : pytest exit code 5 (no tests collected), "ERROR
                    collecting" / "ERRORS" headers, or a local source
                    import-time contract failure.
    "oracle"      : AssertionError-class failures — the only class where the
                    agent should iterate on test logic / oracle definitions.
    "runtime"     : Anything else with a FAILED outcome — runtime exceptions
                    inside test bodies, fixture exceptions, etc.

The classifier is defensive: a low-confidence match falls through to
"runtime" rather than misclassifying. The `primary_signal` field carries
the matched substring so the orchestrator can surface it to the agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class QuickVerifyFailureClass:
    """The failure-class verdict for a single quick-verification run."""

    label: str
    confidence: float  # 0.0 — 1.0
    primary_signal: str  # the substring or returncode that decided the label
    syntax_error_file: Optional[str] = None
    syntax_error_line: Optional[int] = None
    syntax_error_excerpt: Optional[str] = None  # ~5 lines around the error

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "primary_signal": self.primary_signal[:240],
            "syntax_error_file": self.syntax_error_file,
            "syntax_error_line": self.syntax_error_line,
            "syntax_error_excerpt": self.syntax_error_excerpt,
        }


# ``pytest`` and the python interpreter both emit syntax errors with the
# canonical ``  File "path", line N`` header. Match either form.
_SYNTAX_HEADER_RE = re.compile(
    r'(?:^|\n)\s*File "(?P<file>[^"]+)",\s*line\s*(?P<line>\d+)',
    re.MULTILINE,
)
_SYNTAX_BODY_RE = re.compile(
    r"\b(?P<kind>SyntaxError|IndentationError|TabError):\s*(?P<message>[^\n]+)"
)
# Pytest collection-error short form: "tests/test_x.py:1: in <module>"
_COLLECTION_FILE_LINE_RE = re.compile(
    r"\n(?P<file>[\w./\-]+\.py):(?P<line>\d+):\s*in\s+",
)
_ENV_PATTERNS = (
    re.compile(r"\bModuleNotFoundError\b"),
    re.compile(r"\bImportError\b"),
    re.compile(r"No module named\b"),
    re.compile(r"pip install\b"),
    re.compile(r"command not found\b"),
    re.compile(r"\bDistributionNotFound\b"),
    re.compile(r"VersionConflict\b"),
    re.compile(r"rate limit exceeded", re.IGNORECASE),
    re.compile(
        r"\brequests\.exceptions\.(?:HTTPError|ConnectionError|Timeout|ReadTimeout|ConnectTimeout)\b"
    ),
    re.compile(r"\b(?:HTTPError|ConnectionError|ReadTimeout|ConnectTimeout):\s", re.IGNORECASE),
    re.compile(r"could not resolve host", re.IGNORECASE),
    re.compile(r"temporary failure in name resolution", re.IGNORECASE),
    re.compile(r"network is unreachable", re.IGNORECASE),
    re.compile(r"connection (?:refused|reset|aborted)", re.IGNORECASE),
    re.compile(r"max retries exceeded with url", re.IGNORECASE),
    re.compile(r"\bNameResolutionError\b"),
    re.compile(r"\bRemoteDisconnected\b"),
    re.compile(r"\bNo such container\b", re.IGNORECASE),
    re.compile(r"cannot connect to the docker daemon", re.IGNORECASE),
    re.compile(r"container .* is not running", re.IGNORECASE),
    re.compile(r"\bOCI runtime exec failed\b", re.IGNORECASE),
    re.compile(r"docker:\s*Error response from daemon", re.IGNORECASE),
    re.compile(r"Error response from daemon", re.IGNORECASE),
)
_IMPORT_ERROR_WITH_PY_PATH_RE = re.compile(
    r"\bImportError:\s+[^\n]*\((?P<path>[^)\n]+\.py)\)"
)
_SOURCE_COLLECTION_EXCEPTION_RE = re.compile(
    r"\b(?P<kind>AttributeError|NameError|TypeError|ValueError|RecursionError|NotImplementedError):\s*(?P<message>[^\n]+)"
)
_EXTERNAL_IMPORT_PATH_PARTS = (
    "/site-packages/",
    "/dist-packages/",
    "/.venv/",
    "/venv/",
    "/lib/python",
    "\\site-packages\\",
    "\\dist-packages\\",
    "\\.venv\\",
    "\\venv\\",
    "\\lib\\python",
)
_COLLECTION_PATTERNS = (
    re.compile(r"\bERROR collecting\b"),
    re.compile(r"^ERRORS\b", re.MULTILINE),
    re.compile(r"no tests collected", re.IGNORECASE),
)
_ASSERTION_PATTERNS = (
    re.compile(r"\bAssertionError\b"),
    re.compile(r"^E\s+assert\b", re.MULTILINE),
    re.compile(r"^E\s+\+\s+where\b", re.MULTILINE),  # pytest's assert-rewrite
)
_FAILED_OUTCOME_RE = re.compile(r"\bFAILED\b|\bERROR\b")
# Launch/exec failure: the OS or shell could not even START the test process
# (classically E2BIG -> "Argument list too long" when too many test node-ids are
# passed on argv). This is a HARNESS fault, not a code fault. A distinct
# ``harness`` label (NOT ``env``) keeps env-class relaunch routing from being
# silently reused; the engine treats it as INDETERMINATE only when no tests were
# collected, so a real test that merely prints the phrase in an assertion body
# (which still collects + runs) is never laundered into indeterminate. The
# exec/shell-prefixed forms are the strongest anchors (a test body never emits
# them); the bare forms are kept because some kernels/shells print only those.
# Strong markers: an exec/shell launch line that a test body never emits.
_HARNESS_LAUNCH_STRONG_PATTERNS = (
    re.compile(r"exec [^\n:]*:\s*argument list too long", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(?:/bin/)?(?:ba)?sh:.*argument list too long", re.IGNORECASE),
    re.compile(r"\bE2BIG\b"),
)
# Weak marker: the bare phrase can appear inside a real test's assertion text, so
# treat it as a launch failure ONLY when no test actually reported FAILED/ERROR
# (a genuine assertion that prints the phrase still has a FAILED marker).
_HARNESS_LAUNCH_WEAK_PATTERN = re.compile(r"\bArgument list too long\b", re.IGNORECASE)


def classify_quick_verification_failure(
    *,
    output: str,
    returncode: int,
) -> QuickVerifyFailureClass:
    """Classify a single quick-verification result."""
    text = output or ""

    if returncode == 0 and not _FAILED_OUTCOME_RE.search(text):
        return QuickVerifyFailureClass(
            label="ok",
            confidence=1.0,
            primary_signal="returncode=0",
        )
    if returncode == 124:
        return QuickVerifyFailureClass(
            label="timeout",
            confidence=1.0,
            primary_signal="returncode=124",
        )

    # Launch/exec failure (E2BIG / "Argument list too long") -> harness, not code.
    # Checked before syntax/env so an un-launchable run is never mistaken for a
    # source defect. The engine's collected==0 guard is the second safety layer.
    for pattern in _HARNESS_LAUNCH_STRONG_PATTERNS:
        match = pattern.search(text)
        if match:
            return QuickVerifyFailureClass(
                label="harness",
                confidence=0.9,
                primary_signal=match.group(0)[:240],
            )
    weak_launch = _HARNESS_LAUNCH_WEAK_PATTERN.search(text)
    if weak_launch and not _FAILED_OUTCOME_RE.search(text):
        return QuickVerifyFailureClass(
            label="harness",
            confidence=0.8,
            primary_signal=weak_launch.group(0)[:240],
        )

    syntax_body = _SYNTAX_BODY_RE.search(text)
    if syntax_body:
        # Locate the matching `File "..."` header (may appear above the
        # body, may not — we walk back through the most recent header).
        header = _last_match_before(_SYNTAX_HEADER_RE, text, syntax_body.start())
        if header is None:
            header = _COLLECTION_FILE_LINE_RE.search(text)
        if header is not None:
            file_name = header.group("file")
            try:
                line_number = int(header.group("line"))
            except (TypeError, ValueError):
                line_number = None
        else:
            file_name = None
            line_number = None
        excerpt = _excerpt_around(text, anchor_index=syntax_body.start(), context_lines=4)
        return QuickVerifyFailureClass(
            label="syntax",
            confidence=0.95,
            primary_signal=syntax_body.group(0)[:240],
            syntax_error_file=file_name,
            syntax_error_line=line_number,
            syntax_error_excerpt=excerpt,
        )

    if returncode == 5:
        return QuickVerifyFailureClass(
            label="collection",
            confidence=1.0,
            primary_signal="pytest_returncode=5",
        )

    source_import_error = _source_import_error_match(text)
    if source_import_error:
        return QuickVerifyFailureClass(
            label="collection",
            confidence=0.9,
            primary_signal=source_import_error.group(0)[:240],
        )

    source_collection_exception = _source_collection_exception_signal(text)
    if source_collection_exception:
        return QuickVerifyFailureClass(
            label="collection",
            confidence=0.9,
            primary_signal=source_collection_exception[:240],
        )

    for pattern in _ENV_PATTERNS:
        match = pattern.search(text)
        if match:
            return QuickVerifyFailureClass(
                label="env",
                confidence=0.9,
                primary_signal=match.group(0)[:240],
            )

    for pattern in _COLLECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return QuickVerifyFailureClass(
                label="collection",
                confidence=0.85,
                primary_signal=match.group(0)[:240],
            )

    for pattern in _ASSERTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return QuickVerifyFailureClass(
                label="oracle",
                confidence=0.85,
                primary_signal=match.group(0)[:240],
            )

    if _FAILED_OUTCOME_RE.search(text):
        return QuickVerifyFailureClass(
            label="runtime",
            confidence=0.6,
            primary_signal="generic_failed_marker",
        )

    # Non-zero returncode with no recognized signature — the safest report
    # is "runtime" with low confidence so the agent doesn't get a
    # misleading actionable diagnostic.
    return QuickVerifyFailureClass(
        label="runtime",
        confidence=0.3,
        primary_signal=f"returncode={returncode}",
    )


def _source_import_error_match(text: str) -> Optional[re.Match]:
    for match in _IMPORT_ERROR_WITH_PY_PATH_RE.finditer(text or ""):
        path = str(match.group("path") or "").strip()
        if _looks_like_candidate_source_path(path):
            return match
    return None


def _source_collection_exception_signal(text: str) -> Optional[str]:
    """Return a source-path collection exception before incidental env noise.

    Large pytest-xdist runs can concatenate many collection traces. A real
    source import-time AttributeError near a candidate file should not be
    reclassified as an environment problem merely because another worker later
    emitted a ModuleNotFoundError line.
    """

    lines = str(text or "").splitlines()
    for index, line in enumerate(lines):
        exception = _SOURCE_COLLECTION_EXCEPTION_RE.search(line)
        if exception is None:
            continue
        candidates: list[tuple[str, str]] = []
        for prior in range(max(0, index - 10), index + 1):
            location = _COLLECTION_FILE_LINE_RE.search("\n" + lines[prior])
            if location is None:
                continue
            path = str(location.group("file") or "").strip()
            if _looks_like_candidate_source_path(path):
                candidates.append((path, str(location.group("line") or "")))
        if not candidates:
            continue
        source_candidates = [
            (path, line_number)
            for path, line_number in candidates
            if not _looks_like_test_file_path(path)
        ]
        path, line_number = (source_candidates or candidates)[-1]
        return f"{exception.group('kind')}: {exception.group('message')} @ {path}:{line_number}"
    return None


def _looks_like_test_file_path(path: str) -> bool:
    normalized = str(path or "").replace("\\", "/")
    parts = tuple(part.lower() for part in normalized.split("/") if part)
    name = parts[-1] if parts else ""
    return (
        "test" in parts
        or "tests" in parts
        or "__tests__" in parts
        or name == "conftest.py"
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def _looks_like_candidate_source_path(path: str) -> bool:
    normalized = str(path or "").replace("\\", "/")
    lowered = normalized.lower()
    if not normalized.endswith(".py"):
        return False
    if any(part in lowered for part in _EXTERNAL_IMPORT_PATH_PARTS):
        return False
    if lowered.startswith(("/usr/", "/opt/", "/root/.local/")) and "/workspace/" not in lowered:
        return False
    return True


def _last_match_before(
    pattern: re.Pattern,
    text: str,
    end_index: int,
) -> Optional[re.Match]:
    last: Optional[re.Match] = None
    for match in pattern.finditer(text, 0, end_index):
        last = match
    return last


def _excerpt_around(text: str, *, anchor_index: int, context_lines: int) -> str:
    """Return ~`context_lines` lines on either side of `anchor_index`."""
    if not text:
        return ""
    line_starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            line_starts.append(i + 1)
    anchor_line = 0
    for i, start in enumerate(line_starts):
        if start > anchor_index:
            anchor_line = i - 1
            break
    else:
        anchor_line = len(line_starts) - 1
    first_line = max(0, anchor_line - context_lines)
    last_line = min(len(line_starts) - 1, anchor_line + context_lines)
    start_offset = line_starts[first_line]
    end_offset = line_starts[last_line + 1] if last_line + 1 < len(line_starts) else len(text)
    return text[start_offset:end_offset].rstrip("\n")
