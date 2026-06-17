"""Default LLM-backed test/code generators for the apex.modes API.

Phase I.8 — make the public modes API actually usable out of the box.
Previously, ``run_*_*`` functions defaulted to no-op generators that
returned ``[]`` / ``None`` with a warning, forcing every caller to
wire their own LLM-backed callable. This module provides:

    default_test_generator(repo_path, problem_statement, *, config=None)
    default_code_generator(repo_path, problem_statement, test_artifacts, *, config=None)

Both default to the strongest configuration (Codex CLI + gpt-5.5)
per the project directive ("never reduce model size / power"). The
caller can override by passing ``config=ApexConfig(...)``.

Defensive about the obvious failure modes:
    * CLI tool not installed → returns empty + clear error in logs.
    * Schema parse failure → retries (CLI backend already retries 2x).
    * Subprocess crash → returns empty / None and logs the trace.

These defaults are NOT a replacement for the full APEX orchestrator
pipeline (which runs multi-stage with reproducer / localizer /
test_writer / patcher etc.). They're the simplest one-shot LLM call
that produces a real artifact, suitable for IDE plugins / CI gates
that don't need the full ensemble. Callers wanting maximum quality
should pass ``test_generator`` / ``code_generator`` callables that
wrap their own orchestrator setup.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from .core.generated_tests import safe_materialize_test_artifact

logger = logging.getLogger(__name__)


# 10.R: forbidden absolute-path / double-separator patterns that, when
# embedded in user-facing prompts, cause agents to re-inject the host path
# into pytest invocations and produce
# `ImportError while loading conftest '/data/users/...'`-style failures.
# Detected on either POSIX or Windows-style separators; checked with simple
# substring and regex matchers (no glob escapes needed).
_FORBIDDEN_PROMPT_PATTERNS: tuple[str, ...] = (
    "/tmp/",
    "/data/users/",
    "/home/",
    "/Users/",
)
# Double-separator detection — guards against accidental
# os.path.join("/foo", "/bar") -> "/foo//bar" and similar.
_DOUBLE_SEP_PATTERN = re.compile(r"(?<!:)//")


def _lint_prompt_for_absolute_paths(prompt: str, *, context: str = "") -> list[str]:
    """Scan ``prompt`` for forbidden absolute-path / double-separator patterns.

    Returns the list of offending substrings (empty if clean). Logs a warning
    when violations are found — production runs are NOT blocked, since some
    tasks may legitimately mention paths in problem statements; the warning
    surfaces the issue in logs so we can iterate on prompt construction.

    The companion fix (passing relative paths instead of absolute ones in the
    APEX-built portion of the prompt) is what eliminates the bulk of these
    findings; this lint catches anything we missed.
    """
    if not isinstance(prompt, str) or not prompt:
        return []
    findings: list[str] = []
    for pat in _FORBIDDEN_PROMPT_PATTERNS:
        if pat in prompt:
            findings.append(pat)
    if _DOUBLE_SEP_PATTERN.search(prompt):
        findings.append("//")
    if findings:
        logger.warning(
            "prompt_lint: absolute_path_patterns_detected context=%r patterns=%r",
            context or "default",
            findings,
        )
    return findings


# JSON schema enforcing the {test_artifacts: [{path, content}, ...]}
# shape so the agent's response is directly usable. Per "no cost
# reduction", we ask for an unbounded number of artifacts and rich
# design metadata so downstream feedback (axis coverage, F2P, mutation,
# coverage gaps, and minimization) has signal to work with. The previous
# maxItems=10 cap silently truncated large axis-coverage portfolios on
# the way out of the agent — Phase 4A item 4.6 removed it. Operators
# who want a per-invocation cap (e.g. for token-budget reasons) can set
# ``OrchestrationConfig.max_test_artifacts_per_invocation`` and the
# default generator will trim post-hoc with a logged warning.
_TEST_ARTIFACTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "test_artifacts": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Repo-relative test file path (e.g., tests/test_foo.py)",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full source of the test file. "
                        "Must be syntactically valid in the language.",
                    },
                    "strategy": {
                        "type": "string",
                        "enum": [
                            "regression",
                            "contract",
                            "edge",
                            "negative",
                            "property",
                            "metamorphic",
                            "differential",
                            "fuzz_seed",
                        ],
                    },
                    "summary": {"type": "string"},
                    "test_descriptions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "focus_files": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "contract_sources": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "issue",
                                "traceback",
                                "docs",
                                "examples",
                                "types",
                                "existing_tests",
                                "reproduction",
                                "localization",
                            ],
                        },
                    },
                    "contract_targets": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "contract_axes": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "positive_path",
                                "missing_boundary",
                                "negative_malformed",
                                "multi_ordering",
                                "property",
                                "metamorphic",
                                "differential",
                                "fuzz_seed",
                            ],
                        },
                        "description": "Which contract axes this test "
                        "exercises. At least one is required.",
                    },
                    "properties": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "metamorphic_relations": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "fuzz_seeds": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "justification": {"type": "string"},
                    "oracle_origin": {
                        "type": "string",
                        "enum": [
                            "issue",
                            "docs",
                            "existing_tests",
                            "pass_then_invert",
                            "differential",
                            "property",
                            "metamorphic",
                        ],
                    },
                    "pass_then_invert": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "attempted": {"type": "boolean"},
                            "status": {"type": "string"},
                            "passing_variant_summary": {"type": "string"},
                            "inversion_summary": {"type": "string"},
                            "execution_feedback_summary": {"type": "string"},
                        },
                    },
                    "dual_version_verified": {"type": "boolean"},
                },
                "required": [
                    "path",
                    "content",
                    "strategy",
                    "summary",
                    "contract_sources",
                    "contract_targets",
                    "contract_axes",
                    "oracle_origin",
                ],
            },
        },
    },
    "required": ["test_artifacts"],
}


_TEST_GENERATION_SYSTEM_PROMPT = (
    "You are APEX's default test_writer. Given a problem statement and "
    "a repository, produce a portfolio of test cases that catch the "
    "described bug. Tests must FAIL on the current (broken) repo state "
    "and PASS only after a correct fix is applied.\n\n"
    "Rules:\n"
    "  1. Use strict assertions: prefer == against exact expected "
    "values over presence-only checks.\n"
    "  2. Cover all four canonical contract axes: positive_path, "
    "missing_boundary, negative_malformed, multi_ordering. At least "
    "one test per axis when applicable.\n"
    "  3. Include property or metamorphic tests when the API shape "
    "supports invariants such as round-trip, idempotence, monotonicity, "
    "ordering preservation, subset filtering, or parser/serializer "
    "equivalence.\n"
    "  4. Prefer pass-then-invert when expected values are ambiguous: "
    "first identify current passing behavior, then state how the final "
    "oracle inverts it toward the desired contract.\n"
    "  5. Each test file must be self-contained and importable. "
    "Include path-relative imports if needed.\n"
    "  6. Follow the JSON schema EXACTLY. Do not emit prose."
)


# 10.D: default raised from 1200s to 7200s. Agentic CLI sessions with
# multi-turn debugging legitimately need >20 min; the previous cap fired on
# legitimate long-running tasks across every benchmark. Per-benchmark configs
# in configs/benchmark_*.json may still override.
_DEFAULT_LLM_HARD_TIMEOUT_SECONDS = 7200


def _clean_string_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _build_default_config() -> Any:
    """Lazy-import ApexConfig and return a sensible default for the
    modes API. Backend-specific permission defaults are resolved by the
    CLI backend so heterogeneous agent configs never inherit stale flags."""
    from .core.config import ApexConfig, LLMBackend, LLMConfig

    return ApexConfig(
        llm_configs=[
            LLMConfig(
                backend=LLMBackend.CODEX_CLI,
                model="gpt-5.5",
                cli_timeout=1200,
                cli_disable_osx_sandbox=True,
                cli_permission_mode=None,
            )
        ],
    )


# Multi-agent ensemble for testgen candidates. Variety with agentic
# backends comes from running DIFFERENT agents (different training,
# blind spots, internal strategies) — not from sampling the same agent
# multiple times. Cf. TEX-T (Salesforce, 87% SWT-Bench Verified) and
# L*Agent v1 (LogicStar, 84%) which both use heterogeneous agent
# ensembles with cross-candidate selection.
#
# Decisive-Edge B.2: OpenCode-family agents stay in the mapping (callers that
# request them explicitly still resolve), but the default ensemble drops
# them because headline runs show they consistently underperform the
# strong three (codex / claude / gemini) and dilutes selection. Set
# ``RolloutConfig.allow_weak_models=True`` to opt back into the
# OpenCode-family ablation.
AGENT_NAME_TO_CONFIG: dict[str, dict[str, Any]] = {
    "codex": {"backend_name": "CODEX_CLI", "model": "gpt-5.5"},
    "claude": {"backend_name": "CLAUDE_CLI", "model": "opus"},
    "gemini": {"backend_name": "GEMINI_CLI", "model": "gemini-3.1-pro"},
    "opencode": {"backend_name": "OPENCODE_CLI", "model": "meta/avocado-tester"},
    "metacode": {"backend_name": "METACODE_CLI", "model": "meta/avocado-code-latest"},
}

# The strong three: agents that consistently rank in the top tier on
# Commit0-Lite / SWE-Bench Pro / SWT-Bench. Used by
# :func:`default_agent_names` as the default ensemble unless
# ``RolloutConfig.allow_weak_models`` is set.
_STRONG_AGENT_NAMES: tuple[str, ...] = ("codex", "claude", "gemini")
# Lower-performing / experimental agents kept out of defaults but available
# for explicit ablations.
_WEAK_AGENT_NAMES: tuple[str, ...] = ("opencode", "metacode")


def default_agent_names(*, allow_weak_models: bool = False) -> list[str]:
    """Return the default multi-agent ensemble.

    Decisive-Edge B.2: defaults to the strong three
    (``codex``, ``claude``, ``gemini``). When ``allow_weak_models`` is
    True, weaker agent CLIs (``opencode``, ``metacode``) are appended in their
    legacy slot order so callers can run OpenCode-family
    ablation by flipping a single flag.
    """
    if allow_weak_models:
        return list(_STRONG_AGENT_NAMES) + list(_WEAK_AGENT_NAMES)
    return list(_STRONG_AGENT_NAMES)


def build_agent_llm_config(agent_name: str) -> Any:
    """Construct an LLMConfig for a named agent.

    Used by multi-agent ensembles: caller can request
    ``agent_models=["codex", "claude", "gemini", "opencode", "metacode"]`` and the
    candidate pool will assign one config per agent. Each agent
    invocation is itself an internal agentic loop, so genuine output
    variety comes from cross-agent diversity, not from same-agent
    sampling.
    """

    from .core.config import LLMBackend, LLMConfig

    spec = AGENT_NAME_TO_CONFIG.get(agent_name.lower())
    if spec is None:
        raise ValueError(
            f"Unknown agent name {agent_name!r}. Supported: {sorted(AGENT_NAME_TO_CONFIG)}"
        )
    backend = getattr(LLMBackend, spec["backend_name"])
    return LLMConfig(
        backend=backend,
        model=spec["model"],
        cli_timeout=1200,
        cli_disable_osx_sandbox=True,
        cli_permission_mode=None,
    )


def build_agent_ensemble_config(agent_names: list[str]) -> Any:
    """Build an ApexConfig whose llm_configs is one entry per agent.

    The candidate pool maps candidate index → agent_names[i % len].
    Default callers pass a single name (unchanged behavior); ensemble
    callers pass several to get TEX-T-style diversity.
    """

    from .core.config import ApexConfig

    if not agent_names:
        return _build_default_config()
    return ApexConfig(
        llm_configs=[build_agent_llm_config(name) for name in agent_names],
    )


def default_test_generator(
    repo_path: Path,
    problem_statement: str,
    *,
    config: Optional[Any] = None,
) -> list[dict[str, Any]]:
    """LLM-backed default for ``apex.modes.run_*_*(test_generator=...)``.

    Issues a single structured prompt to Codex CLI (or whatever the
    config's first LLM is) asking for a JSON portfolio of test
    artifacts. Returns the parsed list, or ``[]`` if the call fails
    or the response can't be parsed (with a clear log entry).
    """
    try:
        from .core.cli_backend import CLIModelClient
    except Exception as exc:  # pragma: no cover — import-time failure
        logger.warning(
            "default_test_generator: cli_backend unavailable (%s); "
            "returning []. Pass a custom test_generator to bypass.",
            exc,
        )
        return []

    cfg = config or _build_default_config()
    llm_config = cfg.llm_configs[0] if cfg.llm_configs else None
    if llm_config is None or not llm_config.is_cli_backend:
        logger.warning(
            "default_test_generator: only CLI-backed LLMs (codex_cli, "
            "claude_cli, gemini_cli) are supported in the default; "
            "got %r. Returning [].",
            getattr(llm_config, "backend", None),
        )
        return []

    client = CLIModelClient(llm_config)
    # 10.R: embed "." (the working_dir is already pinned via the
    # working_dir=str(repo_path) kwarg below) instead of the absolute host
    # path. Agents that re-injected the absolute path into pytest invocations
    # produced ImportError-while-loading-conftest failures across package-style
    # Python repos on the Commit0 deep run.
    user_prompt = (
        "# Repository\n"
        ".\n\n"
        "# Problem statement\n"
        f"{problem_statement}\n\n"
        "# Task\n"
        "Read the repository, infer the bug from the problem statement, "
        "and produce a JSON portfolio of test artifacts that catches it. "
        "Follow the response schema exactly."
    )
    _lint_prompt_for_absolute_paths(user_prompt, context="default_test_generator")
    try:
        result = client.run_structured_prompt(
            prompt=user_prompt,
            working_dir=str(repo_path),
            schema=_TEST_ARTIFACTS_SCHEMA,
            system_prompt=_TEST_GENERATION_SYSTEM_PROMPT,
            allow_edits=False,
            internet_enabled=False,
            hard_timeout_seconds=_DEFAULT_LLM_HARD_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001
        # Phase 2C 5.5: classify before swallowing. Env failures
        # (network/timeout/install) get logged + return []; non-env
        # failures (programmer errors, schema bugs) are re-raised so
        # they aren't silently masked. The default generator's
        # "never crash" contract still holds for the env class — but
        # bugs surface.
        from .core.failure_classifier import classify_failure as _classify

        classification = _classify(
            stderr=str(exc),
            stdout="",
            returncode=1,
            context={"phase": "test_execution"},
        )
        if classification.failure_class.is_environment:
            logger.warning(
                "default_test_generator: CLI invocation failed (class=%s): %s. Returning [].",
                classification.failure_class.value,
                exc,
            )
            return []
        logger.error(
            "default_test_generator: CLI invocation failed with non-env "
            "exception (class=%s): %s. Re-raising so the bug is visible.",
            classification.failure_class.value,
            exc,
        )
        raise

    raw_response = (
        getattr(result, "parsed_json", None)
        or getattr(result, "structured_response", None)
        or getattr(result, "text", None)
        or getattr(result, "response", None)
    )
    if raw_response is None:
        logger.warning("default_test_generator: CLI returned no parsable response. Returning [].")
        return []

    payload: Any
    if isinstance(raw_response, dict):
        payload = raw_response
    else:
        try:
            payload = json.loads(raw_response)
        except (TypeError, ValueError):
            logger.warning("default_test_generator: response was not valid JSON. Returning [].")
            return []

    artifacts_raw = payload.get("test_artifacts") if isinstance(payload, dict) else None
    if not isinstance(artifacts_raw, list):
        logger.warning(
            "default_test_generator: response did not contain a test_artifacts list. Returning []."
        )
        return []

    cleaned: list[dict[str, Any]] = []
    for raw in artifacts_raw:
        if not isinstance(raw, dict):
            continue
        path = str(raw.get("path") or "").strip()
        content = str(raw.get("content") or "")
        if not path or not content:
            continue
        record: dict[str, Any] = {"path": path, "content": content}
        for key in (
            "strategy",
            "summary",
            "justification",
            "oracle_origin",
        ):
            value = str(raw.get(key) or "").strip()
            if value:
                record[key] = value
        for key in (
            "test_descriptions",
            "focus_files",
            "focus_tests",
            "contract_sources",
            "contract_targets",
            "contract_axes",
            "properties",
            "metamorphic_relations",
            "fuzz_seeds",
        ):
            values = _clean_string_list(raw.get(key))
            if values:
                record[key] = values
        if isinstance(raw.get("pass_then_invert"), dict):
            pass_then_invert = {
                str(key): value
                for key, value in dict(raw.get("pass_then_invert") or {}).items()
                if value not in (None, "", [])
            }
            if pass_then_invert:
                record["pass_then_invert"] = pass_then_invert
        if "dual_version_verified" in raw:
            record["dual_version_verified"] = bool(raw.get("dual_version_verified"))
        cleaned.append(record)
    cleaned = _maybe_truncate_artifacts_per_config(cleaned, cfg)
    return cleaned


def _maybe_truncate_artifacts_per_config(
    artifacts: list[dict[str, Any]], cfg: Any
) -> list[dict[str, Any]]:
    """Honor ``OrchestrationConfig.max_test_artifacts_per_invocation``.

    Phase 4A item 4.6: the JSON schema cap (maxItems=10) was removed so
    large axis-coverage portfolios survive end-to-end. Operators who
    still want a per-invocation cap (token-budget concerns) can set the
    config knob; ``None`` (default) means unbounded. When a cap fires,
    we log a warning so the truncation is visible in run logs rather
    than silent.
    """

    cap_raw = None
    try:
        orchestration = getattr(cfg, "orchestration", None)
        if orchestration is not None:
            cap_raw = getattr(orchestration, "max_test_artifacts_per_invocation", None)
    except Exception:  # pragma: no cover - defensive
        cap_raw = None
    if cap_raw is None:
        return artifacts
    try:
        cap = int(cap_raw)
    except (TypeError, ValueError):
        return artifacts
    if cap <= 0 or len(artifacts) <= cap:
        return artifacts
    logger.warning(
        "default_test_generator: truncating test_artifacts from %d to "
        "max_test_artifacts_per_invocation=%d (configurable on "
        "OrchestrationConfig)",
        len(artifacts),
        cap,
    )
    return artifacts[:cap]


def default_code_generator(
    repo_path: Path,
    problem_statement: str,
    test_artifacts: list[dict[str, Any]],
    *,
    config: Optional[Any] = None,
) -> Optional[str]:
    """LLM-backed default for ``apex.modes.run_*_*(code_generator=...)``.

    Materializes the supplied test artifacts into the repo (so the
    agent can run them), then invokes ``ApexOrchestrator(config).solve()``
    with a problem statement augmented to include the test files as
    success criteria. Returns ``result.patch`` or ``None`` on failure.
    """
    try:
        from .orchestrator import ApexOrchestrator
    except Exception as exc:  # pragma: no cover — import-time failure
        logger.warning(
            "default_code_generator: orchestrator unavailable (%s); "
            "returning None. Pass a custom code_generator to bypass.",
            exc,
        )
        return None

    cfg = config or _build_default_config()
    repo = Path(repo_path)

    # Materialize the test artifacts into the repo so the orchestrator
    # can run them as part of verification. We materialize even if the
    # caller supplied them already-staged — idempotent overwrite.
    materialized_paths: list[str] = []
    for artifact in test_artifacts or []:
        if not isinstance(artifact, dict):
            continue
        rel_path = safe_materialize_test_artifact(repo, artifact)
        if rel_path:
            materialized_paths.append(rel_path)
        else:
            logger.warning(
                "default_code_generator: skipped unsafe or empty test artifact path %r",
                artifact.get("path"),
            )

    augmented_issue = problem_statement
    if materialized_paths:
        paths = ", ".join(materialized_paths)
        augmented_issue = (
            f"{problem_statement}\n\n"
            f"Success criterion: the following test files (already "
            f"present in the repo) must pass after your fix: {paths}"
        )
    _lint_prompt_for_absolute_paths(augmented_issue, context="default_code_generator")

    try:
        result = ApexOrchestrator(cfg).solve(
            repo_path=str(repo),
            issue_description=augmented_issue,
        )
    except Exception as exc:  # noqa: BLE001
        # Phase 2C 5.5: classify before swallowing. Env failures
        # log+return None; non-env failures re-raise so the operator
        # sees the underlying bug rather than getting a silent None.
        from .core.failure_classifier import classify_failure as _classify

        classification = _classify(
            stderr=str(exc),
            stdout="",
            returncode=1,
            context={"phase": "test_execution"},
        )
        if classification.failure_class.is_environment:
            logger.warning(
                "default_code_generator: orchestrator.solve failed (class=%s): %s. Returning None.",
                classification.failure_class.value,
                exc,
            )
            return None
        logger.error(
            "default_code_generator: orchestrator.solve raised non-env "
            "exception (class=%s): %s. Re-raising so the bug surfaces.",
            classification.failure_class.value,
            exc,
        )
        raise

    if not result or not getattr(result, "success", False):
        logger.warning(
            "default_code_generator: orchestrator did not produce a "
            "successful patch. Returning None."
        )
        return None
    return getattr(result, "patch", None)
