"""V5 LLM-as-critic for "right reason" verification (e-Otter++ pattern).

Distinct from ``testgen_judge.py``: that judge picks among already-ranked
F2P-positive candidates from metric summaries. This critic operates one
level lower — it examines an individual test's source code and the
per-(test, patch) failure traces from the dual-version verifier, then
labels each cell as ``right_reason`` (test detected the actual bug) or
``wrong_reason`` (test failed for an unrelated reason: import error,
brittle string match, environment variance, etc.).

This is e-Otter++'s key lift over Echo: cells that look like F→P on
exit-code alone but were actually flaky / brittle are downweighted, and
cells where the failure trace genuinely names the bug surface are
upweighted. Combined with the cross-candidate voter, "right_reason"
oracle scores break ties more cleanly than raw counts.

The critic is intentionally:
  - Optional (``llm_caller=None`` → critic skipped, oracle scores used as-is)
  - Schema-strict (CLI/Anthropic/OpenAI all support ``--schema``-style
    constrained decoding for the JSON output)
  - Per-row, not per-cell (one call summarizes all patch verdicts for
    one test) — keeps critic budget at O(test_count), not O(test×patch)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


CRITIC_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "right_reason_patch_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "wrong_reason_patch_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "overall_quality": {
            "type": "string",
            "enum": ["strong", "adequate", "brittle", "broken"],
        },
        "rationale": {"type": "string"},
    },
    "required": [
        "right_reason_patch_ids",
        "wrong_reason_patch_ids",
        "overall_quality",
        "rationale",
    ],
}


@dataclass(frozen=True)
class CriticVerdict:
    """Per-test critic outcome."""

    test_id: str
    right_reason_patch_ids: list[str] = field(default_factory=list)
    wrong_reason_patch_ids: list[str] = field(default_factory=list)
    overall_quality: str = "adequate"
    rationale: str = ""
    used: bool = False
    error: str = ""

    @property
    def right_reason_score(self) -> int:
        return len(self.right_reason_patch_ids)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CriticLLMCaller = Callable[[str, dict[str, Any]], Optional[dict[str, Any]]]


def critique_test_rows(
    *,
    test_candidates: list[dict[str, Any]],
    dual_version_rows: list[Any],
    llm_caller: Optional[CriticLLMCaller],
    focal_path: str = "",
    focal_source_excerpt: str = "",
) -> list[CriticVerdict]:
    """Critique each test row from the dual-version verifier.

    For every ``TestRow``, builds a prompt from the test source + the
    per-patch verdict notes and asks the LLM to label which patches the
    test "really" detected vs. which were flaky/wrong-reason matches.

    Returns one ``CriticVerdict`` per test in the same order as
    ``test_candidates``. When ``llm_caller`` is None or fails, the
    verdict has ``used=False`` and downstream selectors should fall back
    to raw oracle_score.
    """

    rows_by_id: dict[str, Any] = {getattr(r, "test_id", None): r for r in (dual_version_rows or [])}
    candidates_by_id: dict[str, dict[str, Any]] = {
        str(c.get("test_id") or c.get("agent") or f"test_{i}"): c
        for i, c in enumerate(test_candidates)
    }
    out: list[CriticVerdict] = []
    for test_id, candidate in candidates_by_id.items():
        row = rows_by_id.get(test_id)
        if row is None or llm_caller is None:
            out.append(CriticVerdict(test_id=test_id, used=False))
            continue
        prompt = _build_critic_prompt(
            test_id=test_id,
            test_source=str(candidate.get("artifact_content") or ""),
            focal_path=focal_path,
            focal_source_excerpt=focal_source_excerpt,
            verdicts=getattr(row, "verdicts", []) or [],
        )
        try:
            response = llm_caller(prompt, CRITIC_RESPONSE_SCHEMA)
        except Exception as exc:  # pragma: no cover - LLM/CLI errors are diagnostic
            logger.warning("LLM critic failed for %s: %s", test_id, exc)
            out.append(
                CriticVerdict(
                    test_id=test_id,
                    used=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        if not isinstance(response, dict):
            out.append(CriticVerdict(test_id=test_id, used=False, error="non-dict response"))
            continue
        out.append(
            CriticVerdict(
                test_id=test_id,
                right_reason_patch_ids=[
                    str(p) for p in (response.get("right_reason_patch_ids") or [])
                ],
                wrong_reason_patch_ids=[
                    str(p) for p in (response.get("wrong_reason_patch_ids") or [])
                ],
                overall_quality=str(response.get("overall_quality") or "adequate").lower(),
                rationale=str(response.get("rationale") or "")[:1000],
                used=True,
            )
        )
    return out


def adjusted_oracle_scores(
    *,
    dual_version_rows: list[Any],
    critic_verdicts: list[CriticVerdict],
) -> dict[str, float]:
    """Compose raw oracle_score with critic right_reason_score.

    Strategy: the score for a test = number of patches the test is F→P
    against AND the critic confirmed as right_reason. When the critic
    didn't run (``used=False``) we fall through to raw oracle_score so
    the pipeline degrades gracefully.
    """

    by_id: dict[str, CriticVerdict] = {v.test_id: v for v in critic_verdicts}
    out: dict[str, float] = {}
    for row in dual_version_rows or []:
        test_id = getattr(row, "test_id", None)
        if test_id is None:
            continue
        critic = by_id.get(test_id)
        if critic is None or not critic.used:
            out[test_id] = float(getattr(row, "oracle_score", 0.0) or 0.0)
            continue
        right = set(critic.right_reason_patch_ids)
        wrong = set(critic.wrong_reason_patch_ids)
        f2p_right = sum(
            1
            for v in (getattr(row, "verdicts", []) or [])
            if getattr(v, "is_f2p", False)
            and getattr(v, "patch_id", "") in right
            and getattr(v, "patch_id", "") not in wrong
        )
        out[test_id] = float(f2p_right)
    return out


def _build_critic_prompt(
    *,
    test_id: str,
    test_source: str,
    focal_path: str,
    focal_source_excerpt: str,
    verdicts: list[Any],
) -> str:
    """Render the critic prompt from per-cell verdict trace."""

    cells = []
    for v in verdicts:
        cells.append(
            "  - patch_id={pid}: buggy={bs}, patched={ps}, is_f2p={f2p}, note={note}".format(
                pid=getattr(v, "patch_id", ""),
                bs=getattr(v, "buggy_status", ""),
                ps=getattr(v, "patched_status", ""),
                f2p=getattr(v, "is_f2p", False),
                note=(getattr(v, "note", "") or "")[:120],
            )
        )
    parts = [
        "You are a test-quality critic. Given a generated test and per-patch",
        "execution verdicts (test was run on buggy code, then on patched code",
        "for each candidate fix), classify each patch the test was F→P-against",
        "as RIGHT_REASON (test genuinely targets the bug surface) or",
        "WRONG_REASON (test failed/passed for incidental reasons: imports,",
        "fixtures, environment, brittle string match, off-target assertion).",
        "",
        f"Test id: {test_id}",
    ]
    if focal_path:
        parts.append(f"Focal file under test: {focal_path}")
    if focal_source_excerpt:
        parts.extend(
            [
                "Focal source excerpt:",
                "```python",
                focal_source_excerpt[:1800],
                "```",
                "",
            ]
        )
    parts.extend(
        [
            "Test source:",
            "```python",
            test_source[:6000],
            "```",
            "",
            "Per-patch verdicts:",
            *cells,
            "",
            "Respond with JSON matching the schema. Only include patch_ids",
            "that appear above. overall_quality is your global read of this",
            "test (strong / adequate / brittle / broken).",
        ]
    )
    return "\n".join(parts)


def make_default_critic_caller(
    *,
    judge_agent: str = "claude",
    timeout_seconds: int = 120,
    working_dir: str | Path = ".",
) -> CriticLLMCaller:
    """Build an LLM caller backed by the APEX agentic CLI infrastructure.

    Default to ``claude`` for the critic role — it tends to give the
    most calibrated "right reason" judgements in our internal evals.
    Swap via the ``judge_agent`` arg.
    """

    import json

    from apex._default_generators import build_agent_llm_config
    from apex.core.cli_backend import CLIModelClient

    llm_config = build_agent_llm_config(judge_agent)
    timeout = max(1, int(timeout_seconds))
    llm_config.cli_timeout = timeout
    llm_config.cli_hard_timeout_seconds = timeout
    llm_config.cli_strict_hard_timeout = True

    workdir = Path(working_dir)

    def caller(prompt: str, schema: dict[str, Any]) -> Optional[dict[str, Any]]:
        target_tool_env: dict[str, str] = {}
        try:
            from apex.evaluation.testgeneval_benchmark import _target_authoring_tool_env_overrides

            target_tool_env, _ = _target_authoring_tool_env_overrides(
                workdir=workdir,
                output_dir=workdir / ".apex_llm_critic_tools",
                timeout_seconds=timeout,
            )
        except Exception:
            target_tool_env = {}
        result = CLIModelClient(llm_config).run_structured_prompt(
            prompt=prompt,
            working_dir=str(workdir),
            schema=schema,
            system_prompt=(
                "You are a senior test reviewer evaluating whether a test "
                "fails for the right reason. Be terse and concrete."
            ),
            allow_edits=False,
            internet_enabled=False,
            hard_timeout_seconds=timeout,
            env_overrides=target_tool_env or None,
        )
        parsed = getattr(result, "parsed_json", None)
        if isinstance(parsed, dict):
            return parsed
        text = getattr(result, "text", None) or ""
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            return None

    return caller
