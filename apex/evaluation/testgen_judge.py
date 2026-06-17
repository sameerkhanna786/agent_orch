"""LLM judge for testgen candidate selection.

After the F2P-tuple ranks candidates by ``(any_f2p, mutation_score,
f2p_count, f2p_rate)``, ties are broken today by a heuristic surface-
repair signal. This module adds an optional LLM judge that reads each
tied candidate's measured outcomes (F2P kills, mutation survivors,
sample test paths) and picks the suite most likely to catch real
production bugs.

The judge is intentionally narrow:
    * One LLM call per task (when invoked at all)
    * JSON-only response with a strict schema
    * Only runs on the F2P-tuple top tier — does NOT override candidates
      that catch the bug in favor of ones that don't

Decoupled from the rest of the eval via a pluggable ``llm_caller``
callable so unit tests don't need a real model and so a future commit
can swap CLI / OpenAI / Anthropic backends without touching this
module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# Strict JSON response schema — anchors the model's output and lets
# CLI backends (which support `--schema`) constrain decoding.
JUDGE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "selected_rollout_id": {"type": "integer"},
        "reasoning": {"type": "string"},
    },
    "required": ["selected_rollout_id", "reasoning"],
}


@dataclass
class TestgenJudgeOutcome:
    """The result of a single judge invocation."""

    # Tell pytest this dataclass is NOT a test class — its name happens
    # to start with "Test" but it carries no test methods.
    __test__ = False

    selected_rollout_id: Optional[int]
    reasoning: str
    judge_used: bool
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_rollout_id": self.selected_rollout_id,
            "reasoning": self.reasoning[:1000],
            "judge_used": self.judge_used,
            "error": self.error,
        }


# Type alias for the pluggable LLM caller. Takes (prompt, schema) and
# returns a dict matching JUDGE_RESPONSE_SCHEMA. Returning None / raising
# is interpreted as "judge unavailable" — caller falls through to the
# heuristic comparator.
LLMCaller = Callable[[str, dict[str, Any]], Optional[dict[str, Any]]]


def build_judge_prompt(
    *,
    candidates_summary: list[dict[str, Any]],
    issue_description: str,
    repo_name: str,
) -> str:
    """Render the judge prompt from per-candidate measured outcomes.

    The prompt is deliberately fact-dense and short on instruction —
    the judge's job is to pick from already-ranked candidates, not to
    re-derive the ranking. The summary fields below were chosen to give
    the model what F2P-tuple ranking does not capture: which test files
    actually exist (to spot trivial / over-narrow suites) and the
    raw mutation status counts (to spot one-mutant-killed-the-rest
    luck cases).
    """
    rendered_candidates = []
    for entry in candidates_summary:
        rendered_candidates.append(
            f"- Rollout {entry['rollout_id']}:\n"
            f"    F2P: any_f2p={entry.get('any_f2p')}, "
            f"f2p_count={entry.get('f2p_count')}, "
            f"f2p_rate={entry.get('f2p_rate'):.3f}, "
            f"p2f_regressions={entry.get('p2f_count')}, "
            f"p2p_useless={entry.get('p2p_count')}\n"
            f"    Mutation: score={entry.get('mutation_score'):.3f}, "
            f"killed={entry.get('mutation_killed')}/{entry.get('mutation_total')}, "
            f"survived={entry.get('mutation_survived')}\n"
            f"    Test files ({len(entry.get('test_paths', []))}): "
            f"{', '.join(entry.get('test_paths', [])[:5])}"
            + ("..." if len(entry.get("test_paths", [])) > 5 else "")
        )
    return (
        "You are a test-suite quality judge picking among rollout candidates "
        "that all caught the gold bug (F2P-positive). All candidates already "
        "tied on the (any_f2p, mutation_score) ranking; your job is to break "
        "the tie by picking the suite most likely to catch real production "
        "bugs in the same surface.\n\n"
        f"Repository: {repo_name}\n"
        f"Issue: {issue_description[:600]}\n\n"
        "Candidates:\n"
        + "\n".join(rendered_candidates)
        + "\n\nReturn JSON with the rollout_id you pick and one-paragraph "
        "reasoning. Prefer suites with: (a) zero p2f_regressions, (b) higher "
        "mutation_killed counts, (c) more diverse test_paths (suggests broader "
        "contract surface). Do NOT pick a candidate just because it has more "
        "files — fewer high-quality tests beats many trivial ones."
    )


def summarize_candidate_for_judge(
    *,
    rollout_id: int,
    f2p_payload: dict[str, Any],
    test_paths: list[str],
) -> dict[str, Any]:
    """Extract the fields the judge prompt uses from a per-candidate F2P payload.

    Defensive about missing fields — older payloads (pre-mutation) just
    surface zeros for the mutation columns, which the judge can still
    reason about.
    """
    summary = dict(f2p_payload.get("summary") or {})
    mutation = dict(f2p_payload.get("mutation") or {})
    return {
        "rollout_id": int(rollout_id),
        "any_f2p": bool(summary.get("any_f2p")),
        "f2p_count": int(summary.get("f2p_count") or 0),
        "f2p_rate": float(summary.get("f2p_rate") or 0.0),
        "p2f_count": int(summary.get("p2f_count") or 0),
        "p2p_count": int(summary.get("p2p_count") or 0),
        "mutation_score": float(mutation.get("mutation_score") or 0.0),
        "mutation_killed": int(mutation.get("killed") or 0),
        "mutation_survived": int(mutation.get("survived") or 0),
        "mutation_total": int(mutation.get("total_mutants") or 0),
        "test_paths": list(test_paths or [])[:20],
    }


def judge_testgen_candidates(
    *,
    candidates_summary: list[dict[str, Any]],
    issue_description: str,
    repo_name: str,
    llm_caller: Optional[LLMCaller],
) -> TestgenJudgeOutcome:
    """Pure-function judge entrypoint.

    Returns ``judge_used=False`` (and selected_rollout_id=None) when:
      * No llm_caller is provided (judge disabled at config time)
      * Fewer than 2 candidates (no tie to break)
      * The llm_caller raises or returns malformed JSON

    Otherwise returns the model's choice. Caller is responsible for
    validating the chosen rollout_id is one of the supplied candidates;
    this function does NOT enforce that constraint because letting the
    judge "abstain" (pick something nonsensical) is a valid degradation
    path that the heuristic fallback handles gracefully.
    """
    if llm_caller is None:
        return TestgenJudgeOutcome(
            selected_rollout_id=None,
            reasoning="judge disabled (no llm_caller)",
            judge_used=False,
        )
    if len(candidates_summary) < 2:
        return TestgenJudgeOutcome(
            selected_rollout_id=(
                candidates_summary[0]["rollout_id"] if candidates_summary else None
            ),
            reasoning="single candidate — no tie to break",
            judge_used=False,
        )

    prompt = build_judge_prompt(
        candidates_summary=candidates_summary,
        issue_description=issue_description,
        repo_name=repo_name,
    )
    try:
        response = llm_caller(prompt, JUDGE_RESPONSE_SCHEMA)
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("Testgen judge LLM call failed: %s", exc)
        return TestgenJudgeOutcome(
            selected_rollout_id=None,
            reasoning="",
            judge_used=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    if response is None:
        return TestgenJudgeOutcome(
            selected_rollout_id=None,
            reasoning="",
            judge_used=False,
            error="llm_caller returned None",
        )
    raw_id = response.get("selected_rollout_id")
    raw_reasoning = response.get("reasoning") or ""
    if not isinstance(raw_id, int):
        # Try to recover an int from a string field — some CLI backends
        # emit JSON with stringified numerics.
        try:
            raw_id = int(str(raw_id).strip())
        except (TypeError, ValueError):
            return TestgenJudgeOutcome(
                selected_rollout_id=None,
                reasoning=str(raw_reasoning)[:500],
                judge_used=False,
                error="malformed selected_rollout_id",
            )
    return TestgenJudgeOutcome(
        selected_rollout_id=int(raw_id),
        reasoning=str(raw_reasoning)[:500],
        judge_used=True,
    )
