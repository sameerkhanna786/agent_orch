"""WS3C: a fresh-context LLM reviewer that gates FINAL acceptance.

Complements the deterministic ``adversarial_review`` veto with a separate-family
LLM judgment (SpecRover reviewer / Trae selector pattern). Hard safety contract:

* It is ONLY consulted for an already-accepted candidate, so it can only
  DOWNGRADE (accept -> reject); it can never upgrade a rejected candidate.
* It FAILS OPEN: any reviewer error / malformed verdict leaves acceptance
  unchanged (returns ``accept=True, failed_open=True``), so a flaky reviewer
  never blocks a verified candidate.
* It is DEFAULT OFF (``SelectionConfig.enable_final_acceptance_reviewer``).
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "accept": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["accept"],
    "additionalProperties": True,
}


@dataclass(frozen=True)
class FinalAcceptanceVerdict:
    accept: bool
    reason: str = ""
    used_llm: bool = False
    reviewer_backend: str = ""
    reviewer_model: str = ""
    failed_open: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "accept": bool(self.accept),
            "reason": self.reason,
            "used_llm": bool(self.used_llm),
            "reviewer_backend": self.reviewer_backend,
            "reviewer_model": self.reviewer_model,
            "failed_open": bool(self.failed_open),
        }


def _changed_files(candidate: Any) -> list[str]:
    values = getattr(candidate, "changed_files", None)
    if isinstance(values, (list, tuple, set)):
        return [str(v) for v in values][:50]
    return []


class FinalAcceptanceReviewer:
    """Fresh-context LLM final-acceptance gate (only downgrades; fails open)."""

    def __init__(
        self,
        llm: Any,
        *,
        reviewer_backend: str = "",
        actor_backend: str = "",
        require_distinct_family: bool = True,
        working_dir: str = ".",
        timeout_seconds: int = 90,
    ) -> None:
        self.llm = llm
        self.reviewer_backend = str(reviewer_backend or "")
        self.actor_backend = str(actor_backend or "")
        self.require_distinct_family = bool(require_distinct_family)
        self.working_dir = str(working_dir or ".")
        self.timeout_seconds = int(timeout_seconds)

    def review(
        self,
        candidate: Any,
        verification: Any,
        *,
        issue_description: str = "",
        evidence: Optional[dict[str, Any]] = None,
    ) -> FinalAcceptanceVerdict:
        """Return a verdict. FAILS OPEN (accept=True) on any error."""
        reviewer_model = str(getattr(getattr(self.llm, "llm_config", None), "model", "") or "")
        try:
            prompt = self._build_prompt(candidate, verification, issue_description, evidence)
            raw = self._invoke(prompt)
            verdict = self._parse(raw)
            if verdict is None:
                return FinalAcceptanceVerdict(
                    accept=True,
                    reason="reviewer returned malformed verdict; failing open",
                    used_llm=True,
                    reviewer_backend=self.reviewer_backend,
                    reviewer_model=reviewer_model,
                    failed_open=True,
                )
            accept, reason = verdict
            return FinalAcceptanceVerdict(
                accept=accept,
                reason=reason,
                used_llm=True,
                reviewer_backend=self.reviewer_backend,
                reviewer_model=reviewer_model,
            )
        except Exception as exc:  # noqa: BLE001 - reviewer must never block a candidate
            logger.debug("FinalAcceptanceReviewer failed open: %s", exc, exc_info=True)
            return FinalAcceptanceVerdict(
                accept=True,
                reason=f"reviewer error; failing open: {exc}",
                used_llm=False,
                reviewer_backend=self.reviewer_backend,
                reviewer_model=reviewer_model,
                failed_open=True,
            )

    def _build_prompt(
        self,
        candidate: Any,
        verification: Any,
        issue_description: str,
        evidence: Optional[dict[str, Any]],
    ) -> str:
        patch = str(getattr(candidate, "patch", "") or "")[:8000]
        files = ", ".join(_changed_files(candidate))
        overall = getattr(verification, "overall_score", None)
        return (
            "You are a strict senior code reviewer giving a FINAL accept/reject on a patch that "
            "already passed automated verification. Reject ONLY if the patch is clearly wrong, "
            "incomplete, or solves the wrong problem; otherwise accept.\n\n"
            f"## Issue\n{issue_description.strip()[:4000]}\n\n"
            f"## Changed files\n{files}\n\n"
            f"## Verification score\n{overall}\n\n"
            f"## Patch\n{patch}\n\n"
            'Reply with ONLY a JSON object: {"accept": true|false, "reason": "..."}'
        )

    def _invoke(self, prompt: str) -> str:
        # Dual path: CLIModelClient.run_structured_prompt or LLMClient.chat.
        if hasattr(self.llm, "run_structured_prompt"):
            result = self.llm.run_structured_prompt(
                prompt=prompt,
                working_dir=self.working_dir,
                schema=_REVIEW_SCHEMA,
                allow_edits=False,
                internet_enabled=False,
            )
            if getattr(result, "parsed_json", None) is not None:
                return json.dumps(result.parsed_json)
            return str(getattr(result, "text", "") or "")
        if hasattr(self.llm, "chat"):
            response = self.llm.chat([{"role": "user", "content": prompt}])
            return str(getattr(response, "content", "") or "")
        raise TypeError("reviewer llm exposes neither run_structured_prompt nor chat")

    @staticmethod
    def _parse(raw: str) -> Optional[tuple[bool, str]]:
        raw = str(raw or "").strip()
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end <= start:
                return None
            try:
                data = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return None
        if not isinstance(data, dict) or "accept" not in data:
            return None
        return bool(data.get("accept")), str(data.get("reason") or "")


# ---------------------------------------------------------------------------
# Feature E: perspective-diverse model-critic selection layer.
#
# A set of DISTINCT generic lenses each score a candidate on a 0..1 scale. The
# scores act as a low-priority TIEBREAKER among execution-verified candidates:
# they pick the patch least likely to overfit the visible tests and fail hidden
# / edge tests, but they NEVER override concrete execution evidence (the
# selector only consults them to re-rank within an already-accepted tier). The
# reviewer FAILS OPEN to a neutral 0.5 on any error/timeout so a flaky judge
# can never demote a verified candidate. This is Layer-A general (no
# per-benchmark / per-language conditionals).
# ---------------------------------------------------------------------------

_PERSPECTIVE_NEUTRAL_SCORE = 0.5

_PERSPECTIVE_SCORE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["score"],
    "additionalProperties": True,
}

# Each lens is a generic, benchmark-agnostic perspective. The convention is that
# a HIGHER score is always BETTER for the candidate (less risk / better fit), so
# the scores can be combined and used as a max-tiebreaker without sign juggling.
_PERSPECTIVE_LENS_PROMPTS: dict[str, str] = {
    "minimality": (
        "Lens: MINIMALITY. Judge whether this change is minimal and focused: it "
        "fixes the issue without scope creep, gratuitous refactors, dead code, "
        "or unrelated edits. A HIGHER score means the change is MORE minimal and "
        "targeted; a LOWER score means it sprawls beyond what the issue requires."
    ),
    "spec_conformance": (
        "Lens: SPEC CONFORMANCE. Judge whether this change actually implements "
        "the INTENT of the issue/spec, not merely whatever surface tests check. "
        "A HIGHER score means the change faithfully matches the described intent; "
        "a LOWER score means it games the tests or solves the wrong problem."
    ),
    "edge_case_risk": (
        "Lens: EDGE CASE RISK. Judge how likely this change is to fail HIDDEN or "
        "edge-case tests not shown here (boundary values, empty/None inputs, "
        "concurrency, error paths). A HIGHER score means it is LESS likely to "
        "fail hidden/edge tests; a LOWER score means it is MORE likely to."
    ),
    "regression_risk": (
        "Lens: REGRESSION RISK. Judge how likely this change is to break "
        "unrelated existing behavior elsewhere in the codebase. A HIGHER score "
        "means it is LESS likely to cause regressions; a LOWER score means it is "
        "MORE likely to break unrelated behavior."
    ),
}


def _clamp_unit(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return _PERSPECTIVE_NEUTRAL_SCORE
    if score != score:  # NaN guard
        return _PERSPECTIVE_NEUTRAL_SCORE
    return max(0.0, min(score, 1.0))


class PerspectiveReviewer:
    """Perspective-diverse model critic (lens scoring; fails open to 0.5).

    Modeled on :class:`FinalAcceptanceReviewer`: it reuses the same LLM client
    wiring (``run_structured_prompt`` / ``chat``), the same lenient JSON parse,
    and the same fail-open contract. One LLM call per lens, run concurrently up
    to ``max_workers``. Every lens defaults to a neutral ``0.5`` on any error /
    timeout, so a flaky reviewer is a no-op tiebreaker rather than a blocker.
    """

    def __init__(
        self,
        llm: Any,
        *,
        lenses: Optional[list[str]] = None,
        reviewer_backend: str = "",
        actor_backend: str = "",
        working_dir: str = ".",
        max_workers: int = 4,
        timeout_seconds: int = 90,
    ) -> None:
        self.llm = llm
        requested = list(lenses) if lenses else list(_PERSPECTIVE_LENS_PROMPTS.keys())
        # Keep only known lenses, preserving order and dropping duplicates; fall
        # back to the full default set if the request resolves to nothing.
        seen: set[str] = set()
        resolved: list[str] = []
        for lens in requested:
            key = str(lens or "").strip()
            if key in _PERSPECTIVE_LENS_PROMPTS and key not in seen:
                seen.add(key)
                resolved.append(key)
        self.lenses: list[str] = resolved or list(_PERSPECTIVE_LENS_PROMPTS.keys())
        self.reviewer_backend = str(reviewer_backend or "")
        self.actor_backend = str(actor_backend or "")
        self.working_dir = str(working_dir or ".")
        self.max_workers = max(1, int(max_workers))
        self.timeout_seconds = int(timeout_seconds)

    def score_candidate(
        self,
        candidate: Any,
        issue_description: str = "",
        changed_files: Optional[list[str]] = None,
        test_summary: str = "",
    ) -> dict[str, float]:
        """Return ``{lens: score in 0..1}`` for every configured lens.

        One LLM call per lens, run concurrently up to ``max_workers``. Any error
        or timeout for a lens yields the neutral ``0.5`` for that lens; an error
        scheduling the pool yields ``0.5`` for every lens (FAIL OPEN).
        """

        files = list(changed_files) if changed_files else _changed_files(candidate)
        scores: dict[str, float] = {lens: _PERSPECTIVE_NEUTRAL_SCORE for lens in self.lenses}
        if not self.lenses:
            return scores

        # Overall wall-clock budget for the whole fan-out. A non-positive value
        # means "no enforced deadline" (delegate entirely to the LLM client's own
        # timeout). Any lens still running when the budget elapses keeps its
        # pre-seeded neutral 0.5, so a hung lens can never block selection.
        deadline = float(self.timeout_seconds) if self.timeout_seconds > 0 else None
        workers = min(self.max_workers, len(self.lenses))
        executor: Optional[ThreadPoolExecutor] = None
        try:
            executor = ThreadPoolExecutor(max_workers=workers)
            future_to_lens = {
                executor.submit(
                    self._score_lens,
                    lens,
                    candidate,
                    issue_description,
                    files,
                    test_summary,
                ): lens
                for lens in self.lenses
            }
            try:
                for future in as_completed(future_to_lens, timeout=deadline):
                    lens = future_to_lens[future]
                    try:
                        scores[lens] = future.result()
                    except Exception as exc:  # noqa: BLE001 - per-lens fail-open
                        logger.debug(
                            "PerspectiveReviewer lens %s failed open: %s",
                            lens,
                            exc,
                            exc_info=True,
                        )
                        scores[lens] = _PERSPECTIVE_NEUTRAL_SCORE
            except Exception as exc:  # noqa: BLE001 - timeout/iter fail-open
                # Includes TimeoutError from as_completed: unfinished lenses keep
                # their pre-seeded neutral 0.5; completed lenses retain their real
                # scores. We do NOT block on hung threads here.
                logger.debug(
                    "PerspectiveReviewer fan-out incomplete; using neutral "
                    "scores for unfinished lenses: %s",
                    exc,
                    exc_info=True,
                )
        except Exception as exc:  # noqa: BLE001 - pool-level fail-open
            logger.debug("PerspectiveReviewer failed open (pool): %s", exc, exc_info=True)
            return {lens: _PERSPECTIVE_NEUTRAL_SCORE for lens in self.lenses}
        finally:
            if executor is not None:
                # Non-blocking shutdown: cancel queued lenses and abandon any
                # still-running ones rather than waiting on a hung LLM call, so a
                # stuck lens can never delay selection past the budget.
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:  # pragma: no cover - cancel_futures pre-3.9
                    executor.shutdown(wait=False)
                except Exception:  # noqa: BLE001 - shutdown must never raise out
                    pass
        return scores

    @staticmethod
    def aggregate(scores: dict[str, float]) -> float:
        """Aggregate per-lens scores into a single 0..1 value (mean)."""
        values = [_clamp_unit(v) for v in (scores or {}).values()]
        if not values:
            return _PERSPECTIVE_NEUTRAL_SCORE
        return sum(values) / len(values)

    def _score_lens(
        self,
        lens: str,
        candidate: Any,
        issue_description: str,
        changed_files: list[str],
        test_summary: str,
    ) -> float:
        try:
            prompt = self._build_prompt(
                lens, candidate, issue_description, changed_files, test_summary
            )
            raw = self._invoke(prompt)
            parsed = self._parse(raw)
            if parsed is None:
                return _PERSPECTIVE_NEUTRAL_SCORE
            return parsed
        except Exception as exc:  # noqa: BLE001 - lens scoring must never raise out
            logger.debug("PerspectiveReviewer lens %s error: %s", lens, exc, exc_info=True)
            return _PERSPECTIVE_NEUTRAL_SCORE

    def _build_prompt(
        self,
        lens: str,
        candidate: Any,
        issue_description: str,
        changed_files: list[str],
        test_summary: str,
    ) -> str:
        patch = str(getattr(candidate, "patch", "") or "")[:8000]
        files = ", ".join(str(f) for f in changed_files[:50])
        lens_prompt = _PERSPECTIVE_LENS_PROMPTS.get(lens, "")
        return (
            "You are one of several independent reviewers each judging a code "
            "patch from a single PERSPECTIVE. Score this patch on YOUR lens only, "
            "on a scale from 0.0 (worst) to 1.0 (best), where HIGHER is always "
            "BETTER for the candidate.\n\n"
            f"## Your lens\n{lens_prompt}\n\n"
            f"## Issue\n{str(issue_description).strip()[:4000]}\n\n"
            f"## Changed files\n{files}\n\n"
            f"## Test summary\n{str(test_summary).strip()[:2000]}\n\n"
            f"## Patch\n{patch}\n\n"
            'Reply with ONLY a JSON object: {"score": 0.0-1.0, "reason": "one line"}'
        )

    def _invoke(self, prompt: str) -> str:
        # Dual path mirrors FinalAcceptanceReviewer: CLIModelClient
        # run_structured_prompt or LLMClient.chat.
        if hasattr(self.llm, "run_structured_prompt"):
            result = self.llm.run_structured_prompt(
                prompt=prompt,
                working_dir=self.working_dir,
                schema=_PERSPECTIVE_SCORE_SCHEMA,
                allow_edits=False,
                internet_enabled=False,
            )
            if getattr(result, "parsed_json", None) is not None:
                return json.dumps(result.parsed_json)
            return str(getattr(result, "text", "") or "")
        if hasattr(self.llm, "chat"):
            response = self.llm.chat([{"role": "user", "content": prompt}])
            return str(getattr(response, "content", "") or "")
        raise TypeError("reviewer llm exposes neither run_structured_prompt nor chat")

    @staticmethod
    def _parse(raw: str) -> Optional[float]:
        raw = str(raw or "").strip()
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end <= start:
                return None
            try:
                data = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return None
        if not isinstance(data, dict) or "score" not in data:
            return None
        return _clamp_unit(data.get("score"))


def build_perspective_reviewer(config: Any, llm_client: Any) -> Optional[PerspectiveReviewer]:
    """Construct a :class:`PerspectiveReviewer` from config + an LLM client.

    Returns ``None`` when the feature is disabled (``enable_perspective_review``
    is False) or no LLM client is available, so callers can treat ``None`` as a
    no-op (the selector then skips the tiebreaker entirely).
    """
    selection = getattr(config, "selection", config)
    if not bool(getattr(selection, "enable_perspective_review", False)):
        return None
    if llm_client is None:
        return None
    lenses = list(getattr(selection, "perspective_review_lenses", []) or [])
    return PerspectiveReviewer(
        llm_client,
        lenses=lenses,
        reviewer_backend=str(getattr(selection, "perspective_review_backend", "") or ""),
        max_workers=int(getattr(selection, "perspective_review_max_workers", 4) or 4),
        timeout_seconds=int(
            getattr(selection, "perspective_review_timeout_seconds", 90) or 90
        ),
    )
