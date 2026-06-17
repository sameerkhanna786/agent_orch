"""Verification Amplifier — break the test-suite discrimination ceiling.

When APEX selection has K>1 candidate patches that all pass the existing
test suite, the underlying selector ranks them with AST clustering /
critic features / heuristics. None of these signals can *prove* one
patch correct over another — they can only break ties when the test
suite is silent.

This module turns the selector from "rank with heuristics" into "rank
with discriminating evidence":

    For each (patch_i, patch_j) with i < j, ask an LLM to write a
    test that PASSES on patch_i and FAILS on patch_j. Run that test
    against every patch. The patch that survives the most such
    differential probes wins.

The amplifier is mockable for unit tests. It owns no docker / pytest
plumbing — it delegates execution to a ``test_runner`` callable which
the wiring layer constructs from the existing ``PatchVerifier``
infrastructure.

Cost model (K = number of tied candidates):
    LLM calls: ``min(K * (K - 1) / 2, max_pairs) * max_tests_per_pair``
    Test runs: ``K * (above)``  — every patch executes every generated
    test so the discrimination matrix is filled.

For K = 8 with the default caps (max_pairs = 6, max_tests_per_pair = 2):
    LLM calls    <= 12
    Test runs    <= 96

For K = 3 with defaults:
    LLM calls    <= 6     (3 pairs * 2 tests)
    Test runs    <= 18    (3 patches * 6 tests)

Cost guards (short-circuit BEFORE any LLM call):
    * patches differ on < 5 lines total -> indistinguishable
    * patches touch identical file sets -> indistinguishable

The amplifier returns ``AmplificationResult.confidence``; selectors
should fall back to their existing logic when confidence < 0.6.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

logger = logging.getLogger("apex.orchestration.verification_amplifier")


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DiscriminatingTest:
    """A test designed to distinguish between two specific patches."""

    test_code: str
    test_name: str
    target_patch: int  # the patch this test is supposed to PASS on
    distinguishes_from: int  # the patch this test is supposed to FAIL on
    confidence: float = 0.0  # LLM self-rated confidence (0..1)


@dataclass
class DiscriminationMatrix:
    """Per-(patch, test) verdict matrix.

    ``verdict_matrix[i][k]`` is True iff test ``k`` passed on patch ``i``.
    """

    n_patches: int
    tests: List[DiscriminatingTest] = field(default_factory=list)
    verdict_matrix: List[List[bool]] = field(default_factory=list)

    def discriminating_tests_for_pair(self, i: int, j: int) -> List[int]:
        """Return indices of tests where patch ``i`` and patch ``j`` disagree."""
        if i == j:
            return []
        out: list[int] = []
        for k in range(len(self.tests)):
            if k >= len(self.verdict_matrix[i]) or k >= len(self.verdict_matrix[j]):
                continue
            if self.verdict_matrix[i][k] != self.verdict_matrix[j][k]:
                out.append(k)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_patches": self.n_patches,
            "tests": [
                {
                    "test_name": t.test_name,
                    "target_patch": t.target_patch,
                    "distinguishes_from": t.distinguishes_from,
                    "confidence": t.confidence,
                }
                for t in self.tests
            ],
            "verdict_matrix": [list(row) for row in self.verdict_matrix],
        }


@dataclass
class AmplificationResult:
    """Outcome of one amplification pass."""

    chosen_patch: int
    confidence: float
    discrimination_matrix: DiscriminationMatrix
    cost_inferences: int = 0
    cost_test_runs: int = 0
    short_circuited: bool = False
    short_circuit_reason: Optional[str] = None
    win_counts: List[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chosen_patch": self.chosen_patch,
            "confidence": self.confidence,
            "cost_inferences": self.cost_inferences,
            "cost_test_runs": self.cost_test_runs,
            "short_circuited": self.short_circuited,
            "short_circuit_reason": self.short_circuit_reason,
            "win_counts": list(self.win_counts),
            "matrix": self.discrimination_matrix.to_dict(),
        }


# ---------------------------------------------------------------------------
# Diff helpers (no scipy/numpy — pure Python)
# ---------------------------------------------------------------------------


_DIFF_LINE_RE = re.compile(r"^[+-](?![+-]{2}).*$", re.MULTILINE)
_DIFF_FILE_RE = re.compile(r"^(?:diff --git a/(\S+)|\+\+\+ b/(\S+))", re.MULTILINE)


def _diff_changed_lines(patch: str) -> int:
    """Count +/- lines in a unified diff (excluding file headers ``+++``/``---``)."""
    if not patch:
        return 0
    return len(_DIFF_LINE_RE.findall(patch))


def _diff_file_set(patch: str) -> frozenset[str]:
    """Extract the set of files touched by a unified diff."""
    if not patch:
        return frozenset()
    files: set[str] = set()
    for match in _DIFF_FILE_RE.finditer(patch):
        path = match.group(1) or match.group(2)
        if path:
            files.add(path)
    return frozenset(files)


def _changed_line_payload(patch: str) -> str:
    """Return the concatenated +/- line content for a unified diff."""
    return "\n".join(_DIFF_LINE_RE.findall(patch or ""))


def _patches_indistinguishable(patches: List[str]) -> Optional[str]:
    """Return a short reason string if the patch set looks indistinguishable.

    Cost guards mandated by the spec:

      * Total +/- lines across the smallest patch is fewer than 5 -> "few_diff_lines".
        (A diff with 4 changed lines simply doesn't carry enough behavior
        to make pair-wise discriminating tests worthwhile.)
      * All patches have byte-identical changed-line payload (same file
        set + same +/- lines) -> "identical_file_sets". This is the
        degenerate "all rollouts produced the same diff" case, where
        amplification has nothing to distinguish.

    Returns ``None`` if amplification should proceed.
    """
    if len(patches) < 2:
        return "fewer_than_two_patches"

    # Byte-identical changed-line payload across all patches -> nothing
    # to discriminate. Compare on the +/- line text (not the full diff
    # body, which includes file headers / hunk headers that may vary).
    payloads = {_changed_line_payload(p) for p in patches}
    file_sets = {_diff_file_set(p) for p in patches}
    if len(payloads) == 1 and len(file_sets) == 1:
        return "identical_file_sets"

    # Per-patch line-count guard: if the SMALLEST patch has < 5 +/-
    # lines, the cheapest patch in the tied set is essentially a
    # one-line change — probably not worth the LLM/test cost. Note we
    # use the minimum (not the average), because the amplifier should
    # short-circuit when ANY patch is too thin to be informative.
    line_counts = [_diff_changed_lines(p) for p in patches]
    if min(line_counts) < 5:
        return "few_diff_lines"

    return None


# ---------------------------------------------------------------------------
# LLM prompt + response parsing
# ---------------------------------------------------------------------------


_PROMPT_TEMPLATE = """\
You are inspecting two candidate patches for the same task. Both
patches pass the existing test suite, but you suspect they implement
different behaviors.

Task context:
{task_context}

Patch A (target — your test should PASS against this patch):
```diff
{patch_i}
```

Patch B (foil — your test should FAIL against this patch):
```diff
{patch_j}
```

Write ONE Python pytest test (no fixtures, no class) that exercises
the smallest behavioral difference between A and B. The test must
assert behavior that holds when A is applied and is violated when B
is applied.

Respond with strict JSON:
{{
  "test_name": "test_<short snake_case name>",
  "test_code": "<full source of the test, including any imports>",
  "confidence": <float 0.0 to 1.0 — your confidence the test is
    well-formed and will discriminate>
}}

Do NOT wrap the JSON in markdown fences. Do NOT add commentary."""


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_llm_test_response(raw: str) -> Optional[dict[str, Any]]:
    """Best-effort parse the LLM's JSON response into a dict."""
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    # Strip code-fence wrappers if the model ignored instructions.
    if text.startswith("```"):
        # remove leading fence + optional language tag
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(text)
    if match is None:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# VerificationAmplifier
# ---------------------------------------------------------------------------


# Type aliases for clarity; we deliberately keep them as ``Callable[..., Any]``
# at runtime so callers can pass any compatible function.
LLMCaller = Callable[[str], str]
TestRunner = Callable[[str, str, Path], bool]


class VerificationAmplifier:
    """Generate discriminating tests at selection time and pick a winner.

    Parameters
    ----------
    llm_caller:
        Callable ``(prompt: str) -> str`` that returns the LLM's raw
        response. Should return a JSON string (the prompt instructs
        the model to do so) but the parser tolerates code-fence
        wrappers and extra prose.
    test_runner:
        Callable ``(patch: str, test_code: str, repo_path: Path) -> bool``
        that returns True iff the test passes when the patch is
        applied. Exceptions are caught and treated as a failed test.
    max_pairs:
        Cap on the number of (patch_i, patch_j) pairs we generate
        tests for. Default 6.
    max_tests_per_pair:
        Cap on the number of LLM calls per pair. Default 2 (the
        second call uses a slightly perturbed prompt to encourage
        diversity).
    """

    def __init__(
        self,
        llm_caller: LLMCaller,
        test_runner: TestRunner,
        max_pairs: int = 6,
        max_tests_per_pair: int = 2,
    ):
        if max_pairs < 1:
            raise ValueError("max_pairs must be >= 1")
        if max_tests_per_pair < 1:
            raise ValueError("max_tests_per_pair must be >= 1")
        self.llm_caller = llm_caller
        self.test_runner = test_runner
        self.max_pairs = int(max_pairs)
        self.max_tests_per_pair = int(max_tests_per_pair)

    # -------------------------------------------------------------------
    # Single discriminating test generation
    # -------------------------------------------------------------------

    def generate_discriminating_test(
        self,
        patch_i: str,
        patch_j: str,
        task_context: str,
        *,
        target_index: int = 0,
        distinguishes_from_index: int = 1,
        attempt: int = 0,
    ) -> Optional[DiscriminatingTest]:
        """One LLM call to generate a test that distinguishes ``patch_i`` from ``patch_j``."""
        prompt = _PROMPT_TEMPLATE.format(
            task_context=task_context.strip() or "(no task context provided)",
            patch_i=patch_i,
            patch_j=patch_j,
        )
        if attempt > 0:
            # Light prompt diversification on retries to encourage a
            # different angle of attack.
            prompt += (
                f"\n\nRetry attempt {attempt}: please target a "
                "different behavioral aspect than your previous answer."
            )
        try:
            raw = self.llm_caller(prompt)
        except Exception:
            logger.exception("verification amplifier llm_caller raised; skipping")
            return None

        parsed = _parse_llm_test_response(raw)
        if not parsed:
            logger.debug(
                "verification amplifier failed to parse llm response: %r", raw[:200] if raw else raw
            )
            return None

        test_code = parsed.get("test_code")
        test_name = parsed.get("test_name")
        if not isinstance(test_code, str) or not test_code.strip():
            return None
        if not isinstance(test_name, str) or not test_name.strip():
            test_name = f"test_amp_{target_index}_vs_{distinguishes_from_index}_{attempt}"
        confidence_raw = parsed.get("confidence", 0.5)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        return DiscriminatingTest(
            test_code=test_code,
            test_name=test_name.strip(),
            target_patch=int(target_index),
            distinguishes_from=int(distinguishes_from_index),
            confidence=confidence,
        )

    # -------------------------------------------------------------------
    # End-to-end amplification
    # -------------------------------------------------------------------

    def amplify(
        self,
        patches: List[str],
        task_context: str,
        repo_path: Path,
    ) -> AmplificationResult:
        """Build a discrimination matrix, then pick a winner by win count."""
        n = len(patches)
        empty_matrix = DiscriminationMatrix(
            n_patches=n, tests=[], verdict_matrix=[[] for _ in range(n)]
        )
        if n == 0:
            return AmplificationResult(
                chosen_patch=-1,
                confidence=0.0,
                discrimination_matrix=empty_matrix,
                short_circuited=True,
                short_circuit_reason="no_patches",
                win_counts=[],
            )
        if n == 1:
            return AmplificationResult(
                chosen_patch=0,
                confidence=1.0,
                discrimination_matrix=empty_matrix,
                short_circuited=True,
                short_circuit_reason="single_patch",
                win_counts=[0],
            )

        guard_reason = _patches_indistinguishable(patches)
        if guard_reason is not None:
            logger.info(
                "verification amplifier short-circuit (%s): %d patches",
                guard_reason,
                n,
            )
            return AmplificationResult(
                chosen_patch=0,
                confidence=0.0,
                discrimination_matrix=empty_matrix,
                short_circuited=True,
                short_circuit_reason=guard_reason,
                win_counts=[0] * n,
            )

        # ---------------------------------------------------------------
        # 1) Generate discriminating tests pair-wise.
        # ---------------------------------------------------------------
        tests: List[DiscriminatingTest] = []
        cost_inferences = 0
        pairs_generated = 0
        for i in range(n):
            if pairs_generated >= self.max_pairs:
                break
            for j in range(i + 1, n):
                if pairs_generated >= self.max_pairs:
                    break
                pair_tests = 0
                for attempt in range(self.max_tests_per_pair):
                    cost_inferences += 1
                    candidate = self.generate_discriminating_test(
                        patches[i],
                        patches[j],
                        task_context,
                        target_index=i,
                        distinguishes_from_index=j,
                        attempt=attempt,
                    )
                    if candidate is None:
                        continue
                    tests.append(candidate)
                    pair_tests += 1
                if pair_tests > 0:
                    pairs_generated += 1

        # ---------------------------------------------------------------
        # 2) Run every test against every patch.
        # ---------------------------------------------------------------
        verdict_matrix: List[List[bool]] = [[] for _ in range(n)]
        cost_test_runs = 0
        for test in tests:
            for patch_idx in range(n):
                cost_test_runs += 1
                try:
                    passed = bool(self.test_runner(patches[patch_idx], test.test_code, repo_path))
                except Exception:
                    logger.exception(
                        "verification amplifier test_runner raised on patch %d; treating as fail",
                        patch_idx,
                    )
                    passed = False
                verdict_matrix[patch_idx].append(passed)

        matrix = DiscriminationMatrix(n_patches=n, tests=tests, verdict_matrix=verdict_matrix)

        # ---------------------------------------------------------------
        # 3) Score each patch by differential wins.
        # ---------------------------------------------------------------
        wins = [0] * n
        for k in range(len(tests)):
            for i in range(n):
                if not verdict_matrix[i][k]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if not verdict_matrix[j][k]:
                        # patch i passed test k, patch j failed -> i wins one
                        wins[i] += 1

        # Pick winner: argmax(wins), tiebreak on lower index (stable).
        if not tests:
            # No tests survived generation — nothing to amplify with.
            return AmplificationResult(
                chosen_patch=0,
                confidence=0.0,
                discrimination_matrix=matrix,
                cost_inferences=cost_inferences,
                cost_test_runs=cost_test_runs,
                short_circuited=False,
                short_circuit_reason="no_tests_generated",
                win_counts=wins,
            )

        max_wins = max(wins)
        winner = 0
        for idx, w in enumerate(wins):
            if w == max_wins:
                winner = idx
                break

        # Confidence: winner's wins / total possible wins for the winner.
        # A patch can win at most ``len(tests) * (n - 1)`` differential
        # comparisons (each test, each other patch). When there are no
        # discriminating outcomes, confidence is 0.
        max_possible = len(tests) * (n - 1)
        if max_possible <= 0:
            confidence = 0.0
        else:
            confidence = max_wins / max_possible

        return AmplificationResult(
            chosen_patch=winner,
            confidence=float(confidence),
            discrimination_matrix=matrix,
            cost_inferences=cost_inferences,
            cost_test_runs=cost_test_runs,
            short_circuited=False,
            short_circuit_reason=None,
            win_counts=wins,
        )


__all__ = [
    "AmplificationResult",
    "DiscriminatingTest",
    "DiscriminationMatrix",
    "VerificationAmplifier",
]
