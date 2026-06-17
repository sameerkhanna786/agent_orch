"""Strategy-axis diversity for parallel rollouts (Phase A.4).

Historically the rollout engine diversified K rollouts via temperature
sampling (``RolloutConfig.diversity_temperatures``) and free-form prompt
strategy enums (``RolloutConfig.diversity_prompts``). Both knobs vary
*how the same intent is sampled* but not *what intent is sampled* — so
8 rollouts of the same problem statement collapse onto 8 minor
re-phrasings of the same approach.

This module replaces the dominant axis of diversity with explicit
*strategic axes*. Each rollout is assigned one of seven canonical
strategies via round-robin. The strategy's prompt prefix is prepended
to the agent's task prompt before any agent surface (scaffolded
patcher, CLI agent, in-container V5 agent) consumes it.

The seven axes were chosen to span the qualitative space of "valid
ways a competent engineer might approach a bug fix":

    minimal_fix, refactor, defensive, isolated_helper,
    inverted_logic, two_step_decompose, test_first_red_green

K=8 rollouts produce K different strategic approaches, not K samples
of the same approach. The first axis cycles back at index 7 — the
distribution is uniform under round-robin and reproducible per-rollout.

Public surface:

    STRATEGY_AXES               — canonical axis names
    STRATEGY_PROMPT_PREFIXES    — full multi-sentence prompt prefixes
    assign_strategy(...)        — round-robin assignment
    apply_strategy_prefix(...)  — prepend prefix to a task prompt

The temperature / prompt-style knobs are preserved as tertiary
diversity (``RolloutConfig.diversity_temperatures`` /
``RolloutConfig.diversity_prompts``) for back-compat; callers that
want pure-strategy diversity should configure
``diversity_temperatures=[0.7]`` and let strategies own the variance.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Canonical strategy axes
# ---------------------------------------------------------------------------

STRATEGY_MINIMAL_FIX = "minimal_fix"
STRATEGY_REFACTOR = "refactor"
STRATEGY_DEFENSIVE = "defensive"
STRATEGY_ISOLATED_HELPER = "isolated_helper"
STRATEGY_INVERTED_LOGIC = "inverted_logic"
STRATEGY_TWO_STEP_DECOMPOSE = "two_step_decompose"
STRATEGY_TEST_FIRST_RED_GREEN = "test_first_red_green"

STRATEGY_AXES: List[str] = [
    STRATEGY_MINIMAL_FIX,
    STRATEGY_REFACTOR,
    STRATEGY_DEFENSIVE,
    STRATEGY_ISOLATED_HELPER,
    STRATEGY_INVERTED_LOGIC,
    STRATEGY_TWO_STEP_DECOMPOSE,
    STRATEGY_TEST_FIRST_RED_GREEN,
]


# ---------------------------------------------------------------------------
# Prompt prefixes (3-5 substantive sentences each)
# ---------------------------------------------------------------------------

STRATEGY_PROMPT_PREFIXES: Dict[str, str] = {
    STRATEGY_MINIMAL_FIX: (
        "STRATEGY: TAKE THE MINIMAL FIX APPROACH.\n"
        "Make the smallest coherent change that satisfies the full objective. "
        "Let failing tests, import edges, and public contracts define the "
        "implementation boundary; edit count alone is not a scope signal. "
        "When the evidence points to a connected cluster, repair that cluster "
        "instead of masking only the first symptom. Resist unrelated refactors, "
        "renames, or cleanup, and keep each edited line tied to the objective. "
        "The fix must be obviously correct on inspection because every change "
        "has a direct behavioral reason."
    ),
    STRATEGY_REFACTOR: (
        "STRATEGY: REFACTOR AS PART OF THE FIX.\n"
        "Treat the bug as evidence that the surrounding code shape is "
        "fragile. Restructure the affected function or module so the bug "
        "becomes impossible to reintroduce: extract responsibilities, "
        "clarify control flow, replace ad-hoc branching with a coherent "
        "abstraction. The fix should leave the touched code measurably "
        "cleaner than you found it. Wider edits are acceptable when they "
        "demonstrably remove the class of mistake the bug exemplifies."
    ),
    STRATEGY_DEFENSIVE: (
        "STRATEGY: ADD DEFENSIVE GUARDS AND INPUT VALIDATION.\n"
        "Assume hostile or malformed inputs and harden every entry point "
        "involved in the bug. Add explicit type/None/empty/range checks, "
        "raise clear exceptions on violations, and document the contract "
        "in the docstring. The fix should not just handle the reported "
        "case — it should refuse to silently misbehave on any nearby "
        "input that violates the function's preconditions. Validation "
        "comes first; the corrective logic comes after the guard."
    ),
    STRATEGY_ISOLATED_HELPER: (
        "STRATEGY: EXTRACT AN ISOLATED HELPER FUNCTION.\n"
        "Identify the smallest piece of logic that contains the bug and "
        "lift it into a new private helper function with a precise name "
        "and signature. Fix the bug inside the helper. The original call "
        "site becomes a single delegation. This makes the corrected logic "
        "independently testable, gives the bug fix a clear unit-test "
        "target, and prevents the broken pattern from being copy-pasted "
        "elsewhere in the file."
    ),
    STRATEGY_INVERTED_LOGIC: (
        "STRATEGY: RETHINK THE PREDICATE / CONTROL FLOW.\n"
        "The bug suggests the original control flow models the problem "
        "the wrong way around. Invert the predicate, swap the branches, "
        "or replace cascading if/elif with a structured early-return "
        "ladder. Ask whether the function should be expressed in terms of "
        "what it accepts rather than what it rejects, or vice versa. The "
        "corrected code should read as the natural reading of the spec, "
        "not as a patch on top of a broken framing."
    ),
    STRATEGY_TWO_STEP_DECOMPOSE: (
        "STRATEGY: DO IT IN TWO CLEAR STAGES.\n"
        "Decompose the fix into two visibly separate steps. Stage 1: "
        "compute or normalize the intermediate value the buggy code "
        "should have been operating on, and bind it to a clearly named "
        "local. Stage 2: perform the corrective action against that "
        "named intermediate. The two-stage shape forces the bug surface "
        "into the open — a future reader sees the data the function "
        "actually consumes before it sees what the function does with it."
    ),
    STRATEGY_TEST_FIRST_RED_GREEN: (
        "STRATEGY: WRITE THE TEST FIRST, THEN MAKE IT PASS.\n"
        "Before touching the implementation, write or extend a test that "
        "fails on the current buggy code and passes only when the bug is "
        "fixed (red-green TDD). The test should pin down the exact "
        "behavior the spec demands at the affected boundary. Only after "
        "the failing test is in place may you edit the implementation; "
        "the fix is complete when that test transitions from red to "
        "green and no previously-passing test regresses."
    ),
}


# ---------------------------------------------------------------------------
# Assignment & application
# ---------------------------------------------------------------------------


def _normalized_enabled(enabled_strategies: Optional[List[str]]) -> List[str]:
    """Return a non-empty list of valid strategy names.

    Filters out unknown axes; falls back to :data:`STRATEGY_AXES` when
    the resulting list is empty so callers always get a usable rotation.
    """
    if not enabled_strategies:
        return list(STRATEGY_AXES)
    filtered = [s for s in enabled_strategies if s in STRATEGY_PROMPT_PREFIXES]
    return filtered or list(STRATEGY_AXES)


def assign_strategy(
    rollout_index: int,
    n_rollouts: int,
    enabled_strategies: Optional[List[str]] = None,
) -> str:
    """Round-robin assignment of a strategy axis to a rollout slot.

    With K rollouts and N enabled axes, rollout ``i`` is assigned
    ``enabled[i % N]``. When K <= N every rollout sees a distinct
    strategy; when K > N the cycle repeats from the front.

    ``rollout_index`` is the 0-based rollout id.
    ``n_rollouts`` is reserved for future weighted/coverage-aware
    schemes (e.g. emphasizing minimal_fix in low-K runs); it is not
    consulted by the round-robin scheme but is part of the stable
    public API so callers do not have to refactor when smarter
    assignment lands.
    """
    enabled = _normalized_enabled(enabled_strategies)
    if rollout_index < 0:
        rollout_index = 0
    return enabled[rollout_index % len(enabled)]


def get_prompt_prefix(strategy: str) -> str:
    """Return the substantive prompt prefix for ``strategy``.

    Returns the empty string for unknown axes so callers can apply the
    prefix unconditionally without an additional guard.
    """
    return STRATEGY_PROMPT_PREFIXES.get(strategy, "")


def apply_strategy_prefix(prompt: str, strategy: str) -> str:
    """Prepend the strategy prefix (with a blank-line separator) to ``prompt``.

    Returns ``prompt`` unchanged when ``strategy`` is unknown / empty.
    Idempotent in spirit: callers that pass the same prompt + strategy
    twice will see the prefix added twice — the engine is responsible
    for invoking this exactly once per rollout.
    """
    prefix = get_prompt_prefix(strategy)
    if not prefix:
        return prompt
    if not prompt:
        return prefix
    return f"{prefix}\n\n{prompt}"


__all__ = [
    "STRATEGY_AXES",
    "STRATEGY_PROMPT_PREFIXES",
    "STRATEGY_MINIMAL_FIX",
    "STRATEGY_REFACTOR",
    "STRATEGY_DEFENSIVE",
    "STRATEGY_ISOLATED_HELPER",
    "STRATEGY_INVERTED_LOGIC",
    "STRATEGY_TWO_STEP_DECOMPOSE",
    "STRATEGY_TEST_FIRST_RED_GREEN",
    "assign_strategy",
    "apply_strategy_prefix",
    "get_prompt_prefix",
]
