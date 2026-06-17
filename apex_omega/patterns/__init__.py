"""Composable quality patterns for authored orchestrators (Backbone 2.3).

Each pattern is a host-side function taking ``ctx`` first and is surfaced as a thin
``ctx.<name>`` wrapper. The shared invariants (enforced structurally, not by trust):

  * NONE can set ``accepted``. They mutate candidates ONLY via the soft-write seam
    (``Candidate.set_soft`` / ``Candidate.refute``) or by producing a fresh
    EXECUTION-SCORED ``ctx.solve_attempt`` — so a pattern can downgrade or re-rank, but
    never promote an unverified solve (Cardinal Contract preserved).
  * Each DEGRADES to plain best-of-N at zero knobs (a no-op / single-judge / single
    best), so adding a pattern can never do worse than the floor.
  * Refute is OR/majority-over-skeptics; judging is a sub-execution tiebreak only.
"""

from .critic import CRITIC_SCHEMA, completeness_critic
from .judge import JUDGE_SCHEMA, judge_panel, judge_select
from .loop import loop_until_dry
from .quarantine import quarantined_ask
from .route import CLASSIFY_SCHEMA, classify_and_route
from .synthesize import SYNTH_SCHEMA, synthesize
from .tournament import TOURNEY_SCHEMA, tournament
from .verify import VERDICT_SCHEMA, adversarial_filter, adversarial_verify

__all__ = [
    "adversarial_verify", "adversarial_filter", "judge_panel", "judge_select", "synthesize",
    "loop_until_dry", "completeness_critic", "tournament", "classify_and_route", "quarantined_ask",
    "VERDICT_SCHEMA", "JUDGE_SCHEMA", "SYNTH_SCHEMA", "CRITIC_SCHEMA", "TOURNEY_SCHEMA",
    "CLASSIFY_SCHEMA",
]
