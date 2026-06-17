"""RunGovernor — the single "may we continue?" authority (Backbone 2.1).

NOT a timer. The DEFAULT IS UNBOUNDED (no token/agent budget). The always-on guards
are the per-RUN agent ceiling and a PLATEAU-STOP (k consecutive fan-out rounds with no
pass-rate improvement) — so even an unbounded, pathological ``while True: ctx.parallel(...)``
terminates without a clock and without discarding verified work. A token budget or a soft
agent budget is strictly OPT-IN.
"""

from __future__ import annotations

import math
from typing import Optional


class RunGovernor:
    def __init__(self, *, engine, agent_ceiling: int = 1000, token_budget: Optional[int] = None,
                 agent_budget: Optional[int] = None, plateau_k_dry: int = 2,
                 base_patience: int = 4, nonresult_streak_cut: int = 8,
                 sterile_streak_cut: int = 8, token_cut_fraction: float = 0.35):
        self._engine = engine
        self.agent_ceiling = int(agent_ceiling)
        self.token_budget = token_budget          # opt-in (None = unbounded; mirrors engine.budget.total)
        self.agent_budget = agent_budget          # opt-in soft cap (None = ceiling only)
        self.plateau_k_dry = max(1, int(plateau_k_dry))
        # ---- CUT-LOSSES detector hyperparameters (Backbone 2.4) — FROZEN as ONE set ----
        # These are repo-AGNOSTIC (no decision path reads repo identity) and pre-registered here
        # with rationale; they are NOT fit per-repo. External-validity caveat: they were reasoned
        # against the 4-repo ladder (voluptuous/jinja/mimesis/pydantic) with no held-out repo, so
        # report them as the registered stop-rule, not a tuned optimum (review F3).
        #  - base_patience=4: the SOFT no-progress plateau waits base + ceil(log2(agents+1)) DRY
        #    waves before cutting. base=4 guarantees a width-1 (sequential, e.g. RALPH) lineage
        #    reaches the empirical slow-winner depth — the jinja base-s0 case that stayed flat for
        #    ~6 waves then SOLVED at wave 6 — on BOTH wide-wave (omega) and width-1 (ralph) arms
        #    (review M2). More budget -> more rope; cheap tasks still stop quickly.
        #  - nonresult_streak_cut / sterile_streak_cut = 8 ATTEMPTS (not waves): the hard cuts now
        #    count ATTEMPTS so a width-1 ralph lineage and a width-N omega wave are cut after a
        #    comparable number of ATTEMPTS, keeping the arm comparison apples-to-apples (review M1).
        #  - token_cut_fraction=0.35: opt-in token floor only (inactive on default unbounded runs).
        self.base_patience = max(1, int(base_patience))
        self.nonresult_streak_cut = max(1, int(nonresult_streak_cut))
        self.sterile_streak_cut = max(1, int(sterile_streak_cut))
        self.token_cut_fraction = float(token_cut_fraction)

    def can_start(self, *, reserve: int = 1) -> bool:
        """May a NEW unit of work be dispatched? Gates on the (opt-in) token budget, the
        (opt-in) soft agent budget, and the always-on per-run agent ceiling. Never a clock."""
        if not self._engine.budget.can_start(reserve=reserve):
            return False
        used = self._engine.agents_used()
        if self.agent_budget is not None and used >= self.agent_budget:
            return False
        return used < self.agent_ceiling

    def patience(self, agents_used: int) -> int:
        """Budget-aware no-progress patience: more waves of rope as compute escalates."""
        return self.base_patience + int(math.ceil(math.log2(max(0, int(agents_used)) + 1)))

    def should_continue_waves(self, *, dry_rounds: int) -> bool:
        """Back-compat single-signal form (raw pass_rate plateau). Prefer ``verdict``."""
        return self.can_start() and dry_rounds < self.plateau_k_dry

    def verdict(self, state: dict) -> tuple[bool, str]:
        """The SINGLE cut-losses authority. Returns ``(continue, reason)``. Evaluated each
        wave AFTER the ctx.parallel barrier. Order: hard cuts (objectively dead) first, then
        the budget-aware soft plateau on the BEST distance-to-solve, then the opt-in token
        floor, then the agent ceiling. ``state`` carries:
          dry_rounds                 consecutive WAVES with no BEST gold/pass improvement
          agents_used                agents dispatched so far (drives patience)
          nonresult_streak           consecutive ATTEMPTS producing zero usable work (size-invariant)
          sterile_streak             consecutive ATTEMPTS with no new useful diff AND no improvement
          tokens_since_improvement   output tokens spent since the BEST last improved
        Reasons: cut:* = a genuine non-progress FAILURE (CutLosses); stop:/plateau: = an
        honest "explored, no headroom" stop (PlateauStop)."""
        # HARD CUTS — objectively dead states no amount of identical rollouts escapes.
        if int(state.get("nonresult_streak", 0)) >= self.nonresult_streak_cut:
            return (False, "cut:nonresult-streak")
        if int(state.get("sterile_streak", 0)) >= self.sterile_streak_cut:
            return (False, "cut:sterile-diff-streak")
        # BUDGET-AWARE SOFT PLATEAU — no distance-to-solve gain within the patience window.
        if int(state.get("dry_rounds", 0)) >= self.patience(state.get("agents_used", 0)):
            return (False, "cut:no-progress")
        # OPT-IN TOKEN FLOOR — only when a token budget is set (unbounded runs skip this).
        tb = self.token_budget
        if tb and int(state.get("tokens_since_improvement", 0)) > self.token_cut_fraction * tb:
            return (False, "cut:tokens-since-improvement")
        # AGENT CEILING / BUDGET — honest "no headroom left" stop (not a failure-to-progress).
        if not self.can_start():
            return (False, "stop:agent-ceiling")
        return (True, "continue")

    def to_dict(self) -> dict:
        return {"agent_ceiling": self.agent_ceiling, "token_budget": self.token_budget,
                "agent_budget": self.agent_budget, "plateau_k_dry": self.plateau_k_dry,
                "base_patience": self.base_patience,
                "nonresult_streak_cut": self.nonresult_streak_cut,
                "sterile_streak_cut": self.sterile_streak_cut,
                "token_cut_fraction": self.token_cut_fraction}
