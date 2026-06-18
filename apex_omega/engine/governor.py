"""RunGovernor — the single "may we continue?" authority (Backbone 2.1).

NOT a timer. The DEFAULT IS UNBOUNDED (no token/agent budget). The always-on guards
are the per-RUN agent ceiling and a PLATEAU-STOP (k consecutive fan-out rounds with no
pass-rate improvement) — so even an unbounded, pathological ``while True: ctx.parallel(...)``
terminates without a clock and without discarding verified work. A token budget or a soft
agent budget is strictly OPT-IN.
"""

from __future__ import annotations

import os
from typing import Optional


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    try:
        return int(v) if v and v.strip() else default
    except ValueError:
        return default


class RunGovernor:
    def __init__(self, *, engine, agent_ceiling: int = 1000, token_budget: Optional[int] = None,
                 agent_budget: Optional[int] = None, plateau_k_dry: int = 2,
                 plateau_patience: Optional[int] = None, nonresult_streak_cut: int = 8,
                 sterile_streak_cut: int = 8, token_cut_fraction: float = 0.35,
                 plateau_wall_seconds: Optional[float] = None,
                 plateau_patience_meas: Optional[int] = None,
                 harness_stall_cut: int = 8):
        self._engine = engine
        self.agent_ceiling = int(agent_ceiling)
        self.token_budget = token_budget          # opt-in (None = unbounded; mirrors engine.budget.total)
        self.agent_budget = agent_budget          # opt-in soft cap (None = ceiling only)
        self.plateau_k_dry = max(1, int(plateau_k_dry))
        # ---- CUT-LOSSES detector hyperparameters (Backbone 2.4) — FROZEN as ONE set ----
        # These are repo-AGNOSTIC (no decision path reads repo identity) and pre-registered here
        # with rationale; they are NOT fit per-repo. External-validity caveat: they were reasoned
        # against the 4-repo ladder (voluptuous/jinja/mimesis/pydantic) with no held-out repo, so
        # report them as the registered stop-rule, not a tuned optimum (review F3). ALL cut signals
        # are measured in ATTEMPTS (agents), NOT waves — so the rule is invariant to the wave
        # schedule (omega DOUBLES wave size; ralph is width-1) and the arm comparison stays
        # apples-to-apples (review M1/M2). A wave-based plateau would never fire under a doubling
        # schedule (patience would grow as fast as the wave count) — the attempt unit fixes that.
        #  - plateau_patience=64 ATTEMPTS since the BEST distance-to-solve last improved: the SOFT
        #    no-progress cut. 64 covers the deepest observed slow-but-real winner (omega jinja
        #    base-s0 stayed flat for ~63 agents across 6 doubling waves, then SOLVED) AND the
        #    sequential ralph case (a 7th-attempt solve after 6 flat). Env-overridable
        #    (APEX_OMEGA_PLATEAU_PATIENCE) for smokes/tuning.
        #  - nonresult_streak_cut / sterile_streak_cut = 8 ATTEMPTS: hard cuts for objectively-dead
        #    states (zero usable work; empty/repeated diffs) — fire well before the soft floor.
        #  - token_cut_fraction=0.35: opt-in token floor only (inactive on default unbounded runs).
        self.plateau_patience = max(1, int(plateau_patience if plateau_patience is not None
                                           else _env_int("APEX_OMEGA_PLATEAU_PATIENCE", 64)))
        self.nonresult_streak_cut = max(1, int(nonresult_streak_cut))
        self.sterile_streak_cut = max(1, int(sterile_streak_cut))
        self.token_cut_fraction = float(token_cut_fraction)
        # ---- SPFG+ "Solve-Progress Frontier Governor" arms (Backbone 2.5) ----
        # The cut:no-progress soft plateau is now a FRONTIER DUAL-AND: it fires only when a
        # genuine no-solve-progress plateau is reached on BOTH the VALID-measurement window
        # (valid_measurements_since_improvement >= plateau_patience_meas) AND the journaled
        # VALID-measurement wall clock (seconds_since_frontier_improved >= plateau_wall_seconds).
        # Indeterminate (harness/scorer) measurements are NEUTRAL to both arms — a wall of them
        # trips the DISTINCT cut:harness-stall (indeterminate_streak >= harness_stall_cut), never
        # cut:no-progress. The legacy attempts arm (plateau_patience) is RETAINED as a back-compat
        # backstop and can still cut on attempts alone (it never cuts EARLIER than before); the new
        # frontier dual-AND is an additional, indeterminate-aware path. Defaults from the SHARED
        # frontier module (single-sourced; APEX_FRONTIER_* / LADDER_* env-overridable; no repo
        # identity in any decision path).
        from .frontier import FrontierParams
        _fp = FrontierParams.from_env()
        self.plateau_wall_seconds = float(
            plateau_wall_seconds if plateau_wall_seconds is not None else _fp.w_time)
        self.plateau_patience_meas = max(1, int(
            plateau_patience_meas if plateau_patience_meas is not None else _fp.w_meas))
        self.harness_stall_cut = max(1, int(harness_stall_cut))

    def can_start(self, *, reserve: int = 1) -> bool:
        """May a NEW unit of work be dispatched? Gates on the (opt-in) token budget, the
        (opt-in) soft agent budget, and the always-on per-run agent ceiling. Never a clock."""
        if not self._engine.budget.can_start(reserve=reserve):
            return False
        used = self._engine.agents_used()
        if self.agent_budget is not None and used >= self.agent_budget:
            return False
        return used < self.agent_ceiling

    def should_continue_waves(self, *, dry_rounds: int) -> bool:
        """Back-compat single-signal form (raw pass_rate plateau). Prefer ``verdict``."""
        return self.can_start() and dry_rounds < self.plateau_k_dry

    def verdict(self, state: dict) -> tuple[bool, str]:
        """The SINGLE cut-losses authority. Returns ``(continue, reason)``. Evaluated each
        wave AFTER the ctx.parallel barrier. Order: hard cuts (objectively dead) first, then
        the budget-aware soft plateau on the BEST distance-to-solve, then the opt-in token
        floor, then the agent ceiling. ``state`` carries:
          attempts_since_improvement ATTEMPTS dispatched since the BEST last improved
          agents_used                agents dispatched so far
          nonresult_streak           consecutive ATTEMPTS producing zero usable work (size-invariant)
          sterile_streak             consecutive ATTEMPTS with no new useful diff AND no improvement
          tokens_since_improvement   output tokens spent since the BEST last improved
          valid_measurements_since_improvement  VALID measurements since the FRONTIER last rose (SPFG+)
          seconds_since_frontier_improved       journaled VALID-measurement wall-secs since the rise (SPFG+)
          indeterminate_streak       consecutive harness/scorer-failed measurements (SPFG+; -> harness-stall)
        Reasons: cut:* = a genuine non-progress FAILURE (CutLosses); cut:harness-stall = a
        harness/scorer wall (no real measurement); stop:/plateau: = an honest "explored, no
        headroom" stop (PlateauStop)."""
        # HARD CUTS — objectively dead states no amount of identical rollouts escapes.
        if int(state.get("nonresult_streak", 0)) >= self.nonresult_streak_cut:
            return (False, "cut:nonresult-streak")
        if int(state.get("sterile_streak", 0)) >= self.sterile_streak_cut:
            return (False, "cut:sterile-diff-streak")
        # HARNESS-STALL — a wall of indeterminate (harness/scorer-failed) measurements that
        # never produced a real test outcome. A DISTINCT outcome from a genuine no-progress
        # plateau: the arm was never actually measured, so it must not be booked as
        # cut:no-progress. Fires before the soft plateau (the indeterminate stretch is
        # neutral to the frontier arms, so those would never fire here anyway).
        if int(state.get("indeterminate_streak", 0)) >= self.harness_stall_cut:
            return (False, "cut:harness-stall")
        # SOFT PLATEAU (SPFG+ FRONTIER DUAL-AND) — a GENUINE no-solve-progress plateau: BOTH
        # the VALID-measurement window AND the journaled VALID-measurement wall clock have
        # elapsed since the frontier (best gold-pass count / pass_rate) last rose. Any frontier
        # rise resets BOTH arms; indeterminate measurements advance NEITHER. This is the arm that
        # never force-cuts a still-progressing run (the minitorch case).
        if (int(state.get("valid_measurements_since_improvement", 0)) >= self.plateau_patience_meas
                and float(state.get("seconds_since_frontier_improved", 0.0)) >= self.plateau_wall_seconds):
            return (False, "cut:no-progress")
        # LEGACY ATTEMPTS BACKSTOP — the original attempt-since-improvement no-progress floor,
        # RETAINED for back-compat (it never cuts EARLIER than before; the frontier dual-AND above
        # is the indeterminate-aware primary path).
        if int(state.get("attempts_since_improvement", 0)) >= self.plateau_patience:
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
                "plateau_patience": self.plateau_patience,
                "nonresult_streak_cut": self.nonresult_streak_cut,
                "sterile_streak_cut": self.sterile_streak_cut,
                "token_cut_fraction": self.token_cut_fraction,
                "plateau_wall_seconds": self.plateau_wall_seconds,
                "plateau_patience_meas": self.plateau_patience_meas,
                "harness_stall_cut": self.harness_stall_cut}
