"""loop_until_dry — keep widening best-of-N waves until the work runs dry (Backbone 2.3).

For unknown-size discovery: simple ``while count < N`` schedules miss the tail and
over-spend the head. This calls ``make_round(i) -> [thunk]`` through the PLATEAU-AWARE
``ctx.parallel`` until (a) an accepted candidate appears (completion-first), (b) K
consecutive rounds add no new best pass-rate, (c) the governor plateaus (PlateauStop),
or (d) ``max_rounds``. Returns every candidate produced. With small ``k_dry`` this is
just a couple of best-of-N waves, so it degrades cleanly.
"""

from __future__ import annotations

from ..errors import PlateauStop


def loop_until_dry(ctx, make_round, *, k_dry: int = 2, max_rounds: int = 64,
                   stop_on_accept: bool = True, key=None, seen=None):
    """Keep running rounds until the work runs dry. By default a round is "dry" when it adds no
    new best pass-rate. For unknown-size DISCOVERY, pass ``key=callable`` (and optionally a shared
    ``seen`` set): each round's candidates are deduped vs everything SEEN, the new keys are added
    to ``seen`` BEFORE counting, and a round with NO new keys is dry — the convergence rule from
    the paradigm (dedupe against everything seen, NOT just confirmed, or rejected items reappear
    and the loop never converges)."""
    produced: list = []
    best = -1.0
    dry = 0
    seen = seen if seen is not None else set()
    for i in range(max(1, max_rounds)):
        thunks = list(make_round(i) or [])
        if not thunks:
            break
        try:
            out = ctx.parallel(thunks)
        except PlateauStop:
            break  # the governor halted further fan-out -> stop cleanly
        round_cands = [c for c in out if c is not None]
        produced.extend(round_cands)
        if stop_on_accept and any(getattr(c, "accepted", False) for c in round_cands):
            break
        if key is not None:
            # DISCOVERY mode: dry iff no NEW item key this round (dedupe vs everything SEEN).
            fresh = 0
            for c in round_cands:
                try:
                    k = key(c)
                except Exception:
                    continue
                if k not in seen:
                    seen.add(k)
                    fresh += 1
            if fresh > 0:
                dry = 0
            else:
                dry += 1
                if dry >= max(1, k_dry):
                    break
            continue
        rbest = max((float(c.public_signal_score or 0.0) for c in round_cands), default=-1.0)
        if rbest > best + 1e-9:
            best = rbest
            dry = 0
        else:
            dry += 1
            if dry >= max(1, k_dry):
                break
    return produced
