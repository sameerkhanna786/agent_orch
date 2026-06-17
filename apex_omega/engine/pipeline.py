"""The ``pipeline()`` primitive — the one genuinely net-new build (Fusion Ledger
A2; plan §02.2.3 / §22.2.2).

Per-item staged streaming with NO inter-stage barrier: item A can be in stage 3
while item B is still in stage 1, so wall-clock collapses from
sum-of-slowest-per-stage to slowest-single-chain.  v1 has only barrier waves.

Determinism rule (load-bearing, §02.2.3): cache entries are keyed per
``(item, stage)`` and the returned order is a pure function of the input item
order + stage index — NEVER completion/wall-clock order.  A stage that raises
drops its item to ``None`` and skips that item's remaining stages (fail-loud, the
failure is narrated; a null is never silently treated as success).
"""

from __future__ import annotations

import inspect
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from ..journal.resume import resume_or_run_json


@dataclass
class Stage:
    name: str
    fn: Callable[..., Any]


def _as_stage(s: Any, idx: int) -> Stage:
    if isinstance(s, Stage):
        return s
    name = getattr(s, "__name__", None) or f"stage{idx}"
    return Stage(name=name, fn=s)


def _call_stage(fn: Callable[..., Any], prev: Any, item: Any, index: int) -> Any:
    """Call a stage with as many of (prev, item, index) as it declares."""
    try:
        n = len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        n = 3
    if n >= 3:
        return fn(prev, item, index)
    if n == 2:
        return fn(prev, item)
    return fn(prev)


def run_pipeline(
    engine,
    items: Sequence[Any],
    *stages: Any,
    item_id: Optional[Callable[[Any], str]] = None,
    journal_stages: bool = True,
    budget=None,
) -> list[Any]:
    """Stream ``items`` through ``stages`` with per-item concurrency.

    ``item_id(item) -> str`` provides a stable id for journaling (defaults to the
    item's ``.id`` attribute, else its index).  Stage outputs must be
    JSON-serializable when ``journal_stages`` is True (the typed inter-stage
    artifacts the plan mandates); expensive worker calls inside a stage should go
    through ``engine.agent`` which journals at the agent level (diffs materialized).
    """
    if len(items) > 4096:
        raise ValueError("pipeline: at most 4096 items per call")
    stage_objs = [_as_stage(s, i) for i, s in enumerate(stages)]
    bud = budget if budget is not None else engine.budget

    def _id(it: Any, i: int) -> str:
        if item_id is not None:
            return str(item_id(it))
        return str(getattr(it, "id", i))

    results: dict[int, Any] = {}
    lock = threading.Lock()

    def _run_chain(i: int, item: Any) -> None:
        iid = _id(item, i)
        prev: Any = item
        for sidx, stage in enumerate(stage_objs):
            # Budget governs whether to START the next stage; it never aborts an
            # in-flight stage AND never discards a stage that already produced
            # output (budget.py invariant: never suppress a candidate that has
            # execution evidence). Only an item that never started -> drop sentinel.
            if bud is not None and not bud.can_start():
                engine.log(f"pipeline: budget exhausted before {iid}:{stage.name}; stopping new stages")
                if sidx == 0:
                    prev = None  # nothing ran yet -> no execution evidence -> drop
                # else: keep the last good output (prev holds stages 0..sidx-1)
                break
            try:
                if journal_stages:
                    components = {
                        "kind": "pipeline_stage",
                        "item_id": iid,
                        "stage_index": sidx,
                        "stage_name": stage.name,
                        # Key on the stable item id for stage 0 (raw items may be
                        # non-JSON-native, restoring the Sequence[Any] contract) and
                        # on the prior stage's JSON artifact thereafter, so a changed
                        # upstream re-runs this stage.
                        "scoped_inputs": ({"prev": prev} if sidx > 0 else {"item_id": iid}),
                    }
                    out, _hit = resume_or_run_json(
                        engine.journal, components,
                        lambda p=prev, it=item, ix=i, st=stage: _call_stage(st.fn, p, it, ix),
                        kind="pipeline_stage", node_id=f"{iid}:{stage.name}",
                    )
                else:
                    out = _call_stage(stage.fn, prev, item, i)
            except Exception as exc:  # fail-loud: narrate, drop item to None
                engine.log(f"pipeline: stage {iid}:{stage.name} raised {type(exc).__name__}: {exc}")
                prev = None
                break
            prev = out
            if prev is None:
                # a stage may signal terminal-drop by returning None
                break
        with lock:
            results[i] = prev

    n_workers = max(1, min(engine.max_workers, len(items) or 1))
    with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="apexω-pipe") as pool:
        futs = [pool.submit(_run_chain, i, it) for i, it in enumerate(items)]
        for f in as_completed(futs):
            f.result()  # propagate unexpected executor-level errors (not stage errors)

    # Deterministic return order: by input index, NEVER completion order.
    return [results.get(i) for i in range(len(items))]
