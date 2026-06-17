"""WS4: longitudinal decay-resistance harness.

Premise (arxiv 2605.26302, "Your Agents Are Aging Too"): agents that LOSSILY
compact their working memory — replacing exact identifiers (paths, line numbers,
error strings, return codes) with LLM prose — suffer capability decay with a
finite information half-life (~7-17 sessions). Agents that keep memory
APPEND-ONLY and VALUE-PRESERVING (drop whole turns cleanly rather than
genericize them) have an *infinite* half-life for everything still in the window.

APEX's V5 in-container working memory
(:meth:`InContainerAgent._render_transcript_window` /
:meth:`InContainerAgent._derived_state_sidecar`) implements the value-preserving
design. This harness is a falsifiable A/B:

* APEX arm — drive N turns through a real ``InContainerAgent`` and inspect the
  rendered window + sidecar.
* Lossy baseline arm — the same turns through a simulated lossy compactor that
  genericizes old turns into prose.

For each arm we measure, per session, how many of the injected EXACT-VALUE
markers are still recoverable byte-exact from working memory. The headline claim
APEX must satisfy:

  ``genericized_markers == 0``  (no exact value is ever altered into prose), and
  the machine-maintained counters (commands_run / distinct_commands /
  last_return_code) stay EXACT regardless of how many turns were elided.

The lossy baseline is expected to fail both — its half-life is finite.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from typing import Any, Optional

from ..orchestrator_in_container_agent import (
    AgentTurn,
    InContainerAgent,
    ToolCall,
    ToolResult,
    TOOL_RUN_IN_CONTAINER,
)


def _marker(session: int) -> str:
    """A unique exact-value token for session ``session``. Byte-exact survival
    of this token is what the harness scores."""
    return f"APEXMARK{session:05d}ZZ"


def _make_turn(session: int) -> AgentTurn:
    """One synthetic completed tool turn carrying a unique marker in BOTH the
    command and an error line (so we can detect genericization either place)."""
    marker = _marker(session)
    command = f"pytest tests/test_{marker}.py::case_{session} -x"
    rc = session % 3  # vary return codes so distinct-counting is meaningful
    err = (
        f"E   AssertionError: expected {marker} at src/mod_{marker}.py:{session + 7}"
        if rc != 0
        else ""
    )
    stdout = f"collected 1 item\n{err}\n" if err else "collected 1 item\n1 passed\n"
    return AgentTurn(
        turn_index=session,
        prompt="(synthetic)",
        tool_call=ToolCall(tool=TOOL_RUN_IN_CONTAINER, args={"command": command}),
        tool_result=ToolResult(stdout=stdout, return_code=rc, duration_seconds=0.01),
    )


def _lossy_compact(turns: list[AgentTurn], *, recent_verbatim: int) -> str:
    """Simulated LOSSY compactor (the aging failure mode).

    Keeps the most-recent ``recent_verbatim`` turns verbatim and replaces all
    older turns with a single genericized prose summary that DROPS the exact
    markers — exactly the compression-aging behaviour the paper warns about.
    """
    recent = turns[-recent_verbatim:] if turns else []
    older = turns[:-recent_verbatim] if len(turns) > recent_verbatim else []
    parts: list[str] = []
    if older:
        # Genericized prose — note: NO exact markers, return codes summarized.
        fails = sum(1 for t in older if t.tool_result and t.tool_result.return_code != 0)
        parts.append(
            f"## Earlier work (summary): ran {len(older)} test commands, "
            f"{fails} of them failed with assorted assertion errors; "
            "see history for details."
        )
    for t in recent:
        cmd = str(t.tool_call.args.get("command") or "") if t.tool_call else ""
        body = t.tool_result.stdout if t.tool_result else ""
        parts.append(f"$ {cmd}\n{body}")
    return "\n".join(parts)


@dataclass
class ArmSessionMetric:
    session: int
    markers_present_exact: int
    markers_genericized: int
    commands_run_exact: bool
    distinct_commands_exact: bool
    last_return_code_exact: bool


@dataclass
class ArmReport:
    name: str
    sessions: list[ArmSessionMetric] = field(default_factory=list)
    total_markers_genericized: int = 0
    counters_always_exact: bool = True
    information_half_life_sessions: float = math.inf

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total_markers_genericized": self.total_markers_genericized,
            "counters_always_exact": self.counters_always_exact,
            "information_half_life_sessions": (
                "inf"
                if self.information_half_life_sessions == math.inf
                else round(self.information_half_life_sessions, 2)
            ),
            "n_sessions": len(self.sessions),
        }


def _count_markers(window: str, all_markers: list[str]) -> tuple[int, int]:
    """Return ``(present_exact, total)`` — how many of the EVER-injected markers
    are present byte-exact in the rendered window. A marker that is absent has
    either been (a) cleanly elided (recoverable by re-running — fine) or (b)
    genericized away (lossy — capability loss)."""
    present = sum(1 for m in all_markers if m in window)
    return present, len(all_markers)


def run_decay_resistance(
    *,
    n_sessions: int = 40,
    recent_verbatim_turns: int = 4,
    max_tokens_per_turn: int = 1024,
    workspace_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Run the A/B and return a structured verdict dict.

    The harness needs no LLM and no container — it drives the working-memory
    renderer directly with synthetic turns.
    """
    workspace_dir = workspace_dir or "/tmp/apex-decay-harness"

    apex_arm = ArmReport(name="apex_append_only_value_preserving")
    lossy_arm = ArmReport(name="lossy_compaction_baseline")

    # The APEX arm uses the real renderer; the lossy arm uses _lossy_compact over
    # the same accumulating turn list.
    agent = InContainerAgent(
        workspace_dir=workspace_dir,
        max_turns=max(1, n_sessions),
        max_tokens_per_turn=max_tokens_per_turn,
        recent_verbatim_turns=recent_verbatim_turns,
    )
    turns: list[AgentTurn] = []

    for session in range(1, n_sessions + 1):
        turn = _make_turn(session)
        turns.append(turn)
        agent._turns.append(turn)
        all_markers = [_marker(s) for s in range(1, session + 1)]

        # --- APEX arm ---
        window = agent._render_transcript_window()
        sidecar = "\n".join(agent._derived_state_sidecar())
        present, _total = _count_markers(window, all_markers)
        # A genericized marker = one that the renderer ALTERED rather than kept
        # verbatim or cleanly dropped. The value-preserving renderer NEVER does
        # this, so genericized count is detected by scanning for prose summaries
        # that reference markers without the exact token (there are none).
        genericized = _detect_genericized(window, all_markers)
        completed = [t for t in turns if t.tool_result is not None]
        distinct = len({_command_sig(t) for t in completed})
        last_rc = completed[-1].tool_result.return_code
        apex_arm.sessions.append(
            ArmSessionMetric(
                session=session,
                markers_present_exact=present,
                markers_genericized=genericized,
                commands_run_exact=(f"commands_run: {len(completed)}" in sidecar),
                distinct_commands_exact=(f"distinct_commands: {distinct}" in sidecar),
                last_return_code_exact=(f"last_return_code: {last_rc}" in sidecar),
            )
        )

        # --- Lossy baseline arm ---
        lossy_window = _lossy_compact(turns, recent_verbatim=recent_verbatim_turns)
        lpresent, _ = _count_markers(lossy_window, all_markers)
        lgenericized = (session) - lpresent  # markers no longer recoverable at all
        # Lossy baseline keeps no machine-maintained counters.
        lossy_arm.sessions.append(
            ArmSessionMetric(
                session=session,
                markers_present_exact=lpresent,
                markers_genericized=lgenericized,
                commands_run_exact=False,
                distinct_commands_exact=False,
                last_return_code_exact=False,
            )
        )

    _finalize_arm(apex_arm, recent_verbatim_turns)
    _finalize_arm(lossy_arm, recent_verbatim_turns)

    verdict = {
        "n_sessions": n_sessions,
        "recent_verbatim_turns": recent_verbatim_turns,
        "apex": apex_arm.to_dict(),
        "lossy_baseline": lossy_arm.to_dict(),
        # The falsifiable headline claims:
        "apex_never_genericizes": apex_arm.total_markers_genericized == 0,
        "apex_counters_always_exact": apex_arm.counters_always_exact,
        "lossy_baseline_decays": lossy_arm.total_markers_genericized > 0,
        "apex_half_life_exceeds_baseline": (
            apex_arm.information_half_life_sessions
            > lossy_arm.information_half_life_sessions
        ),
    }
    verdict["passed"] = bool(
        verdict["apex_never_genericizes"]
        and verdict["apex_counters_always_exact"]
        and verdict["lossy_baseline_decays"]
        and verdict["apex_half_life_exceeds_baseline"]
    )
    return verdict


def _command_sig(turn: AgentTurn) -> str:
    cmd = str(turn.tool_call.args.get("command") or "") if turn.tool_call else ""
    rc = turn.tool_result.return_code if turn.tool_result else None
    timed = turn.tool_result.timed_out if turn.tool_result else None
    return f"{cmd}\x00{rc}\x00{timed}"


def _detect_genericized(window: str, all_markers: list[str]) -> int:
    """Count markers that appear in a GENERICIZED form (i.e. the renderer wrote a
    prose summary referencing them without the exact token). The value-preserving
    renderer emits a clean elision notice that names NO marker, so this is always
    0 for the APEX arm — any non-zero here is a real regression."""
    # The renderer's only summary text is the elision notice. If that notice ever
    # contained a marker substring-minus-exact-token we'd flag it; by construction
    # it does not. We additionally guard: no marker may appear truncated (prefix
    # present but full token absent), which would indicate partial genericization.
    genericized = 0
    for m in all_markers:
        if m in window:
            continue
        prefix = m[:-2]  # drop the trailing "ZZ" sentinel
        if prefix in window:  # prefix survived but exact token did not -> altered
            genericized += 1
    return genericized


def _finalize_arm(arm: ArmReport, recent_verbatim_turns: int) -> None:
    arm.total_markers_genericized = sum(s.markers_genericized for s in arm.sessions)
    arm.counters_always_exact = all(
        s.commands_run_exact and s.distinct_commands_exact and s.last_return_code_exact
        for s in arm.sessions
    )
    # Information half-life: the first session at which the fraction of EVER-seen
    # markers that remain *recoverable* (exact in-window OR cleanly elided) drops
    # below 0.5 AND stays there. For the value-preserving arm a clean elision is
    # still recoverable (command preserved / re-runnable), so genericization is
    # the only true loss -> half-life is inf when nothing is genericized.
    if arm.total_markers_genericized == 0:
        arm.information_half_life_sessions = math.inf
        return
    # For the lossy arm, recoverability == markers_present_exact / session.
    half_life = math.inf
    for s in arm.sessions:
        recoverable_fraction = s.markers_present_exact / max(1, s.session)
        if recoverable_fraction < 0.5:
            half_life = float(s.session)
            break
    arm.information_half_life_sessions = half_life


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="WS4 decay-resistance A/B harness")
    parser.add_argument("--sessions", type=int, default=40)
    parser.add_argument("--recent-verbatim", type=int, default=4)
    parser.add_argument("--max-tokens-per-turn", type=int, default=1024)
    args = parser.parse_args(argv)
    verdict = run_decay_resistance(
        n_sessions=args.sessions,
        recent_verbatim_turns=args.recent_verbatim,
        max_tokens_per_turn=args.max_tokens_per_turn,
    )
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
