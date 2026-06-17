"""Vendor-neutral contracts for the APEX-Ω engine.

These dataclasses are the structural boundary between the orchestration-as-code
engine (Section 2) and the normalized Executor (Section 3 / §22.2.1).  They are
deliberately vendor-blind: a worker's authoritative artifact is the git diff it
produces (``ExecResult.fs_diff``); the JSON event stream is telemetry, never the
contract (Fusion Ledger A10/A11, Principle 3).

Field sets are lifted verbatim from APEX_NEXTGEN_PLAN.md §22.2.1 so a builder can
implement an adapter against either ``codex exec`` or ``claude -p`` without
inventing a new shape.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable


# Finalization status enum (plan §02 7-value set).  This is *transport/process*
# disposition — NEVER correctness.  Correctness is decided downstream by
# executing ``fs_diff`` through the verification kernel (Cardinal Contract).
FINALIZATION_STATUSES = (
    "completed",
    "timeout",
    "policy_violation",
    "output_limit",
    "progress_abort",
    "isolation_error",
    "infra_nonresult",
)


# ---------------------------------------------------------------------------
# Token accounting (normalized across vendors)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TokenUsage:
    """Normalized token usage.  Cache-read vs cache-creation are tracked
    separately because the cost story (§16) depends on the ~0.10x cached-read
    multiplier, and the eval plan (§20.4) reports them apart."""

    input: int = 0
    output: int = 0
    cached_input: int = 0
    reasoning: int = 0
    cache_creation: int = 0

    @property
    def total(self) -> int:
        # cached_input is a subset accounting of input on some vendors; we keep
        # the conservative "billable surface" = input + output + reasoning.
        return int(self.input) + int(self.output) + int(self.reasoning)

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            input=self.input + other.input,
            output=self.output + other.output,
            cached_input=self.cached_input + other.cached_input,
            reasoning=self.reasoning + other.reasoning,
            cache_creation=self.cache_creation + other.cache_creation,
        )

    def to_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "TokenUsage":
        if not d:
            return cls()
        return cls(**{k: int(d.get(k, 0)) for k in (
            "input", "output", "cached_input", "reasoning", "cache_creation")})


# ---------------------------------------------------------------------------
# Capability profile (ACP-style negotiation result, §3 / §22.2.1)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CapabilityProfile:
    """Result of the ACP-style ``negotiate()`` handshake.  The hard rule is
    *degrade, do not crash*: a missing capability changes how the Executor
    builds the command, never whether the run happens."""

    vendor: str
    model: str
    cli_version: str = "unknown"
    internet: bool = False
    native_schema: bool = False
    sandbox_levels: tuple[str, ...] = ()
    thinking: bool = False
    bidirectional_stream: bool = False
    mcp: bool = False
    effort_levels: tuple[str, ...] = ()
    # learned capability/cost profile vector (NOT a one-hot vendor id) — A9 in
    # the eval matrix; populated by the controller layer, empty until then.
    profile_vector: tuple[float, ...] = ()
    cost_per_1k_in: float = 0.0
    cost_per_1k_out: float = 0.0

    def supports_effort(self, effort: str | None) -> bool:
        return effort is None or not self.effort_levels or effort in self.effort_levels

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["sandbox_levels"] = list(self.sandbox_levels)
        d["effort_levels"] = list(self.effort_levels)
        d["profile_vector"] = list(self.profile_vector)
        return d


# ---------------------------------------------------------------------------
# Scoped task — the unit of work handed to one leaf worker (§22.2.1)
# ---------------------------------------------------------------------------
@dataclass
class ScopedTask:
    prompt: str
    schema: Optional[dict] = None           # JSON Schema for structured return
    allowed_tools: list[str] = field(default_factory=list)
    sandbox: str = "apex-worktree"          # read-only | workspace-write | apex-worktree
    model: Optional[str] = None             # vendor-resolved at command-build time
    vendor: Optional[str] = None
    effort: Optional[str] = None            # low|medium|high|xhigh|max (degrade if unsupported)
    internet: bool = False
    mcp_servers: list[Any] = field(default_factory=list)
    timeout_seconds: Optional[int] = None
    heartbeat_timeout_seconds: Optional[int] = None   # Backbone 0.2: engine watchdog wall (advisory)
    # opaque per-task inputs folded into the journal input-hash (e.g. repo sha,
    # base diff, scoped file set) so resume is keyed on everything that matters.
    scoped_inputs: dict[str, Any] = field(default_factory=dict)

    def hash_inputs(self) -> dict[str, Any]:
        """The subset of fields that participate in the durable input hash
        (§22.2.3).  Excludes nothing semantic; excludes only ephemeral handles."""
        return {
            "prompt": self.prompt,
            "schema": self.schema,
            "allowed_tools": sorted(self.allowed_tools),
            "sandbox": self.sandbox,
            "model": self.model,
            "vendor": self.vendor,
            "effort": self.effort,
            "internet": self.internet,
            "scoped_inputs": self.scoped_inputs,
        }


# ---------------------------------------------------------------------------
# Exec result — what a worker returns; the diff is the contract (§22.2.1)
# ---------------------------------------------------------------------------
@dataclass
class ExecResult:
    final_message: str = ""
    structured_output: Optional[dict] = None   # validated; None if schema unmet
    usage: TokenUsage = field(default_factory=TokenUsage)
    session_id: Optional[str] = None           # vendor session handle, for resume
    raw_events: list[dict] = field(default_factory=list)  # telemetry, NOT the contract
    fs_diff: str = ""                          # git diff of worktree == authoritative artifact
    vendor: Optional[str] = None
    model: Optional[str] = None
    cli_version: Optional[str] = None
    ok: bool = True                            # transport-level success (never raises; typed failure)
    finalization_status: str = "completed"     # one of FINALIZATION_STATUSES (process disposition, NOT correctness)
    error: Optional[str] = None
    failure_class: Optional[str] = None        # v1 FailureClass name when ok is False
    latency_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_message": self.final_message,
            "structured_output": self.structured_output,
            "usage": self.usage.to_dict(),
            "session_id": self.session_id,
            # raw_events deliberately dropped from the journal artifact — it is
            # telemetry only and can be large; the diff + usage are the contract.
            "fs_diff": self.fs_diff,
            "vendor": self.vendor,
            "model": self.model,
            "cli_version": self.cli_version,
            "ok": self.ok,
            "finalization_status": self.finalization_status,
            "error": self.error,
            "failure_class": self.failure_class,
            "latency_seconds": self.latency_seconds,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ExecResult":
        return cls(
            final_message=d.get("final_message", ""),
            structured_output=d.get("structured_output"),
            usage=TokenUsage.from_dict(d.get("usage")),
            session_id=d.get("session_id"),
            raw_events=list(d.get("raw_events", [])),
            fs_diff=d.get("fs_diff", ""),
            vendor=d.get("vendor"),
            model=d.get("model"),
            cli_version=d.get("cli_version"),
            ok=bool(d.get("ok", True)),
            finalization_status=d.get("finalization_status", "completed"),
            error=d.get("error"),
            failure_class=d.get("failure_class"),
            latency_seconds=float(d.get("latency_seconds", 0.0)),
        )


@runtime_checkable
class Session(Protocol):
    """A spawned vendor worker bound to a worktree (§22.2.1)."""

    def run(self, task: ScopedTask) -> ExecResult: ...

    def observe_diff(self) -> str: ...   # git diff is ground truth


@runtime_checkable
class Executor(Protocol):
    """The load-bearing vendor-neutral abstraction (A10).  Adapters map this
    common surface to native ``codex exec`` / ``claude -p`` flags."""

    def negotiate(self, vendor: str, model: str, version: str) -> CapabilityProfile: ...

    def spawn(self, worktree_cwd: str, vendor: str, model: str, version: str) -> Session: ...
