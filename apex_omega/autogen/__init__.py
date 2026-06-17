"""Generated-code orchestration (plan §7.3 freeze-then-journal).

A planner authors a tailored Python ``orchestrate(ctx)`` from a repo scout; it is
lint/capability-checked, frozen (content-hashed + journaled) for deterministic
replay, executed in a restricted sandbox, and fails open to the verified
best-of-N floor.  Full strategy flexibility (1000s of agents, decomposition,
cross-vendor routing) with an engine-owned acceptance gate.
"""

from .architect import (
    FrozenWorkflow,
    SCOUT_SCHEMA,
    agent_scout,
    autosolve,
    author_orchestration,
    build_author_prompt,
    build_repo_map,
    build_scout_prompt,
    difficulty_profile,
    load_frozen,
)
from .context import OrchestrationContext
from .sandbox import LintResult, extract_code, lint_source, run_orchestration
from .templates import DECOMPOSE_EXEMPLAR, DEFAULT_ORCHESTRATION

__all__ = [
    "autosolve",
    "agent_scout",
    "difficulty_profile",
    "build_scout_prompt",
    "SCOUT_SCHEMA",
    "author_orchestration",
    "build_repo_map",
    "build_author_prompt",
    "load_frozen",
    "FrozenWorkflow",
    "OrchestrationContext",
    "lint_source",
    "run_orchestration",
    "extract_code",
    "LintResult",
    "DEFAULT_ORCHESTRATION",
    "DECOMPOSE_EXEMPLAR",
]
