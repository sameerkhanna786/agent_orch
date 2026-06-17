"""APEX — agentic coding orchestrator.

Top-level public API for external callers (IDE plugins, CI gates,
agentic-coding orchestrators using APEX as a library).

For benchmark harnesses, prefer the apex.evaluation submodule.
For real-world / TDD / non-benchmark use, prefer the three usage
modes exposed here:

    from apex import (
        run_testgen_with_fix,    # Mode 1: have a fix; want tests
        run_codegen_with_tests,  # Mode 2: have tests; want code
        run_generate_both,       # Mode 3: have problem statement only
    )

Each mode returns a unified ModeResult with test_artifacts / patch /
f2p_summary / mutation_summary / minimization_summary so downstream
tooling sees a consistent shape.

For lower-level building blocks (F2P oracle, mutation engine,
minimizer, iteration feedback), import from apex.evaluation directly.
"""

from __future__ import annotations

import importlib.metadata as _im

from .core.status import Status

try:
    __version__: str = _im.version("apex-agents")
except _im.PackageNotFoundError:  # pragma: no cover - source checkout fallback
    __version__ = "0.0.0+unpackaged"
from .modes import (
    AGENT_MODE_CLI_AGENT,
    AGENT_MODE_IN_CONTAINER_V5,
    AGENT_MODE_SCAFFOLDED,
    ALL_AGENT_MODES,
    ALL_MODES,
    MODE_CODEGEN_WITH_TESTS,
    MODE_GENERATE_BOTH,
    MODE_TESTGEN_WITH_FIX,
    AgentMode,
    CodeGenerator,
    ModeResult,
    SurrogatePatcher,
    TestGenerator,
    run_codegen_with_tests,
    run_generate_both,
    run_testgen_with_fix,
)
from .orchestrator import ApexOrchestrator, ApexResult
from .orchestrator_in_container_agent import (
    DEFAULT_MAX_TURNS as IN_CONTAINER_DEFAULT_MAX_TURNS,
)
from .orchestrator_in_container_agent import (
    AgentRunSummary,
    InContainerAgent,
    solve_in_container_agent,
)

__all__ = [
    "__version__",
    "AGENT_MODE_CLI_AGENT",
    "AGENT_MODE_IN_CONTAINER_V5",
    "AGENT_MODE_SCAFFOLDED",
    "ALL_AGENT_MODES",
    "ALL_MODES",
    "AgentMode",
    "AgentRunSummary",
    "ApexOrchestrator",
    "ApexResult",
    "CodeGenerator",
    "IN_CONTAINER_DEFAULT_MAX_TURNS",
    "InContainerAgent",
    "MODE_CODEGEN_WITH_TESTS",
    "MODE_GENERATE_BOTH",
    "MODE_TESTGEN_WITH_FIX",
    "ModeResult",
    "Status",
    "SurrogatePatcher",
    "TestGenerator",
    "run_codegen_with_tests",
    "run_generate_both",
    "run_testgen_with_fix",
    "solve_in_container_agent",
]
