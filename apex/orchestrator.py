"""Back-compat shim for the legacy ``apex.orchestrator`` module path.

Phase 3.2: the 4,000-line monolith was decomposed into the
:mod:`apex.orchestration` package. This file re-exports every
previously-importable name so existing code (`from apex.orchestrator
import X`) and existing test monkeypatches (`monkeypatch.setattr(
"apex.orchestrator.X", ...)`) keep working without change.

If you are writing new code, prefer importing directly from the new
modules — :mod:`apex.orchestration.solver`,
:mod:`apex.orchestration.followups`, etc. — so call sites read more
clearly.
"""

from __future__ import annotations

# --- Core re-exports kept on the legacy module path -------------------
# The Status enum lives in apex.core.status as of Phase 3.2 (so
# ModeResult / V5 in-container agent can import it without touching
# the orchestrator module). Re-export here so legacy callers and tests
# that reference ``apex.orchestrator.Status`` keep working.
from .controller_trace import append_controller_decision  # noqa: F401
from .core.llm_routing import llm_backend_is_available  # noqa: F401
from .core.status import Status  # noqa: F401
from .orchestration.acceptance import (  # noqa: F401
    rollout_has_authoritative_completion_signal,
    rollout_has_expected_coverage_gap,
    rollout_has_local_full_suite_completion_signal,
    rollout_has_strong_progressive_signal,
    selected_result_is_accepted,
)
from .orchestration.escalation import strategy_identity_for_loop_guard  # noqa: F401
from .orchestration.solver import (  # noqa: F401
    _APEX_LOGGING_HANDLER_MARKER,
    _INHERIT_VERIFICATION_TEST_COMMAND,
    _ORCHESTRATOR_LOGGER_NAMESPACE,
    ApexOrchestrator,
    ApexResult,
    _humanize_test_inventory_framework,
    _jaccard_similarity,
    _quick_verification_inventory_context,
    _resolve_verification_test_command,
    logger,
)
from .planning.manager import IssuePlanner  # noqa: F401
from .search.frontier_search import FrontierSearchController  # noqa: F401
from .selection.verifier import PatchVerifier  # noqa: F401

__all__ = [
    "_APEX_LOGGING_HANDLER_MARKER",
    "_INHERIT_VERIFICATION_TEST_COMMAND",
    "_ORCHESTRATOR_LOGGER_NAMESPACE",
    "ApexOrchestrator",
    "ApexResult",
    "FrontierSearchController",
    "IssuePlanner",
    "PatchVerifier",
    "Status",
    "append_controller_decision",
    "llm_backend_is_available",
    "logger",
    "rollout_has_authoritative_completion_signal",
    "rollout_has_expected_coverage_gap",
    "rollout_has_local_full_suite_completion_signal",
    "rollout_has_strong_progressive_signal",
    "selected_result_is_accepted",
    "strategy_identity_for_loop_guard",
]
