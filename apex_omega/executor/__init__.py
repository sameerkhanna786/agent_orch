"""Normalized vendor-neutral Executor plane (Fusion Ledger A10/A11)."""

from .capability import STATIC_CAPABILITY_TABLE, negotiate
from .fake import FakeExecutor, FakeSession
from .v1_executor import V1Executor, V1Session, build_llm_config

__all__ = [
    "negotiate",
    "STATIC_CAPABILITY_TABLE",
    "V1Executor",
    "V1Session",
    "build_llm_config",
    "FakeExecutor",
    "FakeSession",
]
