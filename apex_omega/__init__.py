"""APEX-Ω — a vendor-neutral, deterministic dynamic-workflow ENGINE on which
APEX v1's execution-authoritative kernel is the hardened substrate.

Layers (plan §8):
  L0  engine/     agent / parallel / pipeline / phase / budget  (orchestration-as-code)
  L1  executor/   normalized Executor over codex/claude/gemini/opencode (+capability negotiation)
  L2  kernel/     Cardinal Safety Contract: execution-authoritative selection + verify
  cross-cutting:  journal/ (durable input-hash WAL resume), ablation/ (fail-open flags + SafetyModeConfig)
  eval/           commit0 driver (reuses v1 scoring) + ablation matrix over the target repos
  workflows/      the reference best-of-N / pipeline commit0 programs the engine runs

Five invariants carried verbatim: filesystem-as-source-of-truth,
execution-evidence-authoritative selection, fail-loud-never-fake, durable
resumable journaling, vendor neutrality.
"""

from .errors import ApexOmegaError, FailLoud
from .types import (
    CapabilityProfile,
    ExecResult,
    ScopedTask,
    TokenUsage,
)

__version__ = "0.1.0"

__all__ = [
    "ScopedTask",
    "ExecResult",
    "TokenUsage",
    "CapabilityProfile",
    "ApexOmegaError",
    "FailLoud",
    "__version__",
]
