"""Per-run backend availability portfolio.

The portfolio records backend/command pairs disabled during a benchmark run so
later tasks can avoid repeating expensive or noisy health probes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BackendFingerprint = tuple[str, str]


@dataclass
class BackendPortfolio:
    """Track disabled backend/command pairs for one benchmark run."""

    path: Path | None = None
    retry_after_seconds: float = 0.0
    disabled_backends: set[BackendFingerprint] = field(default_factory=set)
    disable_reason: dict[BackendFingerprint, str] = field(default_factory=dict)
    disabled_at: dict[BackendFingerprint, float] = field(default_factory=dict)

    @staticmethod
    def fingerprint(
        backend_or_config: Any,
        command: Any | None = None,
    ) -> BackendFingerprint:
        """Return the portfolio key for a config or explicit backend/command."""

        if command is None and hasattr(backend_or_config, "backend"):
            config = backend_or_config
            backend = getattr(
                getattr(config, "backend", None), "value", getattr(config, "backend", "")
            )
            command = getattr(config, "resolved_cli_command", getattr(config, "cli_command", ""))
        else:
            backend = getattr(backend_or_config, "value", backend_or_config)
        return (
            str(backend or "").strip().lower(),
            str(command or "").strip().lower(),
        )

    @classmethod
    def for_task_output_dir(cls, task_output_dir: Path | str) -> "BackendPortfolio":
        """Load the run-level portfolio adjacent to a task output directory."""

        return cls.load(Path(task_output_dir).parent / "run_backend_portfolio.json")

    @classmethod
    def load(cls, path: Path | str) -> "BackendPortfolio":
        """Load a portfolio from disk, returning an empty portfolio if absent."""

        portfolio_path = Path(path)
        portfolio = cls(path=portfolio_path)
        if not portfolio_path.exists():
            return portfolio
        try:
            payload = json.loads(portfolio_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return portfolio
        entries = payload.get("disabled_backends") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            return portfolio
        retry_after_seconds = payload.get("retry_after_seconds")
        if retry_after_seconds is not None:
            try:
                portfolio.retry_after_seconds = float(retry_after_seconds)
            except (TypeError, ValueError):
                portfolio.retry_after_seconds = 0.0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            backend = str(entry.get("backend") or "").strip().lower()
            command = str(entry.get("command") or "").strip().lower()
            if not backend or not command:
                continue
            fingerprint = (backend, command)
            portfolio.disabled_backends.add(fingerprint)
            portfolio.disable_reason[fingerprint] = str(entry.get("reason") or "").strip()
            disabled_at = entry.get("disabled_at")
            if disabled_at is not None:
                try:
                    portfolio.disabled_at[fingerprint] = float(disabled_at)
                except (TypeError, ValueError):
                    pass
        return portfolio

    def disable(
        self,
        backend_or_config: Any,
        reason: Any = "",
        *,
        command: Any | None = None,
        now: float | None = None,
    ) -> BackendFingerprint:
        """Disable a backend/command pair and record the latest reason."""

        fingerprint = self.fingerprint(backend_or_config, command)
        self.disabled_backends.add(fingerprint)
        self.disable_reason[fingerprint] = str(reason or "").strip()
        self.disabled_at[fingerprint] = time.time() if now is None else float(now)
        return fingerprint

    def is_disabled(
        self,
        backend_or_config: Any,
        *,
        command: Any | None = None,
        now: float | None = None,
    ) -> bool:
        """Return whether a backend/command pair has been disabled."""

        fingerprint = self.fingerprint(backend_or_config, command)
        if fingerprint not in self.disabled_backends:
            return False
        if self.retry_after_seconds <= 0:
            return True
        disabled_at = self.disabled_at.get(fingerprint)
        if disabled_at is None:
            return True
        current_time = time.time() if now is None else float(now)
        if current_time - disabled_at < self.retry_after_seconds:
            return True
        self._remove_disabled(fingerprint)
        return False

    def reason_for(
        self,
        backend_or_config: Any,
        *,
        command: Any | None = None,
    ) -> str:
        """Return the disable reason for a backend/command pair if present."""

        return self.disable_reason.get(self.fingerprint(backend_or_config, command), "")

    def _remove_disabled(self, fingerprint: BackendFingerprint) -> None:
        self.disabled_backends.discard(fingerprint)
        self.disable_reason.pop(fingerprint, None)
        self.disabled_at.pop(fingerprint, None)

    def save(self) -> None:
        """Persist the portfolio to disk."""

        if self.path is None:
            raise ValueError("BackendPortfolio.save requires a path.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entries = [
            {
                "backend": backend,
                "command": command,
                "reason": self.disable_reason.get((backend, command), ""),
                "disabled_at": self.disabled_at.get((backend, command)),
            }
            for backend, command in sorted(self.disabled_backends)
        ]
        payload = {
            "version": 1,
            "retry_after_seconds": self.retry_after_seconds,
            "disabled_backends": entries,
        }
        tmp_path = self.path.with_name(f"{self.path.name}.tmp")
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(self.path)
