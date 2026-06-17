"""Helpers for locating the upstream SWT-Bench harness entrypoint."""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

SWTBENCH_MODULE_CANDIDATES: tuple[str, ...] = (
    "swt_bench.main",
    "main",
    "src.main",
)


def resolve_swtbench_harness_dir() -> str:
    """Return a local SWT-Bench source checkout to expose on PYTHONPATH.

    The public ``swt-bench`` package currently installs top-level modules
    whose imports expect a sibling ``src`` package. A local source checkout on
    PYTHONPATH makes both the installed wheel and source-tree execution work.
    """

    candidates: list[Path] = []
    override = os.environ.get("APEX_SWTBENCH_HARNESS_DIR", "").strip()
    if override:
        candidates.append(Path(override).expanduser())

    cwd = Path.cwd()
    repo_root = Path(__file__).resolve().parents[2]
    for root in (cwd, repo_root):
        candidates.append(root / ".apex_swtbench_harness")
        candidates.append(root / ".apex_swtbench_harness_logicstar")

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "src" / "main.py").is_file():
            return str(resolved)
    return ""


def make_swtbench_harness_env() -> dict[str, str]:
    env = dict(os.environ)
    harness_dir = resolve_swtbench_harness_dir()
    if harness_dir:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = harness_dir if not existing else harness_dir + os.pathsep + existing
    return env


def resolve_swtbench_module(python: str) -> tuple[str, str]:
    harness_dir = resolve_swtbench_harness_dir()
    return _resolve_swtbench_module_cached(str(python or "python"), harness_dir)


@lru_cache(maxsize=16)
def _resolve_swtbench_module_cached(
    python: str,
    harness_dir: str,
) -> tuple[str, str]:
    diagnostics: list[str] = []
    env = dict(os.environ)
    if harness_dir:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = harness_dir if not existing else harness_dir + os.pathsep + existing
    for module in SWTBENCH_MODULE_CANDIDATES:
        probe = f"import importlib; importlib.import_module({module!r})"
        try:
            completed = subprocess.run(
                [python, "-c", probe],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                env=env,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            diagnostics.append(f"{module}: {type(exc).__name__}: {exc}")
            continue
        if completed.returncode == 0:
            return module, "ok"
        diagnostic = (completed.stderr or completed.stdout or "").strip()
        diagnostics.append(f"{module}: {diagnostic[:500]}")
    return "", " | ".join(diagnostics)
