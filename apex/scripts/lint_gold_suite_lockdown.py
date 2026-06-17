"""Lint guard for gold-suite-visible integrity enforcement surfaces."""

from __future__ import annotations

import argparse
from pathlib import Path

_CHECKS: tuple[tuple[str, str, str], ...] = (
    (
        "apex/rollout/patch_sanitizer.py",
        "GOLD_PROTECTED_TEST",
        "rollout/patch_sanitizer.py must classify GOLD_PROTECTED_TEST paths",
    ),
    (
        "apex/rollout/patch_sanitizer.py",
        "sanitize_patch_text",
        "rollout/patch_sanitizer.py must expose sanitize_patch_text",
    ),
    (
        "apex/rollout/localizer_scope.py",
        "def is_test_path",
        "rollout/localizer_scope.py must expose is_test_path",
    ),
    (
        "apex/rollout/discovery_scope.py",
        "is_test_path",
        "rollout/discovery_scope.py must route paths through is_test_path",
    ),
    (
        "apex/rollout/engine.py",
        "sanitize_patch_text",
        "rollout/engine.py must call sanitize_patch_text",
    ),
    (
        "apex/evaluation/commit0_benchmark.py",
        "sanitize_candidate_worktree",
        "evaluation/commit0_benchmark.py must call sanitize_candidate_worktree",
    ),
    (
        "apex/evaluation/commit0_benchmark.py",
        "docker_internal_network_with_model_proxy_sidecar",
        "evaluation/commit0_benchmark.py must enforce solve-phase network boundary",
    ),
    (
        "apex/core/cli_backend.py",
        "--strict-mcp-config",
        "core/cli_backend.py must launch Claude target runtime with empty MCP config",
    ),
    (
        "apex/core/cli_backend.py",
        "features.plugins=false",
        "core/cli_backend.py must disable Codex target-runtime plugins",
    ),
)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def check_repo(repo_root: Path | str) -> list[str]:
    root = Path(repo_root)
    failures: list[str] = []
    for relative_path, required_text, message in _CHECKS:
        text = _read_text(root / relative_path)
        if required_text not in text:
            failures.append(message)
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify gold-suite-visible lockdown guard surfaces."
    )
    parser.add_argument(
        "repo_root",
        nargs="?",
        default=Path.cwd(),
        type=Path,
        help="Repository root to check.",
    )
    args = parser.parse_args(argv)
    failures = check_repo(args.repo_root)
    if failures:
        for failure in failures:
            print(f"gold-suite-lockdown: {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
