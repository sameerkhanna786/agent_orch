"""``cargo test`` adapter for Rust crates."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Optional

from . import RunResult, register_adapter

_RUST_INFRASTRUCTURE_FILENAMES = {
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain",
    "rust-toolchain.toml",
}
_RUST_STUB_PATTERNS = [
    r"unimplemented!\(",
    r"todo!\(",
    r'panic!\((["\']).*not.{0,5}implemented.*\1\)',
    r'panic!\((["\']).*TODO.*\1\)',
    r"//\s*TODO[: ]",
]


class CargoTestAdapter:
    name = "cargo-test"
    language = "rust"

    def list_tests(self, workspace: Path, env: dict[str, str]) -> set[str]:
        try:
            result = subprocess.run(
                ["cargo", "test", "--", "--list", "--format=terse"],
                cwd=str(workspace),
                env={**env},
                capture_output=True,
                text=True,
                timeout=180,
            )
        except (subprocess.TimeoutExpired, OSError):
            return set()
        nodeids: set[str] = set()
        for line in (result.stdout or "").splitlines():
            line = line.strip()
            if line.endswith(": test") or line.endswith(": benchmark"):
                name, _, _ = line.rpartition(":")
                nodeids.add(name.strip())
        return nodeids

    def build_run_command(
        self,
        workspace: Path,
        test_ids: list[str],
        report_path: Path,
        *,
        executable: Optional[str] = None,
    ) -> str:
        # cargo test accepts space-separated name fragments after `--`;
        # each fragment is a substring filter on the full test path.
        names = " ".join(shlex.quote(t) for t in test_ids)
        return (
            f"cargo test --no-fail-fast "
            f"-- {names} -Z unstable-options --format=json "
            f"> {shlex.quote(str(report_path))}"
        )

    def parse_report(self, report_path: Path) -> RunResult:
        if not report_path.exists():
            return RunResult(returncode=1)
        outcomes: dict[str, str] = {}
        passed = failed = errors = skipped = 0
        try:
            text = report_path.read_text()
        except OSError:
            return RunResult(returncode=1)
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            event_type = event.get("type") or ""
            if event_type != "test":
                continue
            name = event.get("name") or ""
            event_action = event.get("event") or ""
            if event_action == "ok":
                outcomes[name] = "passed"
                passed += 1
            elif event_action == "failed":
                outcomes[name] = "failed"
                failed += 1
            elif event_action == "ignored":
                outcomes[name] = "skipped"
                skipped += 1
        collected = passed + failed + skipped
        return RunResult(
            returncode=0 if failed == 0 else 1,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            collected=collected,
            outcomes=outcomes,
            report_path=str(report_path),
        )

    def extract_failure_excerpt(self, test_id: str, report_path: Path) -> str:
        if not report_path.exists():
            return ""
        try:
            text = report_path.read_text()
        except OSError:
            return ""
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") != "test" or event.get("name") != test_id:
                continue
            stdout = event.get("stdout") or ""
            if stdout:
                return stdout.strip()
        return ""

    def infrastructure_paths(self, workspace: Path) -> set[str]:
        result: set[str] = set()
        for fname in _RUST_INFRASTRUCTURE_FILENAMES:
            candidate = workspace / fname
            if candidate.exists():
                result.add(fname)
        common_mod = workspace / "tests" / "common"
        if common_mod.exists():
            try:
                result.add(str(common_mod.relative_to(workspace)))
            except ValueError:
                pass
        return result

    def stub_patterns(self) -> list[str]:
        return list(_RUST_STUB_PATTERNS)


register_adapter(CargoTestAdapter())
