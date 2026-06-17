"""``go test`` adapter."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Optional

from . import RunResult, register_adapter

_GO_INFRASTRUCTURE_FILENAMES = {"go.mod", "go.sum"}
_GO_STUB_PATTERNS = [
    r'panic\("not implemented"\)',
    r'panic\("TODO"\)',
    r"//\s*TODO[: ]",
    r"return\s+nil,\s*nil\s*//\s*TODO",
]


class GoTestAdapter:
    name = "go-test"
    language = "go"

    def list_tests(self, workspace: Path, env: dict[str, str]) -> set[str]:
        try:
            result = subprocess.run(
                ["go", "test", "-list", ".*", "./..."],
                cwd=str(workspace),
                env={**env},
                capture_output=True,
                text=True,
                timeout=180,
            )
        except (subprocess.TimeoutExpired, OSError):
            return set()
        nodeids: set[str] = set()
        current_pkg = ""
        for line in (result.stdout or "").splitlines():
            line = line.rstrip()
            if not line:
                continue
            # Lines look like:  TestSomething   OR   ok  pkg/foo  0.123s
            if line.startswith("ok ") or line.startswith("FAIL") or line.startswith("?"):
                parts = line.split()
                if len(parts) >= 2:
                    current_pkg = parts[1]
                continue
            if (
                line.startswith("Test")
                or line.startswith("Example")
                or line.startswith("Benchmark")
            ):
                if current_pkg:
                    nodeids.add(f"{current_pkg}::{line.strip()}")
                else:
                    nodeids.add(line.strip())
        return nodeids

    def build_run_command(
        self,
        workspace: Path,
        test_ids: list[str],
        report_path: Path,
        *,
        executable: Optional[str] = None,
    ) -> str:
        # Group ids by package; ``go test -run pat1|pat2 pkg1 pkg2`` runs
        # the named regex in those packages.
        by_pkg: dict[str, list[str]] = {}
        for nid in test_ids:
            pkg, _, name = nid.partition("::")
            if not pkg or not name:
                continue
            by_pkg.setdefault(pkg, []).append(name)
        if not by_pkg:
            return f"go test -json ./... > {shlex.quote(str(report_path))}"
        pkgs = " ".join(shlex.quote(p) for p in sorted(by_pkg))
        # ``go test`` only takes one -run regex per invocation but applies
        # it across all packages, which is fine — the union regex matches
        # exactly the per-package selections.
        all_names = sorted({name for names in by_pkg.values() for name in names})
        run_regex = "^(" + "|".join(re.escape(name) for name in all_names) + ")$"
        return (
            f"go test -json -run {shlex.quote(run_regex)} {pkgs} > {shlex.quote(str(report_path))}"
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
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            action = event.get("Action") or ""
            test = event.get("Test") or ""
            pkg = event.get("Package") or ""
            if not test or action not in {"pass", "fail", "skip"}:
                continue
            nid = f"{pkg}::{test}" if pkg else test
            outcomes[nid] = action
            if action == "pass":
                passed += 1
            elif action == "fail":
                failed += 1
            elif action == "skip":
                skipped += 1
        collected = passed + failed + skipped
        return RunResult(
            returncode=0 if failed == 0 and errors == 0 else 1,
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
        pkg, _, test = test_id.partition("::")
        try:
            text = report_path.read_text()
        except OSError:
            return ""
        # Collect output lines for this test until a fail/pass action.
        captured: list[str] = []
        for line in text.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if (event.get("Test") or "") != test:
                continue
            if pkg and (event.get("Package") or "") != pkg:
                continue
            if event.get("Action") == "output":
                out = event.get("Output") or ""
                if out:
                    captured.append(out.rstrip())
            elif event.get("Action") in {"pass", "fail", "skip"}:
                break
        return "\n".join(captured).strip()

    def infrastructure_paths(self, workspace: Path) -> set[str]:
        result: set[str] = set()
        for fname in _GO_INFRASTRUCTURE_FILENAMES:
            candidate = workspace / fname
            if candidate.exists():
                result.add(fname)
        # Common shared-helper dirs that tests import
        for path in workspace.rglob("testdata"):
            if path.is_dir():
                try:
                    result.add(str(path.relative_to(workspace)))
                except ValueError:
                    continue
        return result

    def stub_patterns(self) -> list[str]:
        return list(_GO_STUB_PATTERNS)


register_adapter(GoTestAdapter())
