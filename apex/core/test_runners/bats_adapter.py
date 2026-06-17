"""Bats (Bash Automated Testing System) adapter."""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree

from . import RunResult, register_adapter

_BASH_INFRASTRUCTURE_FILENAMES = {
    "bats.config",
    ".batsrc",
}
_BASH_STUB_PATTERNS = [
    r'echo\s+[\'"`].*not.{0,5}implemented.*[\'"`]',
    r"#\s*TODO[: ]",
    r"^\s*:\s*$",  # bash no-op
    r"return\s+1\s*#\s*TODO",
]


_BATS_LIST_LINE = re.compile(r"^(?P<file>[^:]+):(?P<line>\d+)\s+(?P<name>.+)$")


class BatsAdapter:
    name = "bats"
    language = "bash"

    def list_tests(self, workspace: Path, env: dict[str, str]) -> set[str]:
        try:
            result = subprocess.run(
                ["bats", "--list-tests", "."],
                cwd=str(workspace),
                env={**env},
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (subprocess.TimeoutExpired, OSError):
            return set()
        nodeids: set[str] = set()
        for line in (result.stdout or "").splitlines():
            m = _BATS_LIST_LINE.match(line.strip())
            if m:
                nodeids.add(f"{m.group('file')}::{m.group('name')}")
        return nodeids

    def build_run_command(
        self,
        workspace: Path,
        test_ids: list[str],
        report_path: Path,
        *,
        executable: Optional[str] = None,
    ) -> str:
        files = sorted({nid.split("::", 1)[0] for nid in test_ids if nid})
        files_arg = " ".join(shlex.quote(f) for f in files) or "."
        return f"bats {files_arg} --formatter junit --output {shlex.quote(str(report_path.parent))}"

    def parse_report(self, report_path: Path) -> RunResult:
        # Bats's --formatter junit writes one .xml per test file in
        # --output dir. If report_path is a directory, walk it; if it's
        # a single file, parse it directly.
        candidates: list[Path] = []
        if report_path.is_dir():
            candidates.extend(sorted(report_path.glob("*.xml")))
        elif report_path.exists():
            candidates.append(report_path)
        if not candidates:
            return RunResult(returncode=1)
        outcomes: dict[str, str] = {}
        passed = failed = errors = skipped = 0
        for xml_file in candidates:
            try:
                tree = ElementTree.parse(xml_file)
            except ElementTree.ParseError:
                continue
            for tc in tree.iter("testcase"):
                classname = tc.attrib.get("classname") or ""
                name = tc.attrib.get("name") or ""
                nid = f"{classname}::{name}" if classname else name
                if tc.find("failure") is not None:
                    outcomes[nid] = "failed"
                    failed += 1
                elif tc.find("error") is not None:
                    outcomes[nid] = "error"
                    errors += 1
                elif tc.find("skipped") is not None:
                    outcomes[nid] = "skipped"
                    skipped += 1
                else:
                    outcomes[nid] = "passed"
                    passed += 1
        collected = passed + failed + errors + skipped
        return RunResult(
            returncode=0 if (failed == 0 and errors == 0) else 1,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            collected=collected,
            outcomes=outcomes,
            report_path=str(report_path),
        )

    def extract_failure_excerpt(self, test_id: str, report_path: Path) -> str:
        candidates: list[Path] = []
        if report_path.is_dir():
            candidates.extend(sorted(report_path.glob("*.xml")))
        elif report_path.exists():
            candidates.append(report_path)
        target_class, _, target_name = test_id.rpartition("::")
        for xml_file in candidates:
            try:
                tree = ElementTree.parse(xml_file)
            except ElementTree.ParseError:
                continue
            for tc in tree.iter("testcase"):
                name = tc.attrib.get("name") or ""
                classname = tc.attrib.get("classname") or ""
                if not (
                    (name == target_name and classname == target_class)
                    or f"{classname}::{name}" == test_id
                ):
                    continue
                for tag in ("failure", "error"):
                    node = tc.find(tag)
                    if node is not None:
                        msg = node.attrib.get("message") or ""
                        body = (node.text or "").strip()
                        return f"{msg}\n{body}".strip()
        return ""

    def infrastructure_paths(self, workspace: Path) -> set[str]:
        result: set[str] = set()
        for fname in _BASH_INFRASTRUCTURE_FILENAMES:
            if (workspace / fname).exists():
                result.add(fname)
        # test_helper.bash files are commonly imported by bats files.
        for path in workspace.rglob("test_helper.bash"):
            try:
                result.add(str(path.relative_to(workspace)))
            except ValueError:
                continue
        return result

    def stub_patterns(self) -> list[str]:
        return list(_BASH_STUB_PATTERNS)


register_adapter(BatsAdapter())
