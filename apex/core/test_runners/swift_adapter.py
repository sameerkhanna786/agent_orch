"""Swift / XCTest adapter.

Tree-sitter-swift isn't bundled in tree-sitter-languages so this
adapter falls back to regex-only stub detection. Test discovery uses
``swift test --list-tests``; runs use ``swift test --filter``. JUnit
XML output goes through ``xcpretty`` or ``swift-test-reporter`` if
available — when those aren't installed we parse stdout instead.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree

from . import RunResult, register_adapter

_SWIFT_INFRASTRUCTURE_FILENAMES = {
    "Package.swift",
    "Package.resolved",
    ".swiftpm",
    "Podfile",
    "Podfile.lock",
    "Cartfile",
    "Cartfile.resolved",
}
_SWIFT_INFRASTRUCTURE_SUFFIXES = (".xcodeproj", ".xcworkspace")
_SWIFT_STUB_PATTERNS = [
    r'fatalError\s*\(\s*[\'"`].*not.{0,5}implemented.*[\'"`]\s*\)',
    r'preconditionFailure\s*\(\s*[\'"`].*not.{0,5}implemented.*[\'"`]\s*\)',
    r'XCTFail\s*\(\s*[\'"`].*TODO.*[\'"`]\s*\)',
    r"//\s*TODO[: ]",
    r"return\s+nil\s*//\s*TODO",
]


_PASSED_LINE = re.compile(r"^Test Case '-\[(\S+) (\S+)\]' passed")
_FAILED_LINE = re.compile(r"^Test Case '-\[(\S+) (\S+)\]' failed")


class SwiftXCTestAdapter:
    name = "swift-xctest"
    language = "swift"

    def list_tests(self, workspace: Path, env: dict[str, str]) -> set[str]:
        try:
            result = subprocess.run(
                ["swift", "test", "--list-tests"],
                cwd=str(workspace),
                env={**env},
                capture_output=True,
                text=True,
                timeout=300,
            )
        except (subprocess.TimeoutExpired, OSError):
            return set()
        nodeids: set[str] = set()
        for line in (result.stdout or "").splitlines():
            line = line.strip()
            if "." in line and "/" not in line and ":" not in line:
                nodeids.add(line)
        return nodeids

    def build_run_command(
        self,
        workspace: Path,
        test_ids: list[str],
        report_path: Path,
        *,
        executable: Optional[str] = None,
    ) -> str:
        # ``swift test --filter`` accepts substrings (one per --filter).
        filters = " ".join(f"--filter {shlex.quote(t)}" for t in test_ids)
        # JUnit XML emission requires xcpretty; fall back to stdout
        # capture if it's not installed (the report file will be empty).
        return (
            f"swift test {filters} 2>&1 | tee {shlex.quote(str(report_path) + '.log')} "
            f"| xcpretty -r junit -o {shlex.quote(str(report_path))} || true"
        )

    def parse_report(self, report_path: Path) -> RunResult:
        # Prefer JUnit XML when present.
        if report_path.exists() and report_path.stat().st_size > 0:
            try:
                tree = ElementTree.parse(report_path)
            except ElementTree.ParseError:
                tree = None
            if tree is not None:
                outcomes: dict[str, str] = {}
                passed = failed = errors = skipped = 0
                for tc in tree.iter("testcase"):
                    classname = tc.attrib.get("classname") or ""
                    name = tc.attrib.get("name") or ""
                    nid = f"{classname}.{name}" if classname else name
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
                return RunResult(
                    returncode=0 if (failed == 0 and errors == 0) else 1,
                    passed=passed,
                    failed=failed,
                    errors=errors,
                    skipped=skipped,
                    collected=passed + failed + errors + skipped,
                    outcomes=outcomes,
                    report_path=str(report_path),
                )
        # Fallback: parse stdout log.
        log_path = report_path.with_suffix(report_path.suffix + ".log")
        if not log_path.exists():
            return RunResult(returncode=1)
        outcomes = {}
        passed = failed = 0
        for line in log_path.read_text(errors="replace").splitlines():
            m = _PASSED_LINE.match(line.strip())
            if m:
                outcomes[f"{m.group(1)}.{m.group(2)}"] = "passed"
                passed += 1
                continue
            m = _FAILED_LINE.match(line.strip())
            if m:
                outcomes[f"{m.group(1)}.{m.group(2)}"] = "failed"
                failed += 1
        return RunResult(
            returncode=0 if failed == 0 else 1,
            passed=passed,
            failed=failed,
            errors=0,
            skipped=0,
            collected=passed + failed,
            outcomes=outcomes,
            report_path=str(report_path),
        )

    def extract_failure_excerpt(self, test_id: str, report_path: Path) -> str:
        # Try JUnit XML first.
        if report_path.exists() and report_path.stat().st_size > 0:
            try:
                tree = ElementTree.parse(report_path)
            except ElementTree.ParseError:
                tree = None
            if tree is not None:
                target_class, _, target_name = test_id.rpartition(".")
                for tc in tree.iter("testcase"):
                    name = tc.attrib.get("name") or ""
                    classname = tc.attrib.get("classname") or ""
                    if not (
                        (name == target_name and classname == target_class)
                        or f"{classname}.{name}" == test_id
                    ):
                        continue
                    for tag in ("failure", "error"):
                        node = tc.find(tag)
                        if node is not None:
                            msg = node.attrib.get("message") or ""
                            body = (node.text or "").strip()
                            return f"{msg}\n{body}".strip()
        # Fall back to log scan.
        log_path = report_path.with_suffix(report_path.suffix + ".log")
        if not log_path.exists():
            return ""
        captured: list[str] = []
        capturing = False
        for line in log_path.read_text(errors="replace").splitlines():
            if test_id in line:
                capturing = True
                captured.append(line)
                continue
            if capturing:
                if line.startswith("Test Case '"):
                    break
                captured.append(line)
        return "\n".join(captured).strip()

    def infrastructure_paths(self, workspace: Path) -> set[str]:
        result: set[str] = set()
        for fname in _SWIFT_INFRASTRUCTURE_FILENAMES:
            if (workspace / fname).exists():
                result.add(fname)
        for path in workspace.iterdir():
            if path.suffix in _SWIFT_INFRASTRUCTURE_SUFFIXES:
                result.add(path.name)
        return result

    def stub_patterns(self) -> list[str]:
        return list(_SWIFT_STUB_PATTERNS)


register_adapter(SwiftXCTestAdapter())
