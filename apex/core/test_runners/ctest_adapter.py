"""C / C++ adapter via CTest + GoogleTest.

CMake-based projects expose tests through ``ctest``. Most repos use
GoogleTest or Catch2 binaries that emit JUnit-XML when invoked with
``--gtest_output=xml:<path>`` or ``-r junit``. This adapter wraps both:
``list_tests`` queries CTest, ``build_run_command`` uses CTest's regex
filter, and ``parse_report`` reads the JUnit-XML CTest emits via
``--output-junit``.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree

from . import RunResult, register_adapter

_C_INFRASTRUCTURE_FILENAMES = {
    "CMakeLists.txt",
    "CMakePresets.json",
    "Makefile",
    "Makefile.am",
    "configure.ac",
    "meson.build",
    "conanfile.py",
    "conanfile.txt",
    "vcpkg.json",
    "compile_commands.json",
}
_C_STUB_PATTERNS = [
    r'assert\s*\(\s*(0|false)\s*&&\s*[\'"`].*not.{0,5}implemented.*[\'"`]',
    r"abort\s*\(\s*\)\s*;?\s*//.*not.{0,5}implemented",
    r'throw\s+std::runtime_error\s*\(\s*[\'"`].*not.{0,5}implemented.*[\'"`]',
    r'throw\s+std::logic_error\s*\(\s*[\'"`].*not.{0,5}implemented.*[\'"`]',
    r"//\s*TODO[: ]",
    r"/\*\s*TODO",
]


class CTestAdapter:
    name = "ctest"
    language = "cpp"

    def list_tests(self, workspace: Path, env: dict[str, str]) -> set[str]:
        try:
            result = subprocess.run(
                ["ctest", "-N"],
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
            line = line.strip()
            # CTest emits lines like ``  Test #5: TestSuite.test_name``.
            if " Test " in line and ":" in line:
                _, _, name = line.partition(":")
                if name.strip():
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
        if not test_ids:
            return f"ctest --output-junit {shlex.quote(str(report_path))}"
        regex = "^(" + "|".join(re.escape(t) for t in test_ids) + ")$"
        return (
            f"ctest -R {shlex.quote(regex)} "
            f"--output-junit {shlex.quote(str(report_path))} "
            f"--output-on-failure"
        )

    def parse_report(self, report_path: Path) -> RunResult:
        if not report_path.exists():
            return RunResult(returncode=1)
        try:
            tree = ElementTree.parse(report_path)
        except ElementTree.ParseError:
            return RunResult(returncode=1)
        outcomes: dict[str, str] = {}
        passed = failed = errors = skipped = 0
        for tc in tree.iter("testcase"):
            name = tc.attrib.get("name") or ""
            classname = tc.attrib.get("classname") or ""
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
        if not report_path.exists():
            return ""
        try:
            tree = ElementTree.parse(report_path)
        except ElementTree.ParseError:
            return ""
        target_class, _, target_name = test_id.rpartition(".")
        for tc in tree.iter("testcase"):
            name = tc.attrib.get("name") or ""
            classname = tc.attrib.get("classname") or ""
            if name != target_name and (classname, name) != (target_class, target_name):
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
        for fname in _C_INFRASTRUCTURE_FILENAMES:
            if (workspace / fname).exists():
                result.add(fname)
        for path in workspace.rglob("CMakeLists.txt"):
            try:
                result.add(str(path.relative_to(workspace)))
            except ValueError:
                continue
        return result

    def stub_patterns(self) -> list[str]:
        return list(_C_STUB_PATTERNS)


register_adapter(CTestAdapter())
