"""C# / F# adapter via ``dotnet test``.

Covers xUnit, NUnit, and MSTest — they all run through the same
``dotnet test`` driver. We use the JUnit logger (``--logger junit``
provided by the JunitXml.TestLogger NuGet that most modern .NET test
projects include); falls back to TRX if a JUnit XML file isn't found.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree

from . import RunResult, register_adapter

_DOTNET_INFRASTRUCTURE_FILENAMES = {
    "Directory.Build.props",
    "Directory.Build.targets",
    "NuGet.config",
    "nuget.config",
    "global.json",
    "Directory.Packages.props",
}
_DOTNET_INFRASTRUCTURE_SUFFIXES = (".csproj", ".fsproj", ".vbproj", ".sln")
_DOTNET_STUB_PATTERNS = [
    r"throw\s+new\s+NotImplementedException\s*\(",
    r'throw\s+new\s+NotSupportedException\s*\(\s*[\'"`].*TODO.*[\'"`]\s*\)',
    r"//\s*TODO[: ]",
    r"return\s+null\s*;\s*//.*TODO",
    r"return\s+default\s*;\s*//.*TODO",
]


class DotNetTestAdapter:
    name = "dotnet-test"
    language = "c_sharp"

    def list_tests(self, workspace: Path, env: dict[str, str]) -> set[str]:
        try:
            result = subprocess.run(
                ["dotnet", "test", "--list-tests", "--nologo", "--verbosity", "quiet"],
                cwd=str(workspace),
                env={**env},
                capture_output=True,
                text=True,
                timeout=300,
            )
        except (subprocess.TimeoutExpired, OSError):
            return set()
        nodeids: set[str] = set()
        in_listing = False
        for line in (result.stdout or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("The following Tests are available:"):
                in_listing = True
                continue
            if in_listing:
                if not stripped:
                    if nodeids:
                        break
                    continue
                if stripped.startswith("Test"):  # legacy header line
                    continue
                nodeids.add(stripped)
        return nodeids

    def build_run_command(
        self,
        workspace: Path,
        test_ids: list[str],
        report_path: Path,
        *,
        executable: Optional[str] = None,
    ) -> str:
        # ``--filter`` accepts a query like ``FullyQualifiedName~Foo|FullyQualifiedName~Bar``.
        if test_ids:
            filter_expr = "|".join(f"FullyQualifiedName~{t}" for t in test_ids)
            filter_arg = f"--filter {shlex.quote(filter_expr)}"
        else:
            filter_arg = ""
        return (
            f"dotnet test --nologo --no-restore --no-build {filter_arg} "
            f"--logger {shlex.quote(f'junit;LogFilePath={report_path}')} "
        ).strip()

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
            if not (
                (name == target_name and classname == target_class)
                or name == test_id
                or f"{classname}.{name}" == test_id
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
        for fname in _DOTNET_INFRASTRUCTURE_FILENAMES:
            if (workspace / fname).exists():
                result.add(fname)
        for path in workspace.rglob("*"):
            if path.is_file() and path.suffix in _DOTNET_INFRASTRUCTURE_SUFFIXES:
                try:
                    result.add(str(path.relative_to(workspace)))
                except ValueError:
                    continue
        return result

    def stub_patterns(self) -> list[str]:
        return list(_DOTNET_STUB_PATTERNS)


register_adapter(DotNetTestAdapter())
