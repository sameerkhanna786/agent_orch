"""PHPUnit adapter."""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree

from . import RunResult, register_adapter

_PHP_INFRASTRUCTURE_FILENAMES = {
    "phpunit.xml",
    "phpunit.xml.dist",
    "composer.json",
    "composer.lock",
    "phpcs.xml",
    "psalm.xml",
}
_PHP_STUB_PATTERNS = [
    r'throw\s+new\s+\\?LogicException\s*\(\s*[\'"`].*not.{0,5}implemented.*[\'"`]',
    r'throw\s+new\s+\\?RuntimeException\s*\(\s*[\'"`].*not.{0,5}implemented.*[\'"`]',
    r"throw\s+new\s+\\?BadMethodCallException\s*\(",
    r"//\s*TODO[: ]",
    r"#\s*TODO[: ]",
]


class PhpUnitAdapter:
    name = "phpunit"
    language = "php"

    def list_tests(self, workspace: Path, env: dict[str, str]) -> set[str]:
        candidates = ("vendor/bin/phpunit", "phpunit", "./phpunit")
        binary = (
            next(
                (str(workspace / c) if (workspace / c).exists() else None for c in candidates),
                None,
            )
            or "phpunit"
        )
        try:
            result = subprocess.run(
                [binary, "--list-tests"],
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
            if line.startswith("- "):
                nodeids.add(line[2:].strip())
        return nodeids

    def build_run_command(
        self,
        workspace: Path,
        test_ids: list[str],
        report_path: Path,
        *,
        executable: Optional[str] = None,
    ) -> str:
        binary = executable or "vendor/bin/phpunit"
        # PHPUnit's `--filter` is a regex over fully-qualified names.
        if test_ids:
            regex = "/(" + "|".join(re.escape(t) for t in test_ids) + ")/"
            filter_arg = f"--filter {shlex.quote(regex)}"
        else:
            filter_arg = ""
        return (
            f"{shlex.quote(binary)} {filter_arg} --log-junit {shlex.quote(str(report_path))}"
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
        if not report_path.exists():
            return ""
        try:
            tree = ElementTree.parse(report_path)
        except ElementTree.ParseError:
            return ""
        target_class, _, target_name = test_id.rpartition("::")
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
        for fname in _PHP_INFRASTRUCTURE_FILENAMES:
            if (workspace / fname).exists():
                result.add(fname)
        return result

    def stub_patterns(self) -> list[str]:
        return list(_PHP_STUB_PATTERNS)


register_adapter(PhpUnitAdapter())
