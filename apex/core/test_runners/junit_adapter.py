"""JUnit (Maven Surefire / Gradle) adapter for Java / Kotlin / Scala."""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree

from . import RunResult, register_adapter

_JAVA_INFRASTRUCTURE_FILENAMES = {
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "settings.gradle",
    "settings.gradle.kts",
    "gradle.properties",
}
_JAVA_STUB_PATTERNS = [
    r"throw new UnsupportedOperationException\(",
    r"throw new RuntimeException\((['\"]).*not.{0,5}implemented.*\1\)",
    r"throw new NotImplementedError\(",
    r"//\s*TODO[: ]",
]


def _detect_build_tool(workspace: Path) -> str:
    if (workspace / "pom.xml").exists():
        return "maven"
    if (workspace / "build.gradle").exists() or (workspace / "build.gradle.kts").exists():
        return "gradle"
    return "maven"


class JUnitAdapter:
    name = "junit"
    language = "java"

    def list_tests(self, workspace: Path, env: dict[str, str]) -> set[str]:
        # Both Maven and Gradle support a "list tests" pre-pass via their
        # surefire/JUnit Platform integration, but the cheapest path is
        # to grep the source tree for ``@Test``-annotated methods. That
        # avoids a multi-minute Maven invocation on every list call.
        nodeids: set[str] = set()
        test_pattern = re.compile(r"^\s*@Test(\s|\(|$)")
        method_pattern = re.compile(
            r"^\s*(?:public|private|protected)?\s*(?:static\s+)?\S+\s+(\w+)\s*\("
        )
        for src in workspace.rglob("src/test/java/**/*.java"):
            try:
                lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            class_name = src.stem
            armed = False
            for line in lines:
                if armed:
                    m = method_pattern.match(line)
                    if m:
                        nodeids.add(f"{class_name}#{m.group(1)}")
                        armed = False
                        continue
                if test_pattern.search(line):
                    armed = True
        return nodeids

    def build_run_command(
        self,
        workspace: Path,
        test_ids: list[str],
        report_path: Path,
        *,
        executable: Optional[str] = None,
    ) -> str:
        # Maven: ``mvn test -Dtest=ClassA#methodA+ClassB#methodB``
        # Gradle: ``./gradlew test --tests ClassA.methodA --tests ClassB.methodB``
        # Reports are emitted as JUnit-XML under target/surefire-reports
        # or build/test-results; report_path here points to a directory
        # we'll later parse via parse_report.
        tool = _detect_build_tool(workspace)
        report_path.mkdir(parents=True, exist_ok=True)
        if tool == "gradle":
            test_args = " ".join(
                f"--tests {shlex.quote(nid.replace('#', '.'))}" for nid in test_ids
            )
            return f"./gradlew test {test_args} --rerun-tasks"
        joined = "+".join(test_ids)
        return f"mvn -B test -Dtest={shlex.quote(joined)}"

    def parse_report(self, report_path: Path) -> RunResult:
        # report_path is the directory that will hold (or already holds)
        # surefire / build/test-results XML files.
        roots: list[Path] = []
        if report_path.is_dir():
            roots.append(report_path)
        for candidate in (
            report_path.parent / "target" / "surefire-reports",
            report_path.parent / "build" / "test-results" / "test",
        ):
            if candidate.exists():
                roots.append(candidate)
        outcomes: dict[str, str] = {}
        passed = failed = errors = skipped = 0
        for root in roots:
            for xml_file in root.rglob("TEST-*.xml"):
                try:
                    tree = ElementTree.parse(xml_file)
                except ElementTree.ParseError:
                    continue
                for tc in tree.iter("testcase"):
                    classname = tc.attrib.get("classname") or ""
                    name = tc.attrib.get("name") or ""
                    nid = f"{classname}#{name}"
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
        roots: list[Path] = []
        if report_path.is_dir():
            roots.append(report_path)
        for candidate in (
            report_path.parent / "target" / "surefire-reports",
            report_path.parent / "build" / "test-results" / "test",
        ):
            if candidate.exists():
                roots.append(candidate)
        target_class, _, target_method = test_id.partition("#")
        for root in roots:
            for xml_file in root.rglob("TEST-*.xml"):
                try:
                    tree = ElementTree.parse(xml_file)
                except ElementTree.ParseError:
                    continue
                for tc in tree.iter("testcase"):
                    if (
                        tc.attrib.get("classname") != target_class
                        or tc.attrib.get("name") != target_method
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
        for fname in _JAVA_INFRASTRUCTURE_FILENAMES:
            candidate = workspace / fname
            if candidate.exists():
                result.add(fname)
        # Test-resources commonly hold fixtures referenced by tests.
        for path in workspace.rglob("src/test/resources"):
            if path.is_dir():
                try:
                    result.add(str(path.relative_to(workspace)))
                except ValueError:
                    continue
        return result

    def stub_patterns(self) -> list[str]:
        return list(_JAVA_STUB_PATTERNS)


register_adapter(JUnitAdapter())
