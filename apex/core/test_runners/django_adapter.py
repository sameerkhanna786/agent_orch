"""Django repository test runner adapter."""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Optional

from . import RunResult, register_adapter

_DJANGO_OUTCOME_RE = re.compile(
    r"^(?P<test>\S+)\s+\((?P<class>[^)]+)\)\s+\.\.\.\s+(?P<status>ok|FAIL|ERROR|skipped)\b",
    re.IGNORECASE,
)


def _path_to_django_label(path: str) -> str:
    rel = str(path or "").split("::", 1)[0].strip().replace("\\", "/")
    if rel.startswith("tests/"):
        rel = rel[len("tests/") :]
    if rel.endswith(".py"):
        rel = rel[:-3]
    return rel.replace("/", ".")


class DjangoAdapter:
    name = "django"
    language = "python"

    def list_tests(self, workspace: Path, env: dict[str, str]) -> set[str]:
        return set()

    def build_run_command(
        self,
        workspace: Path,
        test_ids: list[str],
        report_path: Path,
        *,
        executable: Optional[str] = None,
    ) -> str:
        python = executable or "python"
        labels = sorted(
            {label for test_id in test_ids if (label := _path_to_django_label(test_id))}
        )
        label_args = " ".join(shlex.quote(label) for label in labels)
        return (
            f"{shlex.quote(python)} tests/runtests.py --verbosity 2 --noinput "
            f"{label_args} > {shlex.quote(str(report_path))} 2>&1"
        )

    def parse_report(self, report_path: Path) -> RunResult:
        if not report_path.exists():
            return RunResult(returncode=1)
        try:
            text = report_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return RunResult(returncode=1)
        outcomes: dict[str, str] = {}
        passed = failed = errors = skipped = 0
        for line in text.splitlines():
            match = _DJANGO_OUTCOME_RE.match(line.strip())
            if not match:
                continue
            nodeid = f"{match.group('class')}.{match.group('test')}"
            status = match.group("status").lower()
            if status == "ok":
                outcomes[nodeid] = "passed"
                passed += 1
            elif status == "fail":
                outcomes[nodeid] = "failed"
                failed += 1
            elif status == "error":
                outcomes[nodeid] = "error"
                errors += 1
            else:
                outcomes[nodeid] = "skipped"
                skipped += 1
        return RunResult(
            returncode=0 if failed == 0 and errors == 0 else 1,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            collected=len(outcomes),
            outcomes=outcomes,
            raw_output=text,
            report_path=str(report_path),
        )

    def extract_failure_excerpt(self, test_id: str, report_path: Path) -> str:
        if not report_path.exists():
            return ""
        try:
            text = report_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        if not test_id:
            return text[-2000:]
        index = text.find(test_id)
        if index < 0:
            return text[-2000:]
        return text[index : index + 2000]


register_adapter(DjangoAdapter())
