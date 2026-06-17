"""Jest / Vitest adapter for JavaScript / TypeScript test suites."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Optional

from . import RunResult, register_adapter

_JS_INFRASTRUCTURE_FILENAMES = {
    "jest.config.js",
    "jest.config.ts",
    "jest.config.cjs",
    "jest.config.mjs",
    "jest.config.json",
    "jest.setup.js",
    "jest.setup.ts",
    "vitest.config.js",
    "vitest.config.ts",
    "vitest.config.mjs",
    "vitest.workspace.ts",
    "babel.config.js",
    "babel.config.json",
    ".babelrc",
    ".babelrc.json",
    "tsconfig.json",
}
_JS_STUB_PATTERNS = [
    r"throw new Error\((['\"]).*not.{0,5}implemented.*\1\)",
    r"throw new Error\((['\"]).*TODO.*\1\)",
    r"return undefined;",
    r"return null;\s*//\s*TODO",
    r"//\s*TODO[: ]",
    r"return\s*\{\s*\}\s*;\s*//\s*TODO",
]


class _JsAdapterBase:
    name = "jest"
    language = "javascript"
    runner_invocation = "npx jest"
    list_args = "--listTests"
    json_args = "--json --outputFile={report}"
    report_filename = "jest-report.json"

    def list_tests(self, workspace: Path, env: dict[str, str]) -> set[str]:
        cmd = f"{self.runner_invocation} {self.list_args}"
        try:
            result = subprocess.run(
                shlex.split(cmd),
                cwd=str(workspace),
                env={**env},
                capture_output=True,
                text=True,
                timeout=180,
            )
        except (subprocess.TimeoutExpired, OSError):
            return set()
        return {line.strip() for line in (result.stdout or "").splitlines() if line.strip()}

    def build_run_command(
        self,
        workspace: Path,
        test_ids: list[str],
        report_path: Path,
        *,
        executable: Optional[str] = None,
    ) -> str:
        runner = executable or self.runner_invocation
        # Jest accepts test-file regex patterns via positional args; the
        # exact test name uses ``-t``. We pass the (deduplicated) file
        # paths as patterns so Jest narrows collection to those files,
        # mirroring the pytest "files-not-IDs" trick.
        files = sorted({nid.split("::", 1)[0] for nid in test_ids if nid})
        files_arg = " ".join(shlex.quote(f) for f in files)
        return f"{runner} {files_arg} --json --outputFile={shlex.quote(str(report_path))} --silent"

    def parse_report(self, report_path: Path) -> RunResult:
        if not report_path.exists():
            return RunResult(returncode=1)
        try:
            payload = json.loads(report_path.read_text())
        except (OSError, ValueError):
            return RunResult(returncode=1)
        outcomes: dict[str, str] = {}
        passed = failed = errors = skipped = 0
        for suite in payload.get("testResults") or []:
            file_path = suite.get("name") or suite.get("testFilePath") or ""
            for test in suite.get("testResults") or suite.get("assertionResults") or []:
                title_parts = test.get("ancestorTitles") or []
                title = test.get("title") or test.get("fullName") or ""
                full_id = f"{file_path}::" + " > ".join([*title_parts, title]).strip()
                status = (test.get("status") or "").lower()
                outcomes[full_id] = status
                if status in {"passed", "todo"}:
                    passed += 1
                elif status == "failed":
                    failed += 1
                elif status == "pending" or status == "skipped":
                    skipped += 1
                else:
                    errors += 1
        success = bool(payload.get("success"))
        collected = int(payload.get("numTotalTests") or (passed + failed + errors + skipped))
        return RunResult(
            returncode=0 if success else 1,
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
            payload = json.loads(report_path.read_text())
        except (OSError, ValueError):
            return ""
        for suite in payload.get("testResults") or []:
            file_path = suite.get("name") or suite.get("testFilePath") or ""
            for test in suite.get("testResults") or suite.get("assertionResults") or []:
                title_parts = test.get("ancestorTitles") or []
                title = test.get("title") or test.get("fullName") or ""
                full_id = f"{file_path}::" + " > ".join([*title_parts, title]).strip()
                if full_id != test_id:
                    continue
                msgs = test.get("failureMessages") or []
                if isinstance(msgs, list) and msgs:
                    return "\n".join(str(m) for m in msgs).strip()
        return ""

    def infrastructure_paths(self, workspace: Path) -> set[str]:
        result: set[str] = set()
        for fname in _JS_INFRASTRUCTURE_FILENAMES:
            candidate = workspace / fname
            if candidate.exists():
                result.add(fname)
        for path in workspace.rglob("__mocks__"):
            try:
                result.add(str(path.relative_to(workspace)))
            except ValueError:
                continue
        return result

    def stub_patterns(self) -> list[str]:
        return list(_JS_STUB_PATTERNS)


class JestAdapter(_JsAdapterBase):
    name = "jest"
    runner_invocation = "npx jest"


class VitestAdapter(_JsAdapterBase):
    name = "vitest"
    runner_invocation = "npx vitest run"
    list_args = "list"

    def build_run_command(
        self,
        workspace: Path,
        test_ids: list[str],
        report_path: Path,
        *,
        executable: Optional[str] = None,
    ) -> str:
        runner = executable or self.runner_invocation
        files = sorted({nid.split("::", 1)[0] for nid in test_ids if nid})
        files_arg = " ".join(shlex.quote(f) for f in files)
        return f"{runner} {files_arg} --reporter=json --outputFile={shlex.quote(str(report_path))}"


class MochaAdapter(_JsAdapterBase):
    name = "mocha"
    runner_invocation = "npx mocha"
    list_args = "--dry-run --reporter=min"

    def build_run_command(
        self,
        workspace: Path,
        test_ids: list[str],
        report_path: Path,
        *,
        executable: Optional[str] = None,
    ) -> str:
        runner = executable or self.runner_invocation
        files = sorted({nid.split("::", 1)[0] for nid in test_ids if nid})
        files_arg = " ".join(shlex.quote(f) for f in files)
        return (
            f"{runner} {files_arg} "
            f"--reporter=json --reporter-options=output={shlex.quote(str(report_path))}"
        )

    def parse_report(self, report_path: Path) -> RunResult:
        if not report_path.exists():
            return RunResult(returncode=1)
        try:
            payload = json.loads(report_path.read_text())
        except (OSError, ValueError):
            return RunResult(returncode=1)
        stats = payload.get("stats") or {}
        passed = int(stats.get("passes") or 0)
        failed = int(stats.get("failures") or 0)
        skipped = int(stats.get("pending") or 0)
        collected = int(stats.get("tests") or (passed + failed + skipped))
        outcomes: dict[str, str] = {}
        for test in payload.get("tests") or []:
            outcomes[test.get("fullTitle") or ""] = "failed" if test.get("err") else "passed"
        return RunResult(
            returncode=0 if failed == 0 else 1,
            passed=passed,
            failed=failed,
            errors=0,
            skipped=skipped,
            collected=collected,
            outcomes=outcomes,
            report_path=str(report_path),
        )

    def extract_failure_excerpt(self, test_id: str, report_path: Path) -> str:
        if not report_path.exists():
            return ""
        try:
            payload = json.loads(report_path.read_text())
        except (OSError, ValueError):
            return ""
        for fail in payload.get("failures") or []:
            if (fail.get("fullTitle") or "") != test_id:
                continue
            err = fail.get("err") or {}
            msg = err.get("message") or ""
            stack = err.get("stack") or ""
            return f"{msg}\n{stack}".strip()
        return ""


class OspecAdapter(_JsAdapterBase):
    name = "ospec"
    runner_invocation = "npx ospec"

    def _npm_script_command(self, script: str, files_arg: str) -> str:
        base = f"npm run {shlex.quote(script)}"
        return f"{base} -- {files_arg}".strip() if files_arg else base

    def _runner_command(self, workspace: Path, files_arg: str) -> str:
        package_json = workspace / "package.json"
        if package_json.exists():
            try:
                data = json.loads(package_json.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
            scripts = data.get("scripts") or {}
            if isinstance(scripts, dict):
                if scripts.get("fasttest"):
                    return self._npm_script_command("fasttest", files_arg)
                if scripts.get("test:app"):
                    extra = f"--fast {files_arg}".strip()
                    return self._npm_script_command("test:app", extra)
                if scripts.get("test"):
                    extra = f"--fast {files_arg}".strip()
                    return f"npm test -- {extra}".strip()
        return f"{self.runner_invocation} {files_arg}".strip()

    def build_run_command(
        self,
        workspace: Path,
        test_ids: list[str],
        report_path: Path,
        *,
        executable: Optional[str] = None,
    ) -> str:
        files = sorted({nid.split("::", 1)[0] for nid in test_ids if nid})
        files_arg = " ".join(shlex.quote(f) for f in files)
        paths_marker = json.dumps(files, separators=(",", ":"))
        runner_command = executable or self._runner_command(workspace, files_arg)
        return (
            "{ "
            f"printf '%s\\n' {shlex.quote('__APEX_TEST_PATHS__=' + paths_marker)}; "
            f"{runner_command}; "
            "code=$?; "
            "printf '%s\\n' \"__APEX_EXIT_CODE__=$code\"; "
            "exit $code; "
            f"}} > {shlex.quote(str(report_path))} 2>&1"
        )

    def parse_report(self, report_path: Path) -> RunResult:
        if not report_path.exists():
            return RunResult(returncode=1)
        try:
            text = report_path.read_text()
        except OSError:
            return RunResult(returncode=1)

        test_paths: list[str] = []
        exit_code = 1
        for line in text.splitlines():
            if line.startswith("__APEX_TEST_PATHS__="):
                try:
                    parsed = json.loads(line.split("=", 1)[1])
                    if isinstance(parsed, list):
                        test_paths = [str(item) for item in parsed if str(item)]
                except ValueError:
                    test_paths = []
            elif line.startswith("__APEX_EXIT_CODE__="):
                try:
                    exit_code = int(line.split("=", 1)[1])
                except ValueError:
                    exit_code = 1

        failed_assertions = 0
        total_assertions = 0
        failed_match = re.search(r"(\d+)\s+out of\s+(\d+)\s+assertions?\s+failed", text)
        passed_match = re.search(r"All\s+(\d+)\s+assertions?\s+passed", text)
        if failed_match:
            failed_assertions = int(failed_match.group(1))
            total_assertions = int(failed_match.group(2))
        elif passed_match:
            total_assertions = int(passed_match.group(1))

        failed = 1 if (exit_code != 0 or failed_assertions > 0 or "BAILED OUT" in text) else 0
        passed = 0 if failed else max(1, len(test_paths))
        outcomes = {path: ("failed" if failed else "passed") for path in (test_paths or ["ospec"])}
        return RunResult(
            returncode=exit_code,
            passed=passed,
            failed=failed,
            errors=0,
            skipped=0,
            collected=len(outcomes) if total_assertions or outcomes else 0,
            outcomes=outcomes,
            raw_output=text,
            report_path=str(report_path),
        )

    def extract_failure_excerpt(self, test_id: str, report_path: Path) -> str:
        if not report_path.exists():
            return ""
        try:
            text = report_path.read_text()
        except OSError:
            return ""
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if line.rstrip().endswith(":") and ">" in line:
                end = min(len(lines), index + 12)
                return "\n".join(lines[index:end]).strip()
        return text[-1200:].strip()


register_adapter(JestAdapter())
register_adapter(VitestAdapter())
register_adapter(MochaAdapter())
register_adapter(OspecAdapter())
