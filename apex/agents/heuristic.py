"""
Local deterministic repair fallback used when LLM execution is unavailable.
"""

from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ..core.subprocess_utils import run_shell_command
from ..planning.manager import IssuePlan, RolloutBrief
from .solver import AgentResult


@dataclass
class _AttemptResult:
    summary: str
    changed_files: list[str]
    tests_run: list[str]


def _is_test_like_path(path: str | Path) -> bool:
    parts = [part.lower() for part in Path(path).parts]
    return any("test" in part or "__pycache__" in part for part in parts)


class HeuristicRepairAgent:
    """Small library of test-guided Python repair heuristics."""

    def __init__(
        self,
        working_dir: str,
        test_timeout: int | None = None,
    ):
        self.working_dir = Path(working_dir).resolve()
        self.test_timeout = test_timeout

    def _existing_source_candidates(self, paths: list[str]) -> list[str]:
        candidates: list[str] = []
        for raw_path in paths:
            rel_path = str(raw_path or "").strip().replace("\\", "/")
            if (
                not rel_path
                or not rel_path.endswith(".py")
                or Path(rel_path).is_absolute()
                or _is_test_like_path(rel_path)
            ):
                continue
            target = (self.working_dir / rel_path).resolve()
            try:
                normalized = target.relative_to(self.working_dir).as_posix()
            except ValueError:
                continue
            if target.is_file():
                candidates.append(normalized)
        return list(dict.fromkeys(candidates))

    def _read_candidate_source(self, rel_path: str) -> str | None:
        file_path = self.working_dir / rel_path
        if not file_path.is_file():
            return None
        try:
            return file_path.read_text()
        except OSError:
            return None

    def run(
        self,
        issue_description: str,
        issue_plan: IssuePlan,
        rollout_brief: RolloutBrief,
        test_command: Optional[str],
    ) -> AgentResult:
        if not test_command:
            return AgentResult(
                success=False,
                output="Heuristic fallback requires a test command.",
                submission_tool="submit_patch",
                submission={},
                tokens_used=0,
            )

        keywords = {keyword.lower() for keyword in issue_plan.keywords}
        candidate_files = self._existing_source_candidates(
            list(dict.fromkeys(rollout_brief.focus_files + issue_plan.relevant_files))
        )
        if not candidate_files:
            candidate_files = [
                str(path.relative_to(self.working_dir))
                for path in self.working_dir.rglob("*.py")
                if path.is_file() and not _is_test_like_path(path.relative_to(self.working_dir))
            ]

        baseline = self._run_tests(test_command)
        if baseline.returncode == 0:
            return AgentResult(
                success=True,
                output="Tests already pass; no repair needed.",
                submission_tool="submit_patch",
                submission={
                    "summary": "Tests already pass; no changes required.",
                    "tests_run": [test_command],
                    "changed_files": [],
                    "confidence": 1.0,
                },
                tokens_used=0,
            )
        strategy_chain = self._build_strategy_chain(issue_description, keywords)

        for _, strategy_fn in strategy_chain:
            result = strategy_fn(candidate_files, test_command)
            if result:
                payload = {
                    "summary": result.summary,
                    "tests_run": result.tests_run,
                    "changed_files": result.changed_files,
                    "confidence": 0.35,
                    "followups": [
                        "LLM execution was unavailable, so APEX used the deterministic local repair fallback."
                    ],
                }
                return AgentResult(
                    success=True,
                    output=str(payload),
                    submission_tool="submit_patch",
                    submission=payload,
                    tokens_used=0,
                )

        failure_output = (baseline.stdout + baseline.stderr).strip()
        return AgentResult(
            success=False,
            output=(
                "Heuristic fallback could not repair the issue.\n\n"
                f"Initial test output:\n{failure_output}"
            ),
            submission_tool="submit_patch",
            submission={},
            tokens_used=0,
        )

    def _build_strategy_chain(
        self,
        issue_description: str,
        keywords: set[str],
    ) -> list[tuple[str, Callable[[list[str], str], Optional[_AttemptResult]]]]:
        lowered = issue_description.lower()
        ordered: list[tuple[str, Callable[[list[str], str], Optional[_AttemptResult]]]] = []

        if {"inclusive", "boundary", "range", "upper"} & keywords or "inclusive" in lowered:
            ordered.append(("inclusive-range", self._try_inclusive_range_fix))
        if {"merge", "nested", "defaults", "override", "config"} & keywords or "nested" in lowered:
            ordered.append(("recursive-merge", self._try_recursive_merge_fix))
        if {
            "path",
            "route",
            "slash",
            "normalize",
            "root",
        } & keywords or "trailing slash" in lowered:
            ordered.append(("path-normalization", self._try_path_normalization_fix))

        ordered.append(("operator-mutations", self._try_operator_mutations))
        return ordered

    def _run_tests(self, test_command: str) -> subprocess.CompletedProcess[str]:
        try:
            return run_shell_command(
                test_command,
                self.working_dir,
                timeout=self.test_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            timeout_suffix = ""
            if self.test_timeout is not None:
                timeout_suffix = f"\nHeuristic test run timed out after {self.test_timeout}s."
            return subprocess.CompletedProcess(
                args=exc.cmd,
                returncode=124,
                stdout=stdout,
                stderr=f"{stderr}{timeout_suffix}",
            )

    def _attempt_rewrite(
        self,
        rel_path: str,
        new_content: str,
        test_command: str,
        summary: str,
    ) -> Optional[_AttemptResult]:
        file_path = self.working_dir / rel_path
        if not file_path.is_file():
            return None
        original = file_path.read_text()
        if original == new_content:
            return None

        try:
            if file_path.suffix == ".py":
                ast.parse(new_content)
        except SyntaxError:
            return None

        file_path.write_text(new_content)
        result = self._run_tests(test_command)
        if result.returncode == 0:
            return _AttemptResult(
                summary=summary, changed_files=[rel_path], tests_run=[test_command]
            )

        file_path.write_text(original)
        return None

    def _try_inclusive_range_fix(
        self,
        candidate_files: list[str],
        test_command: str,
    ) -> Optional[_AttemptResult]:
        range_pattern = re.compile(r"range\(([^,\n]+),\s*([^)]+)\)")
        for rel_path in candidate_files:
            content = self._read_candidate_source(rel_path)
            if content is None:
                continue
            lines = content.splitlines()

            combined_lines = lines.copy()
            combined_changed = False
            for index, line in enumerate(combined_lines):
                if "range(" in line and not line.lstrip().startswith("def "):
                    updated = range_pattern.sub(r"range(\1, \2 + 1)", line, count=1)
                    if updated != line:
                        combined_lines[index] = updated
                        combined_changed = True
                if line.lstrip().startswith("if ") and ">=" in line:
                    combined_lines[index] = combined_lines[index].replace(">=", ">", 1)
                    combined_changed = True
                if line.lstrip().startswith("if ") and "<=" in line:
                    combined_lines[index] = combined_lines[index].replace("<=", "<", 1)
                    combined_changed = True
            if combined_changed:
                result = self._attempt_rewrite(
                    rel_path,
                    "\n".join(combined_lines) + ("\n" if content.endswith("\n") else ""),
                    test_command,
                    "Adjusted both the empty-range guard and upper bound handling for inclusive semantics.",
                )
                if result:
                    return result

            candidate_lines = lines.copy()
            for index, line in enumerate(candidate_lines):
                if "range(" in line and not line.lstrip().startswith("def "):
                    updated = range_pattern.sub(r"range(\1, \2 + 1)", line, count=1)
                    if updated != line:
                        candidate_lines[index] = updated
                        result = self._attempt_rewrite(
                            rel_path,
                            "\n".join(candidate_lines) + ("\n" if content.endswith("\n") else ""),
                            test_command,
                            "Adjusted range upper bound to include the terminal value.",
                        )
                        if result:
                            return result
                        break

            for index, line in enumerate(lines):
                if ">=" in line and "return" in line:
                    mutated = lines.copy()
                    mutated[index] = line.replace(">=", ">", 1)
                    result = self._attempt_rewrite(
                        rel_path,
                        "\n".join(mutated) + ("\n" if content.endswith("\n") else ""),
                        test_command,
                        "Relaxed an inclusive boundary guard that was cutting off a valid case.",
                    )
                    if result:
                        return result
                if "<=" in line and "return" in line:
                    mutated = lines.copy()
                    mutated[index] = line.replace("<=", "<", 1)
                    result = self._attempt_rewrite(
                        rel_path,
                        "\n".join(mutated) + ("\n" if content.endswith("\n") else ""),
                        test_command,
                        "Relaxed an inclusive lower-bound guard that was blocking a valid case.",
                    )
                    if result:
                        return result
        return None

    def _try_recursive_merge_fix(
        self,
        candidate_files: list[str],
        test_command: str,
    ) -> Optional[_AttemptResult]:
        for rel_path in candidate_files:
            source = self._read_candidate_source(rel_path)
            if source is None:
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue

            lines = source.splitlines()
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef) or len(node.args.args) != 2:
                    continue
                body_source = "\n".join(lines[node.lineno - 1 : node.end_lineno])
                if ".update(" not in body_source or ".copy()" not in body_source:
                    continue

                replacement = self._build_recursive_merge_function(
                    function_name=node.name,
                    defaults_arg=node.args.args[0].arg,
                    override_arg=node.args.args[1].arg,
                ).splitlines()
                updated_lines = lines[: node.lineno - 1] + replacement + lines[node.end_lineno :]
                result = self._attempt_rewrite(
                    rel_path,
                    "\n".join(updated_lines) + ("\n" if source.endswith("\n") else ""),
                    test_command,
                    "Replaced shallow dict merging with a recursive merge that preserves nested defaults.",
                )
                if result:
                    return result
        return None

    def _build_recursive_merge_function(
        self,
        function_name: str,
        defaults_arg: str,
        override_arg: str,
    ) -> str:
        return "\n".join(
            [
                f"def {function_name}({defaults_arg}, {override_arg}):",
                "    def _clone(value):",
                "        if isinstance(value, dict):",
                "            return {key: _clone(item) for key, item in value.items()}",
                "        if isinstance(value, list):",
                "            return [_clone(item) for item in value]",
                "        return value",
                "",
                f"    merged = _clone({defaults_arg})",
                f"    for key, value in {override_arg}.items():",
                "        if isinstance(value, dict) and isinstance(merged.get(key), dict):",
                f"            merged[key] = {function_name}(merged[key], value)",
                "        else:",
                "            merged[key] = _clone(value)",
                "    return merged",
            ]
        )

    def _try_path_normalization_fix(
        self,
        candidate_files: list[str],
        test_command: str,
    ) -> Optional[_AttemptResult]:
        for rel_path in candidate_files:
            source = self._read_candidate_source(rel_path)
            if source is None:
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue

            lines = source.splitlines()
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef) or len(node.args.args) != 1:
                    continue
                if (
                    node.name.lower() not in {"normalize_path", "canonicalize_path"}
                    and "path" not in node.name.lower()
                ):
                    continue

                replacement = self._build_path_normalizer(
                    function_name=node.name,
                    arg_name=node.args.args[0].arg,
                ).splitlines()
                updated_lines = lines[: node.lineno - 1] + replacement + lines[node.end_lineno :]
                result = self._attempt_rewrite(
                    rel_path,
                    "\n".join(updated_lines) + ("\n" if source.endswith("\n") else ""),
                    test_command,
                    "Normalized path handling to preserve root paths and collapse duplicate slashes safely.",
                )
                if result:
                    return result

                if 'rstrip("/")' in source:
                    candidate = source.replace('rstrip("/")', 'rstrip("/") or "/"', 1)
                    result = self._attempt_rewrite(
                        rel_path,
                        candidate,
                        test_command,
                        "Preserved the root path when trimming trailing slashes.",
                    )
                    if result:
                        return result
        return None

    def _build_path_normalizer(self, function_name: str, arg_name: str) -> str:
        return "\n".join(
            [
                f"def {function_name}({arg_name}):",
                f"    normalized = '/' + {arg_name}.lstrip('/')",
                "    while '//' in normalized:",
                "        normalized = normalized.replace('//', '/')",
                "    normalized = normalized.rstrip('/') or '/'",
                "    return normalized",
            ]
        )

    def _try_operator_mutations(
        self,
        candidate_files: list[str],
        test_command: str,
    ) -> Optional[_AttemptResult]:
        operator_mutations = [
            (">=", ">"),
            ("<=", "<"),
            (">", ">="),
            ("<", "<="),
        ]

        for rel_path in candidate_files:
            source = self._read_candidate_source(rel_path)
            if source is None:
                continue
            lines = source.splitlines()
            for index, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue

                for old, new in operator_mutations:
                    if old not in line:
                        continue
                    if old == ">" and ">=" in line:
                        continue
                    if old == "<" and "<=" in line:
                        continue
                    mutated = lines.copy()
                    mutated[index] = line.replace(old, new, 1)
                    result = self._attempt_rewrite(
                        rel_path,
                        "\n".join(mutated) + ("\n" if source.endswith("\n") else ""),
                        test_command,
                        f"Adjusted a comparison operator from {old} to {new}.",
                    )
                    if result:
                        return result

                if "range(" in line and "+ 1" not in line and not line.lstrip().startswith("def "):
                    candidate_line = re.sub(
                        r"range\(([^,\n]+),\s*([^)]+)\)",
                        r"range(\1, \2 + 1)",
                        line,
                        count=1,
                    )
                    if candidate_line != line:
                        mutated = lines.copy()
                        mutated[index] = candidate_line
                        result = self._attempt_rewrite(
                            rel_path,
                            "\n".join(mutated) + ("\n" if source.endswith("\n") else ""),
                            test_command,
                            "Expanded a range upper bound during local search.",
                        )
                        if result:
                            return result

                if 'rstrip("/")' in line:
                    mutated = lines.copy()
                    mutated[index] = line.replace('rstrip("/")', 'rstrip("/") or "/"', 1)
                    result = self._attempt_rewrite(
                        rel_path,
                        "\n".join(mutated) + ("\n" if source.endswith("\n") else ""),
                        test_command,
                        "Preserved the root path during slash trimming.",
                    )
                    if result:
                        return result
        return None
