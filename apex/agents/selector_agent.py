"""
Agentic patch selector with candidate-specific testing tools.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from ..core.cli_backend import CLIModelClient
from ..core.config import ApexConfig, LLMConfig
from ..core.filesystem import copy_tree
from ..core.llm import AgentLoop, LLMClient, ToolDefinition
from ..core.pytest_utils import (
    build_ephemeral_pytest_command,
    build_pytest_recovery_commands,
    build_runtime_python_command,
    output_indicates_missing_pytest,
    should_disable_pytest_plugin_autoload,
)


class _SelectorSandboxValidationError(ValueError):
    """Raised when selector tool test_code violates path policy."""


def _build_selector_sandbox_env(sandbox_home: Path) -> dict[str, str]:
    allowed_exact = {
        "PATH",
        "LANG",
        "TMPDIR",
        "PYTHONPATH",
        "PYTHONDONTWRITEBYTECODE",
        "SHELL",
        "USER",
        "LOGNAME",
    }
    allowed_prefixes = ("LC_", "PYTEST_")
    sanitized: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in allowed_exact or key.startswith(allowed_prefixes):
            sanitized[key] = value
    sanitized["HOME"] = str(sandbox_home)
    sanitized["TMPDIR"] = sanitized.get("TMPDIR") or str(sandbox_home)
    sanitized["PYTHONDONTWRITEBYTECODE"] = "1"
    return sanitized


def _path_within_allowed_roots(path: Path, allowed_roots: set[Path]) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    for root in allowed_roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _validate_selector_test_code_paths(
    test_code: str,
    *,
    allowed_roots: set[Path],
) -> None:
    """Reject selector test_code that touches paths outside the sandbox."""
    for match in re.finditer(r"['\"](/[A-Za-z0-9_./\\-]+)['\"]", test_code):
        literal = match.group(1)
        if not _path_within_allowed_roots(Path(literal), allowed_roots):
            raise _SelectorSandboxValidationError(
                f"absolute path literal {literal!r} not under sandbox root"
            )
    try:
        tree = ast.parse(test_code)
    except SyntaxError as exc:
        raise _SelectorSandboxValidationError(f"test_code is not parseable Python: {exc}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if value.startswith("/") and not _path_within_allowed_roots(Path(value), allowed_roots):
                raise _SelectorSandboxValidationError(
                    f"AST string {value!r} resolves outside sandbox root"
                )


@dataclass
class _SelectionDecision:
    candidate_id: int
    reasoning: str = ""


@dataclass
class _SelectorVotePlan:
    requested_voters: int
    planned_voters: int
    mode: str = "majority_vote"
    reason: str = ""


class _SelectorToolExecutor:
    def __init__(
        self,
        repo_path: str,
        candidates: list[Any],
        test_command: Optional[str] = None,
        test_timeout: int = 30,
        sandbox_disabled: bool = False,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.candidates = candidates
        self.test_command = test_command
        self.test_timeout = test_timeout
        self.sandbox_disabled = sandbox_disabled
        self.selection: Optional[_SelectionDecision] = None

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        handlers = {
            "view_file": self._view_file,
            "search_files": self._search_files,
            "run_test_on_candidate": self._run_test_on_candidate,
            "run_test_on_all_candidates": self._run_test_on_all_candidates,
            "select_candidate": self._select_candidate,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return f"Unknown selector tool: {tool_name}"
        return handler(**arguments)

    def _view_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: Optional[int] = None,
        candidate_id: Optional[int] = None,
    ) -> str:
        root = self._resolve_root(candidate_id)
        file_path = root / path
        if not file_path.exists():
            return f"File '{path}' does not exist."
        lines = file_path.read_text(errors="replace").splitlines()
        total = len(lines)
        start_line = max(start_line, 1)
        if end_line is None:
            end_line = min(start_line + 99, total)
        end_line = min(end_line, total)
        rendered = [
            f"{index:>6} | {line}"
            for index, line in enumerate(lines[start_line - 1 : end_line], start=start_line)
        ]
        return "\n".join(rendered)

    def _search_files(
        self,
        pattern: str,
        path: Optional[str] = None,
        candidate_id: Optional[int] = None,
    ) -> str:
        root = self._resolve_root(candidate_id)
        search_dir = (root / path).resolve() if path else root
        if shutil.which("rg"):
            result = subprocess.run(
                ["rg", "--files-with-matches", "--color=never", pattern, str(search_dir)],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode not in (0, 1):
                return result.stderr.strip() or "Search failed."
            matches = result.stdout.splitlines()
        else:
            result = subprocess.run(
                ["grep", "-rlE", pattern, str(search_dir)],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode not in (0, 1):
                return result.stderr.strip() or "Search failed."
            matches = result.stdout.splitlines()
        if not matches:
            return "No files matched."
        return "\n".join(matches[:50])

    def _run_test_on_candidate(self, candidate_id: int, test_code: str) -> str:
        if not 0 <= candidate_id < len(self.candidates):
            return f"Invalid candidate_id {candidate_id}"
        return self._run_test(self.candidates[candidate_id].representative.worktree_path, test_code)

    def _run_test_on_all_candidates(self, test_code: str) -> str:
        rows = ["candidate_id | exit_code | summary"]
        for index, candidate in enumerate(self.candidates):
            result = self._run_test(candidate.representative.worktree_path, test_code)
            first_line = result.splitlines()[0] if result else ""
            exit_code = 0
            if first_line.startswith("[exit_code="):
                try:
                    exit_code = int(first_line[len("[exit_code=") : -1])
                except ValueError:
                    exit_code = 1
            rows.append(f"{index} | {exit_code} | {first_line[:120]}")
        return "\n".join(rows)

    def _select_candidate(self, candidate_id: int, reasoning: str = "") -> str:
        self.selection = _SelectionDecision(candidate_id=candidate_id, reasoning=reasoning)
        return f"Selected candidate {candidate_id}."

    def _run_test(self, worktree_path: str, test_code: str) -> str:
        worktree = Path(worktree_path)
        if self.sandbox_disabled:
            return self._run_test_in_worktree(worktree, test_code)
        return self._run_test_sandboxed(worktree, test_code)

    def _run_test_sandboxed(self, worktree: Path, test_code: str) -> str:
        # Build an ephemeral copy of the candidate worktree so adversarial
        # ``test_code`` cannot delete files, exfiltrate env, or reach into
        # other candidate worktrees by absolute path.
        ephemeral_root = Path(tempfile.gettempdir()) / f"apex-sel-{uuid.uuid4().hex}"
        sandbox_root = ephemeral_root / "workspace"
        sandbox_home = ephemeral_root / "home"
        try:
            try:
                copy_tree(
                    worktree,
                    sandbox_root,
                    ignore=shutil.ignore_patterns(
                        "__pycache__", "*.pyc", ".pytest_cache", ".mypy_cache"
                    ),
                    restrict_symlinks_to_root=True,
                )
            except (OSError, shutil.Error) as exc:
                return f"[exit_code=1]\nFailed to prepare sandbox: {exc}"
            sandbox_home.mkdir(parents=True, exist_ok=True)

            allowed_roots = {sandbox_root.resolve(), ephemeral_root.resolve()}
            try:
                _validate_selector_test_code_paths(test_code, allowed_roots=allowed_roots)
            except _SelectorSandboxValidationError as exc:
                return f"[exit_code=1]\nRejected by sandbox policy: {exc}"

            sanitized_env = _build_selector_sandbox_env(sandbox_home)
            test_path = sandbox_root / "_apex_selector_test.py"
            test_path.write_text(test_code)
            is_pytest_style = "def test_" in test_code or "import pytest" in test_code
            disable_plugin_autoload = should_disable_pytest_plugin_autoload(
                self.test_command or "python3 -m pytest -q",
                repo_root=sandbox_root,
            )
            if is_pytest_style:
                command = build_ephemeral_pytest_command(
                    self.test_command,
                    str(test_path.name),
                    disable_plugin_autoload=disable_plugin_autoload,
                ) or (
                    f"{'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ' if disable_plugin_autoload else ''}"
                    f"python3 -m pytest {test_path.name} -q --tb=no"
                )
            else:
                command = (
                    build_runtime_python_command(self.test_command, str(test_path.name))
                    or f"python3 {test_path.name}"
                )
            if disable_plugin_autoload:
                sanitized_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
            else:
                sanitized_env.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)
            return self._invoke_selector_command(
                sandbox_root, command, sanitized_env, is_pytest_style
            )
        finally:
            shutil.rmtree(ephemeral_root, ignore_errors=True)

    def _invoke_selector_command(
        self,
        cwd: Path,
        command: str,
        env: dict[str, str],
        is_pytest_style: bool,
    ) -> str:
        try:
            result = subprocess.run(
                ["bash", "-lc", command],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=self.test_timeout,
                env=env,
            )
            if (
                result.returncode != 0
                and is_pytest_style
                and output_indicates_missing_pytest(result.stdout + result.stderr)
            ):
                for recovery_command in build_pytest_recovery_commands(
                    command,
                    repo_root=cwd,
                ):
                    if recovery_command.strip() == command.strip():
                        continue
                    result = subprocess.run(
                        ["bash", "-lc", recovery_command],
                        cwd=str(cwd),
                        capture_output=True,
                        text=True,
                        timeout=self.test_timeout,
                        env=env,
                    )
                    if result.returncode == 0 or not output_indicates_missing_pytest(
                        result.stdout + result.stderr
                    ):
                        break
            output = "\n".join(part for part in [result.stdout, result.stderr] if part).strip()
            if result.returncode != 0:
                output = f"[exit_code={result.returncode}]\n{output}".strip()
            return output
        except subprocess.TimeoutExpired:
            return f"[exit_code=124]\nCommand timed out after {self.test_timeout} seconds."

    def _run_test_in_worktree(self, worktree: Path, test_code: str) -> str:
        # Legacy unsafe path: kept ONLY behind ``sandbox_disabled`` for ablations.
        test_path = worktree / "_apex_selector_test.py"
        test_path.write_text(test_code)
        is_pytest_style = "def test_" in test_code or "import pytest" in test_code
        disable_plugin_autoload = should_disable_pytest_plugin_autoload(
            self.test_command or "python3 -m pytest -q",
            repo_root=worktree,
        )
        if is_pytest_style:
            command = build_ephemeral_pytest_command(
                self.test_command,
                str(test_path.name),
                disable_plugin_autoload=disable_plugin_autoload,
            ) or (
                f"{'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ' if disable_plugin_autoload else ''}"
                f"python3 -m pytest {test_path.name} -q --tb=no"
            )
        else:
            command = (
                build_runtime_python_command(self.test_command, str(test_path.name))
                or f"python3 {test_path.name}"
            )
        env = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        if disable_plugin_autoload:
            env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        else:
            env.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)
        try:
            return self._invoke_selector_command(worktree, command, env, is_pytest_style)
        finally:
            try:
                test_path.unlink()
            except FileNotFoundError:
                pass

    def _resolve_root(self, candidate_id: Optional[int]) -> Path:
        if candidate_id is None:
            return self.repo_path
        if 0 <= candidate_id < len(self.candidates):
            return Path(self.candidates[candidate_id].representative.worktree_path).resolve()
        return self.repo_path


class SelectorAgent:
    """Multi-turn selector that can inspect code and test candidates."""

    TOOL_DEFINITIONS = [
        ToolDefinition(
            name="view_file",
            description="Read a repository file with bounded context.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "candidate_id": {"type": "integer"},
                },
                "required": ["path"],
            },
        ),
        ToolDefinition(
            name="search_files",
            description="Search repository files for a regex pattern.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "candidate_id": {"type": "integer"},
                },
                "required": ["pattern"],
            },
        ),
        ToolDefinition(
            name="run_test_on_candidate",
            description="Run a Python test script against one candidate patch.",
            parameters={
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "integer"},
                    "test_code": {"type": "string"},
                },
                "required": ["candidate_id", "test_code"],
            },
        ),
        ToolDefinition(
            name="run_test_on_all_candidates",
            description="Run a Python test script against every candidate patch.",
            parameters={
                "type": "object",
                "properties": {"test_code": {"type": "string"}},
                "required": ["test_code"],
            },
        ),
        ToolDefinition(
            name="select_candidate",
            description="Select the strongest candidate patch when confident.",
            parameters={
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "integer"},
                    "reasoning": {"type": "string"},
                },
                "required": ["candidate_id"],
            },
        ),
    ]

    def __init__(self, config: ApexConfig, repo_path: str):
        self.config = config
        self.repo_path = Path(repo_path).resolve()

    def run(
        self,
        candidates: list[Any],
        issue_description: str,
        test_command: Optional[str] = None,
    ) -> int:
        judge_config = self._build_judge_config()
        if judge_config.is_cli_backend:
            try:
                return self._run_cli_selector(judge_config, candidates, issue_description)
            except Exception:
                return self._heuristic_select(candidates)
        if not judge_config.has_api_key:
            return self._heuristic_select(candidates)

        executor = _SelectorToolExecutor(
            str(self.repo_path),
            candidates,
            test_command=test_command,
            test_timeout=self.config.selection.custom_test_timeout_seconds,
            sandbox_disabled=self.config.selection.cross_validation_sandbox_disabled,
        )
        try:
            llm = LLMClient(
                judge_config,
                temperature_override=self.config.selection.judge_temperature,
            )
            loop = AgentLoop(
                llm=llm,
                system_prompt=(
                    "You are a code review expert. Determine which candidate patch is most "
                    "correct. Read code, run differentiating tests, and call select_candidate "
                    "when you are confident."
                ),
                tools=self.TOOL_DEFINITIONS,
                tool_executor=executor.execute,
                max_iterations=self.config.selection.selector_max_iterations,
                finish_tool_names={"select_candidate"},
            )
            loop.set_context_config(self.config.context)
            prompt = self._build_prompt(candidates, issue_description)
            submission = loop.run(prompt)
            if submission is not None:
                candidate_id = submission.arguments.get("candidate_id")
                if isinstance(candidate_id, int) and 0 <= candidate_id < len(candidates):
                    return candidate_id
            if executor.selection and 0 <= executor.selection.candidate_id < len(candidates):
                return executor.selection.candidate_id
        except Exception:
            return self._heuristic_select(candidates)
        return self._heuristic_select(candidates)

    def select_with_majority_voting(
        self,
        candidates: list[Any],
        issue_description: str,
        max_voters: Optional[int] = None,
        test_command: Optional[str] = None,
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> Any:
        if not candidates:
            raise ValueError("select_with_majority_voting requires at least one candidate")

        votes: dict[int, int] = {}
        vote_plan = self._build_vote_plan(candidates, max_voters)
        num_voters = vote_plan.planned_voters
        majority_threshold = (num_voters // 2) + 1
        max_workers = min(
            max(num_voters, 1),
            max(self.config.rollout.parallel_workers, 1),
        )
        self._emit_vote_progress(
            progress_callback,
            vote_plan=vote_plan,
            launched=0,
            votes=votes,
            candidates=candidates,
        )

        if num_voters == 1:
            candidate_id = self.run(candidates, issue_description, test_command)
            if not 0 <= candidate_id < len(candidates):
                candidate_id = self._heuristic_select(candidates)
            votes[candidate_id] = 1
            self._apply_vote_counts(candidates, votes)
            self._emit_vote_progress(
                progress_callback,
                vote_plan=vote_plan,
                launched=1,
                votes=votes,
                candidates=candidates,
            )
            return candidates[candidate_id]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pending: dict[Any, int] = {}
            launched = 0

            def submit_next_voter() -> bool:
                nonlocal launched
                if launched >= num_voters:
                    return False
                future = executor.submit(self.run, candidates, issue_description, test_command)
                pending[future] = launched
                launched += 1
                self._emit_vote_progress(
                    progress_callback,
                    vote_plan=vote_plan,
                    launched=launched,
                    votes=votes,
                    candidates=candidates,
                )
                return True

            while len(pending) < max_workers and submit_next_voter():
                pass

            while pending:
                completed_future = next(iter(as_completed(list(pending))))
                pending.pop(completed_future, None)
                try:
                    candidate_id = completed_future.result()
                except Exception:
                    candidate_id = self._heuristic_select(candidates)
                if not 0 <= candidate_id < len(candidates):
                    candidate_id = self._heuristic_select(candidates)
                votes[candidate_id] = votes.get(candidate_id, 0) + 1
                self._apply_vote_counts(candidates, votes)
                self._emit_vote_progress(
                    progress_callback,
                    vote_plan=vote_plan,
                    launched=launched,
                    votes=votes,
                    candidates=candidates,
                )
                if votes[candidate_id] >= majority_threshold:
                    self._cancel_pending_voters(list(pending))
                    return candidates[candidate_id]
                if self._winner_is_locked(votes, num_voters, candidates):
                    winner_id = self._winner_from_votes(votes, candidates)
                    self._cancel_pending_voters(list(pending))
                    return candidates[winner_id]
                while len(pending) < max_workers and submit_next_voter():
                    pass

        best_id = self._winner_from_votes(votes, candidates)
        self._apply_vote_counts(candidates, votes)
        return candidates[best_id]

    def _build_vote_plan(
        self,
        candidates: list[Any],
        max_voters: Optional[int],
    ) -> _SelectorVotePlan:
        requested_voters = self._resolve_voter_count(candidates, max_voters)
        if requested_voters <= 1:
            return _SelectorVotePlan(
                requested_voters=requested_voters,
                planned_voters=requested_voters,
                mode="single_voter",
                reason="single_candidate_or_budget",
            )

        judge_config = self._build_judge_config()
        if self._deterministic_voting_is_redundant(judge_config):
            reason = (
                "zero_temperature_cli_backend"
                if judge_config.is_cli_backend
                else "deterministic_heuristic_fallback"
            )
            return _SelectorVotePlan(
                requested_voters=requested_voters,
                planned_voters=1,
                mode="single_deterministic_judge",
                reason=reason,
            )
        return _SelectorVotePlan(
            requested_voters=requested_voters,
            planned_voters=requested_voters,
        )

    def _resolve_voter_count(self, candidates: list[Any], max_voters: Optional[int]) -> int:
        candidate_count = len(candidates)
        configured_max = (
            max_voters if max_voters is not None else self.config.selection.selector_max_voters
        )
        upper_bound = max(1, min(max(configured_max, 1), candidate_count * 2))
        if candidate_count <= 1 or upper_bound <= 3:
            return upper_bound

        pressure = self._selection_pressure(candidates)
        if pressure < 0.3:
            return min(upper_bound, 3)
        if pressure < 0.65:
            return min(upper_bound, 4)
        return upper_bound

    def _deterministic_voting_is_redundant(self, judge_config: LLMConfig) -> bool:
        try:
            temperature = float(
                getattr(judge_config, "temperature", self.config.selection.judge_temperature) or 0.0
            )
        except (TypeError, ValueError):
            temperature = 0.0
        if temperature > 0.0:
            return False
        if judge_config.is_cli_backend:
            return True
        return not judge_config.has_api_key

    def _emit_vote_progress(
        self,
        progress_callback: Optional[Callable[[dict[str, Any]], None]],
        *,
        vote_plan: _SelectorVotePlan,
        launched: int,
        votes: dict[int, int],
        candidates: list[Any],
    ) -> None:
        if progress_callback is None:
            return
        payload: dict[str, Any] = {
            "selection_selector_vote_mode": vote_plan.mode,
            "selection_selector_vote_reason": vote_plan.reason,
            "selection_selector_voters_requested": vote_plan.requested_voters,
            "selection_selector_voters_planned": vote_plan.planned_voters,
            "selection_selector_voters_launched": launched,
            "selection_selector_votes_recorded": sum(votes.values()),
        }
        if votes:
            leader_id = self._winner_from_votes(votes, candidates)
            payload["selection_selector_vote_leader"] = leader_id
            payload["selection_selector_vote_leader_votes"] = votes.get(leader_id, 0)
        try:
            progress_callback(payload)
        except Exception:
            return

    def _apply_vote_counts(self, candidates: list[Any], votes: dict[int, int]) -> None:
        for index, cluster in enumerate(candidates):
            cluster.vote_count = votes.get(index, 0)

    def _cancel_pending_voters(
        self,
        futures: list[Any],
        *,
        completed_future: Optional[Any] = None,
    ) -> None:
        for future in futures:
            if future is completed_future:
                continue
            done = getattr(future, "done", None)
            if callable(done) and done():
                continue
            future.cancel()

    def _selection_pressure(self, candidates: list[Any]) -> float:
        if len(candidates) <= 1:
            return 0.0
        scores = sorted(
            (self._heuristic_score(candidate) for candidate in candidates), reverse=True
        )
        top_gap = max(scores[0] - scores[1], 0.0)
        pressure = 1.0 - min(top_gap / 0.25, 1.0)
        if len(candidates) >= 4:
            pressure = min(1.0, pressure + 0.15)
        return max(0.0, pressure)

    def _winner_is_locked(
        self,
        votes: dict[int, int],
        total_voters: int,
        candidates: list[Any],
    ) -> bool:
        if not votes:
            return False
        leader_id = self._winner_from_votes(votes, candidates)
        leader_votes = votes.get(leader_id, 0)
        remaining_votes = max(total_voters - sum(votes.values()), 0)
        challenger_votes = max(
            (count for candidate_id, count in votes.items() if candidate_id != leader_id),
            default=0,
        )
        return leader_votes > challenger_votes + remaining_votes

    def _winner_from_votes(self, votes: dict[int, int], candidates: list[Any]) -> int:
        return max(
            range(len(candidates)),
            key=lambda item: (votes.get(item, 0), self._heuristic_score(candidates[item]), -item),
        )

    def _preferred_comparison_ids(self, candidates: list[Any], limit: int = 3) -> list[int]:
        ranked = sorted(
            range(len(candidates)),
            key=lambda item: (
                self._heuristic_score(candidates[item]),
                candidates[item].size,
                -item,
            ),
            reverse=True,
        )
        return ranked[:limit]

    def _build_prompt(self, candidates: list[Any], issue_description: str) -> str:
        comparison_ids = self._preferred_comparison_ids(candidates)
        sections = [
            "You are a code review expert. Determine which candidate patch is the most correct.",
            "You may inspect repository files, inspect candidate worktrees, and write temporary test scripts to differentiate behavior.",
            "Prefer running the same discriminating test across multiple candidates before voting on a close call.",
            "",
            "Issue:",
            issue_description.strip() or "No issue description provided.",
            "",
            "Differentiation guidance:",
            (
                f"Start by distinguishing candidate IDs {', '.join(str(item) for item in comparison_ids)}."
                if comparison_ids
                else "Inspect the strongest-looking candidates first."
            ),
            "Use run_test_on_all_candidates for the first custom test when multiple candidates look plausible.",
            "Only vote for candidate IDs listed below.",
            "",
            "Candidates:",
        ]
        for index, cluster in enumerate(candidates):
            verification = cluster.verification.to_dict() if cluster.verification else {}
            representative = cluster.representative
            diff_text = representative.patch or ""
            if len(diff_text) > 2500:
                diff_text = diff_text[:2500] + "\n... [diff truncated]"
            sections.extend(
                [
                    f"- Candidate {index}",
                    f"  Worktree: {representative.worktree_path or 'n/a'}",
                    f"  Changed files: {', '.join(representative.changed_files) or 'n/a'}",
                    f"  Verification score: {verification.get('overall_score', 0):.2f}",
                    f"  Cross-validation score: {cluster.cross_validation_score:.2f}",
                    f"  Critic score: {getattr(cluster, 'critic_score', 0.0):.2f}",
                    f"  Heuristic selector score: {self._heuristic_score(cluster):.2f}",
                    f"  Cross-validation row: {verification.get('cross_validation_scores', [])}",
                    f"  Cluster size: {cluster.size}",
                    f"  Test descriptions: {', '.join(representative.test_descriptions) or 'n/a'}",
                    f"  Critic summary: {getattr(cluster, 'critic_summary', '') or 'n/a'}",
                    f"  Verification summary: {json.dumps(verification, sort_keys=True)[:600]}",
                    "  Diff:",
                    diff_text,
                    "",
                ]
            )
        sections.extend(
            [
                "When confident, select the best candidate ID.",
            ]
        )
        return "\n".join(sections)

    def _run_cli_selector(
        self,
        judge_config: LLMConfig,
        candidates: list[Any],
        issue_description: str,
    ) -> int:
        schema = {
            "type": "object",
            "properties": {
                "candidate_id": {"type": "integer"},
                "reasoning": {"type": "string"},
            },
            "required": ["candidate_id"],
        }
        prompt = "\n".join(
            [
                self._build_prompt(candidates, issue_description),
                "",
                "You are allowed to inspect files in the repository and in the candidate worktrees above.",
                "If needed, use inline shell or Python one-liners to compare candidate behavior.",
                "Do not create, modify, or delete files in candidate worktrees.",
                "Return your final decision as JSON matching the schema.",
            ]
        )
        result = CLIModelClient(judge_config).run_structured_prompt(
            prompt=prompt,
            working_dir=str(self.repo_path),
            schema=schema,
            system_prompt=(
                "You are selecting the strongest candidate patch. Use the filesystem and shell "
                "to compare candidates and return the best candidate_id."
            ),
            allow_edits=False,
        )
        if not result.success:
            raise RuntimeError(result.error or "CLI selector failed.")
        payload = result.parsed_json or {}
        candidate_id = payload.get("candidate_id")
        if isinstance(candidate_id, int) and 0 <= candidate_id < len(candidates):
            return candidate_id
        raise RuntimeError("CLI selector returned an invalid candidate id.")

    def _heuristic_select(self, candidates: list[Any]) -> int:
        return max(
            range(len(candidates)),
            key=lambda item: (
                self._heuristic_score(candidates[item]),
                candidates[item].size,
                -item,
            ),
        )

    def _heuristic_score(self, candidate: Any) -> float:
        combined_score = getattr(candidate, "combined_score", None)
        if isinstance(combined_score, (int, float)):
            return float(combined_score)
        verification_score = (
            candidate.verification_score if hasattr(candidate, "verification_score") else 0.0
        )
        cross_validation = getattr(candidate, "cross_validation_score", 0.0)
        critic_score = getattr(candidate, "critic_score", 0.0)
        size = getattr(candidate, "size", 1)
        return (
            (0.50 * verification_score)
            + (0.15 * cross_validation)
            + (0.20 * critic_score)
            + (0.15 * min(size / 3.0, 1.0))
        )

    def _build_judge_config(self) -> LLMConfig:
        primary = self.config.llm_configs[0]
        selector_timeout_seconds = self._selector_vote_timeout_seconds()
        bounded_timeout = max(30, min(int(primary.timeout), selector_timeout_seconds))
        bounded_cli_timeout = max(30, min(int(primary.cli_timeout), selector_timeout_seconds))
        hard_timeout_cap = bounded_cli_timeout + 30
        existing_hard_timeout = (
            int(primary.cli_hard_timeout_seconds)
            if primary.cli_hard_timeout_seconds is not None
            else hard_timeout_cap
        )
        bounded_hard_timeout = max(
            bounded_cli_timeout,
            min(existing_hard_timeout, hard_timeout_cap),
        )
        return LLMConfig(
            model=self.config.selection.judge_model or primary.model,
            backend=primary.backend,
            api_key_env=primary.api_key_env,
            base_url=primary.base_url,
            temperature=self.config.selection.judge_temperature,
            max_tokens=primary.max_tokens,
            timeout=bounded_timeout,
            cli_command=primary.cli_command,
            cli_args=list(primary.cli_args),
            cli_timeout=bounded_cli_timeout,
            cli_hard_timeout_seconds=bounded_hard_timeout,
            cli_stall_window_seconds=selector_timeout_seconds,
            cli_max_inflight_request_seconds=selector_timeout_seconds,
            cli_disable_osx_sandbox=primary.cli_disable_osx_sandbox,
            cli_permission_mode=primary.cli_permission_mode,
            cli_env_overrides=dict(primary.cli_env_overrides),
        )

    def _selector_vote_timeout_seconds(self) -> int:
        caps: list[int] = []
        for value in (
            self.config.selection.verification_timeout_seconds,
            self.config.selection.custom_test_timeout_seconds,
        ):
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                caps.append(parsed)
        if not caps:
            return 120
        return max(45, min(min(caps), 300))
