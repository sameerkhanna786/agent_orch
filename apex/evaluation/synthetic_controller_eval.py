"""Synthetic controller-eval task generation.

The generator creates small feature-deletion tasks from an existing repository
without mutating the source tree. These tasks are intended for fast APEX
controller/prompt/context regression checks before full benchmark runs.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class SyntheticControllerTask:
    repo_root: str
    target_file: str
    symbol_name: str
    start_line: int
    end_line: int
    related_tests: list[str]
    validation_command: str
    task_prompt: str
    mutation_diff: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _iter_python_source_files(repo_root: Path) -> Iterable[Path]:
    for path in sorted(repo_root.rglob("*.py")):
        rel = path.relative_to(repo_root)
        parts = {part.lower() for part in rel.parts}
        test_like = any(
            part in {"test", "tests"} or part.startswith("test_") or part.endswith("_tests")
            for part in parts
        )
        if "__pycache__" in parts or test_like:
            continue
        if any(part.startswith(".") for part in rel.parts):
            continue
        yield path


def _is_stub_like(node: ast.AST) -> bool:
    body = getattr(node, "body", [])
    if not body:
        return True
    executable = body[0]
    if isinstance(executable, ast.Pass):
        return True
    if isinstance(executable, ast.Raise):
        call = executable.exc
        if isinstance(call, ast.Call):
            func = call.func
            if isinstance(func, ast.Name) and func.id == "NotImplementedError":
                return True
    if isinstance(executable, ast.Expr) and isinstance(executable.value, ast.Constant):
        return len(body) <= 1
    return False


def _symbol_name(node: ast.AST, parents: list[str]) -> str:
    name = getattr(node, "name", "")
    if parents:
        return ".".join([*parents, name])
    return str(name)


def _candidate_symbols(tree: ast.AST) -> Iterable[tuple[ast.AST, str]]:
    parents: list[str] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.items: list[tuple[ast.AST, str]] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            parents.append(node.name)
            self.generic_visit(node)
            parents.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_function(node)

        def _visit_function(self, node: ast.AST) -> None:
            name = getattr(node, "name", "")
            if name.startswith("__") and name.endswith("__") and name != "__init__":
                return
            if not getattr(node, "end_lineno", None):
                return
            if _is_stub_like(node):
                return
            span = int(getattr(node, "end_lineno")) - int(getattr(node, "lineno")) + 1
            if span > 80:
                return
            self.items.append((node, _symbol_name(node, parents)))

    visitor = Visitor()
    visitor.visit(tree)
    yield from visitor.items


def _indent_for_body(line: str) -> str:
    prefix = line[: len(line) - len(line.lstrip())]
    return prefix + "    "


def _mutate_symbol(source: str, node: ast.AST) -> str:
    lines = source.splitlines(keepends=True)
    start = int(getattr(node, "lineno"))
    end = int(getattr(node, "end_lineno"))
    header = lines[start - 1]
    indent = _indent_for_body(header)
    replacement = [
        header,
        f"{indent}raise NotImplementedError('synthetic controller eval deletion')\n",
    ]
    return "".join(lines[: start - 1] + replacement + lines[end:])


def _related_tests(repo_root: Path, target_file: str, *, max_tests: int = 4) -> list[str]:
    stem = Path(target_file).stem.lower()
    module_path = str(Path(target_file).with_suffix("")).replace("/", ".")
    matches: list[str] = []
    for path in sorted(repo_root.rglob("*.py")):
        rel = path.relative_to(repo_root)
        rel_text = str(rel)
        if "test" not in rel_text.lower():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            continue
        if stem in rel_text.lower() or stem in content or module_path.lower() in content:
            matches.append(rel_text)
        if len(matches) >= max_tests:
            break
    return matches


def build_synthetic_controller_tasks(
    repo_root: str | Path,
    *,
    validation_command: str = "python -m pytest",
    max_tasks: int = 8,
) -> list[SyntheticControllerTask]:
    root = Path(repo_root).expanduser().resolve()
    tasks: list[SyntheticControllerTask] = []
    for path in _iter_python_source_files(root):
        rel = str(path.relative_to(root))
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            continue
        for node, symbol in _candidate_symbols(tree):
            mutated = _mutate_symbol(source, node)
            diff = "".join(
                difflib.unified_diff(
                    source.splitlines(keepends=True),
                    mutated.splitlines(keepends=True),
                    fromfile=f"a/{rel}",
                    tofile=f"b/{rel}",
                )
            )
            tests = _related_tests(root, rel)
            prompt = (
                "Synthetic controller eval task: restore the deleted behavior in "
                f"{rel}::{symbol}. Use the repo map, related tests, and validation "
                "feedback to implement the smallest correct repair. Do not edit "
                "protected tests to make the task pass."
            )
            if tests:
                prompt += " Related tests: " + ", ".join(tests) + "."
            tasks.append(
                SyntheticControllerTask(
                    repo_root=str(root),
                    target_file=rel,
                    symbol_name=symbol,
                    start_line=int(getattr(node, "lineno")),
                    end_line=int(getattr(node, "end_lineno")),
                    related_tests=tests,
                    validation_command=validation_command,
                    task_prompt=prompt,
                    mutation_diff=diff,
                )
            )
            if len(tasks) >= max(1, int(max_tasks)):
                return tasks
    return tasks


def _write_tasks(tasks: list[SyntheticControllerTask], output: Optional[str]) -> None:
    payload = [task.to_dict() for task in tasks]
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic APEX controller eval tasks.")
    parser.add_argument("--repo", required=True, help="Repository root to sample from.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    parser.add_argument("--max-tasks", type=int, default=8)
    parser.add_argument("--validation-command", default="python -m pytest")
    args = parser.parse_args(argv)
    tasks = build_synthetic_controller_tasks(
        args.repo,
        validation_command=args.validation_command,
        max_tasks=args.max_tasks,
    )
    _write_tasks(tasks, args.output or None)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
