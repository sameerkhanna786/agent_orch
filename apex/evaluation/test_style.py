"""Deterministic test-style inference for test-generation prompts.

The model should mirror the repository's existing test framework instead of
being told to use a benchmark-specific default.  This module keeps that
decision cheap, inspectable, and reusable across benchmark adapters.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True)
class TestStyleProfile:
    runner: str
    language: str
    file_naming: str
    function_naming: str
    fixture_style: str
    assertion_style: str
    decorators: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    runner_source: str = "source"
    runner_config_path: str = ""
    repo_context: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def infer_test_style(
    *,
    existing_test_source: str,
    existing_test_path: str,
    focal_path: str,
    repo_root: Path | None = None,
) -> TestStyleProfile:
    """Infer the test style from the existing test context and siblings."""

    language = _infer_language(existing_test_path or focal_path)
    source = str(existing_test_source or "")
    if len(source.strip()) < 80:
        source = (
            _sibling_test_context(
                existing_test_path=existing_test_path,
                focal_path=focal_path,
                repo_root=repo_root,
            )
            or source
        )

    config_runner, config_path = _infer_config_declared_runner(
        repo_root=repo_root,
        language=language,
    )
    if language == "python":
        style = _infer_python_style(
            source=source,
            existing_test_path=existing_test_path,
            focal_path=focal_path,
        )
        return _with_repo_context(
            _override_runner_from_config(style, config_runner, config_path),
            repo_root=repo_root,
            existing_test_source=source,
            existing_test_path=existing_test_path,
        )
    if language in {"javascript", "typescript"}:
        style = _infer_js_ts_style(
            source=source,
            existing_test_path=existing_test_path,
            language=language,
        )
        return _with_repo_context(
            _override_runner_from_config(style, config_runner, config_path),
            repo_root=repo_root,
            existing_test_source=source,
            existing_test_path=existing_test_path,
        )
    if language == "go":
        return _with_repo_context(
            TestStyleProfile(
                runner=config_runner or "go-test",
                language="go",
                file_naming="*_test.go",
                function_naming="func TestXxx(t *testing.T)",
                fixture_style="testing.T helpers",
                assertion_style="t.Fatalf/t.Errorf or existing assertion helper",
                imports=_extract_plain_import_lines(source),
                runner_source="config" if config_runner else "source",
                runner_config_path=config_path,
            ),
            repo_root=repo_root,
            existing_test_source=source,
            existing_test_path=existing_test_path,
        )
    if language == "java":
        runner = config_runner or ("junit5" if "org.junit.jupiter" in source else "junit4")
        return _with_repo_context(
            TestStyleProfile(
                runner=runner,
                language="java",
                file_naming="*Test.java",
                function_naming="@Test methods",
                fixture_style="@BeforeEach/@Before or existing fixture style",
                assertion_style="JUnit assertions matching existing imports",
                decorators=_extract_decorator_lines(source),
                imports=_extract_plain_import_lines(source),
                runner_source="config" if config_runner else "source",
                runner_config_path=config_path,
            ),
            repo_root=repo_root,
            existing_test_source=source,
            existing_test_path=existing_test_path,
        )
    return _with_repo_context(
        TestStyleProfile(
            runner=config_runner or "unknown",
            language=language,
            file_naming=_file_naming_from_path(existing_test_path or focal_path),
            function_naming="match existing test declarations",
            fixture_style="match existing fixtures",
            assertion_style="match existing assertions",
            imports=_extract_plain_import_lines(source),
            runner_source="config" if config_runner else "source",
            runner_config_path=config_path,
        ),
        repo_root=repo_root,
        existing_test_source=source,
        existing_test_path=existing_test_path,
    )


def render_style_contract(style: TestStyleProfile) -> str:
    profile = runner_profile_for_style(style)
    lines = [
        "## Style contract",
        "Match the test framework already used in this repository.",
        f"- Test runner: {style.runner}",
        f"- File location/name: {style.file_naming}",
        f"- Test function/class shape: {style.function_naming}",
        f"- Fixtures/setup: {style.fixture_style}",
        f"- Assertions: {style.assertion_style}",
        (
            "- Do not introduce imports that are not already present in the "
            "existing test file, focal file, or standard library."
        ),
    ]
    forbidden = forbidden_runner_imports(style)
    if forbidden and style.runner == "sympy-bin-test":
        lines.append(
            "- Forbidden test-framework imports for this style: "
            + ", ".join(f"`{name}`" for name in sorted(forbidden))
        )
    elif forbidden:
        lines.append("- Do not add alternate third-party test framework imports.")
    if profile.validation_strategy:
        lines.append(f"- Validation environment: {profile.validation_strategy}")
    if style.decorators:
        lines.append("- Observed decorators: " + ", ".join(f"`{d}`" for d in style.decorators[:12]))
    if style.notes:
        lines.extend(f"- Note: {note}" for note in style.notes[:8])
    if style.repo_context:
        try:
            from .repo_context import render_repo_context_block

            lines.append(render_repo_context_block(style.repo_context))
        except Exception:
            pass
    return "\n".join(lines)


def _with_repo_context(
    style: TestStyleProfile,
    *,
    repo_root: Path | None,
    existing_test_source: str,
    existing_test_path: str,
) -> TestStyleProfile:
    try:
        from .repo_context import probe_repo_context

        context = probe_repo_context(
            repo_root,
            existing_test_source=existing_test_source,
            existing_test_path=existing_test_path,
        ).to_dict()
    except Exception:
        context = {}
    return replace(style, repo_context=context)


def render_observed_imports_block(style: TestStyleProfile) -> str:
    if not style.imports:
        return ""
    return "\n".join(
        [
            "## Imports observed in existing tests",
            "Reuse these imports when possible instead of adding new test-framework dependencies.",
            _fence_for_language(style.language),
            *style.imports[:40],
            "```",
        ]
    )


def forbidden_runner_imports(style: TestStyleProfile) -> set[str]:
    return set(runner_profile_for_style(style).forbidden_imports)


def runner_profile_for_style(style: TestStyleProfile):
    from apex.core.test_runners import profile_for_runner

    return profile_for_runner(
        getattr(style, "runner", "unknown"),
        language=getattr(style, "language", "python"),
        runner_source=getattr(style, "runner_source", "") or "",
    )


def runner_policy_allows_import(style: TestStyleProfile, name: str) -> bool:
    return runner_profile_for_style(style).allows_import(name)


def imports_forbidden_by_style(source: str, style: TestStyleProfile) -> list[str]:
    forbidden = forbidden_runner_imports(style)
    if not forbidden:
        return []
    if not _high_confidence_runner_mismatch(style):
        return []
    imports = extract_top_level_import_names(source, language=style.language)
    # When the runner is declared by an authoritative source (config file or
    # benchmark test command), the artifact will execute in an environment
    # that won't actually load the alternate framework, so even if the host
    # venv happens to have it installed the import is still wrong.
    authoritative_runner = (style.runner_source or "") in {"config", "command"}
    return sorted(
        name
        for name in imports
        if name in forbidden and (authoritative_runner or not _import_available(name))
    )


def _high_confidence_runner_mismatch(style: TestStyleProfile) -> bool:
    # Configuration files (pyproject.toml [tool.pytest.ini_options], etc.) are
    # the strongest signal. An explicit test command supplied by the benchmark
    # adapter (e.g. ``python -m unittest discover``) is just as authoritative
    # because the operator has already declared the runner in invocation form.
    if (style.runner_source or "") in {"config", "command"}:
        return True
    if (style.runner or "") == "sympy-bin-test":
        return any("sympy.testing.pytest" in item for item in style.imports)
    return False


def _import_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def extract_top_level_import_names(source: str, *, language: str = "python") -> set[str]:
    if (language or "").lower() not in {"python", "py", "python3"}:
        return set(re.findall(r"\b(?:from|import)\s+['\"]?([A-Za-z0-9_@./-]+)", source or ""))
    try:
        tree = ast.parse(source or "")
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add((alias.name or "").split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".", 1)[0])
    return {name for name in names if name}


def _infer_python_style(
    *,
    source: str,
    existing_test_path: str,
    focal_path: str,
) -> TestStyleProfile:
    imports = _extract_python_import_lines(source)
    decorators = _extract_python_decorators(source)
    notes: list[str] = []
    lowered = source.lower()
    has_django = bool(
        re.search(r"\b(?:SimpleTestCase|TransactionTestCase|LiveServerTestCase|TestCase)\b", source)
        and (
            "django.test" in source or "override_settings" in source or "modify_settings" in source
        )
    )
    has_unittest_class = bool(
        re.search(r"class\s+\w+\s*\([^)]*(?:unittest\.)?TestCase", source)
        or re.search(
            r"\b(?:SimpleTestCase|TransactionTestCase|LiveServerTestCase|TestCase)\b", source
        )
    )
    has_sympy_pytest = "sympy.testing.pytest" in source
    has_pytest = bool(
        re.search(r"^\s*(?:import\s+pytest|from\s+pytest\s+import)\b", source, re.M)
        or "pytest.fixture" in source
        or "pytest.mark" in source
    )

    if has_django:
        runner = "django-runtests"
        function_naming = "Django TestCase/SimpleTestCase subclass with test_* methods"
        fixture_style = "unittest setUp/tearDown and Django override_settings/modify_settings"
        assertion_style = "self.assert* assertions used by Django's unittest runner"
        notes.append("uses Django's runtests/unittest style")
    elif has_sympy_pytest:
        runner = "sympy-bin-test"
        function_naming = "module-level test_* functions run by SymPy bin/test"
        fixture_style = "plain functions and helpers used in existing SymPy tests"
        assertion_style = "plain assert plus sympy.testing.pytest helpers such as raises"
        notes.append("use sympy.testing.pytest helpers; do not import bare pytest")
    elif has_unittest_class:
        runner = "unittest"
        function_naming = "unittest.TestCase subclass with test_* methods"
        fixture_style = "setUp/tearDown methods when setup is needed"
        assertion_style = "self.assert* assertions"
    elif has_pytest:
        runner = "pytest"
        function_naming = "module-level test_* functions or pytest-style test classes"
        fixture_style = "pytest fixtures/parametrize only when already present"
        assertion_style = "plain assert and pytest.raises"
    else:
        runner = "pytest"
        function_naming = "module-level test_* functions"
        fixture_style = "plain test functions; introduce fixtures only if necessary"
        assertion_style = "plain assert"
        notes.append("no framework signal observed; pytest is the Python fallback")

    if "self.assert" in source:
        assertion_style = "self.assert* assertions"
    if "override_settings" in source:
        notes.append("observed Django override_settings")
    if "parametrize" in lowered:
        notes.append("observed parametrization; keep decorators single-line if used")

    return TestStyleProfile(
        runner=runner,
        language="python",
        file_naming=_file_naming_from_path(existing_test_path or focal_path),
        function_naming=function_naming,
        fixture_style=fixture_style,
        assertion_style=assertion_style,
        decorators=decorators,
        imports=imports,
        notes=notes,
    )


def _infer_js_ts_style(
    *,
    source: str,
    existing_test_path: str,
    language: str,
) -> TestStyleProfile:
    if re.search(r"\bfrom\s+['\"]vitest['\"]|\bimport\s+.*\bvitest\b", source):
        runner = "vitest"
    elif re.search(r"\bfrom\s+['\"]mocha['\"]|\bdescribe\s*\(", source) and "mocha" in source:
        runner = "mocha"
    else:
        runner = "jest" if "jest" in source or "expect(" in source else "js-test"
    return TestStyleProfile(
        runner=runner,
        language=language,
        file_naming=_file_naming_from_path(existing_test_path) or "*.test.ts",
        function_naming="describe/it or test blocks matching existing tests",
        fixture_style="beforeEach/afterEach when existing tests use them",
        assertion_style="expect(...) assertions matching existing tests",
        imports=_extract_plain_import_lines(source),
        notes=["match .test vs .spec naming from existing path"],
    )


def _override_runner_from_config(
    style: TestStyleProfile,
    config_runner: str,
    config_path: str = "",
) -> TestStyleProfile:
    if not config_runner:
        return style
    if config_runner == style.runner:
        if config_runner and "runner determined from repo config" not in style.notes:
            return TestStyleProfile(
                runner=style.runner,
                language=style.language,
                file_naming=style.file_naming,
                function_naming=style.function_naming,
                fixture_style=style.fixture_style,
                assertion_style=style.assertion_style,
                decorators=list(style.decorators),
                imports=list(style.imports),
                notes=[*style.notes, "runner determined from repo config"],
                runner_source="config",
                runner_config_path=config_path,
            )
        return TestStyleProfile(
            runner=style.runner,
            language=style.language,
            file_naming=style.file_naming,
            function_naming=style.function_naming,
            fixture_style=style.fixture_style,
            assertion_style=style.assertion_style,
            decorators=list(style.decorators),
            imports=list(style.imports),
            notes=list(style.notes),
            runner_source="config",
            runner_config_path=config_path,
        )
    notes = [
        *style.notes,
        f"runner overridden by repo config: {style.runner} -> {config_runner}",
    ]
    runner = config_runner
    function_naming = style.function_naming
    fixture_style = style.fixture_style
    assertion_style = style.assertion_style
    if style.language == "python" and runner == "pytest" and style.runner == "unittest":
        notes.append("pytest can collect unittest.TestCase subclasses in this repository")
    if style.language in {"javascript", "typescript"} and runner in {"jest", "vitest", "mocha"}:
        function_naming = "describe/it or test blocks matching existing tests"
        assertion_style = "expect/assertions matching the configured runner"
    return TestStyleProfile(
        runner=runner,
        language=style.language,
        file_naming=style.file_naming,
        function_naming=function_naming,
        fixture_style=fixture_style,
        assertion_style=assertion_style,
        decorators=list(style.decorators),
        imports=list(style.imports),
        notes=notes,
        runner_source="config",
        runner_config_path=config_path,
    )


def _infer_config_declared_runner(
    *,
    repo_root: Path | None,
    language: str,
) -> tuple[str, str]:
    if repo_root is None:
        return "", ""
    root = Path(repo_root)
    if not root.exists():
        return "", ""
    normalized = (language or "").lower()
    if normalized == "python":
        return _infer_python_config_runner(root)
    if normalized in {"javascript", "typescript"}:
        return _infer_js_config_runner(root)
    if normalized == "go" and (root / "go.mod").exists():
        return "go-test", str(root / "go.mod")
    if normalized == "java":
        if (root / "pom.xml").exists():
            return "junit5", str(root / "pom.xml")
        if (root / "build.gradle").exists() or (root / "build.gradle.kts").exists():
            path = root / (
                "build.gradle" if (root / "build.gradle").exists() else "build.gradle.kts"
            )
            return "junit5", str(path)
    return "", ""


def _infer_python_config_runner(root: Path) -> tuple[str, str]:
    # Repository-owned runners are stronger than incidental config files. Large
    # Django/SymPy-style projects often carry pytest.ini for tooling while the
    # authoritative command remains tests/runtests.py or bin/test.
    if (root / "tests" / "runtests.py").exists():
        return "django-runtests", str(root / "tests" / "runtests.py")
    if (root / "manage.py").exists():
        return "django-runtests", str(root / "manage.py")
    if (root / "bin" / "test").exists() and (root / "sympy").exists():
        return "sympy-bin-test", str(root / "bin" / "test")
    pytest_ini = root / "pytest.ini"
    if pytest_ini.exists():
        return "pytest", str(pytest_ini)
    unittest_cfg = root / "unittest.ini"
    if unittest_cfg.exists():
        return "unittest", str(unittest_cfg)
    pyproject = _read_text(root / "pyproject.toml")
    if re.search(r"(?m)^\s*\[tool\.pytest\.ini_options\]\s*$", pyproject):
        return "pytest", str(root / "pyproject.toml")
    if re.search(r"(?m)^\s*\[tool\.pytest\]\s*$", pyproject):
        return "pytest", str(root / "pyproject.toml")
    if re.search(r"(?m)^\s*\[tool\.unittest\]\s*$", pyproject):
        return "unittest", str(root / "pyproject.toml")
    setup_cfg = _read_text(root / "setup.cfg")
    if re.search(r"(?m)^\s*\[(?:tool:pytest|pytest)\]\s*$", setup_cfg):
        return "pytest", str(root / "setup.cfg")
    if re.search(r"(?m)^\s*\[(?:tool:unittest|unittest)\]\s*$", setup_cfg):
        return "unittest", str(root / "setup.cfg")
    tox_ini = _read_text(root / "tox.ini")
    if re.search(r"(?m)^\s*\[pytest\]\s*$", tox_ini):
        return "pytest", str(root / "tox.ini")
    if re.search(r"(?m)^\s*commands\s*=.*pytest\b", tox_ini):
        return "pytest", str(root / "tox.ini")
    if re.search(r"(?m)^\s*commands\s*=.*(?:unittest|django-admin\s+test|runtests\.py)\b", tox_ini):
        return "unittest", str(root / "tox.ini")
    if (root / "manage.py").exists() and not any(
        (root / marker).exists()
        for marker in ("pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml")
    ):
        return "django-runtests", str(root / "manage.py")
    return "", ""


def _infer_js_config_runner(root: Path) -> tuple[str, str]:
    for name, runner in (
        ("vitest.config.ts", "vitest"),
        ("vitest.config.js", "vitest"),
        ("jest.config.ts", "jest"),
        ("jest.config.js", "jest"),
        ("mocha.opts", "mocha"),
    ):
        if (root / name).exists():
            return runner, str(root / name)
    package_json = root / "package.json"
    if not package_json.exists():
        return "", ""
    try:
        payload = json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "", ""
    scripts = payload.get("scripts") if isinstance(payload, dict) else {}
    deps_blob = json.dumps(
        {
            "scripts": scripts if isinstance(scripts, dict) else {},
            "dependencies": payload.get("dependencies", {}),
            "devDependencies": payload.get("devDependencies", {}),
        },
        sort_keys=True,
    ).lower()
    if "vitest" in deps_blob:
        return "vitest", str(package_json)
    if "jest" in deps_blob:
        return "jest", str(package_json)
    if "mocha" in deps_blob:
        return "mocha", str(package_json)
    return "", ""


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _infer_language(path: str) -> str:
    suffix = Path(str(path or "")).suffix.lower()
    if suffix in {".py", ".pyi"}:
        return "python"
    if suffix in {".js", ".jsx", ".mjs", ".cjs"}:
        return "javascript"
    if suffix in {".ts", ".tsx"}:
        return "typescript"
    if suffix == ".go":
        return "go"
    if suffix == ".java":
        return "java"
    return "python"


def _file_naming_from_path(path: str) -> str:
    name = Path(str(path or "")).name
    if not name:
        return "match existing test-file naming"
    if name.startswith("test_"):
        return "test_*.py" if name.endswith(".py") else f"test_*{Path(name).suffix}"
    if name.endswith("_test.py"):
        return "*_test.py"
    if ".test." in name:
        return f"*.test{Path(name).suffix}"
    if ".spec." in name:
        return f"*.spec{Path(name).suffix}"
    if name.endswith("_test.go"):
        return "*_test.go"
    if name.endswith("Test.java"):
        return "*Test.java"
    return name


def _sibling_test_context(
    *,
    existing_test_path: str,
    focal_path: str,
    repo_root: Path | None,
) -> str:
    if repo_root is None:
        return ""
    root = Path(repo_root)
    rel = existing_test_path or focal_path
    parent = root / Path(str(rel or "")).parent
    if not parent.exists() or not parent.is_dir():
        return ""
    chunks: list[str] = []
    for candidate in sorted(parent.iterdir())[:32]:
        if not candidate.is_file() or candidate.stat().st_size > 64_000:
            continue
        name = candidate.name.lower()
        if not (
            name.startswith("test_")
            or name.endswith("_test.py")
            or ".test." in name
            or ".spec." in name
            or name.endswith("test.java")
        ):
            continue
        try:
            chunks.append(candidate.read_text(encoding="utf-8", errors="ignore")[:4000])
        except OSError:
            continue
    return "\n\n".join(chunks)[:16000]


def _extract_python_import_lines(source: str) -> list[str]:
    source_text = str(source or "")
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return _extract_parseable_python_import_lines(source_text)

    imports: list[str] = []
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        segment = (ast.get_source_segment(source_text, node) or "").strip()
        if not segment:
            try:
                segment = ast.unparse(node).strip()
            except Exception:
                segment = ""
        if segment and segment not in imports:
            imports.append(segment)
    return imports


def _extract_parseable_python_import_lines(source: str) -> list[str]:
    imports: list[str] = []
    for line in str(source or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith(("import ", "from ")):
            continue
        try:
            ast.parse(stripped)
        except SyntaxError:
            continue
        if stripped not in imports:
            imports.append(stripped)
    return imports


def _extract_plain_import_lines(source: str) -> list[str]:
    imports: list[str] = []
    for line in (source or "").splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ", "const ", "let ", "var ")) and (
            "import" in stripped or "require(" in stripped
        ):
            if stripped not in imports:
                imports.append(stripped)
    return imports


def _extract_python_decorators(source: str) -> list[str]:
    decorators: list[str] = []
    for line in (source or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("@") and stripped not in decorators:
            decorators.append(stripped)
    return decorators[:24]


def _extract_decorator_lines(source: str) -> list[str]:
    out: list[str] = []
    for line in (source or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("@") and stripped not in out:
            out.append(stripped)
    return out[:24]


def _fence_for_language(language: str) -> str:
    mapping = {
        "python": "```python",
        "javascript": "```javascript",
        "typescript": "```typescript",
        "go": "```go",
        "java": "```java",
    }
    return mapping.get((language or "").lower(), "```")
