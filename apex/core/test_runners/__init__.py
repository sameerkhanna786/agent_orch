"""Language-neutral test-runner abstraction.

Each adapter implements ``TestRunnerAdapter`` so APEX's solver, verifier,
orchestrator, and benchmark layers don't have to know whether the repo
under test uses pytest, jest, vitest, mocha, ``go test``, ``cargo test``,
JUnit, RSpec, or anything else.

The protocol exposes the small set of operations APEX actually needs:

* ``install`` — optional repo-native dependency/bootstrap step.
* ``discover`` — cheap discovery (the runner's ``--collect-only``
  / ``--listTests`` / ``-list`` equivalent) used to spot test-collection
  drops mid-rollout.
* ``run_paths`` — execute a specific set of test paths/IDs and produce
  a structured report.
* ``parse_results`` — read the structured report (JSON / JUnit-XML /
  whatever) into a uniform :class:`RunResult`.
* ``failure_excerpt`` — pull the verbatim "expected vs actual" text for
  one failing test, used in residual-followup prompts.
* ``infrastructure_paths`` — the framework-specific files that should
  not be edited by the agent (test config, fixtures setup, etc.).
* ``stub_patterns`` — regex/substring tokens that flag unimplemented
  function bodies in this language.

Legacy adapters that still implement ``list_tests`` /
``build_run_command`` / ``parse_report`` / ``extract_failure_excerpt`` are
wrapped by :func:`register_adapter` with compatibility methods. New
language runners should implement the install/discover/run_paths/
parse_results/failure_excerpt shape directly.

Concrete adapters live in this package (``pytest_adapter.py``,
``jest_adapter.py``, ...). Selection happens via :func:`detect_adapter`
which auto-picks based on project files (``pyproject.toml`` /
``package.json`` / ``go.mod`` / ``Cargo.toml`` / ``pom.xml`` /
``Gemfile``) or via :func:`get_adapter` given an explicit name.
"""

from __future__ import annotations

import subprocess
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable


@dataclass
class RunResult:
    """Uniform per-run outcome surface."""

    returncode: int
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    collected: int = 0
    outcomes: dict[str, str] = field(default_factory=dict)
    raw_output: str = ""
    report_path: Optional[str] = None
    timed_out: bool = False
    error: str = ""


@dataclass(frozen=True)
class RunnerProfile:
    """Static policy owned by a selected test runner.

    The runner command decides whether generated tests may rely on pytest,
    unittest/Django helpers, repo-native helpers, or no Python framework at all.
    This profile is intentionally small: synthesis/repair paths can consult it
    without importing benchmark adapters.
    """

    runner: str
    language: str
    test_id_format: str = ""
    validation_strategy: str = "local"
    allowed_imports: frozenset[str] = field(default_factory=frozenset)
    forbidden_imports: frozenset[str] = field(default_factory=frozenset)
    assertion_helpers: frozenset[str] = field(default_factory=frozenset)
    runner_source: str = ""
    static_confidence: float = 0.5

    def allows_import(self, name: str) -> bool:
        root = str(name or "").split(".", 1)[0]
        if root in self.forbidden_imports:
            return False
        return not self.allowed_imports or root in self.allowed_imports

    def allows_helper(self, name: str) -> bool:
        return str(name or "") in self.assertion_helpers

    def to_dict(self) -> dict[str, object]:
        return {
            "runner": self.runner,
            "language": self.language,
            "test_id_format": self.test_id_format,
            "validation_strategy": self.validation_strategy,
            "allowed_imports": sorted(self.allowed_imports),
            "forbidden_imports": sorted(self.forbidden_imports),
            "assertion_helpers": sorted(self.assertion_helpers),
            "runner_source": self.runner_source,
            "static_confidence": self.static_confidence,
        }


def profile_for_runner(
    runner: str,
    *,
    language: str = "python",
    runner_source: str = "",
) -> RunnerProfile:
    normalized = (runner or "unknown").lower()
    lang = (language or "python").lower()
    if lang in {"python", "py", "python3"}:
        if normalized == "pytest":
            return RunnerProfile(
                runner=normalized,
                language="python",
                test_id_format="path::test_name",
                validation_strategy="local",
                allowed_imports=frozenset({"pytest", "numpy", "math", "unittest"}),
                assertion_helpers=frozenset(
                    {
                        "plain_assert",
                        "pytest.raises",
                        "pytest.approx",
                        "numpy.testing.assert_allclose",
                    }
                ),
                runner_source=runner_source,
                static_confidence=0.85,
            )
        if normalized == "sympy-bin-test":
            return RunnerProfile(
                runner=normalized,
                language="python",
                test_id_format="path::test_name",
                validation_strategy="project_env",
                allowed_imports=frozenset({"sympy", "math", "unittest"}),
                forbidden_imports=frozenset({"pytest", "pytest_django", "numpy"}),
                assertion_helpers=frozenset({"plain_assert", "sympy.raises", "math.isclose"}),
                runner_source=runner_source,
                static_confidence=0.9,
            )
        if normalized in {"unittest", "django", "django-runtests"}:
            return RunnerProfile(
                runner=normalized,
                language="python",
                test_id_format="module.Class.test_method",
                validation_strategy="project_env" if "django" in normalized else "local",
                allowed_imports=frozenset({"unittest", "math"}),
                forbidden_imports=frozenset({"pytest", "pytest_django", "numpy"}),
                assertion_helpers=frozenset(
                    {"self.assert*", "self.assertRaises", "self.assertAlmostEqual"}
                ),
                runner_source=runner_source,
                static_confidence=0.85,
            )
        return RunnerProfile(
            runner=normalized,
            language="python",
            test_id_format="path::test_name",
            validation_strategy="local",
            forbidden_imports=frozenset(),
            assertion_helpers=frozenset({"plain_assert"}),
            runner_source=runner_source,
            static_confidence=0.4,
        )
    if lang in {"javascript", "typescript"}:
        return RunnerProfile(
            runner=normalized,
            language=lang,
            test_id_format="file::suite > test",
            validation_strategy="local",
            assertion_helpers=frozenset({"expect"}),
            runner_source=runner_source,
            static_confidence=0.75,
        )
    return RunnerProfile(
        runner=normalized,
        language=lang,
        test_id_format="runner-native",
        validation_strategy="unsupported" if normalized == "unknown" else "local",
        assertion_helpers=frozenset(),
        runner_source=runner_source,
        static_confidence=0.4,
    )


@runtime_checkable
class TestRunnerAdapter(Protocol):
    """Minimal interface APEX needs from any test runner."""

    name: str
    language: str

    def install(
        self,
        workspace: Path,
        env: dict[str, str],
        *,
        timeout_seconds: float,
    ) -> tuple[Optional[Path], str]:
        """Install/bootstrap repo-local test dependencies when supported."""

    def discover(self, workspace: Path, env: dict[str, str]) -> set[str]:
        """Return the set of nodeids/paths the runner can currently discover."""

    def run_paths(
        self,
        workspace: Path,
        test_paths: list[str],
        report_path: Path,
        *,
        timeout_seconds: float,
        env: Optional[dict[str, str]] = None,
        executable: Optional[str] = None,
    ) -> RunResult:
        """Execute ``test_paths`` and return normalized run results."""

    def parse_results(self, report_path: Path) -> RunResult:
        """Read the report file produced by ``run_paths``."""

    def failure_excerpt(self, test_id: str, report_path: Path) -> str:
        """Verbatim "expected vs actual" text for one failing test."""

    def list_tests(self, workspace: Path, env: dict[str, str]) -> set[str]:
        """Legacy alias for ``discover``."""

    def build_run_command(
        self,
        workspace: Path,
        test_ids: list[str],
        report_path: Path,
        *,
        executable: Optional[str] = None,
    ) -> str:
        """Shell command that runs ``test_ids`` and writes a parseable report."""

    def parse_report(self, report_path: Path) -> RunResult:
        """Read the report file produced by ``build_run_command``."""

    def extract_failure_excerpt(self, test_id: str, report_path: Path) -> str:
        """Legacy alias for ``failure_excerpt``."""

    def infrastructure_paths(self, workspace: Path) -> set[str]:
        """Paths the agent must NOT edit (test config, setup files, etc.)."""

    def stub_patterns(self) -> list[str]:
        """Substrings/regexes flagging unimplemented bodies in this language."""

    def runner_profile(self) -> RunnerProfile:
        """Return the runner-owned assertion/import/static-validation policy."""


_REGISTRY: dict[str, "TestRunnerAdapter"] = {}


def _compat_install(
    self: "TestRunnerAdapter",
    workspace: Path,
    env: dict[str, str],
    *,
    timeout_seconds: float,
) -> tuple[Optional[Path], str]:
    return None, "no_install_plan"


def _compat_discover(
    self: "TestRunnerAdapter",
    workspace: Path,
    env: dict[str, str],
) -> set[str]:
    return self.list_tests(workspace, env)


def _compat_parse_results(
    self: "TestRunnerAdapter",
    report_path: Path,
) -> RunResult:
    return self.parse_report(report_path)


def _compat_failure_excerpt(
    self: "TestRunnerAdapter",
    test_id: str,
    report_path: Path,
) -> str:
    return self.extract_failure_excerpt(test_id, report_path)


def _compat_run_paths(
    self: "TestRunnerAdapter",
    workspace: Path,
    test_paths: list[str],
    report_path: Path,
    *,
    timeout_seconds: float,
    env: Optional[dict[str, str]] = None,
    executable: Optional[str] = None,
) -> RunResult:
    try:
        command = self.build_run_command(
            workspace,
            list(test_paths),
            report_path,
            executable=executable,
        )
    except TypeError:
        command = self.build_run_command(workspace, list(test_paths), report_path)
    from apex.core.subprocess_utils import run_shell_command

    try:
        completed = run_shell_command(
            command,
            cwd=str(workspace),
            timeout=timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        partial_stdout = (
            exc.stdout.decode("utf-8", errors="ignore")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        partial_stderr = (
            exc.stderr.decode("utf-8", errors="ignore")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        return RunResult(
            returncode=124,
            errors=1,
            raw_output="\n".join(part for part in (partial_stdout, partial_stderr) if part),
            report_path=str(report_path),
            timed_out=True,
            error=f"adapter '{getattr(self, 'name', '?')}' exceeded {int(timeout_seconds)}s",
        )
    parsed = self.parse_results(report_path)
    raw_output = "\n".join(
        part
        for part in (
            parsed.raw_output,
            completed.stdout or "",
            completed.stderr or "",
        )
        if part
    )
    parsed.raw_output = raw_output
    parsed.returncode = completed.returncode if parsed.returncode is None else parsed.returncode
    return parsed


def _compat_runner_profile(self: "TestRunnerAdapter") -> RunnerProfile:
    return profile_for_runner(
        getattr(self, "name", "unknown"),
        language=getattr(self, "language", "unknown"),
        runner_source="adapter",
    )


def _ensure_runner_facade(adapter: "TestRunnerAdapter") -> "TestRunnerAdapter":
    """Attach the benchmark-agnostic runner facade to legacy adapters."""
    if not hasattr(adapter, "install"):
        setattr(adapter, "install", types.MethodType(_compat_install, adapter))
    if not hasattr(adapter, "discover"):
        setattr(adapter, "discover", types.MethodType(_compat_discover, adapter))
    if not hasattr(adapter, "parse_results"):
        setattr(adapter, "parse_results", types.MethodType(_compat_parse_results, adapter))
    if not hasattr(adapter, "failure_excerpt"):
        setattr(adapter, "failure_excerpt", types.MethodType(_compat_failure_excerpt, adapter))
    if not hasattr(adapter, "run_paths"):
        setattr(adapter, "run_paths", types.MethodType(_compat_run_paths, adapter))
    if not hasattr(adapter, "runner_profile"):
        setattr(adapter, "runner_profile", types.MethodType(_compat_runner_profile, adapter))
    return adapter


def register_adapter(adapter: "TestRunnerAdapter") -> None:
    adapter = _ensure_runner_facade(adapter)
    _REGISTRY[adapter.name] = adapter


def get_adapter(name: str) -> Optional["TestRunnerAdapter"]:
    if not name:
        return None
    return _REGISTRY.get(name.lower())


def list_adapters() -> list[str]:
    return sorted(_REGISTRY.keys())


def detect_adapter(workspace: Path) -> Optional["TestRunnerAdapter"]:
    """Auto-pick an adapter from the project's manifest files.

    The order matters: a polyglot repo (e.g. a Python package with a JS
    sub-app) should still get pytest as the primary if it exists. The
    first match wins; callers can override by passing an explicit name
    to :func:`get_adapter`.
    """
    workspace = Path(workspace)
    if (
        (workspace / "tests" / "runtests.py").exists()
        and (workspace / "django").is_dir()
        and "django" in _REGISTRY
    ):
        return _REGISTRY["django"]
    # Python: pytest if pyproject.toml or pytest.ini or setup.py exists
    if any(
        (workspace / candidate).exists()
        for candidate in ("pyproject.toml", "pytest.ini", "setup.py", "setup.cfg")
    ):
        adapter = _REGISTRY.get("pytest")
        if adapter is not None:
            return adapter
    # JavaScript / TypeScript
    package_json = workspace / "package.json"
    if package_json.exists():
        try:
            import json

            data = json.loads(package_json.read_text())
        except (OSError, ValueError):
            data = {}
        deps = {}
        for key in ("dependencies", "devDependencies"):
            section = data.get(key) or {}
            if isinstance(section, dict):
                deps.update(section)
        # Order: explicit modern runners first, then OSpec/Mocha for
        # repository-native describe/it/o-style suites.
        for runner in ("vitest", "ospec", "jest", "mocha"):
            if runner in deps and runner in _REGISTRY:
                return _REGISTRY[runner]
        # Fall back to Mocha rather than Jest when no manifest runner is
        # explicit: Mocha can execute plain CommonJS describe/it files without
        # a config file, while current Jest refuses to run config-less repos.
        if "mocha" in _REGISTRY:
            return _REGISTRY["mocha"]
    # Go
    if (workspace / "go.mod").exists() and "go-test" in _REGISTRY:
        return _REGISTRY["go-test"]
    # Rust
    if (workspace / "Cargo.toml").exists() and "cargo-test" in _REGISTRY:
        return _REGISTRY["cargo-test"]
    # Java/Kotlin (JVM build tools)
    if (
        any((workspace / m).exists() for m in ("pom.xml", "build.gradle", "build.gradle.kts"))
        and "junit" in _REGISTRY
    ):
        return _REGISTRY["junit"]
    # C / C++ (CMake / Make / Meson / autotools)
    if (
        any(
            (workspace / m).exists()
            for m in ("CMakeLists.txt", "Makefile", "meson.build", "configure.ac")
        )
        and "ctest" in _REGISTRY
    ):
        return _REGISTRY["ctest"]
    # C# / F# / VB (.NET)
    if "dotnet-test" in _REGISTRY:
        for path in workspace.iterdir():
            if path.is_file() and path.suffix in (".csproj", ".fsproj", ".vbproj", ".sln"):
                return _REGISTRY["dotnet-test"]
    # PHP (Composer)
    if (workspace / "composer.json").exists() and "phpunit" in _REGISTRY:
        return _REGISTRY["phpunit"]
    # Swift (Package.swift / Xcode)
    if "swift-xctest" in _REGISTRY:
        if (workspace / "Package.swift").exists():
            return _REGISTRY["swift-xctest"]
        for path in workspace.iterdir():
            if path.suffix in (".xcodeproj", ".xcworkspace"):
                return _REGISTRY["swift-xctest"]
    # Bats (shell-script tests)
    if "bats" in _REGISTRY:
        bats_files = list(workspace.rglob("*.bats"))
        if bats_files:
            return _REGISTRY["bats"]
    # Ruby
    if (workspace / "Gemfile").exists() and "rspec" in _REGISTRY:
        return _REGISTRY["rspec"]
    return None


def adapter_for_task(
    framework_hint: Optional[str],
    workspace: Path,
) -> Optional["TestRunnerAdapter"]:
    """Resolve the adapter for a benchmark task.

    Prefers an explicit ``framework`` field from the dataset; falls back
    to detection from ``workspace`` files. Returns ``None`` if nothing
    matches — caller decides whether to error or fall through.
    """
    if framework_hint:
        explicit = get_adapter(framework_hint)
        if explicit is not None:
            return explicit
    return detect_adapter(workspace)


# Eager imports so `register_adapter` calls happen on module load.
from . import (  # noqa: E402
    bats_adapter,  # noqa: E402,F401
    cargo_test_adapter,  # noqa: E402,F401
    ctest_adapter,  # noqa: E402,F401
    django_adapter,  # noqa: E402,F401
    dotnet_adapter,  # noqa: E402,F401
    go_test_adapter,  # noqa: E402,F401
    jest_adapter,  # noqa: E402,F401
    junit_adapter,  # noqa: E402,F401
    phpunit_adapter,  # noqa: E402,F401
    pytest_adapter,  # noqa: E402,F401
    swebench_adapter,  # noqa: E402,F401
    swebench_pro_adapter,  # noqa: E402,F401
    swift_adapter,  # noqa: E402,F401
)

__all__ = [
    "RunResult",
    "RunnerProfile",
    "TestRunnerAdapter",
    "profile_for_runner",
    "register_adapter",
    "get_adapter",
    "list_adapters",
    "detect_adapter",
    "adapter_for_task",
]
