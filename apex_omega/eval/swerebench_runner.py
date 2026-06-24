"""Self-contained LOCAL runner for the SWE-rebench Mode-C benchmark.

This is NOT a ``Commit0BenchmarkRunner`` subclass.  It mirrors the *surface* that
``commit0_autogen.run_autogen_cell`` (and its swerebench sibling) needs —
``discover_tasks`` / ``_prepare_repo`` / ``evaluate_repo`` — but is fully
self-contained, reads the pinned ``configs/swerebench_slice.json`` registry, and
runs REAL ``pytest-json-report`` over the exact pinned gold node-ids.  No Docker,
no apt; install is uv-only (``uv venv`` + ``uv pip install`` from the repo's own
``install`` recipe + in-repo reqs).

EXECUTION-GROUNDED invariant (the Cardinal Contract):
  * NO path sets ``scoring_source='commit0_test_ids'`` without a REAL
    pytest-json-report run with matched per-test node-ids.
  * Acceptance requires every FAIL_TO_PASS to flip AND every PASS_TO_PASS to be
    preserved: ``failed == errors == missing == 0`` over the gold union.
  * Harness/parser/native crashes populate ``diagnostics`` + ``returncode`` so
    ``scoring.verification_from_commit0_evaluation`` maps them to INDETERMINATE
    (re-run on resume), never a false-zero and never a false-accept.

The ``SweRebenchTask`` is shape-compatible with v1's ``Commit0Task``
(``build_issue_description`` / ``repo_name`` / ``test_cmd`` / ``src_dir`` /
``python_version``) and the evaluation object is a real ``Commit0Evaluation``
imported from v1 (so ``scoring.py`` consumes it UNCHANGED, with
``contract_success()`` callable + ``scored_success`` honest).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
SLICE_PATH = _REPO_ROOT / "configs" / "swerebench_slice.json"

# v1 framing block (single source of truth) — reused so the SWE-rebench prompt
# carries the same binding task-framing as the commit0 path.
try:
    from apex.evaluation.commit0_benchmark import TASK_FRAMING_BLOCK  # noqa: F401
except Exception:  # pragma: no cover - keep the runner importable without apex venv
    TASK_FRAMING_BLOCK = (
        "You are completing missing library functionality so the provided test "
        "suite passes. Treat the visible tests as the specification. Do not edit "
        "tests to make failures disappear."
    )


# ---------------------------------------------------------------------------
# slice registry I/O
# ---------------------------------------------------------------------------
def load_slice(path: str | Path | None = None) -> dict:
    """Load the pinned slice artifact (instance_id -> record). Never re-fetches HF."""
    p = Path(path or SLICE_PATH)
    if not p.exists():
        return {"instances": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def slice_instances(path: str | Path | None = None) -> dict[str, dict]:
    return dict((load_slice(path).get("instances") or {}))


def gold_ids_for(instance_id: str, *, path: str | Path | None = None) -> list[str]:
    """The pinned gold node-id universe (sorted set FAIL_TO_PASS | PASS_TO_PASS).

    This is the gold PROVIDER — it is the SWE-rebench analogue of
    ``_load_expected_test_ids`` and NEVER calls ``commit0.harness.get_pytest_ids``.
    """
    rec = slice_instances(path).get(instance_id) or {}
    ids = rec.get("gold_ids")
    if ids:
        return list(ids)
    # Derive from the parsed fail/pass lists if gold_ids was absent.
    return sorted(set(rec.get("fail_to_pass") or []) | set(rec.get("pass_to_pass") or []))


# ---------------------------------------------------------------------------
# task object (Commit0Task-shape-compatible)
# ---------------------------------------------------------------------------
@dataclass
class SweRebenchTask:
    instance_id: str
    repo: str
    base_commit: str
    python_version: str
    install_command: str
    test_cmd: str
    specification: str = ""
    test_patch: str = ""
    reqs_path: list[str] = field(default_factory=list)
    packages: list[str] = field(default_factory=list)
    pip_packages: list[str] = field(default_factory=list)
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    gold_ids: list[str] = field(default_factory=list)
    created_at: str = ""
    stratum: str = ""
    src_dir: str = ""
    test_dir: str = "tests/"

    @property
    def repo_name(self) -> str:
        return self.repo.split("/")[-1]

    @property
    def src_root(self) -> str:
        # SWE-rebench instances we curate are flat- or src-layout libs; mirror
        # the commit0 convention so the eval PYTHONPATH points at the candidate.
        sd = (self.src_dir or "").strip().rstrip("/")
        if sd and Path(sd).name == "src":
            return sd
        return ""

    @classmethod
    def from_record(cls, rec: dict) -> "SweRebenchTask":
        return cls(
            instance_id=str(rec["instance_id"]),
            repo=str(rec["repo"]),
            base_commit=str(rec["base_commit"]),
            python_version=str(rec.get("python") or "3.11"),
            install_command=str(rec.get("install") or "pip install -e ."),
            test_cmd=str(rec.get("test_cmd") or "pytest"),
            specification=str(rec.get("problem_statement") or rec.get("specification") or ""),
            test_patch=str(rec.get("test_patch") or ""),
            reqs_path=list(rec.get("reqs_path") or []),
            packages=list(rec.get("packages") or []),
            pip_packages=list(rec.get("pip_packages") or []),
            fail_to_pass=list(rec.get("fail_to_pass") or []),
            pass_to_pass=list(rec.get("pass_to_pass") or []),
            gold_ids=list(rec.get("gold_ids")
                          or sorted(set(rec.get("fail_to_pass") or [])
                                    | set(rec.get("pass_to_pass") or []))),
            created_at=str(rec.get("created_at") or ""),
            stratum=str(rec.get("stratum") or ""),
        )

    def build_issue_description(
        self,
        test_command: str,
        *,
        expected_test_count: Optional[int] = None,
        expected_test_ids: Optional[list[str]] = None,
        include_upstream_reference: bool = False,
    ) -> str:
        lines = [
            "Implement the change required by this repository issue so the provided "
            "test suite passes.",
            "Treat this as a repository-completion / bug-fix task grounded in the tests.",
            "",
            TASK_FRAMING_BLOCK,
        ]
        spec = re.sub(r"https?://\S+", "", str(self.specification or "")).strip()
        if spec:
            lines.append(f"Issue / task objective:\n{spec[:8000]}")
        lines.extend([
            "",
            f"Target Python version: {self.python_version}",
            f"Install command already applied by the harness: {self.install_command}",
            f"Repository test command: {test_command}",
            "",
            "Read the existing tests and source to infer the intended behavior.",
            "Treat tests as read-only specification; do not modify tests to make "
            "failures disappear or reduce coverage.",
        ])
        if isinstance(expected_test_count, int) and expected_test_count > 0:
            lines.append(f"Expected gold test count: {expected_test_count}")
            if expected_test_ids:
                lines.append("Gold node-ids (must all pass):")
                lines.extend(f"  {nid}" for nid in expected_test_ids[:60])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# evaluation object (a real v1 Commit0Evaluation, so scoring.py is UNCHANGED)
# ---------------------------------------------------------------------------
def _make_evaluation(**kwargs) -> Any:
    """Build a v1 ``Commit0Evaluation`` (shape scoring.py expects). Falls back to a
    minimal shim only if the apex venv is unavailable (keeps the module importable
    in tooling contexts; the real eval path always has apex)."""
    try:
        from apex.evaluation.commit0_benchmark import Commit0Evaluation
        return Commit0Evaluation(**kwargs)
    except Exception:  # pragma: no cover
        return _EvalShim(**kwargs)


@dataclass
class _EvalShim:  # pragma: no cover - only used without the apex venv
    returncode: int = 1
    output: str = ""
    raw_returncode: Optional[int] = None
    report_path: Optional[str] = None
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    total_tests: int = 0
    scoring_source: str = "pytest_summary"
    evaluation_backend: str = "unknown"
    expected_test_coverage: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        runnable = self.passed + self.failed + self.errors + self.skipped
        return (self.passed / runnable) if runnable else 0.0

    @property
    def scored_success(self) -> bool:
        return self.contract_success()

    def contract_success(self) -> bool:
        return (
            self.scoring_source == "commit0_test_ids"
            and self.total_tests > 0
            and self.failed == 0
            and self.errors == 0
            and int(self.expected_test_coverage.get("missing_expected_test_count", 0)) == 0
            and self.passed >= self.total_tests
        )


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------
_LOCAL_BACKEND = "swerebench_local_pytest_json"


def _resolve_uv() -> list[str]:
    uv = shutil.which("uv")
    if uv:
        return [uv]
    return ["uv"]


class SweRebenchRunner:
    """Self-contained local SWE-rebench runner (no Docker, uv-only)."""

    def __init__(self, *, slice_path: str | Path | None = None,
                 setup_timeout: int = 1800, dep_timeout: int = 1800):
        self._slice_path = Path(slice_path or SLICE_PATH)
        self._instances = slice_instances(self._slice_path)
        self.setup_timeout = setup_timeout
        self.dep_timeout = dep_timeout

    # -- discovery ------------------------------------------------------------
    def discover_tasks(self, *, instance_ids: Optional[list[str]] = None,
                       repos: Optional[list[str]] = None,
                       limit: Optional[int] = None) -> list[SweRebenchTask]:
        """Discover tasks by instance-id (preferred), falling back to ``repos``
        being treated as instance-ids (the ladder passes ``--repos <instance_id>``)."""
        ids = list(instance_ids or [])
        if not ids and repos:
            ids = list(repos)  # the ladder/cli use --repos to carry instance-ids
        if not ids:
            ids = sorted(self._instances.keys())
        tasks: list[SweRebenchTask] = []
        for iid in ids:
            rec = self._instances.get(iid)
            if rec is None:
                continue
            tasks.append(SweRebenchTask.from_record(rec))
            if limit is not None and len(tasks) >= limit:
                break
        return tasks

    # -- prep -----------------------------------------------------------------
    def _run(self, cmd: list[str] | str, *, cwd: Path, env: dict, timeout: int,
             check: bool = True) -> subprocess.CompletedProcess:
        shell = isinstance(cmd, str)
        proc = subprocess.run(
            cmd, cwd=str(cwd), env=env, shell=shell, text=True,
            capture_output=True, timeout=timeout,
        )
        if check and proc.returncode != 0:
            tail = ((proc.stdout or "") + (proc.stderr or ""))[-2000:]
            raise RuntimeError(
                f"command failed (rc={proc.returncode}): "
                f"{cmd if shell else ' '.join(cmd)}\n{tail}")
        return proc

    def _build_runtime_env(self, task: SweRebenchTask, repo_dir: Path,
                           runtime_dir: Path) -> dict[str, str]:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        venv_dir = runtime_dir / ".venv"
        uv = _resolve_uv()
        self._run([*uv, "venv", "--python", task.python_version, str(venv_dir)],
                  cwd=repo_dir, env=dict(os.environ), timeout=self.setup_timeout)
        env = dict(os.environ)
        env["VIRTUAL_ENV"] = str(venv_dir)
        env["PATH"] = f"{venv_dir / 'bin'}{os.pathsep}{env.get('PATH', '')}"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        # Per-task sandbox HOME/TMP so concurrent cells don't race on /tmp or ~/.cache.
        sandbox_home = (runtime_dir / "home").resolve()
        sandbox_tmp = (runtime_dir / "tmp").resolve()
        for path in (sandbox_home, sandbox_tmp, sandbox_home / ".cache",
                     sandbox_home / ".config", sandbox_home / ".local" / "share",
                     sandbox_home / ".local" / "state"):
            path.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(sandbox_home)
        env["TMPDIR"] = str(sandbox_tmp)
        env["TEMP"] = str(sandbox_tmp)
        env["TMP"] = str(sandbox_tmp)
        env["XDG_CACHE_HOME"] = str(sandbox_home / ".cache")
        env["XDG_CONFIG_HOME"] = str(sandbox_home / ".config")
        env["XDG_DATA_HOME"] = str(sandbox_home / ".local" / "share")
        env["XDG_STATE_HOME"] = str(sandbox_home / ".local" / "state")
        # Keep HF offline (no dataset fetch during prep).
        env.setdefault("HF_DATASETS_OFFLINE", "1")
        env.setdefault("HF_HUB_OFFLINE", "1")
        return env

    @staticmethod
    def _clone_url(repo: str) -> str:
        return f"https://github.com/{repo}.git"

    @staticmethod
    def _apply_patch_text(repo_dir: Path, patch_text: str, *, name: str) -> tuple[bool, str]:
        """Apply a unified-diff patch to ``repo_dir`` (git apply, with fallbacks).
        Returns (ok, detail). Used for the gold TEST patch during prep."""
        if not str(patch_text or "").strip():
            return True, "empty patch (noop)"
        pf = repo_dir / f".{name}.patch"
        pf.write_text(patch_text, encoding="utf-8")
        last = ""
        for args in (
            ["git", "apply", "--whitespace=nowarn", str(pf)],
            ["git", "apply", "-3", "--whitespace=nowarn", str(pf)],
            ["patch", "-p1", "-i", str(pf)],
        ):
            proc = subprocess.run(args, cwd=str(repo_dir), text=True, capture_output=True)
            if proc.returncode == 0:
                try:
                    pf.unlink()
                except OSError:
                    pass
                return True, " ".join(args)
            last = ((proc.stdout or "") + (proc.stderr or ""))[-1200:]
        return False, last

    def _prepare_repo(self, task: SweRebenchTask, repo_dir: Path,
                      runtime_dir: Path) -> dict[str, str]:
        """Clone repo@base_commit, checkout -B apex-base, uv venv at python, install
        (uv pip install -e . + in-repo reqs + pip_packages, NO apt), reset+clean,
        return env with VIRTUAL_ENV. Mirrors v1 _prepare_repo (sans Docker/scrub)."""
        repo_dir = Path(repo_dir)
        runtime_dir = Path(runtime_dir)
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        if not (repo_dir / ".git").exists():
            self._run(["git", "clone", "--no-single-branch", self._clone_url(task.repo),
                       str(repo_dir)], cwd=repo_dir.parent, env=dict(os.environ),
                      timeout=self.setup_timeout)
        # Pin to the exact base commit, on a fresh apex-base branch.
        self._run(["git", "fetch", "--depth", "1", "origin", task.base_commit],
                  cwd=repo_dir, env=dict(os.environ), timeout=self.setup_timeout, check=False)
        self._run(["git", "checkout", "-B", "apex-base", task.base_commit],
                  cwd=repo_dir, env=dict(os.environ), timeout=300)
        self._run(["git", "config", "user.email", "apex@example.com"],
                  cwd=repo_dir, env=dict(os.environ), timeout=60)
        self._run(["git", "config", "user.name", "APEX"],
                  cwd=repo_dir, env=dict(os.environ), timeout=60)

        # SWE-bench semantics: the gold TEST state = base + test_patch. The
        # FAIL_TO_PASS tests are DEFINED by test_patch, so apply it now and COMMIT
        # it onto apex-base — this is the read-only spec (the gold tests), NOT the
        # gold code solution. Every forked candidate worktree (forked from
        # apex-base) then inherits the gold tests, exactly like commit0's visible
        # tests. The agent's job is to make these tests pass by editing source.
        if str(task.test_patch or "").strip():
            ok, detail = self._apply_patch_text(repo_dir, task.test_patch, name="gold_test")
            if not ok:
                raise RuntimeError(
                    f"failed to apply gold test_patch for {task.instance_id}: {detail}")
            self._run(["git", "add", "-A"], cwd=repo_dir, env=dict(os.environ),
                      timeout=120, check=False)
            self._run(["git", "commit", "-q", "-m", "apex: gold test_patch (spec tests)",
                       "--allow-empty"], cwd=repo_dir, env=dict(os.environ), timeout=120,
                      check=False)
            # Re-point apex-base at the test-patched commit so worktrees inherit it.
            self._run(["git", "branch", "-f", "apex-base", "HEAD"], cwd=repo_dir,
                      env=dict(os.environ), timeout=60, check=False)

        env = self._build_runtime_env(task, repo_dir, runtime_dir)
        uv_pip = " ".join(_resolve_uv()) + " pip"

        # In-repo requirements files (apt-free). Conda file:// pins from the dataset's
        # `requirements` column are deliberately NOT installed.
        for reqs in dict.fromkeys([*task.packages, *task.reqs_path]):
            reqs = str(reqs).strip()
            if not reqs or (repo_dir / reqs).exists() is False:
                continue
            self._run(f"{uv_pip} install -r {shlex.quote(reqs)}", cwd=repo_dir,
                      env=env, timeout=self.dep_timeout, check=False)
        for pkg in task.pip_packages:
            pkg = str(pkg).strip()
            if pkg:
                self._run(f"{uv_pip} install {shlex.quote(pkg)}", cwd=repo_dir,
                          env=env, timeout=self.dep_timeout, check=False)
        # Test harness deps (always present so collection + json-report work).
        self._run(f"{uv_pip} install pytest pytest-json-report 'setuptools<82'",
                  cwd=repo_dir, env=env, timeout=self.dep_timeout, check=False)
        # The editable project install (the dataset's own recipe; pip -> uv pip).
        install_cmd = self._rewrite_install(task.install_command, uv_pip)
        self._run(install_cmd, cwd=repo_dir, env=env, timeout=self.dep_timeout, check=True)

        # Clean tree (mirror v1) so the worktree the orchestrator forks is pristine.
        self._run(["git", "reset", "--hard", "HEAD"], cwd=repo_dir,
                  env=dict(os.environ), timeout=120, check=False)
        self._run(["git", "clean", "-fdx", "-e", str(runtime_dir.name)],
                  cwd=repo_dir, env=dict(os.environ), timeout=120, check=False)
        return env

    @staticmethod
    def _rewrite_install(install_command: str, uv_pip: str) -> str:
        """Rewrite a ``pip install ...`` recipe to use ``uv pip`` (apt-free)."""
        cmd = str(install_command or "").strip()
        if not cmd:
            return f"{uv_pip} install -e ."
        # Replace a leading 'pip install' / 'python -m pip install' with uv pip.
        cmd = re.sub(r"^\s*python\s+-m\s+pip\s+install", f"{uv_pip} install", cmd)
        cmd = re.sub(r"^\s*pip3?\s+install", f"{uv_pip} install", cmd)
        return cmd

    # -- evaluation (EXECUTION-GROUNDED) -------------------------------------
    def evaluate_repo(self, task: SweRebenchTask, repo_dir: Path, *,
                      artifacts_dir: Path, label: str,
                      python_executable: Optional[str] = None,
                      env: Optional[dict[str, str]] = None,
                      expected_test_ids: Optional[list[str]] = None,
                      timeout_seconds: Optional[int] = None,
                      use_expected_test_scoring: bool = True) -> Any:
        """Run REAL pytest-json-report over the gold node-ids in ``repo_dir`` and
        return a Commit0Evaluation-compatible object.

        Acceptance (contract_success) requires: a real json-report run produced
        per-test outcomes for EVERY gold id, every FAIL_TO_PASS flips to pass,
        every PASS_TO_PASS stays pass, and ``failed == errors == missing == 0``.
        Only then is ``scoring_source`` set to ``commit0_test_ids``.

        Any harness/parser/crash failure returns an evaluation with diagnostics so
        scoring.py maps it to INDETERMINATE (never a false-zero or false-accept).
        """
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        gold = list(expected_test_ids or task.gold_ids or [])
        gold_set = sorted(set(gold))
        f2p = set(task.fail_to_pass or [])
        p2p = set(task.pass_to_pass or [])
        py = python_executable or "python"
        run_env = dict(env or os.environ)
        report_file = artifacts_dir / "report.json"

        if not gold_set:
            return _make_evaluation(
                returncode=1, output="no gold ids", total_tests=0,
                evaluation_backend=_LOCAL_BACKEND,
                diagnostics={"harness_failure": True, "reason": "empty gold id set"})

        # Run pytest over EXACTLY the gold node-ids, with json-report. We pass the
        # ids on the command line so collection is scoped to the gold universe;
        # --continue-on-collection-errors keeps one bad id from masking the rest.
        # NOTE: pytest-json-report auto-loads via its setuptools "pytest11" entry
        # point, so we must NOT also pass ``-p pytest_jsonreport.plugin`` (that
        # double-registers the plugin -> pytest aborts rc=4 before producing any
        # report -> a spurious parser_error). ``--json-report`` alone enables it.
        cmd = [
            py, "-m", "pytest",
            "-p", "no:cacheprovider",
            "--json-report", f"--json-report-file={report_file}",
            "--continue-on-collection-errors",
            "-o", "addopts=",  # neutralize repo addopts (e.g. --cov / --memray plugins)
            "-rA", "--tb=line", "--color=no",
            *gold_set,
        ]
        try:
            proc = subprocess.run(
                cmd, cwd=str(repo_dir), env=run_env, text=True,
                capture_output=True, timeout=timeout_seconds or 1800)
        except subprocess.TimeoutExpired as exc:
            return _make_evaluation(
                returncode=124, output=f"pytest timeout: {exc}", total_tests=len(gold_set),
                evaluation_backend=_LOCAL_BACKEND,
                diagnostics={"harness_failure": True, "timeout": True})
        except OSError as exc:
            return _make_evaluation(
                returncode=1, output=f"pytest spawn failed: {exc}", total_tests=len(gold_set),
                evaluation_backend=_LOCAL_BACKEND,
                diagnostics={"harness_failure": True, "reason": str(exc)})

        rc = int(proc.returncode)
        output_tail = ((proc.stdout or "") + (proc.stderr or ""))[-6000:]

        # Native crash (segfault/abort/signal) -> indeterminate via scoring.py.
        if rc < 0 or rc in (134, 137, 138, 139):
            return _make_evaluation(
                returncode=rc, output=output_tail, total_tests=len(gold_set),
                evaluation_backend=_LOCAL_BACKEND,
                diagnostics={"native_crash_returncode": rc})

        # Parse the json-report for PER-TEST outcomes (no aggregate-only acceptance).
        if not report_file.exists():
            return _make_evaluation(
                returncode=rc or 1, output=output_tail, total_tests=len(gold_set),
                evaluation_backend=_LOCAL_BACKEND,
                diagnostics={"parser_error": True, "reason": "no json report produced"})
        try:
            report = json.loads(report_file.read_text(encoding="utf-8"))
        except Exception as exc:
            return _make_evaluation(
                returncode=rc or 1, output=output_tail, total_tests=len(gold_set),
                evaluation_backend=_LOCAL_BACKEND,
                diagnostics={"parser_error": True, "reason": f"json decode: {exc}"})

        outcomes: dict[str, str] = {}
        for entry in (report.get("tests") or []):
            nid = str(entry.get("nodeid") or "")
            if nid:
                outcomes[nid] = str(entry.get("outcome") or "")
        # Collection errors surface as collectors with outcome=='failed' (no test
        # node), so an uncollected gold id is MISSING, scored against the gold total.
        collection_failed = any(
            isinstance(c, dict) and str(c.get("outcome")) == "failed"
            for c in (report.get("collectors") or []))

        passed = failed = errors = skipped = missing = 0
        f2p_flipped = 0
        p2p_preserved = 0
        per_id: dict[str, str] = {}
        for nid in gold_set:
            outcome = outcomes.get(nid)
            per_id[nid] = outcome or "missing"
            if outcome is None:
                missing += 1
                continue
            if outcome == "passed":
                passed += 1
                if nid in f2p:
                    f2p_flipped += 1
                if nid in p2p:
                    p2p_preserved += 1
            elif outcome == "skipped":
                # Gold contract: a skipped expected test did NOT pass.
                skipped += 1
            elif outcome == "error":
                errors += 1
            else:  # failed / xfailed-as-fail / unknown
                failed += 1

        total = len(gold_set)
        # If pytest itself errored before producing ANY gold outcome AND there was a
        # collection failure, treat as a harness/parser failure (indeterminate),
        # never a genuine all-missing zero. A genuine all-fail (tests collected and
        # ran but failed) is a real residual and stays scored.
        ran_any = (passed + failed + errors + skipped) > 0
        if not ran_any and collection_failed:
            return _make_evaluation(
                returncode=rc or 1, output=output_tail, total_tests=total,
                report_path=str(report_file), evaluation_backend=_LOCAL_BACKEND,
                diagnostics={"harness_failure": True, "reason": "gold collection failed",
                             "per_id_outcomes": per_id})

        # Acceptance: full gold solve = all f2p flipped + all p2p preserved + no
        # failed/errors/skipped/missing over the gold union.
        accept = (
            total > 0
            and failed == 0 and errors == 0 and skipped == 0 and missing == 0
            and f2p_flipped == len(f2p)
            and p2p_preserved == len(p2p)
            and passed == total
        )
        scoring_source = "commit0_test_ids"  # set ONLY here, after a REAL run with matched node-ids
        coverage = {
            "expected_test_count": total,
            "matched_expected_test_count": total - missing,
            "missing_expected_test_count": missing,
            "skipped_expected_test_count": skipped,
            "coverage_preserved": missing == 0,
            "fail_to_pass_total": len(f2p),
            "fail_to_pass_flipped": f2p_flipped,
            "pass_to_pass_total": len(p2p),
            "pass_to_pass_preserved": p2p_preserved,
        }
        evaluation = _make_evaluation(
            returncode=0 if accept else 1,
            raw_returncode=rc,
            output=output_tail,
            report_path=str(report_file),
            passed=passed, failed=failed, errors=errors, skipped=skipped,
            total_tests=total,
            scoring_source=scoring_source,
            evaluation_backend=_LOCAL_BACKEND,
            expected_test_coverage=coverage,
            diagnostics={"per_id_outcomes": per_id,
                         "accept": bool(accept)},
        )
        return evaluation
