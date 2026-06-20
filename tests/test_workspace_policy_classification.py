"""Workspace-discovery guard env-aware allow-list (orchestration_research/WORKSPACE_GUARD_ANALYSIS.md).

The fatal-abort guard mis-classified the agent's OWN runtime infra (CODEX_HOME / uv cache / TMPDIR /
~/.cache) and read-only OS dirs as a fatal cross-isolation escape, aborting ~89% of large-repo
rollouts as false positives. The fix downgrades those to a SOFT ``workspace_discovery``
course-correction (rollout continues) while keeping genuine cheats FATAL by construction: sibling
worktrees, other cells, the ladder root, arbitrary /tmp, and planted ``*_upstream`` copies (G1
precedence). Telemetry is preserved (classify_attempt_integrity still fires sandbox_escape).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from apex.core.cli_backend import CLIModelClient
from apex_omega.autogen.context import classify_attempt_integrity


def _env_tree():
    rt = Path(tempfile.mkdtemp()) / "cells" / "mod" / "runtime"
    home = rt / "home"
    cache = home / ".cache"
    tmp = rt / "tmp"
    for p in (cache / "uv" / "archive-v0", tmp):
        p.mkdir(parents=True, exist_ok=True)
    env = {"HOME": str(home), "CODEX_HOME": str(home), "XDG_CACHE_HOME": str(cache), "TMPDIR": str(tmp)}
    return rt, env


# --------------------------------------------------------------- _agent_runtime_infra_roots
def test_infra_roots_resolves_env_keys_and_drops_root():
    _, env = _env_tree()
    roots = CLIModelClient._agent_runtime_infra_roots(env)
    rootset = {str(r) for r in roots}
    assert str(Path(env["CODEX_HOME"]).resolve()) in rootset
    assert str(Path(env["XDG_CACHE_HOME"]).resolve()) in rootset
    assert str(Path(env["TMPDIR"]).resolve()) in rootset
    assert "/" not in rootset                                   # the filesystem root is never an infra root


def test_infra_roots_none_env_still_covers_host_cache():
    roots = CLIModelClient._agent_runtime_infra_roots(None)
    assert any(str(r).endswith("/.cache") for r in roots)      # dominant host ~/.cache FP covered


# --------------------------------------------------------------- _path_is_agent_runtime_infra
def test_infra_paths_are_soft():
    rt, env = _env_tree()
    roots = CLIModelClient._agent_runtime_infra_roots(env)
    soft = [
        str(Path(env["XDG_CACHE_HOME"]) / "uv" / "archive-v0" / "x"),  # codex uv cache
        str(Path(env["CODEX_HOME"]) / "foo"),                          # CODEX_HOME
        str(Path(env["TMPDIR"]) / "scratch"),                          # rollout TMPDIR
        str(Path.home() / ".cache" / "pip"),                          # host ~/.cache
        "/opt/homebrew/lib/perl5", "/usr/lib/python3.11",             # read-only OS roots
    ]
    for p in soft:
        assert CLIModelClient._path_is_agent_runtime_infra(Path(p), runtime_infra_roots=roots) is True, p


def test_isolation_paths_stay_fatal():
    rt, env = _env_tree()
    roots = CLIModelClient._agent_runtime_infra_roots(env)
    fatal = [
        str(rt.parent.parent / "mod2" / "worktrees" / "wt_x" / "repo"),  # SIBLING worktree (same /var/folders tree)
        "/private/tmp/omega_ladder",                                     # ladder root
        "/tmp/pydantic_upstream/pydantic",                               # planted upstream copy
        str(Path(env["TMPDIR"]) / "pydantic_upstream" / "x"),            # upstream UNDER infra -> G1 precedence
        "/tmp/whatever",                                                 # arbitrary /tmp
        "/var/folders/zz/other/T/tmpY",                                  # ANOTHER rollout's /var/folders tmp
    ]
    for p in fatal:
        assert CLIModelClient._path_is_agent_runtime_infra(Path(p), runtime_infra_roots=roots) is False, p


def test_var_folders_not_blanket_soft():
    # regression for the isolation hole caught in implementation: /var/folders must NOT be a blanket
    # soft root (the eval's own worktrees live there); only THIS rollout's specific TMPDIR is soft.
    assert CLIModelClient._path_is_agent_runtime_infra(
        Path("/var/folders/zz/abc/T/some_other_cell/repo"), runtime_infra_roots=()) is False


def test_upstream_reference_copy_detection():
    assert CLIModelClient._path_resolves_to_upstream_reference_copy(Path("/tmp/pydantic_upstream/x")) is True
    assert CLIModelClient._path_resolves_to_upstream_reference_copy(Path("/x/mimesis_wheel/y")) is True
    assert CLIModelClient._path_resolves_to_upstream_reference_copy(Path("/x/_restore/y")) is True
    assert CLIModelClient._path_resolves_to_upstream_reference_copy(Path("/x/runtime/home/.cache/uv")) is False


def test_infra_check_backcompat_no_roots():
    # without runtime_infra_roots the codex cache is NOT recognized -> reproduces pre-fix fatal
    # (only read-only OS roots / APEX helper targets are soft without the env roots).
    _, env = _env_tree()
    assert CLIModelClient._path_is_agent_runtime_infra(
        Path(env["XDG_CACHE_HOME"]) / "uv", runtime_infra_roots=()) is False
    assert CLIModelClient._path_is_agent_runtime_infra(Path("/usr/lib/x"), runtime_infra_roots=()) is True


# --------------------------------------------------------------- guard severity end-to-end (operand branch)
def _bare_client():
    return CLIModelClient.__new__(CLIModelClient)   # methods only; _process_tree uses no instance state


def _severity_for_find(operand: str, working_dir: str, roots) -> str:
    c = _bare_client()
    c._process_cwd = lambda pid: Path(working_dir)   # in-workspace cwd -> only the operand branch decides
    entries = {1: {"command": f"find {operand} -name x", "argv": ["find", operand, "-name", "x"]}}
    v = c._process_tree_workspace_policy_violation(entries, working_dir, runtime_infra_roots=roots)
    return None if v is None else str(v.get("severity"))


def test_guard_operand_severity():
    rt, env = _env_tree()
    roots = CLIModelClient._agent_runtime_infra_roots(env)
    ws = str(rt.parent / "worktrees" / "wt_a" / "repo")
    Path(ws).mkdir(parents=True, exist_ok=True)
    # codex uv cache operand -> SOFT workspace_discovery
    assert _severity_for_find(str(Path(env["XDG_CACHE_HOME"]) / "uv"), ws, roots) == "workspace_discovery"
    # sibling worktree operand -> FATAL
    sib = str(rt.parent / "worktrees" / "wt_b" / "repo")
    assert _severity_for_find(sib, ws, roots) == "fatal"
    # planted upstream copy -> FATAL even though we pass roots
    assert _severity_for_find("/tmp/pydantic_upstream/p", ws, roots) == "fatal"
    # regression: WITHOUT roots the codex cache operand reverts to fatal (pre-fix behavior)
    assert _severity_for_find(str(Path(env["XDG_CACHE_HOME"]) / "uv"), ws, ()) == "fatal"


# --------------------------------------------------------------- T1 telemetry preservation
class _Res:
    def __init__(self, finalization_status="completed", error=None, fs_diff=""):
        self.finalization_status = finalization_status
        self.error = error
        self.fs_diff = fs_diff


def test_downgraded_escape_still_records_sandbox_escape():
    # the v1 adapter folds the soft reason into `error`; classify_attempt_integrity must still fire.
    res = _Res(error="soft policy: 1 workspace_discovery violation(s); CLI backend helper executed "
                     "repository discovery outside the rollout workspace: find ...")
    integ = classify_attempt_integrity(res)
    assert integ["attempted"] is True
    assert any(s["kind"] == "sandbox_escape" for s in integ["signals"])


def test_clean_completion_records_nothing():
    integ = classify_attempt_integrity(_Res(error=None))
    assert not any(s["kind"] == "sandbox_escape" for s in integ.get("signals", []))
