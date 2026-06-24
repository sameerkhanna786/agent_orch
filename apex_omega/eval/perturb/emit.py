"""Emit the perturbed variant — git repo + bz2 + manifest + sidecar wiring.

The variant must be a REAL git repo whose commit SHAs match the synthetic
dataset row's ``base_commit`` (skeleton) and ``reference_commit`` (gold), because
the harness hard-asserts HEAD == base_commit after ``git checkout -B apex-base``.
So we ``git init`` the variant, commit the perturbed SKELETON (record its SHA =
base_commit), then commit the perturbed REFERENCE (record its SHA =
reference_commit), and write both SHAs into ``manifest.json``.

All wiring is ADDITIVE and namespaced by ``<repo>_perturbed`` so vanilla commit0
stays byte-identical:

* variant repo  -> ``apex_omega/eval/perturb/variants/<repo>_perturbed/repo`` (git)
* manifest      -> ``apex_omega/eval/perturb/variants/<repo>_perturbed/manifest.json``
* gold bz2      -> ``{commit0_pkg}/data/test_ids/<repo>_perturbed.bz2`` (pure data; the
  harness ``get_pytest_ids.main`` reads it by normalized name with ZERO code change)
* synthetic-task sidecar + local-repo-root + registry entry are emitted as a
  ``perturbed_targets.json`` the harness wiring reads (see commit0_benchmark.py).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


VARIANTS_DIR = Path(__file__).resolve().parent / "variants"
# the harness reads this sidecar to register synthetic perturbed tasks + mirrors
PERTURBED_TARGETS_SIDECAR = VARIANTS_DIR / "perturbed_targets.json"


def _git(args: list[str], cwd: Path, env=None) -> str:
    res = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True, env=env,
    )
    return res.stdout.strip()


def _git_commit_tree(repo: Path, message: str) -> str:
    """Stage everything and commit; return the new HEAD SHA."""
    import os
    env = dict(os.environ)
    # deterministic-ish identity (SHA still varies with timestamp; recorded after)
    env.update({
        "GIT_AUTHOR_NAME": "APEX", "GIT_AUTHOR_EMAIL": "apex@example.com",
        "GIT_COMMITTER_NAME": "APEX", "GIT_COMMITTER_EMAIL": "apex@example.com",
    })
    _git(["add", "-A"], repo, env=env)
    _git(["commit", "-q", "--no-verify", "-m", message], repo, env=env)
    return _git(["rev-parse", "HEAD"], repo, env=env)


@dataclass
class EmitResult:
    perturbed_repo_name: str
    variant_repo_path: str
    base_commit: str          # skeleton SHA
    reference_commit: str     # gold SHA
    bz2_path: str
    manifest_path: str
    expected_id_count: int
    src_dir: str
    test_dir: str
    test_cmd: str
    python_version: str
    repo_slug: str            # synthetic task.repo, e.g. "commit-0-perturbed/voluptuous_perturbed"


def emit_variant(
    *,
    perturbed_repo_name: str,         # e.g. "voluptuous_perturbed"
    skeleton_tree: Path,              # perturbed SKELETON checkout (agent's start state)
    reference_tree: Path,             # perturbed REFERENCE checkout (gold)
    expected_ids: list[str],
    commit0_pkg_dir: Path,            # dirname(commit0.__file__)
    src_dir: str,
    test_dir: str,
    test_cmd: str,
    python_version: str,
    name_map_json: dict,
    rename_report: dict,
    seed: int,
    base_repo_slug: str,              # vanilla task.repo, e.g. "commit-0/voluptuous"
) -> EmitResult:
    """Materialize the variant git repo + bz2 + manifest + sidecar entry."""
    import shutil

    # The harness local-mirror fallback looks for ``<mirror_root>/<repo_name>``
    # where repo_name == perturbed_repo_name. So the git mirror must live at
    # ``variants/<perturbed_repo_name>`` directly (its .git at that path), and the
    # manifest is kept under ``variants/_manifests/<perturbed_repo_name>.json`` so
    # it does not pollute the cloned tree.
    repo_git = VARIANTS_DIR / perturbed_repo_name
    manifests_dir = VARIANTS_DIR / "_manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    if repo_git.exists():
        shutil.rmtree(repo_git)
    repo_git.mkdir(parents=True, exist_ok=True)

    # synthetic slug under a perturbed org so a GitHub clone of it is impossible
    # (the harness falls through to the local mirror — see commit0_benchmark wiring)
    repo_slug = f"commit-0-perturbed/{perturbed_repo_name}"

    # 1) commit the perturbed SKELETON => base_commit (the agent's start state)
    shutil.copytree(skeleton_tree, repo_git, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".git", ".ropeproject", "__pycache__", "*.egg-info"))
    _git(["init", "-q", "-b", "main"], repo_git)
    base_commit = _git_commit_tree(repo_git, "perturbed skeleton (base_commit)")

    # 2) overlay the perturbed REFERENCE and commit => reference_commit (gold)
    #    Remove tracked files first so deletions in the reference are reflected.
    for child in repo_git.iterdir():
        if child.name == ".git":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    shutil.copytree(reference_tree, repo_git, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".git", ".ropeproject", "__pycache__", "*.egg-info"))
    reference_commit = _git_commit_tree(repo_git, "perturbed reference (reference_commit / gold)")

    # 3) regenerated gold bz2 (variant-namespaced; can never clobber a vanilla file)
    from .gate import write_expected_ids_bz2
    bz2_path = commit0_pkg_dir / "data" / "test_ids" / f"{perturbed_repo_name.lower().replace('.', '-')}.bz2"
    write_expected_ids_bz2(expected_ids, bz2_path)

    # 4) manifest
    manifest = {
        "perturbed_repo_name": perturbed_repo_name,
        "repo_slug": repo_slug,
        "base_repo_slug": base_repo_slug,
        "base_commit": base_commit,
        "reference_commit": reference_commit,
        "src_dir": src_dir,
        "test_dir": test_dir,
        "test_cmd": test_cmd,
        "python_version": python_version,
        "expected_id_count": len(expected_ids),
        "expected_ids": list(expected_ids),
        "bz2_path": str(bz2_path),
        "seed": seed,
        "name_map": name_map_json,
        "rename_report": rename_report,
        "variant_repo_path": str(repo_git),
    }
    manifest_path = manifests_dir / f"{perturbed_repo_name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 5) update the sidecar the harness reads (synthetic task + mirror root)
    _upsert_sidecar(perturbed_repo_name, manifest)

    return EmitResult(
        perturbed_repo_name=perturbed_repo_name,
        variant_repo_path=str(repo_git),
        base_commit=base_commit,
        reference_commit=reference_commit,
        bz2_path=str(bz2_path),
        manifest_path=str(manifest_path),
        expected_id_count=len(expected_ids),
        src_dir=src_dir,
        test_dir=test_dir,
        test_cmd=test_cmd,
        python_version=python_version,
        repo_slug=repo_slug,
    )


def restore_bz2_from_manifest(perturbed_repo_name: str, commit0_pkg_dir: Path) -> Optional[Path]:
    """Re-drop the gold bz2 for *perturbed_repo_name* into the commit0 data dir
    from its manifest's ``expected_ids``.

    The bz2 lives in the commit0 package's ``data/test_ids`` (outside this repo),
    so a fresh checkout / new venv must restore it before scoring.  Returns the
    bz2 path, or ``None`` if the manifest is missing.
    """
    from .gate import write_expected_ids_bz2

    manifest_path = VARIANTS_DIR / "_manifests" / f"{perturbed_repo_name}.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = manifest.get("expected_ids") or []
    if not ids:
        return None
    bz2_path = commit0_pkg_dir / "data" / "test_ids" / f"{perturbed_repo_name.lower().replace('.', '-')}.bz2"
    write_expected_ids_bz2(ids, bz2_path)
    return bz2_path


def _upsert_sidecar(perturbed_repo_name: str, manifest: dict) -> None:
    """Add/replace this repo's entry in the perturbed_targets sidecar JSON.

    The sidecar carries everything ``discover_tasks`` needs to synthesize a
    ``Commit0Task`` for ``<repo>_perturbed`` without any HuggingFace row, plus the
    local-mirror parent dir.  The harness reads it lazily; if the file is absent,
    vanilla commit0 behaves identically (byte-identical).
    """
    VARIANTS_DIR.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if PERTURBED_TARGETS_SIDECAR.exists():
        try:
            data = json.loads(PERTURBED_TARGETS_SIDECAR.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data.setdefault("_doc", (
        "Sidecar for perturbed-commit0 de-contaminated variants. Read by "
        "apex/evaluation/commit0_benchmark.py (discover_tasks synthetic-task hook + "
        "local-mirror fallback). Absent file => vanilla commit0 unchanged."
    ))
    targets = data.setdefault("targets", {})
    targets[perturbed_repo_name] = {
        "repo": manifest["repo_slug"],
        "base_commit": manifest["base_commit"],
        "reference_commit": manifest["reference_commit"],
        "python_version": manifest["python_version"],
        "src_dir": manifest["src_dir"],
        "test_cmd": manifest["test_cmd"],
        "test_dir": manifest["test_dir"],
        # absolute parent dir holding "<perturbed_repo_name>" git mirror;
        # the harness looks for <mirror_root>/<task.repo_name> with .git inside.
        "mirror_root": str(Path(manifest["variant_repo_path"]).parent),
        "mirror_repo_dirname": Path(manifest["variant_repo_path"]).name,
    }
    PERTURBED_TARGETS_SIDECAR.write_text(json.dumps(data, indent=2), encoding="utf-8")
