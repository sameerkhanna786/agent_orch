"""Instance-id keyed RepoSpec registry for the SWE-rebench Mode-C benchmark.

Built from the pinned ``configs/swerebench_slice.json`` (never re-fetched at eval
time).  Reuses the FROZEN ``RepoSpec`` dataclass from ``registry.py`` so the rest
of the harness treats a SWE-rebench instance exactly like a commit0 repo for
local-runnability gating.  ``local_runnable == not forces_docker``; every curated
SWE-rebench instance is Docker-free by construction (the slice filter requires an
empty ``pre_install``), so ``forces_docker`` is always False here.

This module does NOT touch ``registry.TARGET_REPOS`` — it is a parallel, separate
registry consulted only on the SWE-rebench code path.
"""

from __future__ import annotations

from typing import Optional

# Reuse the SAME frozen RepoSpec dataclass the commit0 registry uses.
from .registry import RepoSpec
from .swerebench_runner import slice_instances


def _spec_from_record(rec: dict) -> RepoSpec:
    # The slice key is the SWE-rebench instance_id (used as the RepoSpec.name so
    # --repos <instance_id> resolves through the same plumbing). Every curated
    # instance is Docker-free (empty pre_install) => forces_docker=False.
    return RepoSpec(
        name=str(rec["instance_id"]),
        in_lite=False,
        python_version=str(rec.get("python") or "3.11"),
        forces_docker=False,
        pre_install=(),
        dataset_fallback_revision=None,
        notes=f"swerebench:{rec.get('repo')}@{str(rec.get('base_commit') or '')[:10]} "
              f"stratum={rec.get('stratum')} created={str(rec.get('created_at') or '')[:10]}",
    )


def _build_registry(slice_path: Optional[str] = None) -> dict[str, RepoSpec]:
    return {iid: _spec_from_record(rec)
            for iid, rec in slice_instances(slice_path).items()}


def all_specs(slice_path: Optional[str] = None) -> dict[str, RepoSpec]:
    return _build_registry(slice_path)


def get(instance_id: str, *, slice_path: Optional[str] = None) -> RepoSpec:
    reg = _build_registry(slice_path)
    return reg[instance_id]


def instance_ids(slice_path: Optional[str] = None) -> list[str]:
    return sorted(_build_registry(slice_path).keys())


def local_runnable_targets(slice_path: Optional[str] = None) -> list[str]:
    return [iid for iid, spec in _build_registry(slice_path).items() if spec.local_runnable]


def fresh_targets(slice_path: Optional[str] = None) -> list[str]:
    return sorted(iid for iid, rec in slice_instances(slice_path).items()
                  if rec.get("stratum") == "fresh")


def older_targets(slice_path: Optional[str] = None) -> list[str]:
    return sorted(iid for iid, rec in slice_instances(slice_path).items()
                  if rec.get("stratum") == "older")
