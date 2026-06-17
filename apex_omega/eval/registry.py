"""The commit0 target-repo registry for APEX-Ω evaluation.

Verified against the locally-cached ``wentingzhao/commit0_combined`` (test split,
56 repos): all 15 targets are present.  ``in_lite`` repos are in commit0's
SPLIT_LITE; ``forces_docker`` repos have an ``apt-get`` pre_install so
``_task_requires_linux_container`` is True (need a Linux container / Docker —
NOT locally runnable on this mac).  Everything else scores locally via
``local_pytest_json_report`` (host per-repo uv venv pytest).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class RepoSpec:
    name: str                       # commit0 short repo name (the split-filter key)
    in_lite: bool                   # member of commit0 SPLIT_LITE
    python_version: str             # setup.python from the dataset row
    forces_docker: bool = False     # apt-get pre_install -> Linux container required
    pre_install: tuple[str, ...] = ()
    dataset_fallback_revision: Optional[str] = None
    notes: str = ""

    @property
    def local_runnable(self) -> bool:
        """Runnable without Docker on this mac."""
        return not self.forces_docker


# The 15 evaluation targets (order = the user's stated focus order).
TARGET_REPOS: tuple[RepoSpec, ...] = (
    RepoSpec("minitorch", True, "3.10", notes="builds numba==0.60 (slower)"),
    RepoSpec("jinja", True, "3.10"),
    RepoSpec("voluptuous", True, "3.10", notes="lightest install; best smoke slice"),
    RepoSpec("web3.py", False, "3.12", forces_docker=True,
             pre_install=("git submodule update --init", "apt-get update", "apt-get install clang"),
             notes="apt clang -> Docker required"),
    RepoSpec("statsmodels", False, "3.10", notes="heavy scientific deps"),
    RepoSpec("babel", True, "3.10", pre_install=("python scripts/download_import_cldr.py",),
             notes="CLDR download pre_install (not apt) -> local OK; repo shim handles CLDR"),
    RepoSpec("pydantic", False, "3.12"),
    RepoSpec("pytest", False, "3.10",
             dataset_fallback_revision="afc4d5f9085597e14e2b2a5bdbae28577ecd7ecb",
             notes="dropped from dataset main after 2024-09-22; needs fallback revision"),
    RepoSpec("networkx", False, "3.12"),
    RepoSpec("mimesis", False, "3.10"),
    RepoSpec("scrapy", False, "3.12", forces_docker=True,
             pre_install=("apt-get update", "apt-get install libxml2-dev libxslt-dev"),
             notes="apt libxml2/libxslt -> Docker required"),
    RepoSpec("seaborn", False, "3.12", notes="matplotlib deps"),
    RepoSpec("sphinx", False, "3.10", forces_docker=True,
             pre_install=("apt-get update", "apt-get install graphviz"),
             notes="apt graphviz -> Docker required"),
    RepoSpec("geopandas", False, "3.10", notes="geo deps (shapely/pyproj)"),
    RepoSpec("cookiecutter", True, "3.10"),
)

TARGET_NAMES: tuple[str, ...] = tuple(r.name for r in TARGET_REPOS)
_BY_NAME = {r.name: r for r in TARGET_REPOS}

# repos that need any dataset fallback revision (passed to Commit0BenchmarkRunner)
DATASET_FALLBACK_REVISIONS: dict[str, str] = {
    r.name: r.dataset_fallback_revision for r in TARGET_REPOS if r.dataset_fallback_revision
}


def get(name: str) -> RepoSpec:
    return _BY_NAME[name]


def local_runnable_targets() -> list[str]:
    """The 12 targets runnable without Docker on this machine."""
    return [r.name for r in TARGET_REPOS if r.local_runnable]


def docker_required_targets() -> list[str]:
    return [r.name for r in TARGET_REPOS if r.forces_docker]


def lite_targets() -> list[str]:
    return [r.name for r in TARGET_REPOS if r.in_lite]


def resolve(names: Optional[list[str]] = None, *, local_only: bool = False) -> list[RepoSpec]:
    """Resolve a list of repo names (or all targets) to RepoSpecs, optionally
    filtering to the Docker-free subset."""
    if names:
        specs = [_BY_NAME[n] for n in names]
    else:
        specs = list(TARGET_REPOS)
    if local_only:
        specs = [s for s in specs if s.local_runnable]
    return specs
