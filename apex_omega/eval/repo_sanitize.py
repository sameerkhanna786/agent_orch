"""Strip upstream identity / version breadcrumbs from a prepared commit0 repo.

A commit0 task is "implement the missing code so the *visible tests* pass." The
prepared container, however, still ships the real package's release metadata —
``version = "17.0.0"`` in setup.py/pyproject, ``__version__`` in the package, an
upstream GitHub/PyPI URL, a CHANGELOG enumerating releases, git version tags, and
a remote pointing at the source. Those breadcrumbs invite a solver to go *fetch /
reconstruct the official upstream version X.Y.Z* instead of implementing from the
tests. That is (a) a cheat vector when the network is reachable and (b) a wasted
attempt when it is not (the workspace jail blocks the fetch -> policy_violation).

The principled fix is to remove the cheat surface at the environment layer rather
than police it with prompts: before any evaluation runs, scrub the version/upstream
pointers from the working tree and commit the result onto ``apex-base`` so every
forked worktree inherits a sanitized container.

Design constraints:
  * Must NOT break import or the test contract. We neutralise version *strings* to
    a placeholder (keeping the field syntactically valid) and only blank dedicated
    upstream-URL fields. If a *visible test* asserts ``__version__`` (i.e. the exact
    version is part of the spec), we leave the package ``__version__`` untouched but
    still remove the upstream locators and the release timeline.
  * Best-effort: any failure must never break repo preparation (caller guards).
  * Idempotent and dependency-free (stdlib only).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

PLACEHOLDER_VERSION = "0.0.0"
PLACEHOLDER_URL = "https://example.invalid"

# Hosts that tell a solver WHERE to fetch the upstream project from.
UPSTREAM_HOSTS = (
    "github.com", "raw.githubusercontent.com", "gitlab.com", "bitbucket.org",
    "pypi.org", "pypi.python.org", "test.pypi.org", "files.pythonhosted.org",
    "readthedocs.io", "readthedocs.org", "rtfd.io",
    "anaconda.org", "conda.anaconda.org",
)

# Packaging / metadata files whose version + URL fields we neutralise in place.
_META_FILES = ("setup.py", "setup.cfg", "pyproject.toml")
# Files that conventionally carry a module-level version literal.
_VERSION_PY_NAMES = ("_version.py", "version.py", "__about__.py", "__version__.py")
# Release-timeline files (delete outright — pure version history).
_TIMELINE_PREFIXES = ("CHANGELOG", "CHANGES", "CHANGE", "HISTORY", "NEWS",
                      "RELEASES", "RELEASE_NOTES", "RELEASENOTES", "WHATSNEW")
# Generated dist metadata that literally embeds Version/Home-page/Download-URL.
_META_GLOBS = ("*.egg-info", "*.dist-info")
_PKG_INFO = "PKG-INFO"

# version = "X" / __version__: 'X' / release="X"  (word-anchored so min_version,
# version_info, api_version are NOT matched). Only string literals are touched, so
# `version=get_version()` is left alone.
_VERSION_LITERAL = re.compile(
    r"(\b(?:__version__|version|release|VERSION)\b\s*[:=]\s*)(['\"])([^'\"\n]+)(['\"])"
)
_URL = re.compile(r"https?://[^\s'\"<>)\]}]+")
# Free-text semver in PROSE docs (README/CHANGELOG-prose/docs) — e.g. "Mimesis 17.0.0 is …" —
# is an upstream-version breadcrumb the key=literal regex above misses. Applied ONLY to prose
# targets (NOT setup.py/cfg/pyproject, whose dependency pins must survive). Over-redaction in
# docs is harmless (docs are not part of the test contract).
_FREETEXT_SEMVER = re.compile(r"\b\d+\.\d+(?:\.\d+)+(?:[._-]?[A-Za-z0-9]+)*\b")


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


def _neutralise_versions(text: str) -> tuple[str, int]:
    n = 0

    def repl(m: re.Match) -> str:
        nonlocal n
        n += 1
        return f"{m.group(1)}{m.group(2)}{PLACEHOLDER_VERSION}{m.group(4)}"

    return _VERSION_LITERAL.sub(repl, text), n


def _neutralise_urls(text: str) -> tuple[str, int]:
    n = 0

    def repl(m: re.Match) -> str:
        nonlocal n
        url = m.group(0)
        if any(h in url for h in UPSTREAM_HOSTS):
            n += 1
            return PLACEHOLDER_URL
        return url

    return _URL.sub(repl, text), n


def _neutralise_freetext_versions(text: str) -> tuple[str, int]:
    """Replace free-text X.Y.Z(+) version strings (prose docs only)."""
    n = 0

    def repl(m: re.Match) -> str:
        nonlocal n
        n += 1
        return PLACEHOLDER_VERSION

    return _FREETEXT_SEMVER.sub(repl, text), n


def _tests_reference_version(repo_dir: Path) -> bool:
    """True if a visible test asserts ``__version__`` — then the exact version is
    part of the spec and we must NOT alter the package version literal."""
    for base in ("tests", "test", "testing"):
        d = repo_dir / base
        if d.is_dir():
            for p in d.rglob("*.py"):
                try:
                    if "__version__" in p.read_text(errors="ignore"):
                        return True
                except OSError:
                    continue
    for p in list(repo_dir.glob("test_*.py")) + list(repo_dir.glob("*_test.py")) + list(repo_dir.glob("conftest.py")):
        try:
            if "__version__" in p.read_text(errors="ignore"):
                return True
        except OSError:
            continue
    return False


def _iter_text_targets(repo_dir: Path):
    """Files that get the URL scrub: metadata + top-level docs/readme + docs tree."""
    seen = set()
    for name in _META_FILES:
        p = repo_dir / name
        if p.is_file():
            seen.add(p)
    for pat in ("README*", "*.rst", "*.md", "*.txt"):
        for p in repo_dir.glob(pat):
            if p.is_file():
                seen.add(p)
    docs = repo_dir / "docs"
    if docs.is_dir():
        for pat in ("**/*.rst", "**/*.md", "**/conf.py"):
            for p in docs.glob(pat):
                if p.is_file():
                    seen.add(p)
    return sorted(seen)


def scrub_upstream_identifiers(repo_dir: Path, *, commit: bool = True) -> dict:
    """Remove upstream/version breadcrumbs from ``repo_dir`` (a git repo whose
    current branch is the eval base). Returns a report dict. Best-effort: callers
    should still wrap in try/except."""
    repo_dir = Path(repo_dir)
    report: dict = {"version_files": [], "url_files": [], "deleted": [],
                    "tags_removed": 0, "remotes_removed": [], "committed": False,
                    "kept_package_version": False, "branches_removed": 0,
                    "reflog_scrubbed": False}
    if not (repo_dir / ".git").exists():
        report["error"] = "not a git repo"
        return report

    keep_pkg_version = _tests_reference_version(repo_dir)
    report["kept_package_version"] = keep_pkg_version
    touched: list[Path] = []

    # 1) Neutralise version literals + upstream URLs in metadata/docs/readme.
    for p in _iter_text_targets(repo_dir):
        try:
            original = p.read_text(errors="ignore")
        except OSError:
            continue
        text, vN = _neutralise_versions(original)
        text, uN = _neutralise_urls(text)
        # PROSE docs (not the metadata files, whose dependency pins must survive) also get the
        # free-text semver scrub so a README "package X.Y.Z" upstream-version hint can't leak.
        if p.name not in _META_FILES:
            text, fN = _neutralise_freetext_versions(text)
            vN += fN
        if text != original:
            try:
                p.write_text(text)
            except OSError:
                continue
            touched.append(p)
            if vN:
                report["version_files"].append(str(p.relative_to(repo_dir)))
            if uN:
                report["url_files"].append(str(p.relative_to(repo_dir)))

    # 2) Neutralise module-level version literals (unless a test pins the version).
    if not keep_pkg_version:
        version_py: set[Path] = set()
        for nm in _VERSION_PY_NAMES:
            version_py.update(repo_dir.rglob(nm))
        # package __init__.py files that declare __version__
        for p in repo_dir.rglob("__init__.py"):
            try:
                if "__version__" in p.read_text(errors="ignore"):
                    version_py.add(p)
            except OSError:
                continue
        for p in sorted(version_py):
            try:
                original = p.read_text(errors="ignore")
            except OSError:
                continue
            text, vN = _neutralise_versions(original)
            if vN and text != original:
                try:
                    p.write_text(text)
                except OSError:
                    continue
                touched.append(p)
                report["version_files"].append(str(p.relative_to(repo_dir)))

    # 3) Delete release-timeline files + generated dist metadata.
    for p in repo_dir.iterdir():
        if p.is_file() and any(p.name.upper().startswith(pref) for pref in _TIMELINE_PREFIXES):
            try:
                p.unlink()
                touched.append(p)
                report["deleted"].append(str(p.relative_to(repo_dir)))
            except OSError:
                pass
    for pat in _META_GLOBS:
        for p in repo_dir.rglob(pat):
            try:
                shutil.rmtree(p) if p.is_dir() else p.unlink()
                touched.append(p)
                report["deleted"].append(str(p.relative_to(repo_dir)))
            except OSError:
                pass
    for p in repo_dir.rglob(_PKG_INFO):
        try:
            p.unlink()
            touched.append(p)
            report["deleted"].append(str(p.relative_to(repo_dir)))
        except OSError:
            pass

    # 4) Git refs that mark versions / point at upstream.
    tags = _git(["tag"], repo_dir)
    if tags.returncode == 0 and tags.stdout.strip():
        for tag in tags.stdout.split():
            if _git(["tag", "-d", tag], repo_dir).returncode == 0:
                report["tags_removed"] += 1
    remotes = _git(["remote"], repo_dir)
    if remotes.returncode == 0 and remotes.stdout.strip():
        for rmt in remotes.stdout.split():
            if _git(["remote", "remove", rmt], repo_dir).returncode == 0:
                report["remotes_removed"].append(rmt)

    # 4b) History-leak scrub: the upstream solution can hide in the reflog or a
    #     stale non-base branch even after tags/remotes are gone. Keep ONLY the
    #     current (eval base) branch, then expire the reflog and gc so the old
    #     objects are unreachable. Best-effort; never fatal.
    cur = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir)
    base_branch = cur.stdout.strip() if (cur.returncode == 0 and cur.stdout.strip()) else "apex-base"
    branches = _git(["for-each-ref", "--format=%(refname:short)", "refs/heads/"], repo_dir)
    if branches.returncode == 0 and branches.stdout.strip():
        for b in branches.stdout.split():
            if b and b != base_branch:
                if _git(["branch", "-D", b], repo_dir).returncode == 0:
                    report["branches_removed"] += 1
    _git(["reflog", "expire", "--expire=now", "--all"], repo_dir)
    if _git(["gc", "--prune=now", "--quiet"], repo_dir).returncode == 0:
        report["reflog_scrubbed"] = True

    # 5) Commit the working-tree scrub onto the current (base) branch so every
    #    worktree forked from it inherits the sanitized container. Stage only the
    #    paths we touched (don't sweep in unrelated prep/shim state).
    if commit and touched:
        rels = []
        for p in touched:
            try:
                rels.append(str(p.relative_to(repo_dir)))
            except ValueError:
                continue
        if rels:
            _git(["add", "-A", "--", *rels], repo_dir)
            staged = _git(["diff", "--cached", "--quiet"], repo_dir)
            if staged.returncode != 0:  # there ARE staged changes
                done = _git(["commit", "--no-verify", "-m",
                             "apex: scrub upstream version identifiers"], repo_dir)
                report["committed"] = done.returncode == 0

    return report
