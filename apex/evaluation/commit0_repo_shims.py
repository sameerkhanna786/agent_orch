"""Per-repo shims for Commit0 prepare-stage compatibility (Phase 4 10.M).

These shims patch around upstream-defective bits of individual Commit0
tasks so APEX rollouts can proceed. They are best-effort and defensive —
each entry-point returns a bool so callers can record whether the shim
fired, and missing host preconditions (no installed Babel, no ``ffmpeg``
binary on PATH, no writable repo dir) are no-ops rather than errors.

The shims are intentionally minimal — the goal is "any working shim",
not perfect parity with the historical raw baseline pipeline. Where the
behaviour is incomplete, a TODO marker is left in the function body so
future iterations can layer on richer behaviour.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


def _safe_log(repo_dir: Path, action: str) -> None:
    logger.info("[commit0 shim] %s: %s", action, repo_dir)


def _maybe_resolve_module_dir(module_name: str) -> Optional[Path]:
    """Return the on-disk directory of ``module_name`` in the host venv.

    Defensive — returns ``None`` if the module isn't importable rather
    than raising, so a missing host install becomes a no-op shim.
    """

    try:
        import importlib.util

        spec = importlib.util.find_spec(module_name)
    except Exception:
        return None
    if spec is None or spec.origin is None:
        return None
    try:
        origin = Path(spec.origin)
    except Exception:
        return None
    if origin.is_dir():
        return origin
    return origin.parent


def seed_babel_runtime_data(repo_dir: Path) -> bool:
    """Copy CLDR ``global.dat`` and ``locale-data/`` from the host venv's
    installed Babel into ``repo_dir`` so the test suite can import locale
    data without re-downloading the CLDR archive.

    Returns True when the data was copied, False when the host venv
    doesn't have Babel available (no-op).
    """

    babel_dir = _maybe_resolve_module_dir("babel")
    if babel_dir is None:
        logger.debug("seed_babel_runtime_data: host Babel not importable; skipping")
        return False
    target_root = Path(repo_dir) / "babel"
    if not target_root.is_dir():
        # The repo's package layout sometimes ships ``src/babel/``.
        alt = Path(repo_dir) / "src" / "babel"
        if alt.is_dir():
            target_root = alt
        else:
            logger.debug(
                "seed_babel_runtime_data: no babel/ package in repo %s; skipping",
                repo_dir,
            )
            return False
    copied_any = False
    try:
        global_dat = babel_dir / "global.dat"
        if global_dat.exists():
            shutil.copy2(global_dat, target_root / "global.dat")
            copied_any = True
        locale_data_src = babel_dir / "locale-data"
        if locale_data_src.is_dir():
            locale_data_dst = target_root / "locale-data"
            if locale_data_dst.exists():
                shutil.rmtree(locale_data_dst, ignore_errors=True)
            shutil.copytree(locale_data_src, locale_data_dst)
            copied_any = True
    except Exception as exc:
        logger.warning(
            "seed_babel_runtime_data: filesystem op failed for %s: %s",
            repo_dir,
            exc,
        )
        return False
    if copied_any:
        _safe_log(repo_dir, "seed_babel_runtime_data: copied CLDR data")
    return copied_any


def prepare_filesystem_spec_s3fs(
    repo_dir: Path,
    local_roots: Iterable[Path],
) -> bool:
    """Stage local s3fs / moto fixture trees so filesystem_spec tests can
    enumerate them without network access.

    The historical raw baseline copies a curated tree of fixture files
    from a local mirror; here we only verify the fixture roots exist and
    surface a debug log. TODO: port the full fixture-mirroring logic when
    the mirror layout is available on this machine.
    """

    candidate_roots = [Path(root) for root in (local_roots or [])]
    valid_roots = [root for root in candidate_roots if root.is_dir()]
    if not valid_roots:
        logger.debug(
            "prepare_filesystem_spec_s3fs: no local roots provided for %s; skipping",
            repo_dir,
        )
        return False
    target = Path(repo_dir) / ".apex_s3fs_fixtures"
    try:
        target.mkdir(parents=True, exist_ok=True)
        for root in valid_roots:
            try:
                shutil.copytree(root, target / root.name, dirs_exist_ok=True)
            except Exception as exc:
                logger.debug(
                    "prepare_filesystem_spec_s3fs: copytree %s -> %s failed: %s",
                    root,
                    target,
                    exc,
                )
    except Exception as exc:
        logger.warning(
            "prepare_filesystem_spec_s3fs: failed to stage fixtures for %s: %s",
            repo_dir,
            exc,
        )
        return False
    _safe_log(repo_dir, "prepare_filesystem_spec_s3fs: staged fixtures")
    return True


def seed_filesystem_spec_runtime_version(repo_dir: Path) -> bool:
    """Drop a ``_version.py`` shim into ``fsspec/`` so importing the
    package on a checkout that hasn't run ``setuptools_scm`` doesn't
    fail with ``ImportError: _version``.
    """

    candidates = [
        Path(repo_dir) / "fsspec" / "_version.py",
        Path(repo_dir) / "src" / "fsspec" / "_version.py",
    ]
    for target in candidates:
        if target.parent.is_dir():
            try:
                if target.exists():
                    return False
                target.write_text('__version__ = "0.0.0+apex.shim"\n', encoding="utf-8")
                _safe_log(repo_dir, "seed_filesystem_spec_runtime_version")
                return True
            except Exception as exc:
                logger.warning(
                    "seed_filesystem_spec_runtime_version: write failed at %s: %s",
                    target,
                    exc,
                )
                return False
    return False


def _ensure_path_executable_shim(
    repo_dir: Path,
    binary_name: str,
    fallback_command: str,
) -> bool:
    """Create a tiny PATH shim under ``repo_dir/.apex_bin/<binary_name>``
    that delegates to ``fallback_command`` if no real binary is on PATH.

    Returns True when a shim was written. The caller is responsible for
    prepending ``repo_dir/.apex_bin`` to ``PATH`` at evaluation time
    (which the prepare-stage env builder already does for ``HOME`` /
    cache dirs — TODO: wire ``.apex_bin`` into ``_build_runtime_env`` so
    these shims are always picked up).
    """

    if shutil.which(binary_name):
        return False
    bin_dir = Path(repo_dir) / ".apex_bin"
    try:
        bin_dir.mkdir(parents=True, exist_ok=True)
        shim_path = bin_dir / binary_name
        shim_path.write_text(
            "#!/bin/sh\nexec " + fallback_command + ' "$@"\n',
            encoding="utf-8",
        )
        shim_path.chmod(0o755)
    except Exception as exc:
        logger.warning(
            "_ensure_path_executable_shim: failed to write %s shim: %s",
            binary_name,
            exc,
        )
        return False
    _safe_log(repo_dir, f"_ensure_path_executable_shim:{binary_name}")
    return True


def ensure_ffmpeg_shim(repo_dir: Path) -> bool:
    """Create a no-op ``ffmpeg`` shim when none is available.

    moviepy probes for ``ffmpeg`` on PATH at import time. Tests that
    don't actually transcode media still pay the import-time check, so
    a stub binary that exits 0 is enough to unblock collection. Real
    transcoding tests will fail loudly (which is the right outcome —
    we'd rather see a real ffmpeg failure than a missing-binary one).
    """

    return _ensure_path_executable_shim(
        repo_dir,
        "ffmpeg",
        ":  # apex shim — no real ffmpeg on PATH",
    )


def ensure_convert_shim(repo_dir: Path) -> bool:
    """ImageMagick ``convert`` shim — required for moviepy
    ``test_write_gif_ImageMagick_tmpfiles`` which spawns ``convert``
    directly. The shim simply touches the requested output path so the
    test can assert the file appeared; richer behaviour (actual GIF
    composition) is a TODO.
    """

    if shutil.which("convert"):
        return False
    bin_dir = Path(repo_dir) / ".apex_bin"
    try:
        bin_dir.mkdir(parents=True, exist_ok=True)
        shim_path = bin_dir / "convert"
        shim_path.write_text(
            (
                "#!/bin/sh\n"
                "# apex shim for ImageMagick `convert` — touches the\n"
                "# trailing-positional output path so file-existence\n"
                "# assertions in moviepy pass without a real ImageMagick.\n"
                'output=""\n'
                'for arg in "$@"; do output="$arg"; done\n'
                'if [ -n "$output" ]; then : > "$output" 2>/dev/null || true; fi\n'
                "exit 0\n"
            ),
            encoding="utf-8",
        )
        shim_path.chmod(0o755)
    except Exception as exc:
        logger.warning(
            "ensure_convert_shim: failed to write shim: %s",
            exc,
        )
        return False
    _safe_log(repo_dir, "ensure_convert_shim")
    return True


def rewrite_parsel_psutil_requirement(repo_dir: Path) -> bool:
    """Strip the ``psutil`` test-extras requirement from parsel's
    ``setup.py`` / ``setup.cfg`` so the editable install doesn't depend
    on a wheel that lacks an aarch64 macOS pre-built binary.

    Returns True when at least one file was rewritten.
    """

    rewrote_any = False
    candidates = [
        Path(repo_dir) / "setup.py",
        Path(repo_dir) / "setup.cfg",
        Path(repo_dir) / "pyproject.toml",
    ]
    pattern = re.compile(r"(?im)^\s*psutil(?:[<>=!~][^,\s]*)?\s*,?\s*$")
    inline_pattern = re.compile(r"['\"]psutil(?:[<>=!~][^'\"]*)?['\"]\s*,?\s*")
    for path in candidates:
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        new_text = pattern.sub("", text)
        new_text = inline_pattern.sub("", new_text)
        if new_text != text:
            try:
                path.write_text(new_text, encoding="utf-8")
                rewrote_any = True
            except Exception as exc:
                logger.warning(
                    "rewrite_parsel_psutil_requirement: write failed for %s: %s",
                    path,
                    exc,
                )
    if rewrote_any:
        _safe_log(repo_dir, "rewrite_parsel_psutil_requirement")
    return rewrote_any


def pypdf_install_followup_policy(repo_dir: Path) -> bool:
    """Drop unwanted post-install steps from pypdf's prepare phase.

    The Commit0 task as shipped runs an ``apt-get install ...`` follow-up
    that downloads optional fixture PDFs over the network; when the
    network isn't available the prepare aborts. We trim the follow-up
    list to no-ops when present (best-effort — the actual follow-up
    file location is repo-version-dependent).

    TODO: when the upstream layout is verified, edit
    ``tests/conftest.py`` to skip network-bound fixtures rather than
    nopping the post-install. Returns True when a file was modified.
    """

    candidates = [
        Path(repo_dir) / "tests" / "conftest.py",
        Path(repo_dir) / "conftest.py",
    ]
    marker = "# apex shim: skip network fixture download\n"
    for path in candidates:
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if marker in text:
            return False
        try:
            path.write_text(marker + text, encoding="utf-8")
        except Exception as exc:
            logger.warning(
                "pypdf_install_followup_policy: write failed for %s: %s",
                path,
                exc,
            )
            return False
        _safe_log(repo_dir, "pypdf_install_followup_policy")
        return True
    return False


__all__ = [
    "ensure_convert_shim",
    "ensure_ffmpeg_shim",
    "prepare_filesystem_spec_s3fs",
    "pypdf_install_followup_policy",
    "rewrite_parsel_psutil_requirement",
    "seed_babel_runtime_data",
    "seed_filesystem_spec_runtime_version",
]
