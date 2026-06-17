"""
Filesystem helpers for copying and cleaning repository trees safely.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable, Iterable, Optional

IgnoreCallable = Optional[Callable[[str, list[str]], Iterable[str]]]


def _path_is_ignored(path: Path, ignore: IgnoreCallable) -> bool:
    if ignore is None:
        return False
    ignored = ignore(str(path.parent), [path.name]) or []
    return path.name in set(ignored)


def remove_path(path: Path) -> None:
    """Remove a filesystem entry without following directory symlinks."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
        return
    if path.exists() or path.is_symlink():
        path.unlink(missing_ok=True)


def copy_tree(
    source: str | Path,
    destination: str | Path,
    *,
    ignore: IgnoreCallable = None,
    dirs_exist_ok: bool = False,
    restrict_symlinks_to_root: bool = False,
) -> None:
    """Copy a tree while preserving symlink structure.

    Preserving symlinks keeps repository layouts intact and avoids failures on
    valid-but-dangling links that appear in real-world repos. When used for a
    sandbox copy, callers can additionally strip links whose targets escape the
    copied root.
    """
    source_path = Path(source)
    destination_path = Path(destination)
    shutil.copytree(
        source_path,
        destination_path,
        symlinks=True,
        ignore=ignore,
        dirs_exist_ok=dirs_exist_ok,
    )
    if restrict_symlinks_to_root:
        remove_symlink_escapes(destination_path)


def copy_path(
    source: str | Path,
    destination: str | Path,
    *,
    ignore: IgnoreCallable = None,
) -> None:
    """Copy a single file, directory, or symlink while preserving link shape."""
    source_path = Path(source)
    destination_path = Path(destination)
    if _path_is_ignored(source_path, ignore):
        return
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if source_path.is_symlink():
        remove_path(destination_path)
        destination_path.symlink_to(os.readlink(source_path))
        return

    if source_path.is_dir():
        if destination_path.exists() and not destination_path.is_dir():
            remove_path(destination_path)
        copy_tree(
            source_path,
            destination_path,
            ignore=ignore,
            dirs_exist_ok=destination_path.exists(),
        )
        return

    if destination_path.is_dir() and not destination_path.is_symlink():
        remove_path(destination_path)
    shutil.copy2(source_path, destination_path, follow_symlinks=False)


def remove_symlink_escapes(root: str | Path) -> None:
    """Remove copied symlinks whose targets resolve outside ``root``."""
    root_path = Path(root).resolve(strict=False)
    stack = [root_path]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                entry_path = Path(entry.path)
                if entry.is_symlink():
                    if not symlink_target_within_root(entry_path, root_path):
                        remove_path(entry_path)
                    continue
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry_path)


def symlink_target_within_root(link_path: str | Path, root: str | Path) -> bool:
    """Return whether a symlink target stays within the provided root."""
    link = Path(link_path)
    root_path = Path(root).resolve(strict=False)
    try:
        raw_target = Path(os.readlink(link))
    except OSError:
        return False
    if raw_target.is_absolute():
        resolved_target = raw_target.resolve(strict=False)
    else:
        resolved_target = (link.parent / raw_target).resolve(strict=False)
    try:
        resolved_target.relative_to(root_path)
        return True
    except ValueError:
        return False
