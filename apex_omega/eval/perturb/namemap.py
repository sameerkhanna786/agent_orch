"""Deterministic, collision-free opaque-name generator for symbol renaming.

DEPENDENCY-FREE on purpose: rope/libcst live only in the build venv, but the
name map is the single source of truth shared by BOTH checkouts (reference +
skeleton), persisted to the manifest, and unit-tested from ``.venv_omega``.

Design:

* Deterministic — keyed by a ``--seed`` so two runs of the build produce the
  identical map (reproducible variants).
* Collision-free — the generated name is checked against the live FQN inventory,
  Python keywords/soft-keywords, builtins, and already-emitted names; on a clash
  the hash slice is extended until unique.
* Structurally parallel — a renamed *function* gets an ``fn_`` prefix, a *class*
  a ``Cls`` prefix (CamelCase preserved so a class still reads as a class), a
  *module* a ``mod_`` prefix.  This keeps reading-load roughly constant so a
  within-perturbed orchestration A/B is not confounded by readability deltas.
* Hierarchy-aware — callers pass one canonical FQN per override-group so an
  overriding method and its base share ONE new name (rope ``in_hierarchy=True``).
"""

from __future__ import annotations

import hashlib
import keyword
from dataclasses import dataclass, field
from typing import Iterable, Optional


# soft keywords (match/case/type/_) must never be generated as identifiers
_SOFT_KEYWORDS = frozenset(getattr(keyword, "softkwlist", ()) or ("match", "case", "type", "_"))
_PY_KEYWORDS = frozenset(keyword.kwlist) | _SOFT_KEYWORDS
# A conservative builtins set so a generated name never shadows ``len``/``list``/...
try:  # builtins is always importable; guard only for exotic envs
    import builtins as _builtins

    _BUILTIN_NAMES = frozenset(dir(_builtins))
except Exception:  # pragma: no cover - defensive
    _BUILTIN_NAMES = frozenset()


class SymbolKind:
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE = "module"


# Prefix per kind. CamelCase for class/method-on-class so the perturbed surface
# preserves the "this is a type" reading cue; snake_case otherwise.
_PREFIX = {
    SymbolKind.FUNCTION: "fn_",
    SymbolKind.METHOD: "fn_",
    SymbolKind.CLASS: "Cls",
    SymbolKind.MODULE: "mod_",
}


@dataclass
class NameMap:
    """old_fqn -> new_name, plus the reverse and the module-rename submap."""

    seed: int
    symbols: dict[str, str] = field(default_factory=dict)        # fqn -> new name
    kinds: dict[str, str] = field(default_factory=dict)          # fqn -> SymbolKind
    modules: dict[str, str] = field(default_factory=dict)        # old module fqn -> new
    _emitted: set[str] = field(default_factory=set)

    def to_json(self) -> dict:
        return {
            "seed": self.seed,
            "symbols": dict(self.symbols),
            "kinds": dict(self.kinds),
            "modules": dict(self.modules),
        }

    @classmethod
    def from_json(cls, data: dict) -> "NameMap":
        nm = cls(seed=int(data.get("seed", 0)))
        nm.symbols = dict(data.get("symbols") or {})
        nm.kinds = dict(data.get("kinds") or {})
        nm.modules = dict(data.get("modules") or {})
        nm._emitted = set(nm.symbols.values()) | set(nm.modules.values())
        return nm


def _digest(fqn: str, seed: int) -> str:
    """Stable hex digest of (fqn, seed).  blake2b keyed by the seed bytes."""
    key = str(seed).encode("utf-8")
    return hashlib.blake2b(fqn.encode("utf-8"), key=key, digest_size=16).hexdigest()


def _is_reserved(name: str, taken: Iterable[str]) -> bool:
    return (
        name in _PY_KEYWORDS
        or name in _BUILTIN_NAMES
        or name in set(taken)
    )


def generate_name(
    fqn: str,
    kind: str,
    seed: int,
    *,
    taken: Iterable[str] = (),
    short_name: Optional[str] = None,
) -> str:
    """Deterministic opaque name for *fqn* of *kind*, unique vs *taken*.

    The hash slice starts at 8 hex chars and extends on collision, so the
    function is total (always returns a unique, valid identifier).
    """
    prefix = _PREFIX.get(kind, "fn_")
    digest = _digest(fqn, seed)
    taken_set = set(taken)
    for width in range(8, len(digest) + 1, 2):
        candidate = f"{prefix}{digest[:width]}"
        if not _is_reserved(candidate, taken_set):
            return candidate
    # Exhausted the digest (astronomically unlikely): salt and retry.
    return generate_name(fqn + "#", kind, seed, taken=taken_set, short_name=short_name)


def build_name_map(
    worklist: list[tuple[str, str]],
    *,
    seed: int,
    reserved_fqns: Iterable[str] = (),
    module_worklist: Optional[list[str]] = None,
) -> NameMap:
    """Build the full :class:`NameMap` from a classified worklist.

    Args:
        worklist: ``[(canonical_fqn, kind), ...]`` — one entry per rename group
            (hierarchies already collapsed to a single canonical fqn upstream).
        seed: reproducibility seed.
        reserved_fqns: short names already present in the repo / builtins that a
            generated name must never collide with (the inventory's leaf names).
        module_worklist: list of intra-repo module FQNs to rename (e.g.
            ``"voluptuous.validators"``).

    Determinism: ``worklist`` is sorted so the emission order — and therefore the
    collision-resolution order — is independent of caller iteration order.
    """
    nm = NameMap(seed=seed)
    # leaf short names of the reserved FQNs (a new name must not equal any of them)
    reserved_leaves = {f.rsplit(".", 1)[-1] for f in reserved_fqns}
    taken: set[str] = set(reserved_leaves)

    # Modules first (stable order) so symbol names can't collide with module names.
    for mod_fqn in sorted(module_worklist or []):
        new = generate_name(mod_fqn, SymbolKind.MODULE, seed, taken=taken)
        nm.modules[mod_fqn] = new
        nm._emitted.add(new)
        taken.add(new)

    for fqn, kind in sorted(worklist):
        if fqn in nm.symbols:
            continue
        new = generate_name(fqn, kind, seed, taken=taken)
        nm.symbols[fqn] = new
        nm.kinds[fqn] = kind
        nm._emitted.add(new)
        taken.add(new)
    return nm
