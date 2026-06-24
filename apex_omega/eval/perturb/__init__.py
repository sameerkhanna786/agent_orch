"""perturbed-commit0 de-contamination pipeline (offline build tool).

Produces ``<repo>_perturbed`` benchmark variants of commit0 repos where the
repo's OWN symbols are consistently alpha-renamed (+ docstrings/comments
neutralized) across BOTH implementation AND tests, so test SEMANTICS are
identical but the surface no longer matches the model's memorized code — forcing
genuine construction, not recall.

This package is an OFFLINE build-time tool, NOT in the eval hot path.  The
rename ENGINE (``rope``) and the symbol CLASSIFIER (``libcst``) live only in a
dedicated build venv (see ``scripts/perturb_commit0.py``); only the lightweight
deterministic name generator (:mod:`namemap`) is dependency-free and importable
from ``.venv_omega`` for unit testing.

Module layout:

* :mod:`inventory`   — libcst FQN classifier -> rename worklist (repo-defined only)
* :mod:`namemap`     — seeded collision-free opaque-name generator (dependency-free)
* :mod:`rename`      — rope engine driver (offsets re-resolved after each apply)
* :mod:`docstrings`  — libcst transformer: neutralize docstrings/comments
* :mod:`gate`        — validation gate (apply-to-reference + 100% gold pass) + bz2 regen
* :mod:`emit`        — git-variant repo + bz2 + manifest + sidecar registry/override
* :mod:`cli`         — one-command driver
"""

from __future__ import annotations

__all__ = ["namemap"]
