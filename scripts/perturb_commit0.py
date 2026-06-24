#!/usr/bin/env python
"""Thin wrapper -> apex_omega.eval.perturb.cli (perturbed-commit0 build tool).

This is an OFFLINE build-time tool.  It requires ``rope`` + ``libcst`` (the
rename engine + classifier) importable; create the dedicated build venv once:

    python3.10 -m venv /tmp/_perturb_venv
    /tmp/_perturb_venv/bin/pip install 'rope==1.14.*' 'libcst==1.8.*'

Then run, e.g. for voluptuous:

    PYTHONPATH=. /tmp/_perturb_venv/bin/python scripts/perturb_commit0.py voluptuous \
        --seed 1337 --repo-slug commit-0/voluptuous \
        --base-commit 81b91c5998e8e5c991d1adf854ecb22ab96376b2 \
        --reference-commit dcaaf3dd68be156253518a045feb1c4172dbd2d5 \
        --top-package voluptuous --test-dir voluptuous/tests \
        --test-cmd pytest --python-version 3.10 --python-exe python3.10 \
        --commit0-pkg-dir "$(/Users/sameertkhanna/Documents/apex/apex/.venv/bin/python -c 'import os,commit0;print(os.path.dirname(commit0.__file__))')"

RENAME-ONLY de-contamination: docstrings/comments (the natural-language SPEC) are
RETAINED; only symbol/module references are renamed (incl. inside docstrings and
``>>>`` doctests, via rope ``docs=True``) so the surface no longer matches
memorized code WITHOUT making the task harder.  Do NOT pass ``--neutralize-docs``
for shipped variants (it strips the spec; opt-in escape hatch only).

The GATE is non-negotiable: a variant is emitted ONLY if its perturbed reference
passes its own gold test suite at 100%.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apex_omega.eval.perturb.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
