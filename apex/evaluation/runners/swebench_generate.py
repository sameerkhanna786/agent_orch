"""SWE-Bench prediction generation CLI entry.

Mirror of ``testgenevallite_generate.py``'s entry shape: a thin wrapper
around the codegen evaluation driver in
``apex.evaluation.swebench_codegen_eval``. Routes by the ``--harness-mode``
flag — ``classic`` / ``verified`` (alias) / ``multilingual`` use the
public ``swebench`` package; ``pro`` delegates to the SWE-Bench Pro
codegen entrypoint.

This module exists so operators can write::

    python -m apex.evaluation.runners.swebench_generate --help

and get the same shape they use for TestGenEvalLite, without having to
remember which sibling module owns which dataset.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

from ..swebench_benchmark import (
    SWEBENCH_HARNESS_MODE_CLASSIC,
    SWEBENCH_HARNESS_MODE_MULTILINGUAL,
    SWEBENCH_HARNESS_MODE_PRO,
)
from ..swebench_codegen_eval import main as classic_main
from ..swebench_pro_codegen_eval import main as pro_main

logger = logging.getLogger("apex.swebench_generate")


def _route(argv: list[str]) -> int:
    """Peek at ``--harness-mode`` and dispatch.

    We do NOT consume any args here; we just sniff the value so we can
    forward the entire argv to the right submain. This keeps each
    submain's argparse setup canonical (so ``--help`` from this entry
    proxies to the chosen submain's full help).
    """

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--harness-mode",
        default=SWEBENCH_HARNESS_MODE_CLASSIC,
    )
    known, _ = parser.parse_known_args(argv)
    mode = (known.harness_mode or "").strip().lower() or SWEBENCH_HARNESS_MODE_CLASSIC
    if mode == SWEBENCH_HARNESS_MODE_PRO:
        return pro_main(argv)
    if mode in (SWEBENCH_HARNESS_MODE_CLASSIC, SWEBENCH_HARNESS_MODE_MULTILINGUAL):
        return classic_main(argv)
    raise SystemExit(
        f"Unknown --harness-mode={mode!r}. Expected one of: classic, multilingual, pro."
    )


def main(argv: Optional[list[str]] = None) -> int:
    """Entrypoint mirror of ``testgenevallite_generate.main``."""

    logging.basicConfig(
        level=os.environ.get("APEX_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if argv is None:
        argv = list(sys.argv[1:])
    if not argv or any(token in ("-h", "--help") for token in argv):
        # Top-level help: print this entry's quick map then proxy.
        sys.stderr.write(
            "swebench_generate routes by --harness-mode:\n"
            "  classic / verified / multilingual  -> apex.evaluation.swebench_codegen_eval\n"
            "  pro                                -> apex.evaluation.swebench_pro_codegen_eval\n\n"
        )
    return _route(list(argv))


if __name__ == "__main__":
    raise SystemExit(main())
