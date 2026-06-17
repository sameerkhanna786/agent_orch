#!/usr/bin/env bash
set -euo pipefail

python -m apex.evaluation.runners.testgenevallite "$@"
