#!/usr/bin/env bash
set -euo pipefail
unset BASH_ENV PYTHONHOME
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec "${PYTHON:-python}" -u scripts/run_experiment.py train "$@"
