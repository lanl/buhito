#!/usr/bin/env bash
# One independent job per demo with --slurm; runs sequentially otherwise.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_demo.sh" --all "$@"
