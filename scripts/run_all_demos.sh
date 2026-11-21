#!/usr/bin/env bash
# Submit one independent Slurm job per demo.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_demo.sh" --all "$@"
