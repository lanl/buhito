#!/usr/bin/env bash
# Run any demo locally, or pass --slurm to submit it to the HPC server.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
if [[ -f "$PROJECT_DIR/.env" ]]; then
    source "$PROJECT_DIR/.env"
    export MAIL_USER="${MAIL_USER:-}"
fi
exec uv run --project "$PROJECT_DIR" --extra demos --extra ml python -m buhito.demo_runner \
    --project "$PROJECT_DIR" --demo-root examples "$@"
