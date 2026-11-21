#!/usr/bin/env bash
# Only discover demos, read resource settings, and submit jobs on the frontend.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
if [[ -f "$PROJECT_DIR/.env" ]]; then
    source "$PROJECT_DIR/.env"
    export MAIL_USER="${MAIL_USER:-}"
fi
exec uv run --project "$PROJECT_DIR" --extra demos --extra ml python "$PROJECT_DIR/src/buhito/demo_runner.py" \
    --project "$PROJECT_DIR" --demo-root examples "$@"
