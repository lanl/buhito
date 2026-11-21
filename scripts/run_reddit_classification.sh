#!/usr/bin/env bash
# Submit only the Reddit classification demo: 4 CPUs, 128G, at most 4 workers.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_demo.sh" \
    --demo examples/reddit_example/reddit_graphlets_example.py "$@"
