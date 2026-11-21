#!/usr/bin/env bash
# Submit just the Reddit individual-graph benchmark with its 256G default.
# Pass --mem/--time before --, and benchmark options after --.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_demo.sh" \
    --demo examples/reddit_example/reddit_individual_graphs_benchmark.py "$@"
