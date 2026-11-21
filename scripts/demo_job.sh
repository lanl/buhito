#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --qos=standard
#SBATCH --account=w26_pfas_g
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --no-requeue

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: demo_job.sh must be submitted through run_demo.sh or run_all_demos.sh." >&2
    exit 1
fi
PROJECT_DIR="$1"
DEMO_ROOT="$2"
shift 2
export PYTHONNOUSERSITE=1
RUNTIME_BASE="${SLURM_TMPDIR:-/tmp}/buhito-${SLURM_JOB_ID}"
export JUPYTER_RUNTIME_DIR="$RUNTIME_BASE/jupyter"
export IPYTHONDIR="$RUNTIME_BASE/ipython"
export MPLCONFIGDIR="$RUNTIME_BASE/matplotlib"
export XDG_CACHE_HOME="$RUNTIME_BASE/cache"
export JOBLIB_TEMP_FOLDER="$RUNTIME_BASE/joblib"
mkdir -p "$JUPYTER_RUNTIME_DIR" "$IPYTHONDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$JOBLIB_TEMP_FOLDER"
cd "$PROJECT_DIR"
# Prepare the environment and execute the selected demo only on the compute node.
exec uv run --project "$PROJECT_DIR" --extra demos --extra ml python -m buhito.demo_worker \
    --project "$PROJECT_DIR" --demo-root "$DEMO_ROOT" "$@"
