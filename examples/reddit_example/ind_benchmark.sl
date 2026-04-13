#!/bin/bash
#
#SBATCH --job-name=ind_bm
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=amd-rome
#
#SBATCH --mail-user=ypimonova@lanl.gov
#SBATCH --mail-type=BEGIN,END,FAIL
#
#SBATCH --qos=normal
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=5200M

set -euo pipefail

# module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate buh

BASE="/projects/un_graphlets/software/buhito/examples"
CURRENTBM="reddit_example/reddit_graphs_benchmarks"
WORKDIR="$BASE/$CURRENTBM"

cd "$SLURM_SUBMIT_DIR"

echo "Running on host: $(hostname)"
echo "Workdir: $WORKDIR"
echo "Job ID: $SLURM_JOB_ID"

timefile="$SLURM_SUBMIT_DIR/time_${SLURM_JOB_ID}_ind_benchmark.txt"
/usr/bin/time -v -o "$timefile" srun python reddit_individual_graphs_benchmark.py "$WORKDIR"



