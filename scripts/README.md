# Demo runners

Run commands from the Buhito repository root. The launchers use `uv run` with
the `demos` and `ml` extras for the submission script and the batch worker. HPC compute nodes must see the same checkout and
virtual environment. Graph rendering requires the system Graphviz `dot` program.

```sh
./scripts/run_demo.sh --list
./scripts/run_all_demos.sh
./scripts/run_demo.sh --demo examples/padre/padre_example.ipynb
./scripts/run_demo.sh --demo examples/plotting_example.py
./scripts/run_demo.sh --demo examples/reddit_example/reddit_graphlets_example.py -- --data-root /absolute/path/to/data
./scripts/run_demo.sh --demo examples/reddit_example/reddit_individual_graphs_benchmark.py -- /absolute/path/to/graph_pickles
```

The shell launchers submit one independent Slurm job per selected demo by default.
`--slurm` remains accepted for compatibility. They require `sbatch`; submission
failure never falls back to local execution. Python demos and notebooks both
run through `demo_job.sh` on compute nodes. The worker refuses to execute without
a Slurm allocation. The frontend runs the submission script through `uv run`;
only the batch worker executes demo code.
`run_all_demos.sh` discovers every `.py` and `.ipynb` recursively in `examples/`,
excluding checkpoints, hidden directories, and old executed copies. All-demo runs
use the demos' default inputs, which must already exist; use `--demo ... -- ...`
for dataset paths and other arguments. A failed Slurm submission
stops submission and reports the failure; jobs already submitted remain queued.

Notebook execution replaces outputs in the original `.ipynb` on success, using
`nbconvert --execute --inplace`. It does not create `_executed` or `.executed`
notebooks. A notebook execution failure leaves its original file unchanged.

Notebook `HPC_RESOURCES` comments set CPU and memory requests. Every notebook
must have a valid marker in its first Markdown cell; Slurm submission stops
before any jobs are queued if a marker is missing or invalid. Other demos default
to 1 CPU and 8G RAM. Override with `--cpus 16 --mem 32G --time 08:00:00`. Set
`--partition`, `--account`, and `--mail-user` as needed; `HPC_PARTITION`,
`HPC_ACCOUNT`, `HPC_TIME`, and `MAIL_USER` are also supported. Slurm output and
errors go to `logs/`. Joblib workers are capped by `SLURM_CPUS_PER_TASK`,
with one native library thread per worker to prevent oversubscription. The scripts load repository `.env` settings when present.

| Demo | Inputs |
| --- | --- |
| `examples/plotting_example.py` | System Graphviz; no dataset |
| `examples/padre/padre_example.ipynb` | Synthetic data generated inside notebook |
| `examples/reddit_example/reddit_graphlets_example.py` | Reddit dataset described in its README |
| `examples/reddit_example/reddit_individual_graphs_benchmark.py` | Labeled NetworkX graph pickles |

The synthetic PADRE notebook requests **4 CPUs and 8G RAM** for its small
synthetic dataset and parallel random forests. Resource requests are configured
budgets, not measured peak requirements; use `--cpus`/`--mem` to tune an actual run.

The batch worker follows the original notebook runner: one node and task,
`standard` QoS, account `w26_pfas_g`, no GPUs, and no requeue. Use `--account`
or `HPC_ACCOUNT` to override the account. Runtime and cache directories use
`SLURM_TMPDIR` (or `/tmp`) inside each job. The frontend requires `uv` and `sbatch`. It runs the submission script through
`uv run` using the project Python, without importing or executing demo code.
