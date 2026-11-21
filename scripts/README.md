# Demo runners

Run commands from the Buhito repository root. `uv` prepares the environment with
the `demos` and `ml` extras. HPC compute nodes must see the same checkout and
virtual environment. Graph rendering requires the system Graphviz `dot` program.

```sh
./scripts/run_demo.sh --list
./scripts/run_all_demos.sh --slurm
./scripts/run_demo.sh --demo examples/padre/padre_example.ipynb --slurm
./scripts/run_demo.sh --demo examples/plotting_example.py --slurm
./scripts/run_demo.sh --demo examples/reddit_example/reddit_graphlets_example.py --slurm -- --data-root /absolute/path/to/data
./scripts/run_demo.sh --demo examples/reddit_example/reddit_individual_graphs_benchmark.py --slurm -- /absolute/path/to/graph_pickles
```

`--slurm` submits one independent job per selected demo. Omit it to execute locally.
`run_all_demos.sh` discovers every `.py` and `.ipynb` recursively in `examples/`,
excluding checkpoints, hidden directories, and old executed copies. All-demo runs
use the demos' default inputs, which must already exist; use `--demo ... -- ...`
for dataset paths and other arguments. Failed local demos are reported with a
nonzero exit status after the remaining demos have run. A failed Slurm submission
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
