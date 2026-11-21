# Reddit download, preparation, and HPC upload

All new tooling lives in `scripts/hpc/`. Downloaded/prepared inputs live in the
ignored dataset directories expected by the demos.

The example authors identify the TU Dortmund graph kernel benchmark collection.
The authoritative listing is https://chrsmrrs.github.io/datasets/docs/datasets/.
The downloader requests exactly one URL:

https://www.chrsmrrs.com/graphkerneldatasets/REDDIT-MULTI-5K.zip

Standard HTTP redirects are followed; the resolved URL is printed at runtime.
No other dataset or executable is downloaded. NetworkX must already be installed;
`uv run` can separately install project dependencies if its environment is missing.

From the Buhito repository root:

```bash
# Print the source URL and output paths; no network or writes.
uv run python scripts/hpc/download_reddit_data.py --dry-run

# Download and prepare the data (run this yourself).
uv run python scripts/hpc/download_reddit_data.py

# Or prepare an archive/directory you downloaded yourself.
uv run python scripts/hpc/prepare_reddit_data.py --source /path/to/REDDIT-MULTI-5K.zip

# Local upload inventory only: no connection.
bash scripts/hpc/upload_data.sh --list

# Preview against HPC (connects; destination directories must already exist).
bash scripts/hpc/upload_data.sh --dry-run

# Upload missing files to ~/buhito on se.
bash scripts/hpc/upload_data.sh
```

Preparation maps the official TU filenames to the existing example filenames:

| Archive member basename | Local path under Buhito |
| --- | --- |
| `REDDIT-MULTI-5K_A.txt` | `data/REDDIT-MULTI-5K/REDDIT-MULTI-5K.edges` |
| `REDDIT-MULTI-5K_graph_indicator.txt` | `data/REDDIT-MULTI-5K/REDDIT-MULTI-5K.graph_idx` |
| `REDDIT-MULTI-5K_graph_labels.txt` | `data/REDDIT-MULTI-5K/REDDIT-MULTI-5K.graph_labels` |

It also creates the first two NetworkX benchmark graphs under
`examples/reddit_example/reddit_graphs_benchmarks/`, with degree node labels and
constant edge labels. This matches the demo's default first-two-graphs benchmark.
Use `--benchmark-count N` to prepare more. Existing local prepared files are kept.

Uploads preserve the same relative paths under `~/buhito`. They use
`rsync --ignore-existing`: existing remote files are never replaced and no
remote files are deleted. Missing required local inputs cause a nonzero exit
status; any other available inputs are still uploaded. An existing remote copy
of an input with no local source is left untouched and cannot be verified by
this local inventory.

Options: `--remote ch`, `--remote-dir buhito`, `--source /path/to/buhito`,
`--ssh-command 'ssh -o ...'`. SSH defaults to Windows `ssh.exe` on WSL, otherwise
`ssh`. Authentication is handled by your SSH client; complete smart-card prompts
locally. Both endpoints need rsync. No upload or download runs automatically.
