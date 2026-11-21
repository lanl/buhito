#!/usr/bin/env python3
"""Run a bounded local Reddit classification smoke test; never submits Slurm jobs.

Uses five complete graphs per class plus the two highest size-3 workload bounds.
Monitors total RSS of the worker process tree and saves evidence under logs/.
"""
import argparse
from array import array
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import resource
import runpy
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]


def choose_graphs(data_root):
    directory = data_root / 'REDDIT-MULTI-5K'
    ids = array('I', (int(line) for line in (directory / 'REDDIT-MULTI-5K.graph_idx').open()))
    degree = array('I', [0]) * len(ids)
    with (directory / 'REDDIT-MULTI-5K.edges').open() as handle:
        for line in handle:
            source = int(line.split(',', 1)[0]) - 1
            degree[source] += 1
    # The TU edge file can repeat adjacency records. These are conservative
    # workload bounds for selecting stress graphs, not exact graphlet counts.
    bounds = Counter()
    for graph_id, value in zip(ids, degree):
        bounds[graph_id] += math.comb(value, 2) + 1
    labels = [int(line) for line in (directory / 'REDDIT-MULTI-5K.graph_labels').open()]
    chosen = set()
    classes = defaultdict(list)
    for index, label in enumerate(labels, 1):
        classes[label].append(index)
    for graph_ids in classes.values():
        chosen.update(graph_ids[:5])
    worst = bounds.most_common(2)
    chosen.update(graph_id for graph_id, _ in worst)
    return sorted(chosen), worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=PROJECT / 'data')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--rss-cap-gib', type=float, default=8)
    parser.add_argument('--report', type=Path, default=PROJECT / 'logs/reddit_classification_smoke.json')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--graph-ids', nargs='+', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        # Address-space limits are inherited by loky workers as a second guard.
        cap = 12 * 1024**3
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        sys.path.insert(0, str(PROJECT / 'src'))
        sys.argv = ['reddit_graphlets_example.py', '--data-root', str(args.data_root),
                    '--smoke', '--n-jobs', str(args.workers), '--graph-ids', *map(str, args.graph_ids)]
        runpy.run_path(str(PROJECT / 'examples/reddit_example/reddit_graphlets_example.py'), run_name='__main__')
        return
    import psutil
    print('Selecting complete stress graphs from the actual Reddit dataset...', flush=True)
    graph_ids, worst = choose_graphs(args.data_root)
    print(f'Selected {len(graph_ids)} graphs; largest workload bounds: {worst}', flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    log = args.report.with_suffix('.log')
    env = dict(os.environ, PYTHONPATH=str(PROJECT / 'src'), PYTHONUNBUFFERED='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               MPLCONFIGDIR=str(args.report.parent / 'classification_matplotlib'))
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--data-root', str(args.data_root),
               '--workers', str(args.workers), '--graph-ids', *map(str, graph_ids)]
    peak = 0
    started = time.monotonic()
    failure = None
    with log.open('w') as output:
        process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        monitored = psutil.Process(process.pid)
        last_update = started
        while process.poll() is None:
            rss = 0
            try:
                for child in [monitored, *monitored.children(recursive=True)]:
                    try:
                        rss += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
            except psutil.NoSuchProcess:
                pass
            peak = max(peak, rss)
            elapsed = time.monotonic() - started
            if rss > args.rss_cap_gib * 1024**3:
                failure = 'Aggregate RSS exceeded the local smoke cap'
            elif elapsed > args.timeout:
                failure = 'Local smoke timeout exceeded'
            if failure:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            if time.monotonic() - last_update > 15:
                print(f'Smoke running: {elapsed:.0f}s, process-tree RSS {rss / 1024**3:.3f} GiB', flush=True)
                last_update = time.monotonic()
            time.sleep(0.2)
    report = dict(graph_ids=graph_ids, worst_workload_bounds=worst, workers=args.workers,
                  elapsed_seconds=time.monotonic() - started, peak_process_tree_rss_gib=peak / 1024**3,
                  returncode=process.returncode, error=failure, output_log=str(log),
                  note='Subset smoke including stress graphs; not a full-dataset peak memory measurement')
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if failure or process.returncode:
        raise SystemExit(f'Smoke failed; see {log}')


if __name__ == '__main__':
    main()
