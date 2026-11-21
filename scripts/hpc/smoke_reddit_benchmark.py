#!/usr/bin/env python3
"""Bounded local memory smoke test of the real Reddit benchmark inputs.

No Slurm submission or SSH connections. Each case runs in a fresh process with
an address-space cap and timeout. Size-4 projections are estimates, not a full run.
"""
import argparse
import json
import math
import os
from pathlib import Path
import pickle
import resource
import runpy
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]


def candidate_bound(graph):
    degrees = dict(graph.degree)
    return (len(graph) + graph.number_of_edges()
            + sum(math.comb(degree, 2) for degree in degrees.values())
            + sum(math.comb(degree, 3) for degree in degrees.values())
            + sum((degrees[u] - 1) * (degrees[v] - 1) for u, v in graph.edges))


def worker(args):
    cap = int(args.memory_cap_gib * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    sys.path.insert(0, str(PROJECT / 'src'))
    module = runpy.run_path(str(PROJECT / 'examples/reddit_example/reddit_individual_graphs_benchmark.py'))
    with args.graph.open('rb') as handle:
        graph = pickle.load(handle)
    original_bound = candidate_bound(graph)
    if args.neighbors:
        hub = max(graph, key=graph.degree)
        graph = graph.subgraph([hub, *sorted(graph.neighbors(hub))[:args.neighbors]]).copy()
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    fn = module['bfs_histogram'] if args.algorithm == 'bfs' else module['bh'].generate_subgraphs_depthwise
    started = time.perf_counter()
    _, counts = module['time_call'](fn, graph, args.size, repeats=1, warmup=0)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    total = sum(counts.values())
    incremental_gib = max(0, peak - baseline) / 1024**2
    print(json.dumps(dict(graph=args.graph.name, algorithm=args.algorithm, size=args.size,
                          nodes=len(graph), edges=graph.number_of_edges(), graphlets=total,
                          seconds=time.perf_counter() - started, peak_rss_gib=peak / 1024**2,
                          baseline_rss_gib=baseline / 1024**2,
                          full_graph_size4_candidate_bound=original_bound,
                          size4_projected_peak_gib=(baseline / 1024**2 + incremental_gib * original_bound / max(1, total)) if args.size == 4 else None)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph-dir', type=Path, default=PROJECT / 'examples/reddit_example/reddit_graphs_benchmarks')
    parser.add_argument('--memory-cap-gib', type=float, default=12)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--report', type=Path, default=PROJECT / 'logs/reddit_memory_smoke.json')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--graph', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--algorithm', choices=['bfs', 'dfs'], help=argparse.SUPPRESS)
    parser.add_argument('--size', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--neighbors', type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    files = sorted(args.graph_dir.glob('*.pkl'))[:2]
    if not files:
        parser.error('No benchmark graph pickles found')
    cases = [(files[0], 3, 0), (files[0], 4, 64), (files[0], 4, 96)]
    if len(files) > 1:
        cases.append((files[1], 4, 0))
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               MPLCONFIGDIR=str(args.report.parent / 'matplotlib'), MPLBACKEND='Agg')
    rows = []
    args.report.parent.mkdir(parents=True, exist_ok=True)
    for graph, size, neighbors in cases:
        for algorithm in ['bfs', 'dfs']:
            print(f'Smoke: {graph.name}, {algorithm}, size={size}, hub neighbors={neighbors or "full graph"}', flush=True)
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--graph', str(graph),
                       '--algorithm', algorithm, '--size', str(size), '--neighbors', str(neighbors),
                       '--memory-cap-gib', str(args.memory_cap_gib)]
            try:
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=args.timeout)
                if result.returncode:
                    raise RuntimeError(result.stderr[-3000:] or f'Exit status {result.returncode}')
                row = json.loads(result.stdout.strip().splitlines()[-1])
                rows.append(row)
                print(json.dumps(row), flush=True)
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                rows.append(dict(graph=graph.name, algorithm=algorithm, size=size, error=str(error)))
            args.report.write_text(json.dumps(rows, indent=2) + '\n')
    if any('error' in row for row in rows):
        raise SystemExit(f'Some smoke cases failed; see {args.report}')
    print(f'Smoke completed. Report: {args.report}')


if __name__ == '__main__':
    main()
