# HPC_RESOURCES: cpus=1 mem=256G
import numpy as np
import networkx as nx
import buhito as bh

from datetime import date
import glob
import pickle

import os
import time
import argparse
import gc
from buhito.depth_graphlet import HashHelper
from pathlib import Path
import statistics as stats
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt


def unique_node_labels(G: nx.Graph, attr: str = "node_label") -> int:
    vals = set()
    for _, data in G.nodes(data=True):
        if attr in data:
            vals.add(data[attr])
    return len(vals)

@dataclass
class BenchRow:
    graph_name: str
    n_nodes: int
    n_edges: int
    n_unique_node_label: int
    max_graphlet_size: int
    algorithm: str
    repeats: int
    time_min_s: float
    time_mean_s: float
    time_std_s: float
    total_graphlets: int

def time_call(fn, *args, repeats: int = 5, warmup: int = 1) -> Tuple[List[float], Any]:
    """Time each call once and retain only its small count histogram.

    Release graphlet locations and the DFS hash cache between repetitions so
    memory use does not grow with warmup/repeat count. Cleanup is not timed.
    """
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup nonnegative")

    def invoke():
        start = time.perf_counter()
        result = fn(*args)
        elapsed = time.perf_counter() - start
        counts = dict(result[1] if len(result) == 2 else result[0])
        del result
        cache_clear = getattr(HashHelper.__call__, 'cache_clear', None)
        if cache_clear is not None:
            cache_clear()
        gc.collect()
        return elapsed, counts

    for _ in range(warmup):
        invoke()
    times = []
    last = None
    for _ in range(repeats):
        elapsed, last = invoke()
        times.append(elapsed)
    return times, last


def bfs_histogram(G, size):
    counts, locations = bh.generate_subgraphs_breadthwise(G, size, return_nodewise=False)
    return locations, counts


def run_benchmark(
    graphs: List[Tuple[str, nx.Graph]],
    max_sizes: List[int],
    repeats: int = 5,
    warmup: int = 1,
    out_dir: str = "benchmark_out",
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, "benchmark_plots")
    os.makedirs(plot_dir, exist_ok=True)

    rows: List[BenchRow] = []

    algos = [
        ("bfs", bfs_histogram),
        ("dfs", bh.generate_subgraphs_depthwise),
    ]

    for temp_tup in graphs:
        graph_name, G = temp_tup
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        n_unique = unique_node_labels(G, "node_label")
        print(f"Benchmarking graph {graph_name}: nodes={n_nodes}, edges={n_edges}, unique_node_label={n_unique}")

        for k in max_sizes:
            for algo_name, fn in algos:
                times, last_result = time_call(fn, G, k, repeats=repeats, warmup=warmup)
                total = sum(last_result.values())

                row = BenchRow(
                    graph_name=graph_name,
                    n_nodes=n_nodes,
                    n_edges=n_edges,
                    n_unique_node_label=n_unique,
                    max_graphlet_size=k,
                    algorithm=algo_name,
                    repeats=repeats,
                    time_min_s=min(times),
                    time_mean_s=stats.mean(times),
                    time_std_s=stats.pstdev(times) if len(times) > 1 else 0.0,
                    total_graphlets=total,
                )
                print(asdict(row))
                rows.append(row)

    df = pd.DataFrame([asdict(r) for r in rows])

    csv_path = os.path.join(out_dir, "benchmark_results.csv")
    df.to_csv(csv_path, index=False)

    txt_path = os.path.join(out_dir, "benchmark_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Graphlet Enumeration Benchmark Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Graphs: {len(graphs)}\n")
        f.write(f"Max sizes: {max_sizes}\n")
        f.write(f"Repeats: {repeats}, Warmup: {warmup}\n\n")

        for gname, _G in graphs:
            sub = df[df["graph_name"] == gname].copy()
            if sub.empty:
                continue
            meta = sub.iloc[0]
            f.write(f"Graph: {gname}\n")
            f.write(f"  nodes={int(meta.n_nodes)}, edges={int(meta.n_edges)}, unique_node_label={int(meta.n_unique_node_label)}\n")

            f.write("  Results (time_mean_s):\n")
            pivot = sub.pivot_table(
                index="max_graphlet_size",
                columns="algorithm",
                values="time_mean_s",
                aggfunc="mean",
            ).sort_index()

            totals = sub.pivot_table(
                index="max_graphlet_size",
                columns="algorithm",
                values="total_graphlets",
                aggfunc="mean",
            ).sort_index()

            f.write(pivot.to_string(float_format=lambda x: f"{x:.6f}") + "\n")
            f.flush()
            f.write("  Total graphlets:\n")
            f.flush()
            f.write(totals.to_string(float_format=lambda x: f"{x:.0f}") + "\n\n")
            f.flush()

    for gname, _G in graphs:
        sub = df[df["graph_name"] == gname].copy()
        if sub.empty:
            continue

        # Plot mean time by k for BFS/DFS
        plt.figure()
        for algo in ["bfs", "dfs"]:
            s = sub[sub["algorithm"] == algo].sort_values("max_graphlet_size")
            plt.plot(s["max_graphlet_size"], s["time_mean_s"], marker="o", label=algo)

        plt.xlabel("max graphlet size")
        plt.ylabel("time mean (s)")
        plt.title(f"Benchmark: {gname}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{gname}_time_vs_maxsize.png"), dpi=200)
        plt.close()

        # Plot total graphlets vs k (pick BFS row, but any is fine)
        plt.figure()
        s = sub[sub["algorithm"] == "bfs"].sort_values("max_graphlet_size")
        plt.plot(s["max_graphlet_size"], s["total_graphlets"], marker="o")
        plt.xlabel("max graphlet size")
        plt.ylabel("total graphlets")
        plt.title(f"{gname}")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{gname}_numgraphlets_vs_maxsize.png"), dpi=200)
        plt.close()

    agg = (
        df.groupby(["algorithm", "max_graphlet_size"])["time_mean_s"]
          .mean()
          .reset_index()
          .sort_values("max_graphlet_size")
    )

    plt.figure()
    for algo in ["bfs", "dfs"]:
        s = agg[agg["algorithm"] == algo]
        plt.plot(s["max_graphlet_size"], s["time_mean_s"], marker="o", label=algo)
    plt.xlabel("max graphlet size")
    plt.ylabel("mean time mean (s) over graphs")
    plt.title("Aggregate benchmark (mean over graphs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "aggregate_time_vs_maxsize.png"), dpi=200)
    plt.close()

    return df

def load_individual_benchmark_graphs(bm_dir, limit=None) -> Dict[int, nx.Graph]:
    bm_locs = sorted(glob.glob(os.path.join(bm_dir, "*.pkl")))
    if not bm_locs:
        raise FileNotFoundError(
            f"No .pkl graph files found in '{os.path.abspath(bm_dir)}'. "
            "Pass the directory containing the Reddit benchmark graph pickles "
            "as the first command-line argument."
        )

    if limit is not None:
        bm_locs = bm_locs[:limit]

    individual_graphs = {}
    for i, loc in enumerate(bm_locs):
        with open(loc, "rb") as f:
            G = pickle.load(f)
        individual_graphs[i] = G
    return individual_graphs

def make_dated_out_dir(out_dir: str, prefix: str | None = None) -> str:
    d = date.today().isoformat()
    base_name = f"{prefix}_{d}" if prefix else d

    candidate = os.path.join(out_dir, base_name)
    if not os.path.exists(candidate):
        os.makedirs(candidate)
        return candidate

    i = 1
    while True:
        candidate = os.path.join(out_dir, f"{base_name}_run{i}")
        if not os.path.exists(candidate):
            os.makedirs(candidate)
            return candidate
        i += 1

def main():
    parser = argparse.ArgumentParser(description="Benchmark BFS/DFS graphlet enumeration on Reddit graphs")
    parser.add_argument('graph_dir', nargs='?', type=Path,
                        default=Path(__file__).resolve().parent / 'reddit_graphs_benchmarks')
    parser.add_argument('--max-sizes', nargs='+', type=int, default=[3, 4])
    parser.add_argument('--max-graphs', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--smoke', action='store_true', help='Full selected graphs through size 3, one repeat, no warmup')
    args = parser.parse_args()
    if args.max_graphs < 1 or any(size < 1 for size in args.max_sizes) or args.repeats < 1 or args.warmup < 0:
        parser.error('Graph counts, sizes, and repeats must be positive; warmup must be nonnegative')
    graphs = load_individual_benchmark_graphs(args.graph_dir, limit=args.max_graphs)
    out_dir = args.out_dir or make_dated_out_dir(str(args.graph_dir.parent), prefix="reddit_benchmark_out")
    run_benchmark(
        [(f'reddit_graph_{i}', graph) for i, graph in graphs.items()],
        max_sizes=[3] if args.smoke else args.max_sizes,
        repeats=1 if args.smoke else args.repeats,
        warmup=0 if args.smoke else args.warmup,
        out_dir=str(out_dir),
    )


if __name__ == '__main__':
    main()
