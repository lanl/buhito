import numpy as np
import networkx as nx
import buhito as bh

from datetime import date
import glob
import pickle

import os
import time
import sys
import statistics as stats
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt


def unique_atom_keys(G: nx.Graph, attr: str = "atom_key") -> int:
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
    n_unique_atom_key: int
    max_graphlet_size: int
    algorithm: str  
    repeats: int
    time_min_s: float
    time_mean_s: float
    time_std_s: float
    total_graphlets: int

def time_call(fn, *args, repeats: int = 5, warmup: int = 1) -> Tuple[List[float], Any]:
    """
    Times fn(*args) multiple times.
    Returns (times, last_result).
    """
    # warmup first
    last = None
    for _ in range(warmup):
        try:
            _, last = fn(*args)
            runopt = 1
        except:
            last, _, _, _ = fn(*args)
            runopt = 2

    times = []
    for _ in range(repeats):
        if runopt == 1:
            t0 = time.perf_counter()
            _, last = fn(*args)
            t1 = time.perf_counter()
        elif runopt == 2:
            t0 = time.perf_counter()
            last, _, _, _ = fn(*args)
            t1 = time.perf_counter()
        else:       
            print("Error: runopt not set correctly.")
            break
        times.append(t1 - t0)

    return times, last

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
        ("bfs", bh.generate_subgraphs_breadthwise),
        ("dfs", bh.generate_subgraphs_depthwise),
    ]

    for temp_tup in graphs:
        graph_name, G = temp_tup
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        n_unique = unique_atom_keys(G, "atom_key")
        print(f"Benchmarking graph {graph_name}: nodes={n_nodes}, edges={n_edges}, unique_atom_key={n_unique}")

        for k in max_sizes:
            for algo_name, fn in algos:
                times, last_result = time_call(fn, G, k, repeats=repeats, warmup=warmup)
                total = sum(last_result.values())

                row = BenchRow(
                    graph_name=graph_name,
                    n_nodes=n_nodes,
                    n_edges=n_edges,
                    n_unique_atom_key=n_unique,
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
            f.write(f"  nodes={int(meta.n_nodes)}, edges={int(meta.n_edges)}, unique_atom_key={int(meta.n_unique_atom_key)}\n")

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

def load_individual_benchmark_graphs(bm_dir) -> Dict[int, nx.Graph]:
    bm_locs = glob.glob(os.path.join(bm_dir, "*.pkl"))
    # load all graphs
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

if __name__ == "__main__":
    bm_dir = os.path.join("examples", "reddit_example", "reddit_graphs_benchmarks")
    bm_dir = str(sys.argv[1]) if len(sys.argv) > 1 else bm_dir
    individual_graphs = load_individual_benchmark_graphs(bm_dir)

    graphs = [(f"reddit_graph_{i}", G) for i, G in individual_graphs.items()][:2]  # limit to first 2 graphs for testing
    out_dir = make_dated_out_dir(os.path.dirname(bm_dir), prefix="reddit_benchmark_out")

    max_sizes = [3, 4]
    df = run_benchmark(
        graphs=graphs,
        max_sizes=max_sizes,
        repeats=3,
        warmup=1,
        out_dir=out_dir,
    )   