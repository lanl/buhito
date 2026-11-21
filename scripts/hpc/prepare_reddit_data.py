#!/usr/bin/env python3
"""Prepare the official TUDataset Reddit archive/directory for Buhito demos."""
import argparse
from array import array
from pathlib import Path, PurePosixPath
import pickle
import shutil
import zipfile

DATASET = 'REDDIT-MULTI-5K'
PROJECT = Path(__file__).resolve().parents[2]
FILES = {'edges': '_A.txt', 'graph_idx': '_graph_indicator.txt', 'graph_labels': '_graph_labels.txt'}


def copy_missing(source, destination):
    if destination.exists():
        print(f'Keeping existing file: {destination}')
        return
    temporary = destination.with_name(destination.name + '.part')
    try:
        with temporary.open('wb') as output:
            shutil.copyfileobj(source, output)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(source, output, benchmark_dir, benchmark_count=2):
    if benchmark_count < 1:
        raise ValueError('benchmark_count must be at least 1')
    # Import before changing files, so missing dependencies fail clearly.
    import networkx as nx

    if not source.exists():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        for suffix, original in FILES.items():
            matches = sorted(source.rglob(DATASET + original))
            converted = source / f'{DATASET}.{suffix}'
            if converted.is_file():
                matches = [converted]
            if len(matches) != 1:
                raise ValueError(f'Expected one {DATASET + original} under {source}')
            with matches[0].open('rb') as handle:
                copy_missing(handle, output / f'{DATASET}.{suffix}')
    else:
        with zipfile.ZipFile(source) as archive:
            for suffix, original in FILES.items():
                matches = [name for name in archive.namelist() if PurePosixPath(name).name == DATASET + original]
                if len(matches) != 1:
                    raise ValueError(f'Archive must contain exactly one {DATASET + original}')
                # Copy named members only; do not extract archive-provided paths.
                with archive.open(matches[0]) as handle:
                    copy_missing(handle, output / f'{DATASET}.{suffix}')
    with (output / f'{DATASET}.graph_labels').open() as handle:
        graph_count = sum(1 for line in handle if line.strip())
    graph_ids = array('I')
    graphs = {index: nx.Graph() for index in range(1, min(graph_count, benchmark_count) + 1)}
    with (output / f'{DATASET}.graph_idx').open() as handle:
        for node, line in enumerate(handle):
            graph_id = int(line)
            if graph_id < 1 or graph_id > graph_count:
                raise ValueError(f'Invalid graph ID at node {node + 1}')
            graph_ids.append(graph_id)
            if graph_id in graphs:
                graphs[graph_id].add_node(node)
    with (output / f'{DATASET}.edges').open() as handle:
        for line in handle:
            a, b = (int(value.strip()) - 1 for value in line.split(','))
            if not (0 <= a < len(graph_ids) and 0 <= b < len(graph_ids)):
                raise ValueError('Edge references a node outside the graph indicator')
            graph_id = graph_ids[a]
            if graph_ids[b] != graph_id:
                raise ValueError('Edge crosses graph boundaries')
            if graph_id in graphs:
                graphs[graph_id].add_edge(a, b)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    for graph_id, graph in graphs.items():
        nx.set_node_attributes(graph, dict(graph.degree), 'node_label')
        nx.set_edge_attributes(graph, 'link', 'edge_label')
        target = benchmark_dir / f'reddit_graph_{graph_id - 1:04d}.pkl'
        if target.exists():
            print(f'Keeping existing file: {target}')
            continue
        temporary = target.with_name(target.name + '.part')
        try:
            with temporary.open('wb') as handle:
                pickle.dump(graph, handle, protocol=4)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'Prepared {target}: {len(graph)} nodes, {graph.number_of_edges()} edges')
    print(f'Classification data: {output}\nBenchmark graphs: {benchmark_dir}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Downloaded ZIP or extracted TUDataset directory')
    parser.add_argument('--output', type=Path, default=PROJECT / 'data' / DATASET)
    parser.add_argument('--benchmark-dir', type=Path, default=PROJECT / 'examples/reddit_example/reddit_graphs_benchmarks')
    parser.add_argument('--benchmark-count', type=int, default=2, help='Number of benchmark graphs to prepare (default: 2, matching the benchmark demo)')
    args = parser.parse_args()
    prepare(args.source.expanduser(), args.output, args.benchmark_dir, args.benchmark_count)


if __name__ == '__main__':
    main()
