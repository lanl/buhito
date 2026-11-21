from __future__ import annotations

# HPC_RESOURCES: cpus=4 mem=128G

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import spmatrix

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from buhito.featurizers.dfs_graphlet_featurizer import DFSGraphletFeaturizer
from buhito.transformers import GraphletTransformer

# Configuration
@dataclass
class DataConfig:
    data_root: str = "./data"
    dataset_name: str = "REDDIT-MULTI-5K"
    graph_ids: Tuple[int, ...] | None = None


@dataclass
class FingerprintConfig:
    max_subgraph_size: int = 3
    n_jobs: int = min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    chunk_size: int = 1  # keep only one graph per dispatched task


@dataclass
class TrainConfig:
    train_fraction: float = 0.8
    random_state_split: int = 195
    random_state_model: int = 42
    max_iter: int = 5000

# Data loading & graph construction

def load_reddit_multi_5k_graphs(cfg: DataConfig) -> Tuple[List[nx.Graph], np.ndarray]:
    """Load graphs without retaining a full edge table and duplicate edge lists.

    Optional one-based graph IDs select a smoke subset while preserving global
    node IDs, node degrees, and graph labels for those complete graphs.
    """
    ds_dir = os.path.join(cfg.data_root, cfg.dataset_name)
    graph_idx = pd.read_csv(os.path.join(ds_dir, f'{cfg.dataset_name}.graph_idx'),
                            header=None, dtype=np.int32)[0].to_numpy()
    labels = pd.read_csv(os.path.join(ds_dir, f'{cfg.dataset_name}.graph_labels'), header=None)[0].to_numpy()
    available = set(map(int, np.unique(graph_idx)))
    selected = available if cfg.graph_ids is None else set(cfg.graph_ids)
    if not selected or not selected <= available:
        raise ValueError('Select valid one-based graph IDs')
    ordered = sorted(selected)
    by_id = {graph_id: nx.Graph() for graph_id in ordered}
    for node, graph_id in enumerate(graph_idx):
        graph = by_id.get(int(graph_id))
        if graph is not None:
            graph.add_node(node)
    chunks = pd.read_csv(os.path.join(ds_dir, f'{cfg.dataset_name}.edges'),
                         header=None, names=['source', 'target'], dtype=np.int32,
                         chunksize=100_000)
    for chunk in chunks:
        for source, target in chunk.itertuples(index=False, name=None):
            source, target = source - 1, target - 1
            graph_id = int(graph_idx[source])
            graph = by_id.get(graph_id)
            if graph is not None and graph_idx[target] == graph_id:
                graph.add_edge(source, target)
    graphs = list(by_id.values())
    for graph in graphs:
        nx.set_node_attributes(graph, dict(graph.degree), 'node_label')
        nx.set_edge_attributes(graph, 'link', 'edge_label')
    return graphs, labels[np.asarray(ordered) - 1]


# ----------------------------------------------------------------------
# Featurization
# ----------------------------------------------------------------------


def build_featurizer(fp_cfg: FingerprintConfig) -> GraphletTransformer:
    """
    Construct a GraphletTransformer that uses DFSGraphletFeaturizer.

    This follows the notebook logic but wraps it in a reusable function.
    """
    graphlet_featurizer = DFSGraphletFeaturizer(max_len=fp_cfg.max_subgraph_size)
    featurizer = GraphletTransformer(
        featurizer=graphlet_featurizer,
        verbose=0,
        n_jobs=fp_cfg.n_jobs,
        chunk_size=fp_cfg.chunk_size,
        retain_bit_info=False,
    )
    return featurizer


def featurize_graphs(
    featurizer: GraphletTransformer,
    graphs: List[nx.Graph],
) -> spmatrix:
    """
    Transform graphs into a sparse matrix of graphlet counts.
    """
    return featurizer.transform(graphs)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: TrainConfig,
) -> None:
    """
    Fit a multinomial logistic regression classifier and print metrics.
    """
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=cfg.max_iter,
        random_state=cfg.random_state_model,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("=== Classification report ===")
    print(classification_report(y_test, y_pred, digits=3))

    print("=== Confusion matrix ===")
    print(confusion_matrix(y_test, y_pred))


# ----------------------------------------------------------------------
# Main script
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Graphlet fingerprint example on REDDIT-MULTI-5K (buhito)."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Directory containing the REDDIT-MULTI-5K folder.",
    )
    parser.add_argument('--n-jobs', type=int, default=FingerprintConfig().n_jobs,
                        help='Concurrent graph workers; default at most 4')
    parser.add_argument('--graph-ids', type=int, nargs='+', help='Selected complete graphs, using one-based IDs')
    parser.add_argument('--max-iter', type=int, default=5000)
    parser.add_argument('--smoke', action='store_true', help='Use ten graphs per class and at most 100 solver iterations')
    args = parser.parse_args()
    if args.n_jobs < 1:
        parser.error('--n-jobs must be positive; use an explicit bounded worker count')
    if os.environ.get('SLURM_CPUS_PER_TASK') and args.n_jobs > int(os.environ['SLURM_CPUS_PER_TASK']):
        parser.error('--n-jobs exceeds the Slurm CPU allocation')

    data_cfg = DataConfig(data_root=args.data_root)
    fp_cfg = FingerprintConfig(n_jobs=args.n_jobs)
    train_cfg = TrainConfig(max_iter=min(100, args.max_iter) if args.smoke else args.max_iter)
    if args.graph_ids:
        data_cfg.graph_ids = tuple(args.graph_ids)
    elif args.smoke:
        label_path = os.path.join(data_cfg.data_root, data_cfg.dataset_name,
                                  f'{data_cfg.dataset_name}.graph_labels')
        all_labels = pd.read_csv(label_path, header=None)[0].to_numpy()
        data_cfg.graph_ids = tuple(int(index + 1) for label in np.unique(all_labels)
                                  for index in np.flatnonzero(all_labels == label)[:10])

    print("Loading graphs...")
    graphs, labels = load_reddit_multi_5k_graphs(data_cfg)
    n_graphs = len(graphs)
    print(f"Loaded {n_graphs} graphs.")

    # Train/test split over graphs
    all_indices = np.arange(n_graphs)
    train_inds, test_inds = train_test_split(
        all_indices,
        train_size=train_cfg.train_fraction,
        random_state=train_cfg.random_state_split,
        stratify=labels,
    )

    train_graphs = [graphs[i] for i in train_inds]
    test_graphs = [graphs[i] for i in test_inds]
    del graphs  # The train/test lists now own the selected input graphs.
    train_labels = labels[train_inds]
    test_labels = labels[test_inds]

    print(f"Train size: {len(train_graphs)}, test size: {len(test_graphs)}")

    print("Building graphlet featurizer...")
    featurizer = build_featurizer(fp_cfg)

    print("Featurizing graphs (this may take a while)...")
    X_train = featurizer.fit_transform(train_graphs)
    del train_graphs
    X_test = featurizer.transform(test_graphs)
    del test_graphs

    print(f"Count-only workers: {fp_cfg.n_jobs}; graphlet locations are not retained.")
    print(f"Feature matrix shapes: X_train={X_train.shape}, X_test={X_test.shape}")

    print("Training classifier...")
    train_and_evaluate(X_train, train_labels, X_test, test_labels, train_cfg)


if __name__ == "__main__":
    main()
