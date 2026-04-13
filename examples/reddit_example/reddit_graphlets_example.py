from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from minervachem.fingerprinters import GraphletFingerprinter
from minervachem.transformers import FingerprintFeaturizer

# Configuration
@dataclass
class DataConfig:
    data_root: str = "./data"
    dataset_name: str = "REDDIT-MULTI-5K"


@dataclass
class FingerprintConfig:
    max_subgraph_size: int = 3
    n_jobs: int = -3          
    chunk_size: str = "auto"   # let FingerprintFeaturizer decide


@dataclass
class TrainConfig:
    train_fraction: float = 0.8
    random_state_split: int = 195
    random_state_model: int = 42
    max_iter: int = 5000

# Data loading & graph construction

def load_reddit_multi_5k_graphs(cfg: DataConfig) -> Tuple[List[nx.Graph], np.ndarray]:
    """
    Load REDDIT-MULTI-5K graphs as NetworkX Graph objects with optional
    node / edge attributes that are useful for graphlet fingerprints.

    The files follow the TU Dortmund GNN benchmark layout.

    Returns
    -------
    graphs : list of nx.Graph
        One NetworkX graph per Reddit thread.
    labels : np.ndarray of shape (n_graphs,)
        Integer graph labels.
    """
    ds_dir = os.path.join(cfg.data_root, cfg.dataset_name)
    edges_path = os.path.join(ds_dir, f"{cfg.dataset_name}.edges")
    graph_idx_path = os.path.join(ds_dir, f"{cfg.dataset_name}.graph_idx")
    graph_labels_path = os.path.join(ds_dir, f"{cfg.dataset_name}.graph_labels")

    # Edges file: each row "src, dst" in 1-based node indices.
    edges_df = pd.read_csv(edges_path, sep=",", header=None, names=["source", "target"])

    # For each node (1..N) we have a graph index telling us which graph it belongs to.
    # We read this as a 1D array; entries are graph indices starting at 1.
    graph_idx = pd.read_csv(graph_idx_path, header=None)[0].values

    # Graph-level labels
    graph_labels = pd.read_csv(graph_labels_path, header=None)[0].values

    # Map: node_id (0-based) -> graph_id (1-based), by construction of graph_idx file.
    node_to_graph = graph_idx

    # Prepare mapping graph_id -> list of node ids (0-based)
    from collections import defaultdict

    graph_to_nodes: Dict[int, List[int]] = defaultdict(list)
    for node_id, g_id in enumerate(node_to_graph):
        graph_to_nodes[g_id].append(node_id)

    # Prepare mapping graph_id -> list of edges (0-based node ids)
    graph_to_edges: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for _, row in edges_df.iterrows():
        src_1b, tgt_1b = int(row["source"]), int(row["target"])
        # Convert to 0-based indices
        src = src_1b - 1
        tgt = tgt_1b - 1
        g_id = node_to_graph[src]
        # Only keep edges inside the same graph
        if node_to_graph[tgt] == g_id:
            graph_to_edges[g_id].append((src, tgt))

    # Create individual NetworkX graphs
    graphs: List[nx.Graph] = []
    for g_id in sorted(graph_to_nodes.keys()):
        G = nx.Graph()
        # Nodes are simple integer indices; you can attach attributes if desired.
        G.add_nodes_from(graph_to_nodes[g_id])
        G.add_edges_from(graph_to_edges[g_id])

        # Example attributes in the spirit of your notebook:
        #   atom_key    = degree
        #   atom_symbol = a single dummy element
        #   bond_key    = 'SINGLE' for all edges
        for node in G.nodes():
            G.nodes[node]["atom_key"] = G.degree[node]
            G.nodes[node]["atom_symbol"] = "A"
        nx.set_edge_attributes(G, "SINGLE", "bond_key")

        graphs.append(G)

    # Labels are 1-based in the original file; scikit-learn is fine with that.
    labels = graph_labels

    return graphs, labels


# ----------------------------------------------------------------------
# Featurization
# ----------------------------------------------------------------------


def build_featurizer(fp_cfg: FingerprintConfig) -> FingerprintFeaturizer:
    """
    Construct a FingerprintFeaturizer that uses GraphletFingerprinter.

    This follows the notebook logic but wraps it in a reusable function.
    """
    fingerprinter = GraphletFingerprinter(max_len=fp_cfg.max_subgraph_size)
    featurizer = FingerprintFeaturizer(
        fingerprinter=fingerprinter,
        verbose=0,
        n_jobs=fp_cfg.n_jobs,
        chunk_size=fp_cfg.chunk_size,
    )
    return featurizer


def featurize_graphs(
    featurizer: FingerprintFeaturizer,
    graphs: List[nx.Graph],
) -> np.ndarray:
    """
    Transform a list of NetworkX graphs into a 2D numpy array of features.
    """
    X = featurizer.transform(graphs)
    # FingerprintFeaturizer may return a scipy sparse matrix;
    # convert to dense numpy array for convenience.
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    return X


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
        multi_class="multinomial",
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
        description="Graphlet fingerprint example on REDDIT-MULTI-5K (minervachem)."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Directory containing the REDDIT-MULTI-5K folder.",
    )
    args = parser.parse_args()

    data_cfg = DataConfig(data_root=args.data_root)
    fp_cfg = FingerprintConfig()
    train_cfg = TrainConfig()

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
    train_labels = labels[train_inds]
    test_labels = labels[test_inds]

    print(f"Train size: {len(train_graphs)}, test size: {len(test_graphs)}")

    print("Building graphlet featurizer...")
    featurizer = build_featurizer(fp_cfg)

    print("Featurizing graphs (this may take a while)...")
    X_train = featurizer.fit_transform(train_graphs)
    X_test = featurizer.transform(test_graphs)

    if not isinstance(X_train, np.ndarray):
        X_train = X_train.toarray()
    if not isinstance(X_test, np.ndarray):
        X_test = X_test.toarray()

    print(f"Feature matrix shapes: X_train={X_train.shape}, X_test={X_test.shape}")

    print("Training classifier...")
    train_and_evaluate(X_train, train_labels, X_test, test_labels, train_cfg)


if __name__ == "__main__":
    main()
