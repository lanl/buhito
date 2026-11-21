"""Depth-first graphlet features for arbitrary NetworkX graphs."""
import networkx as nx
from collections import defaultdict
from buhito.depth_graphlet import (
    HashHelper, active_neighbors, generate_subgraphs_depthwise,
    generate_subgraphs_from_node, search_subgraphs_from_cluster,
)


class DFSGraphletFeaturizer:
    def __init__(self, max_len=3, node_key="node_label", edge_key="edge_label"):
        self.size = max_len
        self.node_key = node_key
        self.edge_key = edge_key
        self.verbose = False

    def __call__(self, G):
        return compute_subgraphs_dfs(G, self.size, node_key=self.node_key, edge_key=self.edge_key)


def compute_subgraphs_dfs(G, maxlen, *, node_key="node_label", edge_key="edge_label"):
    if not isinstance(G, nx.Graph):
        raise ValueError(f"Expected NetworkX graph, got {type(G)}")
    subsets, counts = generate_subgraphs_depthwise(
        G, maxlen, node_key=node_key, edge_key=edge_key,
    )
    locations = defaultdict(list)
    for nodes, bit in subsets:
        locations[(len(nodes), bit)].append(nodes)
    return dict(sorted(counts.items())), dict(sorted(locations.items()))
