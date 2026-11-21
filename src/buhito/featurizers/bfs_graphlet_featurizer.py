"""Breadth-first graphlet features for arbitrary NetworkX graphs."""
from buhito.breadth_graphlet import custom_hash, graphlet_hash, generate_subgraphs_breadthwise


class BFSGraphletFeaturizer:
    def __init__(self, max_len=3, return_nodewise=True,
                 node_key="node_label", edge_key="edge_label"):
        self.size = max_len
        self.return_nodewise = return_nodewise
        self.node_key = node_key
        self.edge_key = edge_key
        self.verbose = False

    def __call__(self, G):
        return generate_subgraphs_breadthwise(
            G, self.size, return_nodewise=self.return_nodewise,
            node_key=self.node_key, edge_key=self.edge_key,
        )
