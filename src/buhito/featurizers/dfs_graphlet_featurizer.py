"""Depth-first graphlet features for arbitrary NetworkX graphs."""
import networkx as nx
from collections import Counter, defaultdict
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

    def counts_only(self, G):
        """Count graphlets without retaining their locations or terminal hashes."""
        helper = HashHelper(G, self.size, node_key=self.node_key, edge_key=self.edge_key)
        counts = Counter()
        for nodes in iter_connected_subgraphs(G, self.size):
            # Lower-order hashes are reused; terminal hashes are consumed once.
            hashed = helper._hash(nodes) if len(nodes) == self.size else helper(nodes)
            counts[(len(nodes), hashed)] += 1
        return dict(sorted(counts.items()))

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


def iter_connected_subgraphs(G, maxlen):
    """Yield each connected induced node set once, without a global seen set.

    Within each root, previously expanded candidates are excluded from later
    branches. Removing roots from allowed assigns every set one starting node.
    """
    if maxlen < 1:
        raise ValueError("maxlen must be positive")
    allowed = set(G)

    def extend(nodes, candidates, excluded):
        remaining = set(candidates)
        blocked = set(excluded)
        while remaining:
            node = remaining.pop()
            cluster = nodes | {node}
            yield frozenset(cluster)
            if len(cluster) < maxlen:
                extension = (remaining | (set(G.neighbors(node)) & allowed)) - cluster - blocked
                yield from extend(cluster, extension, blocked)
            blocked.add(node)

    for root in G:
        yield frozenset({root})
        if maxlen > 1:
            yield from extend({root}, (set(G.neighbors(root)) & allowed) - {root}, set())
        allowed.remove(root)
