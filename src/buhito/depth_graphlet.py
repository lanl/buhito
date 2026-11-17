"""
Depth-first calculation of graphlet histogram with recursive hash.
"""
import hashlib
import re


import networkx as nx
from collections import Counter, defaultdict

from buhito.utilities import sorted_hash_counts

def graph_adjacency(G):
    """Neighbor sets of every node, without self loops. Built once per graph."""
    return {v: set(G.neighbors(v)) - {v} for v in G.nodes}

def generate_subgraphs_from_node(G, node, depth, whitelist, adj=None):
    """
    Generate every connected node set with at most `depth` nodes that contains `node`
    and otherwise only nodes of `whitelist`, each set exactly once (as a frozenset).

    This is the ESU rule (Wernicke): each partial set carries an extension set
    (candidates that may be added next) and its closed neighborhood. A candidate
    taken out of the extension set is never handed back to later siblings, and a
    child only gains the *exclusive* neighbors of the node just added (neighbors
    not already adjacent to the set). Every connected set containing `node` is
    therefore produced by exactly one growth path, so no "found" set is needed.
    Callers remove `node` from `whitelist` after this generator is exhausted so
    that later roots never produce a set containing it.
    """
    if adj is None:
        adj = graph_adjacency(G)
    root = frozenset((node,))
    yield root
    if depth<=1: # stop recursion at some arbitrary integer.
        # Note: if you change this integer, you must change the hasher recursive call to generate_subgraphs!
        return

    allowed = set(whitelist) # copies; any container works
    allowed.discard(node)
    extension = adj[node] & allowed
    # stack frames: (cluster, extension set, closed neighborhood of cluster, size of cluster)
    stack = [(root, extension, set(extension), 1)]
    while stack:
        cluster, extension, neighborhood, size = stack.pop()
        while extension:
            n = extension.pop() # n leaves the extension set for all later siblings
            next_cluster = cluster | {n}
            yield next_cluster
            if size + 1 < depth:
                new_neighbors = adj[n] & allowed
                stack.append((next_cluster,
                              extension | (new_neighbors - neighborhood), # exclusive neighbors of n
                              neighborhood | new_neighbors,
                              size + 1))

    return


def generate_subgraphs_depthwise(G, maxlen, hash_helper=None, whitelist=None, memo_scope='graph'):
    """
    G: networkx graph for molecule
    maxlen: size of
    hash_helper: used to compute hashes when this function is called recursively
    whitelist: subset of atoms in the molecule to consider.
    memo_scope: 'graph' keeps the hash memo for the whole call, 'root' clears it after
                every root node (bounded working memory, identical results).
    """
    # If hash_helper is None, we are generating fresh.
    # if hash_helper is not None, we are helping ourselves.

    if memo_scope not in ('graph', 'root'):
        raise ValueError(f"memo_scope must be 'graph' or 'root', got {memo_scope!r}")

    retain_counts=(hash_helper is None)
    if retain_counts:
        hash_helper=HashHelper(G,maxlen)
    adj = getattr(hash_helper, 'adj', None)
    if adj is None:
        adj = graph_adjacency(G)

    all_subsets = set()
    if whitelist is None:
        whitelist = set(G.nodes)
    else:
        whitelist = set(whitelist) # copies if set, removes 'frozen' if frozenset

    for n in list(whitelist): # changes during iteration
        for subset in generate_subgraphs_from_node(G,n,maxlen,whitelist,adj):
            # each subset arrives exactly once, no duplicate check needed.
            all_subsets.add((subset,hash_helper(subset)))

        whitelist.remove(n) # now never look at that node again
        if retain_counts and memo_scope == 'root':
            # hashes are a pure function of the labeled induced subgraph, so entries
            # may be dropped and recomputed by later roots without changing any value.
            hash_helper.memo.clear()
    if retain_counts:
        res = Counter((len(ss),h) for ss,h in all_subsets)
    else:
        res = None
    return all_subsets,res

class HashHelper():
    def __init__(self, graph, maxlen):
        self.graph=graph
        self.maxlen=maxlen
        self.adj=graph_adjacency(graph)
        # Memo of node set -> hash. Owned by this per-graph helper (not a process-wide
        # cache), so it dies with the graph and can be cleared per root.
        self.memo={}

    @staticmethod
    def custom_hash(obj):
        s = str(obj).encode()
        hash_int = int.from_bytes(hashlib.sha1(s).digest(), 'big')
        return hash_int

    def __call__(self, indices):
        # very important, this is memoized.
        memo = self.memo
        h = memo.get(indices)
        if h is not None:
            return h

        if len(indices)==1:
            # Just hash atom key
            a=list(indices)[0]
            akey = self.graph.nodes[a]['atom_key']
            h = self.custom_hash(akey)

            memo[indices] = h
            return h

        if len(indices)==2:
            # Include bond key in hash
            a1,a2=indices
            h1, h2 = map(self,((a1,),(a2,)))
            btype = self.graph.edges[a1,a2]['bond_key']
            h3 = self.custom_hash(btype)
            h = self.custom_hash(tuple(sorted((h1, h2, h3))))
            memo[indices] = h
            return h

        # hash and count of all connected proper substructures of this structure.
        # The rooted enumeration restricted to `indices` with bound len-1 yields each
        # connected proper subset exactly once, so the list is the exact multiset.
        sub_hashes = []
        allowed = set(indices)
        for n in list(allowed): # changes during iteration
            for sub in generate_subgraphs_from_node(self.graph,n,len(indices)-1,allowed,self.adj):
                sub_hashes.append(self(sub))
            allowed.remove(n)
        # combine hashes from substructures to form new hash for this: sort once and
        # run-length encode, same (hash, count) pairs as tuple(sorted(Counter(...).items())).
        this_hashkey = sorted_hash_counts(sub_hashes)
        h = self.custom_hash(this_hashkey)

        memo[indices] = h
        return h

