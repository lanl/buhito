"""
Breadth-first calculation of graphlet histogram with recursive hash.

"""
import re
import hashlib

import networkx as nx
from collections import Counter, defaultdict

from buhito.utilities import sorted_hash_counts

class BFSGraphletFeaturizer():     
    def __init__(self, max_len=3, return_nodewise=True):
        """A class to produce graphlets on graphs using breadth-first search.

        The size component of the bit ID corresponds to the number of nodes in the graphlet. 
        
        :param max_len: int, the largest number of nodes to consider for induced subgraphs
        """
        
        self.size = max_len
        self.return_nodewise = return_nodewise
        self.verbose = False

        
    def __call__(self, G):
        # Pass the flag through: the node-wise tables are only built when requested,
        # so the result is (fp, bi, node_fps, node_bi) or (fp, bi) accordingly.
        return generate_subgraphs_breadthwise(G=G,
                                              depth=self.size,
                                              return_nodewise=self.return_nodewise,
                                              full_hash=True,
                                              node_key="atom_key",
                                              edge_key="bond_key")
    

def custom_hash(obj):
    s = str(obj).encode()
    hash_int = int.from_bytes(hashlib.sha1(s).digest(), 'big')
    return hash_int

def graphlet_hash(glet: frozenset,
                  G: nx.Graph,
                  ancestor_graph: dict[frozenset,list[frozenset]],
                  current_hashes: dict[frozenset,int],
                  node_key: str,
                  edge_key: str,
                  hash_function: callable=custom_hash):
    """_summary_

    Args:
        glet (frozenset): The set of nodes being hashed.
        G (nx.Graph): The graph the nodes are in.
        ancestor_graph (dict[frozenset,list[frozenset]]): List of ancestor graphlets of for each graphlet
        current_hashes (dict[frozenset,int]): List of hashes for previously hashed graphlets, written to as output.
        node_key (str): The key used to color nodes.
        edge_key (str): The key used to color edges.
        hash_function (callable, optional): Customizable hook for hashing. Defaults to custom_hash.

    Raises:
        ValueError: If called on a graphlet has already been hashed.

    Returns:
        _type_: hash for graphlet glet
    """
    
    if glet in current_hashes:
        raise ValueError("Should only hash each thing once!")
    if len(glet)==1:
        # Just hash atom key
        a = list(glet)[0]
        akey = G.nodes[a][node_key]
        h = hash_function(akey)

        return h
    
    elif len(glet)==2:
            # Include bond key in hash
            # Get the two nodes and create single-node graphlets
            nodes = sorted(list(glet))  # Sort for consistent ordering
            g1 = frozenset({nodes[0]})
            g2 = frozenset({nodes[1]})
            
            # These should have been hashed already
            if g1 not in current_hashes or g2 not in current_hashes:
                # Fallback: hash them now if not already hashed
                for single_node_glet in [g1, g2]:
                    if single_node_glet not in current_hashes:
                        n = list(single_node_glet)[0]
                        nkey = G.nodes[n][node_key]
                        current_hashes[single_node_glet] = hash_function(nkey)
            
            h1 = current_hashes[g1]
            h2 = current_hashes[g2]
            a1 = list(g1)[0]
            a2 = list(g2)[0]
            btype = G.edges[a1,a2][edge_key]
            h3 = hash_function(btype)
            h = hash_function(tuple(sorted((h1, h2, h3))))
            
            return h
    else:
        ancestors = ancestor_graph[glet]

        # note: modifying the hash to use chain-of-counters (different set of counts)
        # does not significantly improve speed.
        # The essential idea there was to track the hash-counter for each graphlet and use only
        # the parents' counters (instead of all ancestors) to compute the hash for this graphlet.
        # that strategy would also be a different assumption than the reconstruction hypothesis.

        # Sort the ancestor hashes once and run-length encode them; same (hash, count)
        # pairs as tuple(sorted(Counter(...).items())) without building a Counter.
        this_hashkey = sorted_hash_counts(map(current_hashes.__getitem__, ancestors))
        h = hash_function(this_hashkey)
        return h


def generate_subgraphs_breadthwise(G, depth, *, whitelist=None, start_nodes=None, return_nodewise=True, full_hash=True, node_key="atom_key",edge_key="bond_key"):
    """_summary_

    Args:
        G (_type_): input graph in networkx.
        depth (_type_): _description_
        whitelist (_type_, optional): Use induced graph with the set of nodes in whitelist. Defaults to None.
        start_nodes (_type_, optional): Nodes to start BFS from. If None, start from all nodes in whitelist. Defaults to None.
        return_nodewise (bool, optional): Return node-wese fingerprints and bitinfo. Defaults to True.
        full_hash (bool, optional): Use all ancestors for hashing. Defaults to True. If False, only use parents for hashing.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    
    if depth < 1:
        raise ValueError(f"Code doesn't handle depth value {depth!r}, only depth >= 1")
    if whitelist is None:
        whitelist = set(G.nodes)
    if start_nodes is None:
        start_nodes = whitelist
    else:
        start_nodes = set(start_nodes) & whitelist  # Ensure start_nodes are in whitelist

    hash_function = custom_hash

    # Level-local integer layout: within one level a graphlet is identified by its
    # position in a group of parallel lists (node set, frontier, parent ids, hash).
    # Parent links are small-int indices into the previous level's lists, so adding a
    # link is one list append and gathering parent hashes is an indexed read. The only
    # keyed map is the per-level deduplication index, which is dropped once the level is
    # complete (node sets of different sizes can never coincide, so a whole-run `seen`
    # set is unnecessary). Only the previous and the current level stay resident; under
    # full_hash a single write-once map node set -> hash keeps earlier hashes reachable.
    # The enumeration set, the parent / ancestor multisets fed to each hash, the base
    # cases and the label serialization are unchanged, so hash values are unchanged.

    # Outputs are accumulated per level as soon as the level is hashed.
    # Plain dicts filled with get-or-default: entries appear in the same first-seen order as
    # with Counter/defaultdict, so the outputs are unchanged and no final cast is needed.
    fingerprint: dict[tuple, int] = {}
    bitinfo: dict[tuple, list] = {}
    # The node-wise tables are a projection of the two above (every graphlet is credited to
    # each of its nodes). Only allocate and fill them when the caller asked for them.
    node_fingerprints: dict[int, dict[tuple, int]] = {} if return_nodewise else None
    node_bitinfos: dict[int, dict[tuple, list]] = {} if return_nodewise else None

    def emit_level(size, level_nodes, level_hashes):
        for glet, h in zip(level_nodes, level_hashes):
            key = (size, h)
            fingerprint[key] = fingerprint.get(key, 0) + 1
            glets = bitinfo.get(key)
            if glets is None:
                bitinfo[key] = [glet]
            else:
                glets.append(glet)
            if return_nodewise:
                for node in glet:
                    nfp = node_fingerprints.get(node)
                    if nfp is None:
                        # First graphlet containing this node.
                        node_fingerprints[node] = {key: 1}
                        node_bitinfos[node] = {key: [glet]}
                        continue
                    nfp[key] = nfp.get(key, 0) + 1
                    nbi = node_bitinfos[node]
                    glets = nbi.get(key)
                    if glets is None:
                        nbi[key] = [glet]
                    else:
                        glets.append(glet)

    # setup size 1 graphlets
    # Initialize ALL size-1 graphlets (in G.nodes order; the position is the id).
    prev_nodes: list[frozenset] = []
    prev_front: list[set] = []
    prev_hash: list[int] = []
    singleton: dict = {}      # node -> its size-1 graphlet object (reused as ancestor)
    node_hash: dict = {}      # node -> hash of its size-1 graphlet (size-2 base case)
    for node in G.nodes:
        if node not in whitelist:
            continue
        frontier = set()
        for neigh in G.neighbors(node):
            if neigh not in whitelist or neigh == node:
                continue
            frontier.add(neigh)
        glet = frozenset([node])
        # Just hash atom key
        h = hash_function(G.nodes[node][node_key])
        prev_nodes.append(glet)
        prev_front.append(frontier)
        prev_hash.append(h)
        singleton[node] = glet
        node_hash[node] = h
    emit_level(1, prev_nodes, prev_hash)

    if full_hash:
        # Write-once map keeping every earlier hash reachable for the all-ancestors rule,
        # plus the ancestor node sets of the previous level (positional).
        hash_by_set: dict[frozenset,int] = dict(zip(prev_nodes, prev_hash))
        prev_anc: list[set] = [set() for _ in prev_nodes]

    # If start_nodes is specified, we'll only expand from those nodes in the main loop

    # Now continue with larger graphlets. Not parallelizable.
    for size in range(2, depth+1): # stop when size is equal to depth.

        index: dict[frozenset,int] = {}   # per-level dedup: node set -> position
        nodes: list[frozenset] = []
        front: list[set] = []
        par: list[list[int]] = []         # positions in the previous level
        # Frontiers are only needed to expand this level further.
        need_frontier = size < depth

        for i, glet in enumerate(prev_nodes):
            # For each previous-level graphlet (parents/ancestors are complete and hashed).

            # For size-2, only expand from start_nodes if specified
            if size == 2 and start_nodes is not None:
                # Only expand from graphlets that contain a start_node
                if not any(node in start_nodes for node in glet):
                    continue

            frontier = prev_front[i]
            # Construct successor graphlets
            for neigh in frontier:
                # Make the next graphlet by adding a frontier element to the cluster
                next_glet = set(glet)
                next_glet.add(neigh)
                next_glet = frozenset(next_glet)

                j = index.get(next_glet)
                if j is None:
                    # If this graphlet is new, set up the new frontier (not for the last level).
                    if need_frontier:
                        next_frontier = frontier.copy()
                        next_frontier.remove(neigh)
                        for next_neigh in G.neighbors(neigh):
                            if next_neigh in whitelist and next_neigh not in next_glet:
                                next_frontier.add(next_neigh)
                        front.append(next_frontier)

                    # Record this graphlet and its first backlink.
                    index[next_glet] = len(nodes)
                    nodes.append(next_glet)
                    par.append([i])
                else:
                    # Further backlink: one integer append.
                    par[j].append(i)
        del index

        if not nodes:
            break

        # Hash the level: every backlink into it now exists.
        hashes: list[int] = []
        if size == 2:
            # Include bond key in hash (same base case as graphlet_hash).
            if full_hash:
                anc: list[set] = []
            for glet in nodes:
                a1, a2 = sorted(glet)  # Sort for consistent ordering
                h1 = node_hash[a1]
                h2 = node_hash[a2]
                btype = G.edges[a1,a2][edge_key]
                h3 = hash_function(btype)
                hashes.append(hash_function(tuple(sorted((h1, h2, h3)))))
                if full_hash:
                    # single-node ancestors needed for the all-ancestors rule
                    anc.append({singleton[a1], singleton[a2]})
        elif full_hash:
            # Hashing based on all ancestors: distinct ancestor node sets of all parents.
            anc = []
            for parents in par:
                ancestors = set()
                for p in parents:
                    ancestors.add(prev_nodes[p])
                    ancestors |= prev_anc[p]
                this_hashkey = sorted_hash_counts(map(hash_by_set.__getitem__, ancestors))
                hashes.append(hash_function(this_hashkey))
                anc.append(ancestors)
        else:
            # Only hash with the parent graph, not full ancestors.
            for parents in par:
                this_hashkey = sorted_hash_counts(map(prev_hash.__getitem__, parents))
                hashes.append(hash_function(this_hashkey))

        emit_level(size, nodes, hashes)

        # Advance one level; the lists of level size-1 become garbage.
        if full_hash:
            hash_by_set.update(zip(nodes, hashes))
            prev_anc = anc
        prev_nodes, prev_front, prev_hash = nodes, front, hashes

    if return_nodewise:
        return fingerprint, bitinfo, node_fingerprints, node_bitinfos
    else:
        return fingerprint, bitinfo
