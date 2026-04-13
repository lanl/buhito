"""
Breadth-first calculation of graphlet histogram with recursive hash.

"""
import re
import hashlib

import networkx as nx
from collections import Counter, defaultdict


def custom_hash(obj):
    s = str(obj).encode()
    hash_int = int(hashlib.sha1(s).hexdigest(), 16)
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
            g1, g2 = ancestor_graph[glet]
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

        sub_hashes = Counter(current_hashes[a] for a in ancestors)
        this_hashkey = tuple(sorted(sub_hashes.items()))
        h = hash_function(this_hashkey)
        return h


def generate_subgraphs_breadthwise(G, depth, *, whitelist=None, return_nodewise=True, full_hash=True, node_key="atom_key",edge_key="bond_key"):
    """_summary_

    Args:
        G (_type_): input graph in networkx.
        depth (_type_): _description_
        whitelist (_type_, optional): Use induced graph with the set of nodes in whitelist. Defaults to None.
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

    current_graphlets: set[frozenset] = set(frozenset([n]) for n in G.nodes if n in whitelist)

    all_graphlets = set()
    parent_graph: dict[frozenset, set[frozenset]] = defaultdict(set) # immediate backlinks between graphlets
    ancestor_graph: dict[frozenset, set[frozenset]] = defaultdict(set) # all backlinks between graphlets
    current_graphlets: dict[frozenset,set] = {} # dict of graphlet and frontier
    graphlet_hashes: dict[frozenset,int] = {}
    
    if full_hash:
        # Hashing based on all ancestors.
        def add_hash(glet):
            graphlet_hashes[glet] = graphlet_hash(glet, G, ancestor_graph, graphlet_hashes, node_key=node_key, edge_key=edge_key)

        def link_glet(glet, next_glet):
            # Add backlinks for next graphlet to the current one.
            parent_graph[next_glet].add(glet)
            ancestor_graph[next_glet].add(glet)
            ancestor_graph[next_glet] |= ancestor_graph[glet] # Add ancestors of this parent.
    else:
        # Only hash with the parent graph, not full ancestors.
        def add_hash(glet):
            graphlet_hashes[glet] = graphlet_hash(glet, G, parent_graph, graphlet_hashes, node_key=node_key, edge_key=edge_key)

        def link_glet(glet, next_glet):
            parent_graph[next_glet].add(glet)
    
    # setup size 1 graphlets
    for node in G.nodes:
        if node not in whitelist:
            continue
        frontier = set()
        for neigh in G.neighbors(node):
            if neigh not in whitelist or neigh == node:
                continue
            frontier.add(neigh)
        glet = frozenset([node])

        current_graphlets[glet] = frontier
        all_graphlets.add(glet)

    # Now continue with larger graphlets. Not parallelizable.
    for size in range(2, depth+1): # stop when size is equal to depth.

        next_graphlets: dict[frozenset,set] = {}
        # Somewhat parallelizable but requires some locks.
        for glet, frontier in current_graphlets.items():
            # For each current graphlet
            
            # Since that parents/ancestors from previous rounds are complete, hash the graphlet.
            add_hash(glet)
            # Construct successor graphlets
            for neigh in frontier:
                # Make the next graphlet by adding a frontier element to the cluster
                next_glet = set(glet)
                next_glet.add(neigh)
                next_glet = frozenset(next_glet)

                if next_glet not in all_graphlets:
                    # If this graphlet is new, set up the new frontier.
                    next_frontier = frontier.copy()
                    next_frontier.remove(neigh)
                    for next_neigh in G.neighbors(neigh):
                        if next_neigh in whitelist and next_neigh not in next_glet:
                            next_frontier.add(next_neigh)

                    # Record this graphlet. 
                    next_graphlets[next_glet] = next_frontier
                    all_graphlets.add(next_glet)
                
                link_glet(glet, next_glet)
                # # Add backlinks for next graphlet to the current one.
                # parent_graph[next_glet].add(glet)
                # ancestor_graph[next_glet].add(glet)
                # ancestor_graph[next_glet] |= ancestor_graph[glet] # Add ancestors of this parent.
                
        # Set the current graphlets to the next round, and continue.
        current_graphlets = next_graphlets

    # Finally, hash the last set of graphlets
    for glet in current_graphlets:
        add_hash(glet)

    # Transpose for fingerprint and bit_info
    fingerprint: dict[int, int] = Counter()
    bitinfo: dict[int,list] = defaultdict(list)
    node_fingerprints: dict[int,dict[int,int]] = defaultdict(Counter)
    node_bitinfos: dict[int,dict[int,list]] = defaultdict(lambda: defaultdict(list))

    # 1) Accumulating the information this wayseems hard to parallelize without directly locking 
    # ea ch bucket inside fingerprint and bitinfo, but it could be done.
    # 2) There might be an efficient strategy for simply splitting (arbitrary) sets of graphlets
    # to different threads and then merging fingerprint and bitinfo recursively from each threads? 
    # 3) There might be a way to do the counting in the midst of a paralellizable sorting, merge-sort or similar.
    for glet, h in graphlet_hashes.items():
        size = len(glet)
        key = (size, h)
        fingerprint[key] += 1
        bitinfo[key].append(glet)
        # This part might be more easily parallelized if we build the graph of descendants incrementally.
        # If we did that, we could walk through the fingerprints for each node independently without 
        # concern for thread safety. 
        for node in glet:
            node_fingerprints[node][key] +=1
            node_bitinfos[node][key].append(glet)

    # Cast to more conventional data structures, no actual changes.
    fingerprint = dict(fingerprint)
    bitinfo = dict(bitinfo)
    node_fingerprints = {k:dict(v) for k,v in node_fingerprints.items()}
    node_bitinfos = {k:dict(v) for k,v in node_bitinfos.items()}
    
    return fingerprint, bitinfo, node_fingerprints, node_bitinfos
