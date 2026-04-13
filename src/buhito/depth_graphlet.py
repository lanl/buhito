"""
Depth-first calculation of graphlet histogram with recursive hash.
"""
from functools import lru_cache
import hashlib
import re


import networkx as nx
from collections import Counter, defaultdict

# version with nice notation.
# def active_neighbors(G,node,cur_neighbors,whitelist,yields_with):
#     # The next things to check will be the neighbors of this neighbor one,
#     # so long as they aren't already going to be checked!
#     possible = cur_neighbors | set(G.neighbors(node))
#     allowed = (possible - yields_with) & whitelist
#     return allowed
def active_neighbors(G,node,cur_neighbors,whitelist,yields_with):
    """
    Things in current neighbors or neighbors of a given node,
    as long as in whitelist and not already in current cluster.
    """
    """note! this definition is inlined in search_subgraphs from cluster"""
    return {n for n in (*cur_neighbors,*G.neighbors(node))
                  if n in whitelist
                  and n not in yields_with}

def generate_subgraphs_from_node(G, node, depth, whitelist):
    """
    Generate subgraphs starting from a specific node.
    Essentially just sets up calculation for searching from cluster.
    """
    blacklist = ()
    yields_with = {node}
    neighbors = active_neighbors(G,node,set(),whitelist,yields_with)

    yield from search_subgraphs_from_cluster(G,depth,neighbors,yields_with,whitelist,set())

def search_subgraphs_from_cluster(G, depth, neighbors, yields_with, whitelist, found):
    "Generate subgraphs from a given cluster."
    # Depth-first type search
    f_yw = frozenset(yields_with)
    if f_yw in found:
        # Already seen this thing, ignore it
        return
    found.add(f_yw)

    yield yields_with
    if depth<=1: # stop recursion at some arbitrary integer.
        # Note: if you change this integer, you must change the hasher recursive call to generate_subgraphs!
        return

    # go deeper!

    for n in neighbors:
        # next_neighbors = active_neighbors(G, n, neighbors, whitelist, yields_with)
        # the above gets called enough that inlining actually helps.
        yields_with.add(n) # add to current set for yielding
        next_neighbors={n for n in (*neighbors, *G.neighbors(n))
         if n in whitelist
         and n not in yields_with}
        # inception!
        yield from search_subgraphs_from_cluster(G,depth-1,next_neighbors,yields_with,
                                                 whitelist,found)
        yields_with.remove(n) # remove from current set for yielding

    return
    

def generate_subgraphs_depthwise(G, maxlen, hash_helper=None, whitelist=None):
    """
    G: networkx graph for molecule
    maxlen: size of
    hash_helper: used to compute hashes when this function is called recursively
    whitelist: subset of atoms in the molecule to consider.
    """
    # If hash_helper is None, we are generating fresh.
    # if hash_helper is not None, we are helping ourselves.

    retain_counts=(hash_helper is None)
    if retain_counts:
        hash_helper=HashHelper(G,maxlen)

    all_subsets = set()
    if whitelist is None:
        whitelist = set(G.nodes)
    else:
        whitelist = set(whitelist) # copies if set, removes 'frozen' if frozenset

    for n in list(whitelist): # changes during iteration
        for subset in generate_subgraphs_from_node(G,n,maxlen,whitelist):
            subset = frozenset(subset)
            h = hash_helper(subset)
            if (subset,h) in all_subsets:
                print(f"Subset {subset} with hash {h} is in all subsets already!")
            all_subsets.add((subset,h))         
            
        whitelist.remove(n) # now never look at that node again
    if retain_counts:
        res = Counter((len(ss),h) for ss,h in all_subsets)
    else:
        res = None
    return all_subsets,res

class HashHelper():
    def __init__(self, graph, maxlen):
        self.graph=graph
        self.maxlen=maxlen

    @staticmethod
    def custom_hash(obj):
        s = str(obj).encode()
        hash_int = int(hashlib.sha1(s).hexdigest(), 16)
        return hash_int

    @lru_cache(maxsize=None) # very important, this is memoized.
    def __call__(self, indices):

        if len(indices)==1:
            # Just hash atom key
            a=list(indices)[0]
            akey = self.graph.nodes[a]['atom_key']
            h = self.custom_hash(akey)

            return h
        
        if len(indices)==2:
            # Include bond key in hash
            a1,a2=indices
            h1, h2 = map(self,((a1,),(a2,)))
            btype = self.graph.edges[a1,a2]['bond_key']
            h3 = self.custom_hash(btype)
            h = self.custom_hash(tuple(sorted((h1, h2, h3))))
            return h
        
        # hash abd count of all substructures of this structure that are at most one smaller than this one.
        # Note that the his call passes itself back to generate_subgraphs to increase the efficiency
        # of memoization.
        sub_graph_set,_ = generate_subgraphs_depthwise(self.graph,len(indices)-1,hash_helper=self,whitelist=indices)
        sub_graph_set = [sg for sg, h in sub_graph_set]
        sub_hashes = Counter(self(idxs) for idxs in sub_graph_set)
        # combine hashes from substructures to form new hash for this.
        this_hashkey = tuple(sorted(sub_hashes.items()))
        h = self.custom_hash(this_hashkey)

        return h
                                 

