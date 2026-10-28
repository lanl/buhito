import networkx as nx
import pynauty
from collections import Counter
import random

import sys
from pathlib import Path
import pytest

def _pynauty_label(nbunchs,
            vertex_coloring = {},
            edge_coloring = None,
            **kwargs):
    """
    Get canonical label.  

    Canonical labels for unlabeled graphs are of the form :
        frozenset({(vertex, frozenset(neighbors)) for vertex, neighbors in canon_adj.items()})
    Canonical labels for labeled graphs have the same information followed with the dictionary of the coloring.
    This ensures that differently colored graphlets that are otherwise isomorphic will have different labels.  
    (NB: Pynauty only uses the fact that there is a labeling of a given shape, but it does not take into account
    label values.)  By including the coloring dictionary, we ensure that label values are included in the final.
    """
    if vertex_coloring or edge_coloring :
        for g_inputs, c in nbunchs:
            g = pynauty.Graph(**g_inputs)
            canon_perm = pynauty.canon_label(g)
            vertex_map = { canon_perm[i] : i for i in range(g.number_of_vertices) }
            canon_adj = _map_adj_dict_restrict(vertex_map, g._adjacency_dict)
            canon_coloring = tuple(sorted((vertex_map[i], color) for (i, color) in c))
            yield frozenset({(vertex, frozenset(neighbors)) for vertex, neighbors in canon_adj.items()}), canon_coloring
    else:
        for g_inputs in nbunchs:
            g_inputs["vertex_coloring"] = [{x for x in range(g_inputs["number_of_vertices"]) }]
            g = pynauty.Graph(**g_inputs)
            canon_perm = pynauty.canon_label(g)
            vertex_map = { canon_perm[i] : i for i in range(g.number_of_vertices) }
            canon_adj = _map_adj_dict_restrict(vertex_map, g._adjacency_dict)
            yield frozenset({(vertex, frozenset(neighbors)) for vertex, neighbors in canon_adj.items()})



def _pynauty_inputs(nbunchs, 
        adj = None,
        graphlet_sizes = [2],
        vertex_coloring = {},
        edge_coloring = None,
        **kwargs):
    """ 
    adj                 : dict_of_dicts - the graph adjacencies
    graphlet_sizes      : positive int or list of positive ints
    vertex_coloring     : dictionary of vertices -> labels
    edge_coloring       : name of edge attribute in nx.Graph to use for edge labelling
    
    nbunchs are (level, vertex) describing the leaf of current subtree in search.
    Output is everything pynuaty needs to produces a canonical label.

    NB: This routine does the magic of getting the inputs for the canonical
    label of the _line graph_ of the graph for graphs that have colored edges.
    (Pynauty only supports coloring vertices natively).
    """ 
    max_size = max(graphlet_sizes)
    vertex_maps = {i : {} for i in range(max_size + 1)}
    adjacencies = {i : {} for i in range(max_size + 1)}
    if vertex_coloring:
        vertex_labels = list({v for k,v in vertex_coloring.items()})
    else:
        vertex_labels = []
    if edge_coloring:
        edge_labels = list({v for k,v in edge_coloring.items()})
    else:
        edge_labels = []
    if vertex_coloring or edge_coloring:
        try:
            labels = sorted(vertex_labels + edge_labels + ["unlabelled"] )
        except: 
            print(vertex_labels, edge_labels)
            raise ValueError
        label_index_lookup ={labels[i] : i for i in range(len(labels))}
        colorings   = {i : [ set({}) for j in range(len(labels)) ] for i in range(max_size +1)}
        color_maps = {i : {} for i in range(max_size + 1)}
    ############
    # algorithm:
    for level, node in nbunchs:
        vertex_maps[level] = vertex_maps[level - 1].copy()
        adjacencies[level] = {i : adjacencies[level - 1][i].copy() 
            for i in adjacencies[level - 1]}
        neighbors_unmapped = set(adj[node]).intersection( set( vertex_maps[level] ))
        # get correct node number for new node
        if edge_coloring: # we need to add the appropriate line graph vertices too
            try:
                node_mapped = max({x for x in adjacencies[level]}) + 1
            except:
                node_mapped = 0
        else:
            neighbors = {vertex_maps[level][n_u] for n_u in neighbors_unmapped}
            node_mapped = level - 1
            # adjacencies
            adjacencies[level][node_mapped] = neighbors
            for n in neighbors:
                adjacencies[level][n].add(node_mapped)
        # update node mapping
        vertex_maps[level][node] = node_mapped
        # update color bookkeeping
        if vertex_coloring or edge_coloring:
            color_maps[level] = color_maps[level - 1].copy()
            colorings[level] = [ color.copy() for color in colorings[level - 1] ]
            new_label = vertex_coloring.get(node, "unlabelled")
            color_maps[level][node_mapped] = new_label
            colorings[level][label_index_lookup[new_label]].add(node_mapped)
            added_edges = 0
            if edge_coloring:
                # adjacencies
                adjacencies[level][node_mapped] = set(range(node_mapped + 1, node_mapped + 
                        1 + len(neighbors_unmapped)))
                for n_u in neighbors_unmapped:
                    added_edges += 1
                    neighbor = vertex_maps[level][n_u]
                    adjacencies[level][node_mapped + added_edges] = {node_mapped, neighbor}
                    adjacencies[level][neighbor].add(node_mapped + added_edges)
                    new_edge_label = edge_coloring.get(frozenset({node, n_u}), "unlabelled")
                    colorings[level][label_index_lookup[new_edge_label]].add(node_mapped + added_edges)
                    color_maps[level][node_mapped + added_edges] = new_edge_label
            if level in graphlet_sizes: 
                yield {"number_of_vertices" : node_mapped + 1 + added_edges, "adjacency_dict" : adjacencies[level],
                    "vertex_coloring" : [color_class for color_class in colorings[level] 
                    if len(color_class) > 0 ]}, tuple(sorted(color_maps[level].items()))
        # yield the graph and its coloring.  We need the coloring (labels) to ensure 
        # different labelings result in different final outputs (whether certificate or labelings).
        elif level in graphlet_sizes:
            yield {"number_of_vertices" : level, "adjacency_dict" : adjacencies[level]}



def _count(adj,
       graphlet_sizes = [2],
       reject_probability = None,
       return_distribution = True, 
       blacklist = set({}),
       subtree_starts = None,
       **kwargs):
    """ 
    adj                 : dict_of_dicts - the graph adjcencies
    graphlet_sizes      : positive int or list of positive ints
    reject_probability  : rejection sampling with probability or full count
    return_distribution : return distribution of graphlets per size (defualt) otherwise counts of graphlets
    blacklist           : list of nodes to ignore
    subtree_starts      : only include graphlets with at least one of these nodes
    """ 
    # initialize state
    max_size = max(graphlet_sizes)
    l = 0
    state = {}
    state[l] = {}
    if not subtree_starts: # include all nodes
        subtree_starts = [x for x in adj]
    state[l]['neighbors'] = subtree_starts
    state[l]['blacklist'] = blacklist
    tree = []
    remove = None
    # count!
    while l >= 0:
        neighbors, blacklist = state[l]['neighbors'], state[l]['blacklist']
        if len(neighbors) == 0: # remove node & decrement
            try:
                tree.pop()
            except:
                pass
            l -= 1 # decrement level
        else: # add node from nb to tree
            node = neighbors.pop()  # this removes from state[l]['neighbors']
            if reject_probability and random.random() <= reject_probability: # reject node; move on to next
                blacklist = blacklist.union({node}) # add to state[l]['blacklist']
            else: 
                tree.append(node)
                yield l+1, node
                if len(tree) < max_size:  # proceed to next level
                    if len(tree) == 1: # just one node, disregard level 0 neighbors
                        next_neighbors = {x for x in adj[node]
                                         if x not in tree and x not in blacklist}
                    else: # add neighbors to list to search from
                        next_neighbors = {x for x in adj[node]
                                         if x not in tree and x not in blacklist}.union(
                                         neighbors.difference(blacklist))
                    state[ l+1 ] = {'neighbors' : next_neighbors, 'blacklist' : blacklist}
                    state[ l ]['blacklist'] = blacklist.union({node})
                    l += 1
                else:  # reached max size
                    tree.pop()

def _map_adj_dict_restrict(mapping, dictionary):
    result = {}
    for k in mapping:
        result[mapping[k]] = set(mapping[i] for i in dictionary[k] if i in mapping)
    return result    

def _simple_census(kwargs):
    graphlets = _count(**kwargs)
    inputs = _pynauty_inputs(graphlets, **kwargs)
    labels = _pynauty_label(inputs, **kwargs)
    return Counter(labels)


def take_census(graph,
                graphlet_sizes = [2],
                vertex_coloring = {},
                edge_coloring = None,
                reject_probability = None,
                return_distribution = True, 
                blacklist = set({}),
                subtree_starts = None,
                use_pool = False,
                save = None,
                ):
    """
    """
    if type(graph) == nx.Graph:
        adj = nx.to_dict_of_dicts(graph)
        if type(vertex_coloring) == str: # from networkx attributes
            label = vertex_coloring
            vertex_coloring = {v : graph[v].get(label, "unlabelled") for v in graph}
        if type(edge_coloring) == str: # from networkx attributes
            label = edge_coloring
            edge_coloring = {frozenset({u,v}) : graph[u][v].get(label, "unlabelled") for u,v in graph.edges()}
    else:
        adj = graph
    kwargs = {  "adj" : adj,
                "graphlet_sizes" : graphlet_sizes,
                "vertex_coloring" : vertex_coloring,
                "edge_coloring" : edge_coloring,
                "reject_probability" : reject_probability,
                "blacklist" : blacklist,
                "subtree_starts" : subtree_starts,
                }
    if use_pool:
        from multiprocessing import Pool, cpu_count
        counter = Counter()
        results = []
        with Pool(cpu_count() - 1) as workers:
            for node in adj:
                kw = kwargs.copy()
                kw["subtree_starts"] = set({node})
                kw["blacklist"] = blacklist.copy()
                blacklist.add(node)
                results.append(workers.apply_async(_simple_census,  [kw] ))
            workers.close()
            workers.join()
            for r in results:
                counter.update( r.get() )
        if save:
            with open(save + ".counter", "w") as myfile:
                print(counter, file = myfile)
        return counter
    else:
        counter = _simple_census(kwargs)
        if save:
            with open(save + ".counter", "w") as myfile:
                print(counter, file = myfile)
        return counter


if __name__ == '__main__':
    # Following are tests
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root / 'src'))
    test_file = project_root / 'tests' / 'test_census_graphlet.py'
    raise SystemExit(pytest.main([str(test_file)]))
