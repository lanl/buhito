import networkx as nx
import pynauty
# from functools import lru_cache
# import hashlib
from collections import Counter
import random
# from multiprocessing import Pool, cpu_count


def map_adj_dict_restrict(mapping, dictionary):
    result = {}
    for k in mapping:
        result[mapping[k]] = set(mapping[i] for i in dictionary[k] if i in mapping)
    return result    



class Census():
    """ Class of algorithms to perform a graphlet census """
    def __init__(self, graph,
                graphlet_sizes = [2],
                vertex_coloring = {},
                edge_coloring = None,
                reject_probability = None,
                return_distribution = True, 
                blacklist = set({}),
                must_include = None,
                use_pool = None,
                ):
        """ 
        graph               : nx.Graph or dict_of_dicts
        graphlet_sizes      : positive int or list of positive ints
        vertex_coloring     : dictionary of vertices -> labels
        edge_coloring       : name of edge attribute in nx.Graph to use for edge labelling
        reject_probability  : rejection sampling with probability or full count
        return_distribution : return distribution of graphlets per size (defualt) otherwise counts of graphlets
        blacklist          : list of nodes to ignore
        starting_nodes      : only include graphlets with at least one of these nodes
        use_pool            : use multiprocessing.Pool to count
        """
        if type(graph) == nx.Graph:
            self.adj = nx.to_dict_of_dicts(graph)
            ## can get coloring info from networkx
            if type(vertex_coloring) == str: # from networkx attributes
                label = vertex_coloring
                self.vertex_coloring = {v : g[v].get(label, "unlabelled") for v in g}
            else: self.vertex_coloring = vertex_coloring
            if type(edge_coloring) == str: # from networkx attributes
                label = edge_coloring
                self.edge_coloring = {frozenset({u,v}) : g[u][v].get(label, "unlabelled") for u,v in g.edges()}
            else: self.edge_coloring = edge_coloring
        else:
            self.adj = graph
            self.vertex_coloring = vertex_coloring
            self.edge_coloring = edge_coloring
        self.graphlet_sizes = graphlet_sizes
        self.max_size = max(self.graphlet_sizes)
        self.vertex_maps = {i : {} for i in range(self.max_size + 1)}
        self.adjacencies = {i : {} for i in range(self.max_size + 1)}
        if vertex_coloring:
            vertex_labels = list({v for k,v in self.vertex_coloring.items()})
        else:
            vertex_labels = []
        if edge_coloring:
            edge_labels = list({v for k,v in self.edge_coloring.items()})
        else:
            edge_labels = []
        if vertex_coloring or edge_coloring:
            try:
                labels = sorted(vertex_labels + edge_labels + ["unlabelled"] )
            except: 
                print(vertex_labels, edge_labels)
                raise ValueError
            self.label_index_lookup ={labels[i] : i for i in range(len(labels))}
            self.colorings   = {i : [ set({}) for j in range(len(labels)) ] for i in range(self.max_size +1)}
            self.color_maps = {i : {} for i in range(self.max_size + 1)}
        self.blacklist = blacklist
        self.reject_probability = reject_probability


    def _count(self, subtree_starts=None):
        """
        Algorithm doing the counting. 
            1) Contain at least one node from subtree_starts (default value is all nodes)
            2) Cannot contain any node from self.blacklist
            3) If self.reject_probability is > 0 then choose report graphlet H 
                       satisfying (1) and (2) on k nodes with probability $(1-p)^k$.
        """
        # initialize state
        l = 0
        state = {}
        state[l] = {}
        if not subtree_starts: # include all nodes
            subtree_starts = [x for x in self.adj]
        state[l]['neighbors'] = subtree_starts
        state[l]['blacklist'] = self.blacklist
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
                if self.reject_probability and random.random() <= self.reject_probability: # reject node; move on to next
                    blacklist = blacklist.union({node}) # add to state[l]['blacklist']
                else: 
                    tree.append(node)
                    yield l+1, node
                    if len(tree) < self.max_size:  # proceed to next level
                        if len(tree) == 1: # just one node, disregard level 0 neighbors
                            next_neighbors = {x for x in self.adj[node]
                                             if x not in tree and x not in blacklist}
                        else: # add neighbors to list to search from
                            next_neighbors = {x for x in self.adj[node]
                                             if x not in tree and x not in blacklist}.union(
                                             neighbors.difference(blacklist))
                        state[ l+1 ] = {'neighbors' : next_neighbors, 'blacklist' : blacklist}
                        state[ l ]['blacklist'] = blacklist.union({node})
                        l += 1
                    else:  # reached max size
                        tree.pop()

    def _pynauty_inputs(self, nbunchs):
        """
        Input is level, vertex describing the leaf of current subtree in search.
        Output is everything pynuaty needs to produces a canonical label.

        NB: This routine does the magic of getting the canonical label of the _line graph_ of the 
            graph for graphs that have colored edges.  (Pynauty only supports coloring vertices 
            natively).
        """
        for level, node in nbunchs:
            self.vertex_maps[level] = self.vertex_maps[level - 1].copy()
            self.adjacencies[level] = {i : self.adjacencies[level - 1][i].copy() 
                for i in self.adjacencies[level - 1]}
            neighbors_unmapped = set(self.adj[node]).intersection( set( self.vertex_maps[level] ))
            # get correct node number for new node
            if self.edge_coloring: # we need to add the appropriate line graph vertices too
                try:
                    node_mapped = max({x for x in self.adjacencies[level]}) + 1
                except:
                    node_mapped = 0
            else:
                neighbors = {self.vertex_maps[level][n_u] for n_u in neighbors_unmapped}
                node_mapped = level - 1
                # adjacencies
                self.adjacencies[level][node_mapped] = neighbors
                for n in neighbors:
                    self.adjacencies[level][n].add(node_mapped)
            # update node mapping
            self.vertex_maps[level][node] = node_mapped
            # update color bookkeeping
            if self.vertex_coloring or self.edge_coloring:
                self.color_maps[level] = self.color_maps[level - 1].copy()
                self.colorings[level] = [ color.copy() for color in self.colorings[level - 1] ]
                new_label = self.vertex_coloring.get(node, "unlabelled")
                self.color_maps[level][node_mapped] = new_label
                self.colorings[level][self.label_index_lookup[new_label]].add(node_mapped)
                added_edges = 0
                if self.edge_coloring:
                    # adjacencies
                    self.adjacencies[level][node_mapped] = set(range(node_mapped + 1, node_mapped + 
                            1 + len(neighbors_unmapped)))
                    for n_u in neighbors_unmapped:
                        added_edges += 1
                        neighbor = self.vertex_maps[level][n_u]
                        self.adjacencies[level][node_mapped + added_edges] = {node_mapped, neighbor}
                        self.adjacencies[level][neighbor].add(node_mapped + added_edges)
                        new_edge_label = self.edge_coloring.get(frozenset({node, n_u}), "unlabelled")
                        self.colorings[level][self.label_index_lookup[new_edge_label]].add(node_mapped + added_edges)
                        self.color_maps[level][node_mapped + added_edges] = new_edge_label
                if level in self.graphlet_sizes: 
                    yield {"number_of_vertices" : node_mapped + 1 + added_edges, "adjacency_dict" : self.adjacencies[level],
                        "vertex_coloring" : [color_class for color_class in self.colorings[level] 
                        if len(color_class) > 0 ]}, tuple(sorted(self.color_maps[level].items()))
            # yield the graph and its coloring.  We need the coloring (labels) to ensure 
            # different labelings result in different final outputs (whether certificate or labelings).
            elif level in self.graphlet_sizes:
                yield {"number_of_vertices" : level, "adjacency_dict" : self.adjacencies[level]}

    def _pynauty_label(self, nbunchs):
        """
        Get canonical label.  

        Canonical labels for unlabeled graphs are of the form :
            frozenset({(vertex, frozenset(neighbors)) for vertex, neighbors in canon_adj.items()})
        Canonical labels for labeled graphs have the same information followed with the dictionary of the coloring.
        This ensures that differently colored graphlets that are otherwise isomorphic will have different labels.  
        (NB: Pynauty only uses the fact that there is a labeling of a given shape, but it does not take into account
        label values.)  By including the coloring dictionary, we ensure that label values are included in the final.
        """
        if self.vertex_coloring or self.edge_coloring:
            for g_inputs, c in nbunchs:
                g = pynauty.Graph(**g_inputs)
                canon_perm = pynauty.canon_label(g)
                vertex_map = { canon_perm[i] : i for i in range(g.number_of_vertices) }
                canon_adj = map_adj_dict_restrict(vertex_map, g._adjacency_dict)
                canon_coloring = tuple(sorted((vertex_map[i], color) for (i, color) in c))
                yield frozenset({(vertex, frozenset(neighbors)) for vertex, neighbors in canon_adj.items()}), canon_coloring
        else:
            for g_inputs in nbunchs:
                g_inputs["vertex_coloring"] = [{x for x in range(g_inputs["number_of_vertices"]) }]
                g = pynauty.Graph(**g_inputs)
                canon_perm = pynauty.canon_label(g)
                vertex_map = { canon_perm[i] : i for i in range(g.number_of_vertices) }
                canon_adj = map_adj_dict_restrict(vertex_map, g._adjacency_dict)
                yield frozenset({(vertex, frozenset(neighbors)) for vertex, neighbors in canon_adj.items()})

    def take(self):
        """
        Take the graphlet census.
        """
        graphlets = self._count()
        inputs = self._pynauty_inputs(graphlets)
        labels = self._pynauty_label(inputs)
        return Counter(labels)

    def featurizer(self):
        return dict(self.take()), None


if __name__ == '__main__':
    g = nx.petersen_graph()
    # unlabelled, all graphlets
    c = Census(g, graphlet_sizes = [x for x in range(1,10)] )
    assert set({(len(info), number) for info, number in c.take().items()}) == set({ (9, 10), (3, 30), (6, 15), (7, 20), (6, 10), (8, 15), (7, 30), (5, 60), (8, 30), (4, 10), (6, 60), (4, 60), (1, 10), (7, 60), (5, 12), (2, 15) })  
    # with a vertex coloring
    c = Census(g, vertex_coloring = {1 : "one"}, graphlet_sizes = [3])
    assert c.take() == Counter({(frozenset({(2, frozenset({0, 1})), (0, frozenset({2})), (1, frozenset({2}))}), ((0, 'unlabelled'), (1, 'unlabelled'), (2, 'unlabelled'))): 21, (frozenset({(0, frozenset({1})), (2, frozenset({1})), (1, frozenset({0, 2}))}), ((0, 'one'), (1, 'unlabelled'), (2, 'unlabelled'))): 6, (frozenset({(2, frozenset({0})), (1, frozenset({0})), (0, frozenset({1, 2}))}), ((0, 'one'), (1, 'unlabelled'), (2, 'unlabelled'))): 3})
    # with edge coloring
    c = Census(g, edge_coloring={frozenset({1,2}) : "one"}, graphlet_sizes = [1,2,3])
    assert c.take() == Counter({(frozenset({(0, frozenset({3})),
                     (1, frozenset({4})),
                     (2, frozenset({3, 4})),
                     (3, frozenset({0, 2})),
                     (4, frozenset({1, 2}))}),
          ((0, 'unlabelled'),
           (1, 'unlabelled'),
           (2, 'unlabelled'),
           (3, 'unlabelled'),
           (4, 'unlabelled'))): 26,
         (frozenset({(0, frozenset({2})),
                     (1, frozenset({2})),
                     (2, frozenset({0, 1}))}),
          ((0, 'unlabelled'), (1, 'unlabelled'), (2, 'unlabelled'))): 14,
         (frozenset({(0, frozenset())}), ((0, 'unlabelled'),)): 10,
         (frozenset({(0, frozenset({1, 2})),
                     (1, frozenset({0, 4})),
                     (2, frozenset({0})),
                     (3, frozenset({4})),
                     (4, frozenset({1, 3}))}),
          ((0, 'one'),
           (1, 'unlabelled'),
           (2, 'unlabelled'),
           (3, 'unlabelled'),
           (4, 'unlabelled'))): 4,
         (frozenset({(0, frozenset({1, 2})),
                     (1, frozenset({0})),
                     (2, frozenset({0}))}),
          ((0, 'one'), (1, 'unlabelled'), (2, 'unlabelled'))): 1})
