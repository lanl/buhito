class CensusFeaturizer():
    def __init__(self, 
                graphlet_sizes = [2],
                vertex_coloring = {},
                edge_coloring = None,
                reject_probability = None,
                return_distribution = True, 
                blacklist = set({}),
                subtree_starts = None,
                use_pool = None,
                ):
        self.kwargs = { 
            graphlet_sizes : graphlet_sizes,
            vertex_coloring : vertex_coloring,
            edge_coloring : edge_coloring,
            reject_probability : reject_probability,
            blacklist : blacklist,
            must_include : must_include,
            use_pool : use_pool
            }

    def __call__(self, graph):
        return dict(take_census(graph, **self.kwargs)), None

