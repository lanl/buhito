class CensusFeaturizer():
    def __init__(self, 
                graphlet_sizes = [2],
                vertex_coloring = {},
                edge_coloring = None,
                reject_probability = None,
                return_distribution = True, 
                blacklist = set({}),
                must_include = None,
                use_pool = None,
                ):
        self.graphlet_sizes = graphlet_sizes
        self.vertex_coloring = vertex_coloring
        self.edge_coloring = edge_coloring
        self.reject_probability = reject_probability
        self.blacklist = blacklist
        self.must_include = must_include
        self.use_pool = use_pool

    def __call__(self, graph):
        census = Census(graph, 
                        graphlet_sizes = self.graphlet_sizes, 
                        vertex_coloring =  self.vertex_coloring, 
                        edge_coloring =  self.edge_coloring,
                        reject_probability = self.reject_probability,
                        blacklist = self.blacklist,
                        must_include = self.must_include,
                        use_pool = self.use_pool,
                        )
        return dict(census.take()), None

