"""Graphlet inclusion relationships and coefficient projection on arbitrary graphs."""
from collections import defaultdict
from copy import deepcopy
import numpy as np
import networkx as nx


def group_nodes_by_size(bi, decreasing=True):
    grouped = defaultdict(list)
    for (size, _), locations in bi.items():
        grouped[size].extend(tuple(sorted(nodes)) for nodes in locations)
    return sorted(grouped, reverse=decreasing), dict(grouped)


class GraphletDAG:
    def __init__(self, bi, layerwise=True, bit_ids=None, coef=None, bit_to_coef_map=None):
        self.bi = bi
        self.layerwise = layerwise
        self.bit_ids = bit_ids
        self.coef = coef
        if bit_to_coef_map is None and bit_ids is not None and coef is not None:
            bit_to_coef_map = dict(zip(bit_ids, coef))
        self.bit_to_coef_map = bit_to_coef_map
        self.sizes, self.nodes_by_size = group_nodes_by_size(bi)
        self._build_ssg()
        if self.bit_to_coef_map is not None:
            self._add_coefs_to_nodes()
        self.projected_coefs = None
        self._has_been_projected = False

    def _create_graph_nodes(self):
        self.G = nx.DiGraph()
        for bit, locations in self.bi.items():
            for nodes in locations:
                self.G.add_node(tuple(sorted(nodes)), bit=bit, size=bit[0])

    def _up_projection_weight(self, node, child):
        return 1 / len(self._G.pred[node])

    def project_to_layer(self, layer):
        if self.bit_to_coef_map is None:
            raise RuntimeError("No coefficients supplied for projection")
        if layer not in (1, 2):
            raise ValueError("Projection supports layers 1 and 2")
        if self._has_been_projected:
            self._undo_projection()
        self._G = deepcopy(self.G)
        self._has_been_projected = True
        for size in self.sizes:
            if size > layer:
                self._project_from_layer(size, final_target_layer=layer)
        if layer == 2 and 1 in self.nodes_by_size:
            self._project_from_layer(1, 'up', final_target_layer=layer)
        nodes = sorted(node for node in self.G if len(node) == layer)
        self.projected_coefs = np.array([self.G.nodes[node]['coef'] for node in nodes])
        return self.projected_coefs.copy()

    def _create_graph_edges(self):
        """create the substructure inclusion edges. if self.layerwie, only between size n and n-1"""

        for i, size in enumerate(self.sizes[:-1]):
            for nodes in self.nodes_by_size[size]:
                # either just the next smallest size or all smaller sizes
                next_sizes = [self.sizes[i + 1]] if self.layerwise else self.sizes[i + 1:]
                for next_size in next_sizes:
                    for nodes_prime in self.nodes_by_size[next_size]:
                        if set(nodes_prime).issubset(set(nodes)):
                            self.G.add_edge(nodes, nodes_prime)


    def _build_ssg(self):
        self._create_graph_nodes()
        self._create_graph_edges()
        return


    @property
    def node_labels(self):
        labels = {}
        for node in self.G:
            if len(node) > 3:
                label = '{' + str(node[0]) + ',...,' + str(node[-1]) + '}'
            else:
                label = str(set(node))
            labels[node] = label
        return labels


    def _add_coefs_to_nodes(self):
        for node in self.G.nodes:
            coef = self.bit_to_coef_map.get(self.G.nodes[node]['bit'], 0)
            self.G.add_node(node, coef=coef)
        return


    def _project_from_layer(self,
                            l,
                            direction='down',
                            final_target_layer=None,
                            weighted=True
                            ):
        """Project attrs from layer l to l +/- 1 in layerwise case to final_target_layer otherwise"""
        G = self.G
        layer_l = self.nodes_by_size[l]
        layer_next = G.succ if direction == 'down' else G.pred
        if not self.layerwise:
            # then only project to the target layer
            _layer_next = {}
            for k, v in layer_next.items():
                if len(k) == l:
                    _layer_next[k] = {x: y for x, y in layer_next[k].items() if len(x) == final_target_layer}

            layer_next = _layer_next

        if direction == 'up' and l > 1:
            raise RuntimeError('only projecting up to edges is supported')

        # for each node in this layer
        for node_l in layer_l:
            # grab the coef and look at the children
            coef_l = G.nodes[node_l]['coef']
            children = dict(layer_next[node_l])

            # for each child
            for child in children.keys():
                if direction == 'up' and weighted:
                    delta = coef_l * self._up_projection_weight(node_l, child)
                    # also reverse the edge direction to show direction of projection
                    G.remove_edge(child, node_l)
                    G.add_edge(node_l, child)
                else:
                    # always evenly divide among children on down projection
                    delta = coef_l / len(children)
                coef_new = G.nodes[child]['coef'] + delta
                G.add_node(child, coef=coef_new)


    def _undo_projection(self):
        self.G = self._G
        self._has_been_projected = False



def rescale_layout(pos, scale=1):
    """Returns scaled position array to (-scale, scale) in all axes.
    The function acts on NumPy arrays which hold position information.
    Each position is one row of the array. The dimension of the space
    equals the number of columns. Each coordinate in one column.
    To rescale, the mean (center) is subtracted from each axis separately.
    Then all values are scaled so that the largest magnitude value
    from all axes equals `scale` (thus, the aspect ratio is preserved).
    The resulting NumPy Array is returned (order of rows unchanged).
    Parameters
    ----------
    pos : numpy array
        positions to be scaled. Each row is a position.
    scale : number (default: 1)
        The size of the resulting extent in all directions.
    Returns
    -------
    pos : numpy array
        scaled positions. Each row is a position.
    See Also
    --------
    rescale_layout_dict
    """
    # Find max length over all dimensions
    lim = 0  # max coordinate for all axes
    for i in range(pos.shape[1]):
        pos[:, i] -= pos[:, i].mean()
        lim = max(abs(pos[:, i]).max(), lim)
    # rescale to (-scale, scale) in all directions, preserves aspect
    if lim > 0:
        for i in range(pos.shape[1]):
            pos[:, i] *= scale / lim
    return pos



def _process_params(G, center, dim):
    # Some boilerplate code.
    import numpy as np

    if not isinstance(G, nx.Graph):
        empty_graph = nx.Graph()
        empty_graph.add_nodes_from(G)
        G = empty_graph

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)

    if len(center) != dim:
        msg = "length of center coordinates must match dimension of layout"
        raise ValueError(msg)

    return G, center



def multipartite_layout(G, subset_key="subset", align="vertical", scale=1, center=None):
    """Position nodes in layers of straight lines.
    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.
    subset_key : string (default='subset')
        Key of node data to be used as layer subset.
    align : string (default='vertical')
        The alignment of nodes. Vertical or horizontal.
    scale : number (default: 1)
        Scale factor for positions.
    center : array-like or None
        Coordinate pair around which to center the layout.
    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node.
    Examples
    --------
    >>> G = nx.complete_multipartite_graph(28, 16, 10)
    >>> pos = nx.multipartite_layout(G)
    Notes
    -----
    This algorithm currently only works in two dimensions and does not
    try to minimize edge crossings.
    Network does not need to be a complete multipartite graph. As long as nodes
    have subset_key data, they will be placed in the corresponding layers.
    """
    import numpy as np

    if align not in ("vertical", "horizontal"):
        msg = "align must be either vertical or horizontal."
        raise ValueError(msg)

    G, center = _process_params(G, center=center, dim=2)
    if len(G) == 0:
        return {}

    layers = {}
    for v, data in G.nodes(data=True):
        try:
            layer = data[subset_key]
        except KeyError:
            msg = "all nodes must have subset_key (default='subset') as data"
            raise ValueError(msg)
        layers[layer] = [v] + layers.get(layer, [])

    # Sort by layer, if possible
    try:
        layers = sorted(layers.items())
    except TypeError:
        layers = list(layers.items())

    pos = None
    nodes = []
    width = len(layers)
    for i, (_, layer) in enumerate(layers):
        layer = list(sorted(layer))
        height = len(layer)
        xs = np.repeat(i, height)
        ys = np.arange(0, height, dtype=float)
        offset = ((width - 1) / 2, (height - 1) / 2)
        layer_pos = np.column_stack([xs, ys]) - offset
        if pos is None:
            pos = layer_pos
        else:
            pos = np.concatenate([pos, layer_pos])
        nodes.extend(layer)
    pos = rescale_layout(pos, scale=scale) + center
    if align == "horizontal":
        pos = pos[:, ::-1]  # swap x and y coords
    pos = dict(zip(nodes, pos))
    return pos



def group_bi_by_size(bi, reverse=True):
    out = defaultdict(lambda: [])
    for k, v in bi.items():
        size, hash_ = k
        out[size].append(k)

    sizes = list(sorted((out.keys())))
    if reverse:
        sizes = list(reversed(sizes))
    return sizes, dict(out)
