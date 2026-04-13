import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


def to_pydot_custom(G, node_label_attr="atom_key", edge_label_attr="bond_key", show_edge_labels=False, 
                    rankdir="LR"): # LR, TB, BT, RL):
    # prog = dot, neato, fdp, sfdp, etc. (for plotting)
    P = nx.nx_pydot.to_pydot(G)
    P.set_rankdir(rankdir)
    P.set_node_defaults(shape="circle")  
    P.set_edge_defaults() 

    for n in G.nodes:
        pd_node = P.get_node(str(n))
        if not pd_node:
            continue
        pd_node = pd_node[0]
        lbl = G.nodes[n].get(node_label_attr, str(n))
        pd_node.set_label(str(lbl))
        pd_node.set_shape("circle")

    if show_edge_labels:
        if G.is_multigraph():
            for u, v, k, data in G.edges(keys=True, data=True):
                lbl = data.get(edge_label_attr, "")
                pd_edges = P.get_edge(str(u), str(v))
                if pd_edges:
                    pd_edges[0].set_label(str(lbl))
        else:
            for u, v, data in G.edges(data=True):
                lbl = data.get(edge_label_attr, "")
                pd_edges = P.get_edge(str(u), str(v))
                if pd_edges:
                    pd_edges[0].set_label(str(lbl))

    return P

def get_graphs_with_graphlet(bit, 
                             bit_ids, 
                             X): 
    """
    Outputs the indices of the graphs with at least one instance of graphlet identified by its hash 
    """
    i = bit_ids.index(bit)
    ii, jj = X.nonzero()
    rows_with_bit = ii[np.where(jj==i)]
    return rows_with_bit

def draw_graphlet(bit=None, gr=None, node_ixs=None, gr_idx=0,
                  featurizer=None, X=None, train_df=None,
                  node_label_attr="atom_symbol", edge_label_attr="bond_key"):
    """
    Draw a graphlet as a pydot graph.

    a) node_ixs and gr define an induced subgraph, the function draws it
    b) if only bit is provided, find a molecule containing the bit using get_graphs_with_graphlet function, 
    extract the node indices, build the induced subgraph and return a pydot graph.

    Parameters:
    - bit: a bit identifier (tuple (num of nodes, hash)) or an integer index in bit_ids_
    - gr: a NetworkX graph
    - node_ixs: iterable of node indices to keep
    - gr_idx: index selecting which molecule to use (default 0).
    - featurizer, X, train_df: objects after featurization.
    """
    if node_ixs is not None:
        if gr is None:
            raise ValueError("Graph must be provided when node_ixs is provided.")
        sub = gr.subgraph(node_ixs).copy()
        return to_pydot_custom(sub, node_label_attr=node_label_attr, edge_label_attr=edge_label_attr, show_edge_labels=True)

    if bit is None:
        raise ValueError("Either node_ixs or bit must be provided.")

    if isinstance(bit, int):
        bit = featurizer.bit_ids_[bit]

    # find graphs with bit
    grs_with_bit = get_graphs_with_graphlet(bit, featurizer.bit_ids_, X)
    if len(grs_with_bit) == 0:
        raise ValueError(f"No graphs found containing bit {bit}")

    if gr_idx >= len(grs_with_bit):
        print(f"gr_idx {gr_idx} is out of range for the number of graphs containing the bit. Defaulting to the first graph.")
        gr_idx = 0
    chosen = grs_with_bit[gr_idx]
    graph_full = train_df.iloc[chosen]['nxg']
    try:
        node_inds = featurizer.bi_fit_[chosen][bit][0]
    except Exception as e:
        raise RuntimeError("Could not extract node indices for the requested bit from featurizer.bi_fit_.") from e
    inds_to_keep = set(node_inds)
    subgraph = graph_full.subgraph(inds_to_keep).copy()
    return to_pydot_custom(subgraph, node_label_attr=node_label_attr, edge_label_attr=edge_label_attr, show_edge_labels=True)