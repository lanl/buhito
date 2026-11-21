import pytest
import networkx as nx
from buhito.featurizers.bfs_graphlet_featurizer import BFSGraphletFeaturizer
from buhito.featurizers.dfs_graphlet_featurizer import DFSGraphletFeaturizer
from buhito.transformers import GraphletTransformer


class TestBFSGraphletFeaturizer:
    def test_init(self):
        featurizer = BFSGraphletFeaturizer(max_len=3)
        assert featurizer.size == 3
        assert featurizer.return_nodewise is True

    def test_call_petersen_graph(self):
        G = nx.petersen_graph()
        labels = {n: "node" for n in G.nodes}
        nx.set_node_attributes(G, labels, "node_label")
        labels_e = {n: f"1" for n in G.edges}
        nx.set_edge_attributes(G, labels_e, "edge_label")

        featurizer = BFSGraphletFeaturizer(max_len=3, return_nodewise=False)
        fp, bi = featurizer(G)

        assert isinstance(fp, dict)
        assert isinstance(bi, dict)
        assert len(fp) > 0
        assert len(bi) > 0

    def test_call_nodewise(self):
        G = nx.petersen_graph()
        labels = {n: "node" for n in G.nodes}
        nx.set_node_attributes(G, labels, "node_label")
        labels_e = {n: f"1" for n in G.edges}
        nx.set_edge_attributes(G, labels_e, "edge_label")

        featurizer = BFSGraphletFeaturizer(max_len=3, return_nodewise=True)
        fp, bi, node_fps, node_bi = featurizer(G)

        assert isinstance(fp, dict)
        assert isinstance(bi, dict)
        assert isinstance(node_fps, dict)
        assert isinstance(node_bi, dict)
        assert len(node_fps) == len(G.nodes)
        assert len(node_bi) == len(G.nodes)


class TestDFSGraphletFeaturizer:
    def test_init(self):
        featurizer = DFSGraphletFeaturizer(max_len=3)
        assert featurizer.size == 3

    def test_call_petersen_graph(self):
        G = nx.petersen_graph()
        labels = {n: "node" for n in G.nodes}
        nx.set_node_attributes(G, labels, "node_label")
        labels_e = {n: f"1" for n in G.edges}
        nx.set_edge_attributes(G, labels_e, "edge_label")

        featurizer = DFSGraphletFeaturizer(max_len=3)
        fp, bi = featurizer(G)

        assert isinstance(fp, dict)
        assert isinstance(bi, dict)
        assert len(fp) > 0
        assert len(bi) > 0


class TestFeaturizerConsistency:
    def test_bfs_dfs_consistency(self):
        G = nx.petersen_graph()
        labels = {n: "node" for n in G.nodes}
        nx.set_node_attributes(G, labels, "node_label")
        labels_e = {n: f"1" for n in G.edges}
        nx.set_edge_attributes(G, labels_e, "edge_label")

        bfs_featurizer = BFSGraphletFeaturizer(max_len=3, return_nodewise=False)
        dfs_featurizer = DFSGraphletFeaturizer(max_len=3)

        bfs_fp, bfs_bi = bfs_featurizer(G)
        dfs_fp, dfs_bi = dfs_featurizer(G)

        assert set(bfs_fp.keys()) == set(dfs_fp.keys())


class TestGraphletTransformer:
    def test_init(self):
        featurizer = BFSGraphletFeaturizer(max_len=3)
        transformer = GraphletTransformer(featurizer=featurizer)
        assert transformer.featurizer == featurizer

    def test_fit_transform(self):
        G = nx.petersen_graph()
        labels = {n: "node" for n in G.nodes}
        nx.set_node_attributes(G, labels, "node_label")
        labels_e = {n: f"1" for n in G.edges}
        nx.set_edge_attributes(G, labels_e, "edge_label")

        featurizer = BFSGraphletFeaturizer(max_len=3, return_nodewise=False)
        transformer = GraphletTransformer(featurizer=featurizer, n_jobs=-1)

        X = transformer.fit_transform([G])

        assert hasattr(transformer, 'n_bits_')
        assert hasattr(transformer, 'bit_ids_')
        assert X.shape[0] == 1
        assert X.shape[1] == transformer.n_bits_
