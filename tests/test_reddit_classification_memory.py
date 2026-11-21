"""Verify bounded feature extraction without changing graphlet counts."""
import gc
import itertools
from pathlib import Path
import runpy
import weakref
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from scipy import sparse

from buhito import demo_runner
from buhito.depth_graphlet import HashHelper
from buhito.featurizers.dfs_graphlet_featurizer import DFSGraphletFeaturizer, iter_connected_subgraphs
from buhito.transformers import GraphletTransformer

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('size', [1, 2, 3, 4])
def test_streaming_connected_sets_and_counts_match_full_output(size):
    for seed in range(5):
        graph = nx.gnp_random_graph(8, 0.4, seed=seed)
        nx.set_node_attributes(graph, dict(graph.degree), 'node_label')
        nx.set_edge_attributes(graph, 'link', 'edge_label')
        groups = list(iter_connected_subgraphs(graph, size))
        expected = {frozenset(nodes) for length in range(1, size + 1)
                    for nodes in itertools.combinations(graph, length)
                    if nx.is_connected(graph.subgraph(nodes))}
        assert len(groups) == len(set(groups))
        assert set(groups) == expected
        featurizer = DFSGraphletFeaturizer(size)
        assert featurizer.counts_only(graph) == featurizer(graph)[0]


def test_hash_cache_does_not_keep_finished_graphs_alive():
    graph = nx.path_graph(5)
    reference = weakref.ref(graph)
    helper = HashHelper(graph, 3)
    helper(frozenset({0, 1, 2}))
    del helper, graph
    gc.collect()
    assert reference() is None


def test_count_only_transformer_matches_features_without_locations():
    graphs = [nx.path_graph(4), nx.cycle_graph(4)]
    full = GraphletTransformer(DFSGraphletFeaturizer(), n_jobs=1, chunk_size=1)
    lean = GraphletTransformer(DFSGraphletFeaturizer(), n_jobs=1, chunk_size=1, retain_bit_info=False)
    expected = full.fit_transform(graphs)
    actual = lean.fit_transform(graphs)
    assert sparse.issparse(actual)
    assert lean.bit_ids_ == full.bit_ids_
    assert np.array_equal(actual.toarray(), expected.toarray())
    assert all(info is None for info in lean.bi_fit_)
    assert np.array_equal(lean.transform(graphs).toarray(), expected.toarray())
    assert all(info is None for info in lean.bi_transform_)


def test_classifier_resources_and_sparse_output(monkeypatch):
    script = REPO / 'examples/reddit_example/reddit_graphlets_example.py'
    submitted = []
    def capture(command, **kwargs):
        submitted.append(command)
        return SimpleNamespace(stdout='fake', returncode=0)
    monkeypatch.setattr(demo_runner.subprocess, 'run', capture)
    demo_runner.main(['--project', str(REPO), '--demo-root', 'examples', '--demo', str(script.relative_to(REPO))])
    assert submitted[0][submitted[0].index('--mem') + 1] == '128G'
    assert submitted[0][submitted[0].index('--cpus-per-task') + 1] == '4'
    module = runpy.run_path(str(script))
    transformer = module['build_featurizer'](module['FingerprintConfig'](n_jobs=1))
    transformer.fit([nx.path_graph(3)])
    assert sparse.issparse(module['featurize_graphs'](transformer, [nx.path_graph(3)]))
