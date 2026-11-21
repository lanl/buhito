"""Keep memory optimizations compatible with graphlet outputs and job settings."""
from pathlib import Path
import runpy
import subprocess
from types import SimpleNamespace

import networkx as nx
import pytest

from buhito import demo_runner
from buhito.breadth_graphlet import generate_subgraphs_breadthwise
from buhito.featurizers.dfs_graphlet_featurizer import DFSGraphletFeaturizer

REPO = Path(__file__).resolve().parents[1]
BENCHMARK = REPO / 'examples/reddit_example/reddit_individual_graphs_benchmark.py'


@pytest.mark.parametrize('size', [1, 2, 3, 4])
def test_graphlet_outputs_survive_frontier_removal(size):
    graph = nx.wheel_graph(7)
    nx.set_node_attributes(graph, dict(graph.degree), 'node_label')
    counts, locations, node_counts, node_locations = generate_subgraphs_breadthwise(graph, size)
    plain_counts, plain_locations = generate_subgraphs_breadthwise(graph, size, return_nodewise=False)
    expected_counts, expected_locations = DFSGraphletFeaturizer(size)(graph)
    assert counts == plain_counts == expected_counts
    normalize = lambda mapping: {bit: set(map(frozenset, nodes)) for bit, nodes in mapping.items()}
    assert normalize(locations) == normalize(plain_locations) == normalize(expected_locations)
    for node in graph:
        for bit, nodes in locations.items():
            expected = {frozenset(group) for group in nodes if node in group}
            assert node_counts.get(node, {}).get(bit, 0) == len(expected)
            assert set(map(frozenset, node_locations.get(node, {}).get(bit, []))) == expected


def test_timing_calls_once_and_supports_no_warmup():
    module = runpy.run_path(str(BENCHMARK))
    calls = []
    def fn():
        calls.append(1)
        return {'location': []}, {(1, 123): 2}
    times, counts = module['time_call'](fn, warmup=0, repeats=2)
    assert len(times) == len(calls) == 2
    assert counts == {(1, 123): 2}
    def fail():
        calls.append(1)
        raise RuntimeError('algorithm failed')
    with pytest.raises(RuntimeError, match='algorithm failed'):
        module['time_call'](fail, warmup=0, repeats=1)
    assert len(calls) == 3


def test_single_job_submission_requests_256g(monkeypatch):
    submitted = []
    def capture(command, **kwargs):
        submitted.append(command)
        return SimpleNamespace(stdout='fake-job', returncode=0)
    monkeypatch.setattr(demo_runner.subprocess, 'run', capture)
    demo_runner.main(['--project', str(REPO), '--demo-root', 'examples', '--demo', str(BENCHMARK.relative_to(REPO))])
    assert len(submitted) == 1
    assert submitted[0][submitted[0].index('--mem') + 1] == '256G'
    assert str(BENCHMARK.relative_to(REPO)) in submitted[0]
