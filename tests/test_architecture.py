"""Prevent reverse dependencies and chemical assumptions in the graph library."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import networkx as nx
import numpy as np
import pytest
from buhito.featurizers.bfs_graphlet_featurizer import BFSGraphletFeaturizer
from buhito.featurizers.dfs_graphlet_featurizer import DFSGraphletFeaturizer
from buhito.graphlet_dags import GraphletDAG
from buhito.transformers import GraphletTransformer


def test_every_module_imports_without_chemistry(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    code = '''
        import importlib.abc, importlib, sys, pkgutil
        class NoChemistry(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'minervachem', 'rdkit', 'architector'}:
                    raise ImportError('Forbidden chemistry dependency: ' + fullname)
        sys.meta_path.insert(0, NoChemistry())
        import buhito
        from pathlib import Path
        package = Path(buhito.__file__).parent
        for path in package.rglob('*.py'):
            relative = path.relative_to(package).with_suffix('')
            parts = relative.parts[:-1] if relative.name == '__init__' else relative.parts
            importlib.import_module('.'.join(('buhito', *parts)))
        import numpy as np
        from sklearn.linear_model import LinearRegression
        from buhito.regressors import HierarchicalResidualModel
        from buhito.padre import PADRERegressor
        X = np.arange(12.).reshape(6, 2)
        y = X.sum(axis=1)
        model = HierarchicalResidualModel(LinearRegression(), levels=np.array([1, 2]))
        model.fit(X, y)
        assert np.allclose(model.predict(X), y)
        model = PADRERegressor(LinearRegression())
        model.fit(X, y)
        assert np.allclose(model.predict(X), y)
    '''
    result = subprocess.run([sys.executable, '-c', textwrap.dedent(code)],
                            env=dict(os.environ, PYTHONPATH=str(repo / 'src'), MPLCONFIGDIR=str(tmp_path)),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('custom_labels', [False, True])
def test_bfs_dfs_on_generic_graphs(custom_labels):
    graph = nx.cycle_graph(5)
    kwargs = {}
    if custom_labels:
        nx.set_node_attributes(graph, dict(graph.degree), 'role')
        nx.set_edge_attributes(graph, 'friend', 'relation')
        kwargs = dict(node_key='role', edge_key='relation')
    bfs = BFSGraphletFeaturizer(4, return_nodewise=False, **kwargs)
    dfs = DFSGraphletFeaturizer(4, **kwargs)
    assert bfs(graph)[0] == dfs(graph)[0]
    assert dfs(graph)[0] == dfs(nx.relabel_nodes(graph, {i: i + 10 for i in graph}))[0]


def test_unlabelled_transform_and_empty_input():
    transformer = GraphletTransformer(DFSGraphletFeaturizer(), n_jobs=None)
    matrix = transformer.fit_transform([nx.path_graph(3), nx.cycle_graph(3)])
    assert matrix.shape[0] == 2 and matrix.shape[1] > 0
    assert transformer.transform([]).shape == (0, matrix.shape[1])


def test_generic_projection_preserves_total_coefficient():
    _, locations = DFSGraphletFeaturizer()(nx.cycle_graph(3))
    coefficients = {bit: float(bit[0]) for bit in locations}
    expected = sum(coefficients[bit] * len(nodes) for bit, nodes in locations.items())
    dag = GraphletDAG(locations, bit_to_coef_map=coefficients)
    assert np.isclose(dag.project_to_layer(1).sum(), expected)
    assert np.isclose(dag.project_to_layer(2).sum(), expected)


def test_search_with_nonchemical_state():
    from buhito.mcts import Node, utcsearch

    class CountingState:
        def __init__(self, turn=0):
            self.turn = turn
            self.num_moves = 0 if turn else 1

        def terminal(self):
            return self.turn == 1

        def next_state(self):
            self.num_moves = 0
            return CountingState(self.turn + 1)

        def reward(self):
            return 5.0

    root = Node(CountingState())
    chosen = utcsearch(4, root, alpha=1, deterministic=True, scalar=1)
    assert chosen.state.terminal()
    assert root.visits > 1
