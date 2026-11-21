"""Exercise the Reddit workflow with chemistry packages unavailable."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_reddit_workflows_without_chemistry(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    script = r'''
        import importlib.abc
        import pathlib
        import runpy
        import sys

        class NoChemistry(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'minervachem', 'rdkit'}:
                    raise ImportError(f'Chemistry dependency forbidden: {fullname}')

        sys.meta_path.insert(0, NoChemistry())
        import networkx as nx
        import numpy as np

        repo, work = map(pathlib.Path, sys.argv[1:])
        example = runpy.run_path(str(repo / 'examples/reddit_example/reddit_graphlets_example.py'))
        dataset = work / 'REDDIT-MULTI-5K'
        dataset.mkdir()
        # Two graph classes with 10 examples each; all nodes use global 1-based IDs.
        edges, graph_ids, labels = [], [], []
        for i in range(20):
            graph = nx.path_graph(4) if i % 2 else nx.star_graph(3)
            edges.extend(f'{4*i+u+1},{4*i+v+1}\n' for u, v in graph.edges)
            graph_ids.extend([f'{i+1}\n'] * 4)
            labels.append(f'{i % 2 + 1}\n')
        for suffix, rows in [('edges', edges), ('graph_idx', graph_ids), ('graph_labels', labels)]:
            (dataset / f'REDDIT-MULTI-5K.{suffix}').write_text(''.join(rows))

        graphs, y = example['load_reddit_multi_5k_graphs'](example['DataConfig'](data_root=str(work)))
        assert len(graphs) == 20
        assert all('node_symbol' not in attrs for g in graphs for _, attrs in g.nodes(data=True))
        train = list(range(16))
        test = list(range(16, 20))
        transformer = example['build_featurizer'](example['FingerprintConfig'](n_jobs=1))
        X_train = transformer.fit_transform([graphs[i] for i in train]).toarray()
        X_test = example['featurize_graphs'](transformer, [graphs[i] for i in test])
        assert X_train.shape[0] == 16 and X_train.shape[1] > 0
        assert X_test.shape == (4, X_train.shape[1])
        assert np.isfinite(X_train).all()
        example['train_and_evaluate'](X_train, y[train], X_test, y[test], example['TrainConfig']())

        benchmark = runpy.run_path(str(repo / 'examples/reddit_example/reddit_individual_graphs_benchmark.py'))
        results = benchmark['run_benchmark']([('tiny', graphs[0])], [2], repeats=1, out_dir=str(work / 'benchmark'))
        assert set(results.algorithm) == {'bfs', 'dfs'}
        assert results.total_graphlets.nunique() == 1
        assert (work / 'benchmark/benchmark_results.csv').exists()
        assert 'minervachem' not in sys.modules
        assert 'rdkit' not in sys.modules
    '''
    env = dict(os.environ, PYTHONPATH=str(repo / 'src'), MPLBACKEND='Agg', MPLCONFIGDIR=str(tmp_path / 'mpl'))
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script), str(repo), str(tmp_path)],
        env=env, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
