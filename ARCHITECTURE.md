# Package boundary

Buhito owns algorithms on generic graphs and numerical data. It never imports
MinervaChem, RDKit, or Architector. MinervaChem depends on Buhito and supplies
chemical meaning and molecular adapters.

| Buhito | MinervaChem |
| --- | --- |
| BFS/DFS graphlet enumeration and hashing | RDKit/SMILES/MOL2 conversion |
| Configurable graph labels and graphlet features | Atom/bond labels, hydrogen handling, element filtering |
| Sparse feature matrices and feature elimination | Molecular fingerprinter adapters |
| Hierarchical regression, PADRE, unseen-feature correction | Chemical property datasets and benchmarks |
| Graphlet inclusion DAGs and general coefficient projection | Bond-order/valence weights and molecular rendering |
| MCTS search over a state interface | Molecular search states, rewards, and search visualization |
| General graph and regression plots, numerical utilities | Chemical notebooks, QM9 preprocessing, molecular pipelines |
| Reddit and synthetic PADRE examples | Molecular examples |

The canonical graph algorithms live in `breadth_graphlet.py` and
`depth_graphlet.py`. Featurizer classes call these implementations. The default
attribute names are `node_label` and `edge_label`; missing labels have constant
value `1`, allowing plain NetworkX graphs. Use `node_key` and `edge_key` to choose
attributes, or `None` to ignore labels. Attribute values must be hashable.

MinervaChem explicitly selects `atom_key` and `bond_key`. Its historical molecular
fingerprint IDs and filtering behavior are preserved. Its old general-purpose
imports (for example `minervachem.regressors` and `minervachem.padre`) re-export
Buhito objects for compatibility; implementation belongs to Buhito.

## Migration

- General graph code should import directly from `buhito`.
- Callers using old graph labels can pass `node_key="atom_key"` and
  `edge_key="bond_key"` explicitly, or rename their graph attributes.
- The chemical converter moved to `minervachem.converters.smiles_to_nx`.
- QM9 code/data moved to `minervachem/demos/qm9/`.
- The synthetic PADRE notebook moved to `examples/padre/padre_example.ipynb`.
- Historical Reddit benchmark outputs retain their original column names; new
  runs use graph terminology.

Install `buhito[ml]` for optional FLAML/LightGBM tuning, Seaborn coefficient plots,
significant-figure formatting, and sparse/distributed PADRE mappers. Their imports
are lazy, so the graph library and basic regressors work without these engines.
The optional Dask/LightGBM predictor additionally needs a Dask dataframe stack
compatible with the installed pandas. Recent Dask dataframe releases require
pandas 2, while these projects currently pin pandas below 2. The Dask array
mappers work with the current environment; the dataframe-dependent LightGBM
integration test skips when that optional stack cannot import.

Install `buhito[demos]` for Jupyter execution and Python Graphviz bindings. The
plotting demo also needs the system `dot` executable.

MinervaChem's `uv` development source points to `../buhito`, so keep the two
checkouts beside each other on the HPC server as well as locally. Published
MinervaChem metadata declares the dependency on `buhito[ml]`; release the updated
Buhito before releasing MinervaChem. The local source setting is for development
and is not encoded into the built wheel's requirements.

`tests/test_architecture.py` imports every Buhito module with chemistry imports
blocked, exercises unlabeled/custom-labeled graphs, generic regression and search,
and checks projection behavior. MinervaChem tests preserve baseline molecular
fingerprints and exercise the adapters.
