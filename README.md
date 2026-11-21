# buhito
<img src="https://raw.githubusercontent.com/lanl/buhito/main/assets/buhito_icon.svg" alt="Logo" width="500">

**buhito** is a Python library for enumerating graphlets and analyzing graphlet-based features from graphs.

## Installation
```bash
git clone https://github.com/lanl/buhito.git
cd buhito
uv sync --all-extras
```

## Quick Start

### Enumerate graphlets with BFS

```python
import networkx as nx
from buhito import generate_subgraphs_breadthwise

G = nx.petersen_graph()

nx.set_node_attributes(G, {n: "node" for n in G.nodes}, "node_label")
nx.set_edge_attributes(G, {e: "1" for e in G.edges}, "edge_label")

fp, bitinfo = generate_subgraphs_breadthwise(
    G,
    depth=4,
    return_nodewise=False,
    full_hash=True,
    node_key="node_label",
    edge_key="edge_label"
)

print("Graphlet fingerprint length:", len(fp))
```

### Enumerate graphlets with DFS

```python
from buhito import generate_subgraphs_depthwise

subsets, counts = generate_subgraphs_depthwise(G, maxlen=4)
print("Found graphlet subsets:", len(subsets))
```

## Examples

Available example scripts:

- `examples/plotting_example.py` — visualize graphlets from the Petersen graph
- [Reddit graph classification and enumeration benchmark](examples/reddit_example/README.rst)

From the repository root:

```sh
uv run python examples/reddit_example/reddit_graphlets_example.py --data-root path/to/data_root
uv run python examples/reddit_example/reddit_individual_graphs_benchmark.py path/to/graph_pickles
```

Buhito provides general graph algorithms and has no MinervaChem or RDKit dependency.
Chemical conversion and the QM9 benchmark live in MinervaChem, which depends on
Buhito. The SMILES converter is now `minervachem.converters.smiles_to_nx`.

See [package boundaries and migration notes](ARCHITECTURE.md) and
[local/HPC demo runners](scripts/README.md).

## Testing

```bash
pytest tests/
```

## Package structure

- `src/buhito/` — core features
- `src/buhito/plotting.py` — graphlet visualization helpers

## Authors

- Yulia Pimonova (LANL)
- Nicholas Lubbers (LANL)
- Nathan Lemons (LANL)
- Pieter Swart (LANL)

## Citation

If you use `buhito` in your work, please cite this repository. A publication citation is coming.

## Copyright and licensing

`buhito` is released under the BSD-3 License. See `LICENSE.txt` for the full license.


The copyright to `buhito` is owned by Triad National Security, LLC
and is released for open source use as project number O5043.


© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are.
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare.
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit.
others to do so.
