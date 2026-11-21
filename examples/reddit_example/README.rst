Example workflow for computing graphlet fingerprints on the
REDDIT-MULTI-5K graph classification dataset and training a simple classifier.
The main workflow is a single top-to-bottom script: reddit_graphlets_example.py


**Requirements** (Python packages)
    - numpy
    - pandas
    - networkx
    - scikit-learn
    - buhito  (for DFSGraphletFeaturizer & GraphletTransformer)

**Dataset**
    One needs the REDDIT-MULTI-5K dataset in the following layout:

        data_root/
            REDDIT-MULTI-5K/
                REDDIT-MULTI-5K.edges
                REDDIT-MULTI-5K.graph_idx
                REDDIT-MULTI-5K.graph_labels

    You can obtain this dataset from the TU Dortmund graph kernel
    benchmark collection.

**Usage**
    python reddit_graphlets_example.py --data-root path/to/data_root

    The script will
    1. load the graphs,
    2. compute graphlet fingerprints for each graph,
    3. train a LogisticRegression classifier,
    4. print a classification report & confusion matrix.


The ``node_label`` attributes are node degrees and all ``edge_label`` attributes
have the same value. No chemical properties, RDKit, or MinervaChem are used.

From the Buhito repository root::

    uv run python examples/reddit_example/reddit_graphlets_example.py --data-root path/to/data_root
    uv run python examples/reddit_example/reddit_individual_graphs_benchmark.py path/to/graph_pickles

The benchmark accepts NetworkX graph pickles with the same node and edge labels.
The Slurm submission script is included; adjust its dataset path and environment
for your cluster. Historical benchmark reports are retained alongside the scripts.
