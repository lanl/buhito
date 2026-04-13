Example workflow for computing graphlet fingerprints on the
REDDIT-MULTI-5K graph classification dataset and training a simple classifier.
The main workflow is a single top-to-bottom script: reddit_graphlets_example.py


**Requirements** (Python packages)
    - numpy
    - pandas
    - networkx
    - scikit-learn
    - minervachem  (for GraphletFingerprinter & FingerprintFeaturizer)

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

