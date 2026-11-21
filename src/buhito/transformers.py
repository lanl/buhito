from time import time

import numpy as np
import scipy as sp

from joblib import Parallel, delayed, effective_n_jobs

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from tqdm.auto import tqdm

from buhito.utilities import uniquify_lol, evenly_distribute_jobs
from buhito.sparse_eliminator import map_useless_features_sparse


class GraphletTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,
                 featurizer,
                 return_dense=False,
                 return_float=False,
                 n_jobs=None,
                 chunk_size='even',
                 verbose=0,
                 retain_bit_info=True,
                ):

        """sklearn transformer that applies buhito featurizers to networkx graphs.

        Args:
            featurizer: a buhito.featurizers.Featurizer object
            return_dense (bool): whether transform returns a dense matrix
            return_float (bool):  whether transform returns a matrix of floats
            n_jobs (int): number of joblib jobs
            chunk_size (int or string): joblib chunk size. If "even", evaluates to n_graphs // n_cores
            verbose (int): verbosity level (currently 0 or 1)
            retain_bit_info (bool): retain graphlet locations for interpretation;
                disable for count-only ML to reduce memory.

        Attributes:
            self.n_bits_ (int): number of fingerprint fragments found during fit. Sorted like self.bit_ids_
            self.bit_ids_ (list): list of fragment identifiers for all fragments found in fit. Sorted in increasing order
            self.bit_sizes_ (list): list of sizes of each fragment found during fit. Sorted like self.bit_ids_
            self.bit_indices_ (dict): maps bit IDs to their column index
            self.n_unseen_ (int): number of new fingerprint fragments found during transform. Sorted like self.bit_ids_
            self.X_unseen_ (np.ndarray or sp.sparse.matrix): matrix of fingerprint fragments found during transform
            self.bit_indices_unseen_ (dict): maps unseen bit IDs to their column index
            self.bit_sizes_unseen_ (list): list of sizes of fragments found during transform.
            self.bi_fit_ (list): list of bit maps (map bit IDs to to lists of nodes) found during fit
            self.bi_transform_ (list): list of bit maps (map bit IDs to to lists of nodes) found during transform
            self.transform_time_ (float): time per graph taken during transform in seconds
        """
        self.featurizer = featurizer
        self.retain_bit_info = retain_bit_info
        self.return_dense = return_dense
        self.return_float = return_float
        self.n_jobs = n_jobs
        self.chunk_size = chunk_size
        self.verbose = verbose
        self._cache = {}
        self._in_fit_transform = False

        # set during .fit
        self.n_bits_ = None
        self.bit_ids_ = None
        self.bit_sizes_ = None
        self.bit_indices_ = None

        # set during.transform
        self.n_unseen_ = None
        self.X_unseen_ = None
        self.bit_ids_unseen_ = None
        self.bit_indices_unseen_ = None
        self.bit_sizes_unseen_ = None

    def fit(self, X, y=None):
        """Find feature vectors in iterable of graphs

        Note: It is highly recommended to call from fit_transform with large datasets, as fingerprints
              must be recomputed during transform otherwise.

        Args:
            X: an iterable of graphs
            y: exists for compatibility with sklearn pipelines, does nothing

        Returns:
            self (FingerprintFeaturizer): this instance
        """

        if self.verbose:
            print(f'Finding fingerprints in fit')
        X = self._as_list(X)
        fps, self.bi_fit_, bit_ids = self._get_fps(X)

        # unnest, get unique BitIDs, and sort bit IDs

        self.bit_ids_ = sorted(uniquify_lol(bit_ids))
        (self.bit_indices_,
         self.bit_sizes_) = self._get_bit_indices_and_sizes(self.bit_ids_)
        self.n_bits_ = len(self.bit_ids_)

        if self.verbose:
            print(f'N bits: {self.n_bits_}')

        # the cache is used in between fit and transform
        # during self.fit_transform to avoid computing the FPs twice
        if self._in_fit_transform:
            self._cache['fps'] = fps
        self.is_fitted_ = True
        return self

    def transform(self, X, return_unseen=False):
        """Transform an iterable of graphs to a matrix of fingerprints.

        Must be called after .fit

        Args:
            X: an iterable of graphs
            return_unseen: (bool) if True, return X_unseen containing fragments not seen during .fit
        Returns:
            X (np.ndarray or sp.sparse.matrix of int or float): a matrix, format depends on self.return_sparse and
                                                                return_float
            X_unseen (optional): like X, but for fragments not identified during fit
        """
        X = self._as_list(X)
        start = time()
        check_is_fitted(self, "is_fitted_")
        if self._in_fit_transform:
            fps = self._cache['fps']
            # bits_seen = self.bit_ids_
            self.bit_ids_unseen_ = []
        else:
            gs = self._as_list(X)
            self._print('finding fingerprints in transform')
            fps, self.bi_transform_, bits = self._get_fps(gs)
            bits = sorted(uniquify_lol(bits))
            self.bit_ids_unseen_ = sorted(list(set(bits) - set(self.bit_ids_)))
            self.n_unseen_ = len(self.bit_ids_unseen_)
            if self.n_unseen_ > 0:
                (self.bit_indices_unseen_,
                 self.bit_sizes_unseen_) = self._get_bit_indices_and_sizes(self.bit_ids_unseen_)
            else:
                self.bit_indices_unseen_, self.bit_sizes_unseen_ = {}, []

        self._print('Converting bits to array form')
        M_seen = self._to_sparse(fps, self.bit_indices_)
        if self.bit_ids_unseen_:
            self._print('Converting unseen bits to array form')
            self.X_unseen_ = self._to_sparse(fps, self.bit_indices_unseen_)
        else:
            self.X_unseen_ = None
        if self.return_dense:
            self._print('Converting from sparse to dense')
            M_seen = M_seen.toarray()
            if self.X_unseen_ is not None:
                self.X_unseen_ = self.X_unseen_.toarray()
        else:
            M_seen = M_seen.tocsr()
            if self.X_unseen_ is not None:
                self.X_unseen_ = self.X_unseen_.tocsr()

        if self.return_float:
            M_seen = M_seen.astype(float)
            if self.X_unseen_ is not None:
                self.X_unseen_ = self.X_unseen_.astype(float)

        self.transform_time_ = (time() - start) / max(1, len(X))  # .shape[0]
        self._print(f"Sparse Transform time: {round(self.transform_time_, 3)} s/graph")

        if return_unseen:
            return M_seen, self.X_unseen_
        else:
            return M_seen

    def fit_transform(self, X, y=None, fit_params=None):
        """Fit to iterable of graphs, then transform it return a matrix of fingerprints

        Args:
            X: an iterable of graphs
            y: exists for compatibility with sklearn pipelines, does nothing
            fit_params: exists for compatibility with sklearn pipelines, does nothing

        Returns:
            X (np.ndarray or sp.sparse.matrix of int or float): a matrix, format depends on self.return_sparse and
                                                                return_float
            X_unseen (optional): like X, but for fragments not identified during fit
        """
        X = self._as_list(X)
        self._in_fit_transform = True
        self.fit(X, y)
        out = self.transform(X)
        self._in_fit_transform = False
        self._cache = {}
        return out

    def extract_smaller_fp(self, fps, new_max_subgraph_size=3, return_bit_ids_updated=False):
        if self.featurizer.size < new_max_subgraph_size:
            print("Selected max substructure size is bigger than in available fingerprinter.\n "
                  "Please, invoke new fingerprinting procedure.")
            return

        inds_of_smaller_size = np.flatnonzero(self.bit_sizes_ <= new_max_subgraph_size)
        new_fps = fps.tocsc()[:, inds_of_smaller_size].tocsr()
        if return_bit_ids_updated:
            updated_bits_ids = [self.bit_ids_[i] for i in inds_of_smaller_size]
            return new_fps, updated_bits_ids
        else:
            return new_fps


    def get_feature_names_out(self):
        check_is_fitted(self, "is_fitted_")
        return np.array([str(b) for b in self.bit_ids_], dtype=object)

    @staticmethod
    def _get_bit_indices_and_sizes(bit_ids):
            """Given a list of bit ids

            return
                * dict mapping list elements to indexes (for constant time arg lookup)
                * np array  of size elements of the IDs
            """
            # error check for the structure of bit_ids
            if bit_ids and not (isinstance(bit_ids[0], tuple) and len(bit_ids[0]) == 2):
                raise ValueError("Expected bit_ids to be 2-tuples like (size, id).")

            # this gives us constant time lookup of the bit IDS
            bit_indices = {v: i for i, v in enumerate(bit_ids)}

            # a list of bit sizes
            bit_sizes = np.asarray([bit[0] for bit in bit_ids], dtype=int)

            return bit_indices, bit_sizes

    def _get_fps(self, gs):
        """Applies self._get_fp in parallel over graphs
        :param gs: iterable of graphs
        :return: Tuple[List[Dict], List[Dict], List[List[Tuple]]]

                 The elemenets of these tuple are:
                     * List of fingerprints
                     * List of bit maps (map bit IDs to to lists of nodes)
                     * List of bit IDs
        """

        # progress_bar = tqdm if self.verbose else lambda x: x

        if self.chunk_size == 'even':
            real_n_jobs = effective_n_jobs(self.n_jobs)
            batch_size = evenly_distribute_jobs(len(gs), real_n_jobs)
        else:
            batch_size = self.chunk_size
        p = Parallel(n_jobs=self.n_jobs,
                     verbose=self.verbose if self.verbose else 0,
                     prefer='processes',
                     batch_size=batch_size,
                     return_as='generator',
                     pre_dispatch='n_jobs',
                     # backend='loky',
                    )
        if self.verbose:
            print(f"Number of cores used: {effective_n_jobs(p.n_jobs)}")
        f = delayed(self._get_fp)

        fps = tqdm(p(f(G,
                  self.featurizer,
                  self.retain_bit_info,
                 )
                for G in gs), desc="Constructing Fingerprints", total=len(gs), disable=not self.verbose)
        results = list(fps)
        if not results:
            return (), (), ()
        fps, bis, bits = zip(*results)
        return fps, bis, bits

    @staticmethod
    def _get_fp(G,
                featurizer,
                retain_bit_info=True,
               ):
        """Calls the featurizer on an individual graph,
        returning a fingerprint and a list of tuples of form (bit, r)
        """
        if not retain_bit_info and hasattr(featurizer, 'counts_only'):
            fp, bi = featurizer.counts_only(G), None
        else:
            fp, bi = featurizer(G)
            if not retain_bit_info:
                bi = None
        bits = list(fp.keys())
        return fp, bi, bits

    def _to_sparse(self, fps, used_bits): #COO instead of DOK
        rows, cols, data = [], [], []
        for i, fp in tqdm(enumerate(fps),
                          'Converting FPs to sparse',
                          total=len(fps),
                          disable=not self.verbose):
            for bit, count in fp.items():
                j = used_bits.get(bit)
                if j is not None:
                    rows.append(i); cols.append(j); data.append(count)
        return sp.sparse.coo_matrix((data, (rows, cols)), shape=(len(fps), len(used_bits)), dtype=int)


    @staticmethod
    def _as_list(X):
        return X if isinstance(X, list) else list(X)

    def _print(self, msg):
        if self.verbose:
            print(msg)



class SparseFeatureEliminator(BaseEstimator, TransformerMixin):

    def __init__(self, overflow_action='warn', verbose=False):
        self.overflow_action = overflow_action
        self.verbose = verbose

    def fit(self, X, y=None):
        if self.verbose:
            print('sparse feature elim')
        self.useful_, self.mapr_ = map_useless_features_sparse(
                                        X,
                                        overflow_action=self.overflow_action
                                    )
        self.n_useful_ = self.useful_.sum()
        self.frac_useful_ = self.useful_.mean()
        return self

    def transform(self, X):
        return X[:, self.useful_]
