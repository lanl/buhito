# partition_interpretability.py


import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Draw
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# =============================================================================
# BUHITO
# =============================================================================

from buhito.featurizers.bfs_graphlet_featurizer import (
    BFSGraphletFeaturizer,
)

from buhito.transformers import (
    GraphletTransformer,
)

from buhito.converters import (
    smiles_to_nx,
)

def smiles_to_buhito_graph(
    smiles: str,
    add_hs: bool = True,
    output_2d_pos: bool = False,
):
    """
    Convert a SMILES string to the NetworkX graph representation
    expected by Buhito.
    """

    result = smiles_to_nx(
        smiles,
        add_hs=add_hs,
        output_2d_pos=output_2d_pos,
    )

    # Buhito versions may return either the graph itself
    # or a tuple/list containing the graph.
    if isinstance(
        result,
        (tuple, list),
    ):
        graph = result[0]
    else:
        graph = result

    return graph

# =============================================================================
# PARTITION INTERPRETABILITY - Equation (2)
# =============================================================================

@dataclass
class PartitionInterpretation:
    """
    Stores the partition-wise decomposition from Equation (2):
    f(G) = Σ_p F_p + Σ_{p1,p2} F_{p1,p2} + Σ_{p1,p2,p3} F_{p1,p2,p3} + ...
    
    Attributes:
        within_partition: F_p terms (graphlets within single partition)
        between_partition: F_{p1,p2} terms (graphlets spanning 2 partitions)
        higher_order: F_{p1,p2,p3,...} terms (graphlets spanning 3+ partitions)
        total_prediction: Total model prediction for this molecule
        partition: Mapping from atom index to partition ID
        atom_baseline: Atom-level baseline energy (structure-only mode)
    """
    within_partition: Dict[int, float]
    between_partition: Dict[Tuple[int, int], float]
    higher_order: Dict[Tuple[int, ...], float]
    total_prediction: float
    partition: Dict[int, int]
    atom_baseline: float = 0.0  # NEW: For structure-only mode
    
    @property
    def score(self) -> float:
        """
        Interpretability score: minimize higher-order contributions.
        Higher score = more interpretable (more within-partition, less interaction)
        
        Returns:
            Score in [0, 1], higher is better
        """
        # Use sum of absolute contributions (consistent with breakdown)
        within_total = sum(abs(v) for v in self.within_partition.values())
        between_total = sum(abs(v) for v in self.between_partition.values())
        higher_total = sum(abs(v) for v in self.higher_order.values())
        total = within_total + between_total + higher_total + 1e-10
        
        within_frac = within_total / total
        between_frac = between_total / total
        higher_frac = higher_total / total
        
        # Sum to 1.0
        score = within_frac - 0.5 * between_frac - 2.0 * higher_frac
        
        # Normalize to [0, 1]
        # Best: within=1.0 → score = 1.0
        # Worst: higher=1.0 → score = -2.0
        # Range: [-2.0, 1.0] → map to [0, 1]
        return max(0.0, min(1.0, (score + 2.0) / 3.0))
    
    
    
    def get_contribution_breakdown(self) -> Dict[str, float]:
        """
        Get breakdown of contributions by type.
        
        Returns:
            Dictionary with 'within', 'between', and 'higher' contributions
        """
        within_total = sum(abs(v) for v in self.within_partition.values())
        between_total = sum(abs(v) for v in self.between_partition.values())
        higher_total = sum(abs(v) for v in self.higher_order.values())
        total = within_total + between_total + higher_total + 1e-10
        
        return {
            'within': within_total,
            'between': between_total,
            'higher': higher_total,
            'within_frac': within_total / total,
            'between_frac': between_total / total,
            'higher_frac': higher_total / total
        }



def partition_connectivity_penalty(
    mol: Chem.Mol,
    partition: Dict[int, int],
    disconnected_singleton_weight: float = 2.0
) -> float:
    """
    Penalize disconnected components within a partition.

    A disconnected component containing exactly one atom receives
    disconnected_singleton_weight. Larger disconnected components are
    penalized by their number of atoms.

    An entire partition containing only one atom is not penalized.

    The final penalty is normalized by the total number of atoms.
    """
    if disconnected_singleton_weight < 1.0:
        raise ValueError(
            "disconnected_singleton_weight must be at least 1.0."
        )

    n_atoms = mol.GetNumAtoms()

    if n_atoms == 0:
        return 0.0

    atoms_by_partition = defaultdict(set)

    for atom_idx, partition_id in partition.items():
        atoms_by_partition[int(partition_id)].add(
            int(atom_idx)
        )

    total_penalty = 0.0

    for partition_atoms in atoms_by_partition.values():

        if not partition_atoms:
            continue

        # A one-atom partition is a connected partition and receives
        # no penalty.

        unvisited = set(partition_atoms)
        components = []

        while unvisited:
            start_atom = unvisited.pop()
            queue = deque([start_atom])
            component = {start_atom}

            while queue:
                atom_idx = queue.popleft()
                atom = mol.GetAtomWithIdx(atom_idx)

                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()

                    if (
                        neighbor_idx in partition_atoms
                        and neighbor_idx in unvisited
                    ):
                        unvisited.remove(neighbor_idx)
                        queue.append(neighbor_idx)
                        component.add(neighbor_idx)

            components.append(component)

        # This includes ordinary connected partitions and one-atom
        # partitions.
        if len(components) == 1:
            continue

        largest_component_index = max(
            range(len(components)),
            key=lambda index: len(components[index])
        )

        # Penalize every disconnected component outside the largest
        # connected component.
        for component_index, component in enumerate(components):

            if component_index == largest_component_index:
                continue

            component_size = len(component)

            if component_size == 1:
                total_penalty += disconnected_singleton_weight
            else:
                total_penalty += component_size

    return total_penalty / n_atoms


def connectivity_adjusted_score(
    mol: Chem.Mol,
    interpretation: PartitionInterpretation,
    connectivity_weight: float = 0.10,
    disconnected_singleton_weight: float = 2.0
) -> float:
    """
    Combine the raw interpretability score with the disconnected-component
    penalty.

    Entire partitions containing one atom are not penalized.
    """
    penalty = partition_connectivity_penalty(
        mol=mol,
        partition=interpretation.partition,
        disconnected_singleton_weight=(
            disconnected_singleton_weight
        )
    )

    adjusted_score = (
        interpretation.score
        - connectivity_weight * penalty
    )

    return max(
        0.0,
        min(1.0, adjusted_score)
    )


class PartitionInterpreter:

    def __init__(
        self,
        featurizer,
        model,
        remove_single_atom_contributions=False,
    ):
        """
        Partition interpreter for a fitted Buhito
        GraphletTransformer.
        """

        self.featurizer = (
            featurizer
        )

        self.model = (
            model
        )

        self.remove_single_atom_contributions = bool(
            remove_single_atom_contributions
        )

        # Molecule-dependent caches.
        self._graph_cache = {}

        self._instance_cache = {}
        self._contributing_instance_cache = {}
        self._atom_to_instances_cache = {}

        self._prediction_cache = {}
        self._full_prediction_cache = {}

        self._fingerprint_cache = {}
        self._bitinfo_cache = {}

        # -------------------------------------------------------------
        # Validate fitted Buhito transformer.
        # -------------------------------------------------------------

        if not hasattr(
            featurizer,
            "bit_ids_"
        ):
            raise TypeError(
                "PartitionInterpreter requires a fitted "
                "Buhito GraphletTransformer with bit_ids_."
            )

        if not hasattr(
            featurizer,
            "bit_sizes_"
        ):
            raise AttributeError(
                "The fitted Buhito GraphletTransformer must "
                "expose bit_sizes_."
            )

        self.bit_ids = list(
            featurizer.bit_ids_
        )

        # -------------------------------------------------------------
        # Model coefficients.
        # -------------------------------------------------------------

        coefficients = np.asarray(
            model.coef_,
            dtype=float
        ).reshape(-1).copy()

        if (
            len(coefficients)
            != len(self.bit_ids)
        ):
            raise ValueError(
                f"Dimension mismatch: "
                f"{len(coefficients)} coefficients "
                f"vs {len(self.bit_ids)} "
                "Buhito graphlet features."
            )

        bit_sizes = np.asarray(
            featurizer.bit_sizes_
        ).reshape(-1)

        if (
            len(bit_sizes)
            != len(self.bit_ids)
        ):
            raise ValueError(
                f"Dimension mismatch: "
                f"{len(bit_sizes)} bit sizes "
                f"vs {len(self.bit_ids)} bit IDs."
            )

        self.original_coefficients = (
            coefficients.copy()
        )

        self.coefficients = (
            coefficients.copy()
        )

        # -------------------------------------------------------------
        # Size-1 graphlets.
        # -------------------------------------------------------------

        self.atom_level_indices = set(
            np.flatnonzero(
                bit_sizes == 1
            )
            .astype(int)
            .tolist()
        )

        if not self.atom_level_indices:
            raise ValueError(
                "No size-1 graphlet features were found in "
                "featurizer.bit_sizes_."
            )

        if (
            self
            .remove_single_atom_contributions
        ):

            atom_indices = np.asarray(
                sorted(
                    self.atom_level_indices
                ),
                dtype=int
            )

            self.coefficients[
                atom_indices
            ] = 0.0

    def get_bitinfo(
        self,
        mol: Chem.Mol,
    ) -> Dict:
        """
        Return the Buhito bit-information dictionary for a registered molecule.
        """

        cache_key = (
            self._get_molecule_cache_key(
                mol
            )
        )

        # Populates both the fingerprint and bit-info caches.
        self._get_cached_fingerprint(
            mol
        )

        return self._bitinfo_cache[
            cache_key
        ]

    def register_graph(
        self,
        mol: Chem.Mol,
        graph
    ) -> None:
        """
        Register the Buhito NetworkX graph corresponding to an RDKit molecule.

        The Buhito graph node IDs must exactly match the RDKit atom indices.
        """

        n_atoms = mol.GetNumAtoms()

        expected_nodes = set(
            range(n_atoms)
        )

        actual_nodes = {
            int(node)
            for node in graph.nodes
        }

        if actual_nodes != expected_nodes:

            missing_nodes = (
                expected_nodes
                - actual_nodes
            )

            extra_nodes = (
                actual_nodes
                - expected_nodes
            )

            raise ValueError(
                "Buhito graph node IDs must match "
                "RDKit atom indices exactly. "
                f"Missing nodes: {sorted(missing_nodes)}; "
                f"extra nodes: {sorted(extra_nodes)}."
            )

        cache_key = (
            self._get_molecule_cache_key(
                mol
            )
        )

        self._graph_cache[
            cache_key
        ] = graph

    def _get_cached_graph(
        self,
        mol: Chem.Mol
    ):
        """
        Return the Buhito graph registered for this molecule.
        """

        cache_key = (
            self._get_molecule_cache_key(
                mol
            )
        )

        if (
            cache_key
            not in self._graph_cache
        ):
            raise RuntimeError(
                "No Buhito graph has been registered for this "
                "RDKit molecule. Call interpreter.register_graph("
                "mol, graph) before computing partition contributions."
            )

        return self._graph_cache[
            cache_key
        ]

    @staticmethod
    def _as_scalar(value) -> float:
        """Convert a scalar or one-element array-like result to float."""
        array = np.asarray(value, dtype=float).reshape(-1)

        if array.size != 1:
            raise ValueError(
                "Expected a scalar prediction, but received "
                f"{array.size} values."
            )

        return float(array[0])

    def _get_cached_fingerprint(
        self,
        mol: Chem.Mol
    ):
        """
        Transform one registered Buhito graph and cache both:

            1. fitted fingerprint row
            2. graphlet-instance bit information
        """

        cache_key = (
            self._get_molecule_cache_key(
                mol
            )
        )

        if (
            cache_key
            not in self._fingerprint_cache
        ):

            graph = (
                self._get_cached_graph(
                    mol
                )
            )

            fingerprint_matrix = (
                self.featurizer.transform(
                    [graph]
                )
            )

            # Convert to CSR because later code relies on
            # .indices and .data.
            fingerprint_matrix = (
                fingerprint_matrix.tocsr()
            )

            fingerprint = (
                fingerprint_matrix[
                    0
                ]
            )

            self._fingerprint_cache[
                cache_key
            ] = fingerprint

            # -------------------------------------------------------------
            # Buhito keeps graphlet locations from the transform.
            # -------------------------------------------------------------

            if not hasattr(
                self.featurizer,
                "bi_transform_"
            ):
                raise RuntimeError(
                    "The fitted Buhito GraphletTransformer "
                    "did not expose bi_transform_ after transform()."
                )

            if (
                len(
                    self.featurizer
                    .bi_transform_
                )
                != 1
            ):
                raise RuntimeError(
                    "Expected one bit-information dictionary "
                    "for one transformed graph."
                )

            self._bitinfo_cache[
                cache_key
            ] = (
                self.featurizer
                .bi_transform_[0]
            )

        return (
            self._fingerprint_cache[
                cache_key
            ]
        )
        
    
    
    def _enumerate_graphlet_instances_buhito(
        self,
        mol: Chem.Mol
    ) -> List[Dict]:
        """
        Enumerate all fitted Buhito graphlet instances for one molecule.

        Each instance is returned as:

            {
                "feature_idx": int,
                "bit_id": ...,
                "atoms": set[int],
                "coefficient": float,
                "graphlet_size": int,
            }

        Buhito graph node IDs are assumed to match RDKit atom indices.
        """

        fingerprint = (
            self._get_cached_fingerprint(
                mol
            )
        )

        cache_key = (
            self._get_molecule_cache_key(
                mol
            )
        )

        bitinfo = (
            self._bitinfo_cache[
                cache_key
            ]
        )

        instances = []

        # One sparse fingerprint row.
        fingerprint = (
            fingerprint.tocsr()
        )

        active_indices = (
            fingerprint.indices
        )

        active_counts = (
            fingerprint.data
        )

        for (
            feature_idx,
            count
        ) in zip(
            active_indices,
            active_counts
        ):

            feature_idx = int(
                feature_idx
            )

            # Graphlet counts should be integer-valued.
            rounded_count = int(
                round(
                    float(count)
                )
            )

            if not np.isclose(
                float(count),
                rounded_count
            ):
                raise RuntimeError(
                    "Expected an integer graphlet count, "
                    f"but feature {feature_idx} has "
                    f"count={count}."
                )

            count = rounded_count

            if (
                feature_idx < 0
                or
                feature_idx
                >= len(self.bit_ids)
            ):
                raise RuntimeError(
                    "Fingerprint contains an invalid "
                    f"feature index: {feature_idx}."
                )

            bit_id = (
                self.bit_ids[
                    feature_idx
                ]
            )

            graphlet_size = int(
                self.featurizer
                .bit_sizes_[
                    feature_idx
                ]
            )

            coefficient = float(
                self.coefficients[
                    feature_idx
                ]
            )

            # -------------------------------------------------------------
            # Find every actual graphlet occurrence.
            # -------------------------------------------------------------

            if bit_id not in bitinfo:

                raise RuntimeError(
                    "Active fingerprint bit is missing from "
                    "Buhito bit-information mapping.\n"
                    f"feature_idx={feature_idx}\n"
                    f"bit_id={bit_id}"
                )

            atom_occurrences = (
                bitinfo[
                    bit_id
                ]
            )

            atom_sets = []

            for occurrence in (
                atom_occurrences
            ):

                atom_set = {
                    int(atom_idx)
                    for atom_idx
                    in occurrence
                }

                atom_sets.append(
                    atom_set
                )

            # -------------------------------------------------------------
            # Quantitative consistency check.
            # -------------------------------------------------------------

            if (
                len(atom_sets)
                != count
            ):

                raise RuntimeError(
                    "Buhito graphlet count and bit-information "
                    "instance count disagree.\n"
                    f"feature_idx={feature_idx}\n"
                    f"bit_id={bit_id}\n"
                    f"fingerprint count={count}\n"
                    f"recovered instances="
                    f"{len(atom_sets)}"
                )

            # -------------------------------------------------------------
            # Store each occurrence separately.
            # -------------------------------------------------------------

            for atom_set in (
                atom_sets
            ):

                if (
                    len(atom_set)
                    != graphlet_size
                ):
                    raise RuntimeError(
                        "Buhito graphlet size does not match "
                        "the number of recovered nodes.\n"
                        f"feature_idx={feature_idx}\n"
                        f"expected size="
                        f"{graphlet_size}\n"
                        f"nodes="
                        f"{sorted(atom_set)}"
                    )

                instances.append({
                    "feature_idx": (
                        feature_idx
                    ),
                    "bit_id": (
                        bit_id
                    ),
                    "atoms": (
                        atom_set
                    ),
                    "coefficient": (
                        coefficient
                    ),
                    "graphlet_size": (
                        graphlet_size
                    ),
                })

        return instances
   
    
    def _get_molecule_cache_key(self, mol: Chem.Mol) -> bytes:
        """
        Create a stable cache key for the complete RDKit molecule.

        The binary representation preserves explicit hydrogens, bond orders,
        atom properties, and molecular structure.
        """
        return mol.ToBinary()
    
    def _get_cached_graphlet_instances(
        self,
        mol: Chem.Mol
    ) -> List[Dict]:

        cache_key = (
            self._get_molecule_cache_key(
                mol
            )
        )

        if cache_key not in self._instance_cache:

            instances = (
                self
                ._enumerate_graphlet_instances_buhito(
                    mol
                )
            )

            self._instance_cache[
                cache_key
            ] = instances

            contributing_instances = [
                instance
                for instance
                in instances
                if not np.isclose(
                    float(
                        instance[
                            "coefficient"
                        ]
                    ),
                    0.0
                )
            ]

            self._contributing_instance_cache[
                cache_key
            ] = (
                contributing_instances
            )

            atom_to_instances = [
                []
                for _
                in range(
                    mol.GetNumAtoms()
                )
            ]

            for (
                instance_idx,
                instance
            ) in enumerate(
                contributing_instances
            ):

                for atom_idx in (
                    instance["atoms"]
                ):

                    if (
                        atom_idx < 0
                        or
                        atom_idx
                        >= mol.GetNumAtoms()
                    ):

                        raise RuntimeError(
                            "Buhito node index does not correspond "
                            "to an RDKit atom index: "
                            f"{atom_idx}."
                        )

                    atom_to_instances[
                        atom_idx
                    ].append(
                        instance_idx
                    )

            self._atom_to_instances_cache[
                cache_key
            ] = tuple(
                tuple(indices)
                for indices
                in atom_to_instances
            )

        return self._instance_cache[
            cache_key
        ]

    def enumerate_graphlet_instances(
        self,
        mol: Chem.Mol,
        include_zero_coefficients: bool = True
    ) -> List[Dict]:
        """
        Return cached graphlet instances.

        Parameters
        ----------
        include_zero_coefficients
            If True, return all enumerated graphlets.
            If False, return only graphlets with active nonzero coefficients.
        """
        cache_key = self._get_molecule_cache_key(mol)

        # Populates all associated instance caches.
        self._get_cached_graphlet_instances(mol)

        if include_zero_coefficients:
            return self._instance_cache[cache_key]

        return self._contributing_instance_cache[cache_key]
    
    def get_atom_to_instance_indices(
        self,
        mol: Chem.Mol
    ) -> Tuple[Tuple[int, ...], ...]:
        """
        Return cached graphlet-instance indices for each atom.

        result[atom_idx] contains the indices of contributing graphlets that
        contain that atom.
        """
        cache_key = self._get_molecule_cache_key(mol)

        # Ensure all instance caches have been initialized.
        self._get_cached_graphlet_instances(mol)

        return self._atom_to_instances_cache[cache_key]

    def _classify_instance(
    self,
    instance: Dict,
    partition: Dict[int, int]
) -> Tuple[str, Tuple[int, ...]]:
        """
        Classify one graphlet instance under a partition.

        Returns
        -------
        category
            "within", "between", or "higher"
        key
            Tuple representation of the partition IDs spanned.

            Examples:
                ("within", (0,))
                ("between", (0, 1))
                ("higher", (0, 1, 2))
        """
        partitions_spanned = tuple(
            sorted({
                partition[atom_idx]
                for atom_idx in instance["atoms"]
            })
        )

        n_parts = len(partitions_spanned)

        if n_parts == 0:
            raise ValueError(
                "Graphlet instance does not contain any partitioned atoms."
            )

        if n_parts == 1:
            return "within", partitions_spanned

        if n_parts == 2:
            return "between", partitions_spanned

        return "higher", partitions_spanned

    def _get_partitions_spanned_for_atoms(
    self,
    atom_set,
    partition: Dict[int, int]
) -> Set[int]:
        """
        Return the partition IDs spanned by a graphlet atom set.
        """
        return {
            partition[int(atom_idx)]
            for atom_idx in atom_set
        }


    @staticmethod
    def _update_contribution(
        contributions: Dict,
        key,
        amount: float,
        tolerance: float = 1e-12
    ) -> None:
        """
        Add amount to a contribution dictionary and remove numerical zeros.
        """
        new_value = contributions.get(key, 0.0) + amount

        if abs(new_value) <= tolerance:
            contributions.pop(key, None)
        else:
            contributions[key] = new_value
    # =========================================================================
    # Main Partition Contribution Method
    # =========================================================================
    
    def compute_partition_contributions(self, mol, 
                                       partition: Dict[int, int]) -> PartitionInterpretation:
        """
        Implement Equation (2): decompose f(G) into partition-wise terms.
        
        CORRECTED: Now processes each graphlet INSTANCE separately per Reference [26].
        
        From Proposal_20260440ER-3.pdf: "we assign contributions to the prediction for 
        each element of the partition, as well as between partitions, forming an 
        interpretation graph"
        
        For each graphlet instance i with contribution f_g:
        - If i spans partition p only: add f_g to F_p
        - If i spans partitions p1, p2: add f_g to F_{p1,p2}
        - If i spans p1, p2, p3, ...: add f_g to F_{p1,p2,...}
        
        Args:
            mol: RDKit molecule object
            partition: Mapping from atom index to partition ID
            
        Returns:
            PartitionInterpretation object with decomposed contributions
        """
        
        n_atoms = mol.GetNumAtoms()
        expected_atoms = set(range(n_atoms))
        provided_atoms = set(partition)

        missing_atoms = expected_atoms - provided_atoms
        extra_atoms = provided_atoms - expected_atoms

        if missing_atoms or extra_atoms:
            raise ValueError(
                "Invalid partition mapping. "
                f"Missing atoms: {sorted(missing_atoms)}; "
                f"unexpected atoms: {sorted(extra_atoms)}."
            )

        # Graphlet instances depend on the molecule, not the partition.
        # Enumerate them once and reuse them for all candidate partitions.
        instances = self.enumerate_graphlet_instances(
            mol,
            include_zero_coefficients=False
        )
       

        # Initialize contribution dictionaries
        within_partition = defaultdict(float)
        between_partition = defaultdict(float)
        higher_order = defaultdict(float)
        
        for instance in instances:
            contribution = float(instance["coefficient"])

            category, key_tuple = self._classify_instance(
                instance,
                partition
            )

            if category == "within":
                within_partition[key_tuple[0]] += contribution

            elif category == "between":
                between_partition[key_tuple] += contribution

            else:
                higher_order[key_tuple] += contribution
        

        cache_key = self._get_molecule_cache_key(mol)

        if cache_key not in self._prediction_cache:
            fingerprint = self._get_cached_fingerprint(mol)

            active_prediction = self._as_scalar(
                fingerprint @ self.coefficients
            )
            self._prediction_cache[cache_key] = active_prediction

        total_pred = self._prediction_cache[cache_key]

        decomposed_prediction = (
            sum(within_partition.values())
            + sum(between_partition.values())
            + sum(higher_order.values())
        )

        decomposition_error = abs(
            decomposed_prediction - total_pred
        )

        if decomposition_error > 1e-6:
            print(
                "Warning: graphlet-instance decomposition does not match "
                f"the active-coefficient prediction. "
                f"Difference={decomposition_error:.6e}"
            )

        if cache_key not in self._full_prediction_cache:
            fingerprint = self._get_cached_fingerprint(mol)

            self._full_prediction_cache[cache_key] = self._as_scalar(
                fingerprint @ self.original_coefficients
            )

        full_prediction = self._full_prediction_cache[cache_key]
        atom_baseline = full_prediction - total_pred

        return PartitionInterpretation(
            within_partition=dict(within_partition),
            between_partition=dict(between_partition),
            higher_order=dict(higher_order),
            total_prediction=total_pred,
            partition=partition.copy(),
            atom_baseline=atom_baseline
        )
    
    def compute_single_atom_move(
    self,
    mol: Chem.Mol,
    current_interpretation: PartitionInterpretation,
    atom_idx: int,
    new_partition_id: int
) -> PartitionInterpretation:
        """
        Incrementally update a partition interpretation after moving one atom.

        Only graphlet instances containing atom_idx are reclassified. All other
        graphlets retain their previous within/between/higher classification.

        Parameters
        ----------
        mol
            RDKit molecule.
        current_interpretation
            Interpretation for the current partition.
        atom_idx
            Atom being moved.
        new_partition_id
            Destination partition ID.

        Returns
        -------
        PartitionInterpretation
            Updated decomposition for the candidate partition.
        """
        n_atoms = mol.GetNumAtoms()

        if atom_idx < 0 or atom_idx >= n_atoms:
            raise IndexError(
                f"atom_idx={atom_idx} is outside the valid range "
                f"0 to {n_atoms - 1}."
            )

        current_partition = current_interpretation.partition

        if atom_idx not in current_partition:
            raise ValueError(
                f"Atom {atom_idx} is missing from the current partition."
            )

        old_partition_id = current_partition[atom_idx]

        # No move means the interpretation is unchanged.
        if old_partition_id == new_partition_id:
            return current_interpretation

        # Copy only the small aggregate contribution dictionaries.
        within_partition = dict(
            current_interpretation.within_partition
        )
        between_partition = dict(
            current_interpretation.between_partition
        )
        higher_order = dict(
            current_interpretation.higher_order
        )

        # Copy the atom partition and apply the candidate move.
        candidate_partition = current_partition.copy()
        candidate_partition[atom_idx] = int(new_partition_id)

        cache_key = self._get_molecule_cache_key(mol)

        # Ensure contributing graphlets and atom lookup are cached.
        self._get_cached_graphlet_instances(mol)

        instances = self._contributing_instance_cache[cache_key]
        affected_instance_indices = (
            self._atom_to_instances_cache[cache_key][atom_idx]
        )

        for instance_idx in affected_instance_indices:
            instance = instances[instance_idx]
            contribution = float(instance["coefficient"])

            # -------------------------------------------------------------
            # Remove the graphlet's classification under the old partition.
            # -------------------------------------------------------------
            old_category, old_key_tuple = self._classify_instance(
                instance,
                current_partition
            )

            if old_category == "within":
                old_key = old_key_tuple[0]
                self._update_contribution(
                    within_partition,
                    old_key,
                    -contribution
                )

            elif old_category == "between":
                old_key = (
                    old_key_tuple[0],
                    old_key_tuple[1]
                )
                self._update_contribution(
                    between_partition,
                    old_key,
                    -contribution
                )

            else:
                self._update_contribution(
                    higher_order,
                    old_key_tuple,
                    -contribution
                )

            # -------------------------------------------------------------
            # Add the graphlet's classification under the new partition.
            # -------------------------------------------------------------
            new_category, new_key_tuple = self._classify_instance(
                instance,
                candidate_partition
            )

            if new_category == "within":
                new_key = new_key_tuple[0]
                self._update_contribution(
                    within_partition,
                    new_key,
                    contribution
                )

            elif new_category == "between":
                new_key = (
                    new_key_tuple[0],
                    new_key_tuple[1]
                )
                self._update_contribution(
                    between_partition,
                    new_key,
                    contribution
                )

            else:
                self._update_contribution(
                    higher_order,
                    new_key_tuple,
                    contribution
                )

        return PartitionInterpretation(
            within_partition=within_partition,
            between_partition=between_partition,
            higher_order=higher_order,
            total_prediction=current_interpretation.total_prediction,
            partition=candidate_partition,
            atom_baseline=current_interpretation.atom_baseline
        )

    def clear_cache(self):

        self._instance_cache.clear()
        self._contributing_instance_cache.clear()
        self._atom_to_instances_cache.clear()

        self._prediction_cache.clear()
        self._full_prediction_cache.clear()
        self._fingerprint_cache.clear()

        self._bitinfo_cache.clear()
        self._graph_cache.clear()




# =============================================================================
# PARTITION OPTIMIZATION
# =============================================================================

    
class PartitionOptimizer:
    """
    Search for partitions that maximize interpretability score.
    
    Provides THREE greedy optimization strategies:
    1. optimize_partition_random - Random moves (simple greedy)
    2. optimize_partition_distance - Best-neighbor selection (systematic greedy)
    3. optimize_partition_metropolis_hastings
    Metropolis-Hastings search that usually accepts score improvements and
    occasionally accepts worse partitions to escape local optima.
   
    """
    
    def __init__(
        self,
        interpreter: PartitionInterpreter,
        min_partitions: int = 2,
        max_partitions: int = 6,
        connectivity_weight: float = 0.10,
        disconnected_singleton_weight: float = 2.0
    ):
        self.interpreter = interpreter
        self.min_partitions = min_partitions
        self.max_partitions = max_partitions
        self._graph_cache = {}

        self.connectivity_weight = float(
            connectivity_weight
        )

        self.disconnected_singleton_weight = float(
            disconnected_singleton_weight
        )

        if self.connectivity_weight < 0:
            raise ValueError(
                "connectivity_weight must be nonnegative."
            )

        if self.disconnected_singleton_weight < 1.0:
            raise ValueError(
                "disconnected_singleton_weight must be at least 1.0."
            )

        self._greedy_helper = GreedyWithInitialization(
            interpreter=interpreter,
            max_partitions=max_partitions,
            connectivity_weight=self.connectivity_weight,
            disconnected_singleton_weight=(
                self.disconnected_singleton_weight
            )
        )
        
    def optimize_partition_random(
        self,
        mol,
        n_iterations: int = 100,
        seed: Optional[int] = None,
        verbose: bool = True
    ) -> PartitionInterpretation:
        """
        Random-start greedy hill climbing using the connectivity-adjusted score.
        """
        initial_partition = ChemicalPartitioner.random_partition(
            mol,
            n_clusters=self.max_partitions,
            seed=seed
        )

        # Compute this regardless of verbose because it is needed below.
        initial_interp = (
            self.interpreter.compute_partition_contributions(
                mol,
                initial_partition
            )
        )

        initial_penalty = partition_connectivity_penalty(
            mol=mol,
            partition=initial_partition,
            disconnected_singleton_weight=(
                self.disconnected_singleton_weight
            )
        )

        initial_adjusted_score = connectivity_adjusted_score(
            mol=mol,
            interpretation=initial_interp,
            connectivity_weight=self.connectivity_weight,
            disconnected_singleton_weight=(
                self.disconnected_singleton_weight
            )
        )

        if verbose:
            print(
                "\nRandom-Start Greedy Search "
                "(Random Moves)"
            )

            print(
                f"   Starting: "
                f"k={len(set(initial_partition.values()))}, "
                f"raw={initial_interp.score:.4f}, "
                f"penalty={initial_penalty:.4f}, "
                f"adjusted={initial_adjusted_score:.4f}"
            )

        best_interp = (
            self._greedy_helper.optimize_with_initialization(
                mol=mol,
                initial_partition=initial_partition,
                n_iterations=n_iterations,
                seed=seed,
                verbose=verbose
            )
        )

        if verbose:
            final_k = len(
                set(best_interp.partition.values())
            )

            final_penalty = partition_connectivity_penalty(
                mol=mol,
                partition=best_interp.partition,
                disconnected_singleton_weight=(
                self.disconnected_singleton_weight
            )
            )

            final_adjusted_score = connectivity_adjusted_score(
                mol=mol,
                interpretation=best_interp,
                connectivity_weight=self.connectivity_weight,
                disconnected_singleton_weight=(
                self.disconnected_singleton_weight
            )
            )

            adjusted_improvement = (
                final_adjusted_score
                - initial_adjusted_score
            )

            if abs(initial_adjusted_score) > 1e-12:
                improvement_pct = (
                    100.0
                    * adjusted_improvement
                    / abs(initial_adjusted_score)
                )
            else:
                improvement_pct = 0.0

            print(
                f"\nConverged: k={final_k}"
            )

            print(
                f"   Raw score: "
                f"{best_interp.score:.4f}"
            )

            print(
                f"   Connectivity penalty: "
                f"{final_penalty:.4f}"
            )

            print(
                f"   Adjusted score: "
                f"{final_adjusted_score:.4f}"
            )

            print(
                f"   Adjusted-score improvement: "
                f"{adjusted_improvement:+.4f} "
                f"({improvement_pct:+.1f}%)"
            )

        return best_interp
    
    def optimize_partition_distance(
        self,
        mol,
        n_clusters: Optional[int] = None,
        n_iterations: int = 100,
        seed: Optional[int] = None,
        verbose: bool = True
    ) -> PartitionInterpretation:
        """
        Distance-based initialization followed by systematic best-neighbor
        greedy optimization using the connectivity-adjusted score.
        """
        n_atoms = mol.GetNumAtoms()

        if n_atoms == 0:
            raise ValueError(
                "Cannot partition a molecule with zero atoms."
            )

        if n_clusters is None:
            n_clusters = self.max_partitions

        n_clusters = int(
            min(n_clusters, n_atoms)
        )

        if n_clusters < 1:
            raise ValueError(
                "n_clusters must be at least 1."
            )

        # Create the distance-based initial partition.
        initial_partition = ChemicalPartitioner.distance_partition(
            mol=mol,
            n_clusters=n_clusters,
            seed=seed
        )

        # Compute these regardless of verbose.
        initial_interp = (
            self.interpreter.compute_partition_contributions(
                mol,
                initial_partition
            )
        )

        initial_penalty = partition_connectivity_penalty(
            mol=mol,
            partition=initial_partition,
            disconnected_singleton_weight=(
                self.disconnected_singleton_weight
            )
        )

        initial_adjusted_score = connectivity_adjusted_score(
            mol=mol,
            interpretation=initial_interp,
            connectivity_weight=self.connectivity_weight,
            disconnected_singleton_weight=(
                self.disconnected_singleton_weight
            )
        )

        if verbose:
            print(
                "\nDistance-Based Greedy Search "
                "with Best-Neighbor Selection"
            )

            print(
                f"   Starting: "
                f"k={len(set(initial_partition.values()))}, "
                f"raw={initial_interp.score:.4f}, "
                f"penalty={initial_penalty:.4f}, "
                f"adjusted={initial_adjusted_score:.4f}"
            )

            print(
                "   Algorithm: systematic search of all valid "
                "moves into directly neighboring partitions"
            )

        best_interp = (
            self._greedy_helper.optimize_with_best_neighbor(
                mol=mol,
                initial_partition=initial_partition,
                n_iterations=n_iterations,
                seed=seed,
                verbose=verbose
            )
        )

        if verbose:
            final_penalty = partition_connectivity_penalty(
                mol=mol,
                partition=best_interp.partition,
                disconnected_singleton_weight=(
                self.disconnected_singleton_weight
            )
            )

            final_adjusted_score = connectivity_adjusted_score(
                mol=mol,
                interpretation=best_interp,
                connectivity_weight=self.connectivity_weight,
                disconnected_singleton_weight=(
                self.disconnected_singleton_weight
            )
            )

            final_k = len(
                set(best_interp.partition.values())
            )

            improvement = (
                final_adjusted_score
                - initial_adjusted_score
            )

            if abs(initial_adjusted_score) > 1e-12:
                improvement_pct = (
                    100.0
                    * improvement
                    / abs(initial_adjusted_score)
                )
            else:
                improvement_pct = 0.0

            print(
                f"\nConverged: k={final_k}"
            )

            print(
                f"   Raw score: "
                f"{best_interp.score:.4f}"
            )

            print(
                f"   Connectivity penalty: "
                f"{final_penalty:.4f}"
            )

            print(
                f"   Adjusted score: "
                f"{final_adjusted_score:.4f}"
            )

            print(
                f"   Adjusted-score improvement: "
                f"{improvement:+.4f} "
                f"({improvement_pct:+.1f}%)"
            )

        return best_interp
    
      
    def optimize_partition_metropolis_hastings(
    self,
    mol: Chem.Mol,
    n_clusters: Optional[int] = None,
    n_iterations: int = 1000,
    beta: float = 10.0,
    initial_partition: Optional[Dict[int, int]] = None,
    initialization: str = "distance",
    seed: Optional[int] = None,
    verbose: bool = True
) -> PartitionInterpretation:
        """
        Optimize a molecular partition using Metropolis-Hastings with
        one-way neighboring-partition moves.

        One proposal is:

            1. Randomly choose one molecular bond.
            2. Randomly choose one endpoint as the candidate atom.
            3. Let the other endpoint define the destination partition.
            4. If the atoms are in different partitions, move ONLY the
            candidate atom into the neighbor's partition.
            5. Reject the proposal if moving the candidate would empty
            its current partition.
            6. Accept or reject the candidate using the Metropolis rule.

        Unlike the previous implementation, partition labels are NOT swapped.

        Example
        -------
        Before:

            atom_a -> P0
            atom_b -> P1

        If atom_a is selected as the candidate:

            atom_a -> P1
            atom_b -> P1

        atom_b is unchanged.

        The number of active partitions is preserved by preventing the final
        atom of a partition from moving.

        The optimization objective is the connectivity-adjusted score.

        For the proposal mechanism used here, the Metropolis acceptance rule is

            acceptance = min(
                1,
                exp(
                    beta
                    * (
                        candidate_score
                        - current_score
                    )
                )
            ) 

        The highest-scoring partition encountered is returned.
        """

        # =====================================================================
        # VALIDATION
        # =====================================================================

        if n_iterations < 1:
            raise ValueError(
                "n_iterations must be at least 1."
            )

        if beta < 0:
            raise ValueError(
                "beta must be nonnegative."
            )

        rng = np.random.default_rng(seed)

        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        if n_atoms == 0:
            raise ValueError(
                "Cannot partition a molecule with zero atoms."
            )

        if n_bonds == 0:
            raise ValueError(
                "Cannot perform neighboring-atom moves on a "
                "molecule containing no bonds."
            )

        # =====================================================================
        # NUMBER OF PARTITIONS
        # =====================================================================

        if n_clusters is None:
            n_clusters = self.max_partitions

        n_clusters = int(n_clusters)

        if n_clusters < 1:
            raise ValueError(
                "n_clusters must be at least 1."
            )

        if n_clusters > n_atoms:
            raise ValueError(
                f"n_clusters={n_clusters} exceeds the number "
                f"of atoms ({n_atoms})."
            )

        # =====================================================================
        # INITIAL PARTITION
        # =====================================================================

        if initial_partition is None:

            initialization = initialization.lower()

            if initialization == "distance":

                current_partition = (
                    ChemicalPartitioner.distance_partition(
                        mol=mol,
                        n_clusters=n_clusters,
                        seed=seed
                    )
                )

            elif initialization == "random":

                current_partition = (
                    ChemicalPartitioner.random_partition(
                        mol=mol,
                        n_clusters=n_clusters,
                        seed=seed
                    )
                )

            else:

                raise ValueError(
                    "initialization must be either "
                    "'distance' or 'random'."
                )

        else:

            current_partition = {
                int(atom_idx): int(partition_id)
                for atom_idx, partition_id
                in initial_partition.items()
            }

            expected_atoms = set(range(n_atoms))
            provided_atoms = set(current_partition)

            if provided_atoms != expected_atoms:

                missing = (
                    expected_atoms
                    - provided_atoms
                )

                extra = (
                    provided_atoms
                    - expected_atoms
                )

                raise ValueError(
                    "Invalid initial_partition. "
                    f"Missing atoms: {sorted(missing)}; "
                    f"unexpected atoms: {sorted(extra)}."
                )

            active_initial_partitions = set(
                current_partition.values()
            )

            if (
                len(active_initial_partitions)
                != n_clusters
            ):

                raise ValueError(
                    "initial_partition contains "
                    f"{len(active_initial_partitions)} "
                    "active partitions, but "
                    f"n_clusters={n_clusters}."
                )

        # =====================================================================
        # VERIFY ACTIVE PARTITIONS
        # =====================================================================

        active_partitions = tuple(
            sorted(
                set(current_partition.values())
            )
        )

        if len(active_partitions) != n_clusters:

            raise RuntimeError(
                "Initialization did not create the requested "
                f"number of partitions. Requested "
                f"{n_clusters}, obtained "
                f"{len(active_partitions)}."
            )

        if n_clusters == 1:

            return (
                self.interpreter
                .compute_partition_contributions(
                    mol,
                    current_partition
                )
            )

        # =====================================================================
        # FIXED BOND LIST
        # =====================================================================
        #
        # Every bond is always available for selection.
        #
        # After selecting a bond, one of its two endpoints is chosen uniformly
        # as the candidate atom.
        #
        # Thus the basic directed proposal is:
        #
        #       candidate atom -> partition of bonded neighbor
        #
        # =====================================================================

        molecular_bonds = [
            (
                int(bond.GetBeginAtomIdx()),
                int(bond.GetEndAtomIdx())
            )
            for bond in mol.GetBonds()
        ]

        # =====================================================================
        # INITIAL INTERPRETATION
        # =====================================================================

        current_interp = (
            self.interpreter
            .compute_partition_contributions(
                mol,
                current_partition
            )
        )

        current_raw_score = (
            current_interp.score
        )

        current_penalty = (
            partition_connectivity_penalty(
                mol=mol,
                partition=current_partition,
                disconnected_singleton_weight=(
                    self.disconnected_singleton_weight
                )
            )
        )

        current_score = (
            connectivity_adjusted_score(
                mol=mol,
                interpretation=current_interp,
                connectivity_weight=(
                    self.connectivity_weight
                ),
                disconnected_singleton_weight=(
                    self.disconnected_singleton_weight
                )
            )
        )

        # =====================================================================
        # BEST STATE
        # =====================================================================

        best_interp = current_interp
        best_score = current_score
        best_raw_score = current_raw_score
        best_penalty = current_penalty

        best_iteration = 0

        # =====================================================================
        # STATISTICS
        # =====================================================================

        accepted_moves = 0
        accepted_improving_moves = 0
        accepted_worse_moves = 0

        same_partition_proposals = 0
        singleton_source_proposals = 0
        valid_move_proposals = 0

        # =====================================================================
        # PRINT INITIAL INFORMATION
        # =====================================================================

        if verbose:

            print(
                "\nMetropolis-Hastings "
                "Neighbor-Move Partition Search"
            )

            print(
                f"  Initialization: {initialization}"
                if initial_partition is None
                else
                "  Initialization: user supplied"
            )

            print(
                f"  Starting partitions: "
                f"{n_clusters}"
            )

            print(
                f"  Molecular bonds: "
                f"{n_bonds}"
            )

            print(
                f"  Starting raw score: "
                f"{current_raw_score:.6f}"
            )

            print(
                f"  Starting connectivity penalty: "
                f"{current_penalty:.6f}"
            )

            print(
                f"  Starting adjusted score: "
                f"{current_score:.6f}"
            )

            print(
                f"  beta: "
                f"{beta:.4f}"
            )

            print(
                "  Proposal: move one candidate atom "
                "into the partition of a directly bonded neighbor"
            )

            print(
                "  Neighbor atom remains unchanged"
            )

        # =====================================================================
        # METROPOLIS-HASTINGS ITERATIONS
        # =====================================================================

        for iteration in range(
            1,
            n_iterations + 1
        ):

            # -----------------------------------------------------------------
            # 1. Select a molecular bond uniformly.
            # -----------------------------------------------------------------

            bond_index = int(
                rng.integers(
                    0,
                    n_bonds
                )
            )

            atom_a, atom_b = (
                molecular_bonds[bond_index]
            )

            # -----------------------------------------------------------------
            # 2. Randomly select ONE endpoint as the candidate.
            #
            # The other endpoint only determines the destination partition.
            # -----------------------------------------------------------------

            if rng.random() < 0.5:

                candidate_atom = atom_a
                neighbor_atom = atom_b

            else:

                candidate_atom = atom_b
                neighbor_atom = atom_a

            source_partition_id = (
                current_partition[
                    candidate_atom
                ]
            )

            destination_partition_id = (
                current_partition[
                    neighbor_atom
                ]
            )

            # -----------------------------------------------------------------
            # 3. Candidate and neighbor must currently belong to different
            #    partitions.
            # -----------------------------------------------------------------

            if (
                source_partition_id
                == destination_partition_id
            ):

                same_partition_proposals += 1
                continue

            # -----------------------------------------------------------------
            # 4. Preserve exactly k nonempty partitions.
            #
            # Since this is no longer a swap, moving the final atom out of a
            # partition would cause that partition to disappear.
            # -----------------------------------------------------------------

            source_partition_size = sum(
                partition_id
                == source_partition_id
                for partition_id
                in current_partition.values()
            )

            if source_partition_size <= 1:

                singleton_source_proposals += 1
                continue

            valid_move_proposals += 1

            # -----------------------------------------------------------------
            # 5. MOVE ONLY THE CANDIDATE.
            #
            # Before:
            #
            #     candidate -> source partition
            #     neighbor  -> destination partition
            #
            # After:
            #
            #     candidate -> destination partition
            #     neighbor  -> destination partition
            #
            # The neighbor never moves.
            # -----------------------------------------------------------------

            candidate_interp = (
                self.interpreter
                .compute_single_atom_move(
                    mol=mol,
                    current_interpretation=(
                        current_interp
                    ),
                    atom_idx=(
                        candidate_atom
                    ),
                    new_partition_id=(
                        destination_partition_id
                    )
                )
            )

            candidate_partition = (
                candidate_interp.partition
            )

            # -----------------------------------------------------------------
            # 6. Safety check: k must remain fixed.
            # -----------------------------------------------------------------

            candidate_active_partitions = set(
                candidate_partition.values()
            )

            if (
                len(candidate_active_partitions)
                != n_clusters
            ):

                raise RuntimeError(
                    "Single-atom neighbor move unexpectedly "
                    "changed the number of active partitions."
                )

            # -----------------------------------------------------------------
            # 7. Evaluate candidate.
            # -----------------------------------------------------------------

            candidate_raw_score = (
                candidate_interp.score
            )

            candidate_penalty = (
                partition_connectivity_penalty(
                    mol=mol,
                    partition=(
                        candidate_partition
                    ),
                    disconnected_singleton_weight=(
                        self
                        .disconnected_singleton_weight
                    )
                )
            )

            candidate_score = (
                connectivity_adjusted_score(
                    mol=mol,
                    interpretation=(
                        candidate_interp
                    ),
                    connectivity_weight=(
                        self.connectivity_weight
                    ),
                    disconnected_singleton_weight=(
                        self
                        .disconnected_singleton_weight
                    )
                )
            )

            # -----------------------------------------------------------------
            # 8. Score change.
            # -----------------------------------------------------------------

            score_change = (
                candidate_score
                - current_score
            )

            # -----------------------------------------------------------------
            # 9. Metropolis acceptance.
            #
            # Better moves:
            #     accepted automatically.
            #
            # Worse moves:
            #     accepted with probability exp(beta * score_change).
            # -----------------------------------------------------------------

            log_acceptance_ratio = (
                beta
                * score_change
            )

            if log_acceptance_ratio >= 0.0:

                accept_move = True

            else:

                log_uniform = np.log(
                    rng.uniform(
                        0.0,
                        1.0
                    )
                )

                accept_move = (
                    log_uniform
                    < log_acceptance_ratio
                )

            # -----------------------------------------------------------------
            # 10. Accept candidate.
            # -----------------------------------------------------------------

            if accept_move:

                previous_score = (
                    current_score
                )

                current_interp = (
                    candidate_interp
                )

                current_partition = (
                    current_interp.partition
                )

                current_score = (
                    candidate_score
                )

                current_raw_score = (
                    candidate_raw_score
                )

                current_penalty = (
                    candidate_penalty
                )

                accepted_moves += 1

                if (
                    current_score
                    >= previous_score
                ):

                    accepted_improving_moves += 1

                else:

                    accepted_worse_moves += 1

                # -------------------------------------------------------------
                # Track highest-scoring state encountered.
                # -------------------------------------------------------------

                if current_score > best_score:

                    best_interp = (
                        current_interp
                    )

                    best_score = (
                        current_score
                    )

                    best_raw_score = (
                        current_raw_score
                    )

                    best_penalty = (
                        current_penalty
                    )

                    best_iteration = (
                        iteration
                    )

            # -----------------------------------------------------------------
            # Progress reporting.
            # -----------------------------------------------------------------

            if verbose and (
                iteration <= 5
                or iteration % 100 == 0
                or iteration == n_iterations
            ):

                if valid_move_proposals > 0:

                    acceptance_rate = (
                        accepted_moves
                        / valid_move_proposals
                    )

                else:

                    acceptance_rate = 0.0

                print(
                    f"  Iteration {iteration:5d}: "
                    f"current={current_score:.6f}, "
                    f"best={best_score:.6f}, "
                    f"acceptance={acceptance_rate:.3f}"
                )

        # =====================================================================
        # FINAL STATISTICS
        # =====================================================================

        if valid_move_proposals > 0:

            acceptance_rate = (
                accepted_moves
                / valid_move_proposals
            )

        else:

            acceptance_rate = 0.0

        if verbose:

            print(
                "\nMetropolis-Hastings "
                "neighbor-move search completed"
            )

            print(
                f"  Total iterations: "
                f"{n_iterations}"
            )

            print(
                f"  Valid neighbor moves: "
                f"{valid_move_proposals}"
            )

            print(
                f"  Same-partition proposals: "
                f"{same_partition_proposals}"
            )

            print(
                f"  Rejected singleton-source proposals: "
                f"{singleton_source_proposals}"
            )

            print(
                f"  Accepted moves: "
                f"{accepted_moves}"
            )

            print(
                f"  Accepted improving moves: "
                f"{accepted_improving_moves}"
            )

            print(
                f"  Accepted worse moves: "
                f"{accepted_worse_moves}"
            )

            print(
                f"  Acceptance rate: "
                f"{acceptance_rate:.4f}"
            )

            print(
                f"  Best iteration: "
                f"{best_iteration}"
            )

            print(
                f"  Best raw score: "
                f"{best_raw_score:.6f}"
            )

            print(
                f"  Best connectivity penalty: "
                f"{best_penalty:.6f}"
            )

            print(
                f"  Best adjusted score: "
                f"{best_score:.6f}"
            )

        return best_interp

# =============================================================================
# GREEDY INITIALIZATION COMPARISON 
# =============================================================================

class GreedyWithInitialization:
    """
    Tests whether initialization (distance-based) 
    improves greedy search outcomes.
    
    Provides TWO optimization strategies:
    1. optimize_with_initialization - Random moves (for optimize_partition_random)
    2. optimize_with_best_neighbor - Systematic best-neighbor (for optimize_partition_distance)
    """
    
    def __init__(
        self,
        interpreter: PartitionInterpreter,
        max_partitions: int = 5,
        connectivity_weight: float = 0.10,
        disconnected_singleton_weight: float = 2.0
    ):
        self.interpreter = interpreter
        self.max_partitions = max_partitions

        self.connectivity_weight = float(
            connectivity_weight
        )

        self.disconnected_singleton_weight = float(
            disconnected_singleton_weight
        )

        if self.connectivity_weight < 0:
            raise ValueError(
                "connectivity_weight must be nonnegative."
            )

        if self.disconnected_singleton_weight < 1.0:
            raise ValueError(
                "disconnected_singleton_weight must be at least 1.0."
            )


    def _objective(
        self,
        mol: Chem.Mol,
        interpretation: PartitionInterpretation
    ) -> float:
        return connectivity_adjusted_score(
            mol=mol,
            interpretation=interpretation,
            connectivity_weight=self.connectivity_weight,
            disconnected_singleton_weight=(
                self.disconnected_singleton_weight
            )
        )
    
    def optimize_with_initialization(
        self, 
        mol, 
        initial_partition: Dict[int, int],
        n_iterations: int = 100,
        seed: Optional[int] = None,
        verbose: bool = False
    ) -> PartitionInterpretation:
        """
        Greedy optimization with RANDOM move selection.
        
        Used by optimize_partition_random().
        
        Algorithm:
        1. Start with initial_partition
        2. For n_iterations:
           - Pick random atom
           - Try moving to random partition
           - Accept if score improves (greedy hill-climbing)
        3. Return best partition found
        
        Args:
            mol: RDKit molecule
            initial_partition: Starting partition {atom_idx: partition_id}
            n_iterations: Number of greedy moves to attempt
            seed: Random seed for reproducibility
            verbose: Print progress
            
        Returns:
            PartitionInterpretation with optimized partition
        """
        rng = np.random.default_rng(seed)
        
        n_atoms = mol.GetNumAtoms()
        
        # Start from provided initial partition
        current_partition = initial_partition.copy()
        current_interp = self.interpreter.compute_partition_contributions(mol, current_partition)
        
        current_raw_score = current_interp.score

        current_score = self._objective(
            mol,
            current_interp
        )

        initial_score = current_score

        best_interp = current_interp
        best_score = current_score
        
        improvements = 0
        
        if verbose:
            print(f"  Initial score: {current_score:.4f}")
            print(f"  Using RANDOM move selection...")
        
        # Greedy hill-climbing with RANDOM moves
        for iteration in range(n_iterations):
            # Pick random atom to reassign
            atom_to_move = int(
                rng.integers(0, n_atoms)
            )
            # Get current partition assignment
            current_p = current_partition[atom_to_move]

            active_partitions = tuple(
                sorted(set(current_partition.values()))
            )

            partition_counts = {
                partition_id: 0
                for partition_id in active_partitions
            }

            for partition_id in current_partition.values():
                partition_counts[partition_id] += 1

            # Moving this atom would delete its current partition.
            if partition_counts[current_p] <= 1:
                continue

            available_partitions = [
                partition_id
                for partition_id in active_partitions
                if partition_id != current_p
            ]
            
            if not available_partitions:
                continue
            
            new_p = rng.choice(available_partitions)
            
            # Create candidate partition
            candidate_interp = (
                self.interpreter.compute_single_atom_move(
                    mol=mol,
                    current_interpretation=current_interp,
                    atom_idx=atom_to_move,
                    new_partition_id=new_p
                )
            )

            candidate_score = self._objective(
                mol,
                candidate_interp
            )
            
            # Greedy acceptance: only accept improvements
            if candidate_score > current_score:
                current_interp = candidate_interp
                current_partition = current_interp.partition
                current_score = candidate_score
                improvements += 1
                
                # Track best
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_interp = candidate_interp
        
        if verbose:
            final_penalty = partition_connectivity_penalty(
                mol=mol,
                partition=best_interp.partition,
                disconnected_singleton_weight=(
                self.disconnected_singleton_weight
            )
            )

            print(
                f"  Final interpretability score: "
                f"{best_interp.score:.4f}"
            )

            print(
                f"  Final connectivity penalty: "
                f"{final_penalty:.4f}"
            )

            print(
                f"  Final adjusted score: "
                f"{best_score:.4f}"
            )
        
        return best_interp
    
    def optimize_with_best_neighbor(
    self,
    mol,
    initial_partition: Dict[int, int],
    n_iterations: int = 100,
    seed: Optional[int] = None,
    verbose: bool = False
) -> PartitionInterpretation:
        """
        Neighbor-restricted systematic greedy optimization.

        At each iteration:

        1. Examine every atom.
        2. Find the partitions occupied by its directly bonded neighbors.
        3. Consider moving the atom only into those neighboring partitions.
        4. Evaluate every valid neighbor-based move.
        5. Accept the move producing the greatest increase in the
        connectivity-adjusted score.

        The number of active partitions is preserved. An atom cannot leave its
        current partition if it is the final atom assigned to that partition.

        Parameters
        ----------
        mol
            RDKit molecule.

        initial_partition
            Initial atom-to-partition mapping.

        n_iterations
            Maximum number of accepted greedy moves.

        seed
            Random seed controlling atom visitation order and tie resolution.

        verbose
            Print optimization progress.

        Returns
        -------
        PartitionInterpretation
            Best partition interpretation found.
        """
        rng = np.random.default_rng(seed)

        n_atoms = mol.GetNumAtoms()

        if n_atoms == 0:
            raise ValueError(
                "Cannot optimize a molecule with zero atoms."
            )

        expected_atoms = set(range(n_atoms))
        provided_atoms = set(initial_partition)

        if provided_atoms != expected_atoms:
            missing_atoms = expected_atoms - provided_atoms
            extra_atoms = provided_atoms - expected_atoms

            raise ValueError(
                "Invalid initial_partition. "
                f"Missing atoms: {sorted(missing_atoms)}; "
                f"unexpected atoms: {sorted(extra_atoms)}."
            )

        current_partition = {
            int(atom_idx): int(partition_id)
            for atom_idx, partition_id in initial_partition.items()
        }

        current_interp = (
            self.interpreter.compute_partition_contributions(
                mol,
                current_partition
            )
        )

        current_score = self._objective(
            mol,
            current_interp
        )

        initial_score = current_score

        best_interp = current_interp
        best_score = current_score

        improvements = 0
        iterations_completed = 0

        if verbose:
            print(
                f"  Initial adjusted score: "
                f"{current_score:.4f}"
            )
            print(
                "  Using NEIGHBOR-RESTRICTED "
                "best-move selection..."
            )

        for iteration in range(n_iterations):
            iterations_completed = iteration + 1

            best_move = None
            best_move_score = current_score

            active_partitions = tuple(
                sorted(set(current_partition.values()))
            )

            partition_counts = {
                partition_id: 0
                for partition_id in active_partitions
            }

            for partition_id in current_partition.values():
                partition_counts[partition_id] += 1

            # Randomized ordering only affects ties between equal-scoring moves.
            atom_order = rng.permutation(n_atoms)

            candidate_moves_evaluated = 0

            for atom_to_move in atom_order:
                atom_to_move = int(atom_to_move)
                current_partition_id = current_partition[atom_to_move]

                # Moving this atom would remove its current partition.
                if partition_counts[current_partition_id] <= 1:
                    continue

                atom = mol.GetAtomWithIdx(atom_to_move)

                # Only consider partitions represented among directly bonded
                # neighbors of this atom.
                neighboring_partition_ids = {
                    current_partition[int(neighbor.GetIdx())]
                    for neighbor in atom.GetNeighbors()
                    if (
                        current_partition[int(neighbor.GetIdx())]
                        != current_partition_id
                    )
                }

                # The atom is not on a partition boundary.
                if not neighboring_partition_ids:
                    continue

                # Sort for deterministic destination ordering.
                for new_partition_id in sorted(
                    neighboring_partition_ids
                ):
                    candidate_interp = (
                        self.interpreter.compute_single_atom_move(
                            mol=mol,
                            current_interpretation=current_interp,
                            atom_idx=atom_to_move,
                            new_partition_id=int(
                                new_partition_id
                            )
                        )
                    )

                    candidate_score = self._objective(
                        mol,
                        candidate_interp
                    )

                    candidate_moves_evaluated += 1

                    if (
                        candidate_score
                        > best_move_score + 1e-12
                    ):
                        best_move_score = candidate_score

                        best_move = (
                            atom_to_move,
                            int(new_partition_id),
                            candidate_interp
                        )

            if best_move is None:
                if verbose:
                    print(
                        f"  Converged after "
                        f"{iterations_completed} iterations: "
                        "no improving neighbor-based move exists."
                    )

                break

            (
                atom_moved,
                new_partition_id,
                new_interp
            ) = best_move

            score_gain = (
                best_move_score - current_score
            )

            current_interp = new_interp
            current_partition = current_interp.partition
            current_score = best_move_score

            improvements += 1

            if current_score > best_score:
                best_score = current_score
                best_interp = current_interp

            if verbose and (
                iteration < 5
                or iteration % 10 == 0
            ):
                print(
                    f"    Iter {iteration}: "
                    f"moved atom {atom_moved} "
                    f"to neighboring partition "
                    f"{new_partition_id}, "
                    f"score={current_score:.4f} "
                    f"(+{score_gain:.4f}), "
                    f"candidates="
                    f"{candidate_moves_evaluated}"
                )

        if verbose:
            final_penalty = (
                partition_connectivity_penalty(
                    mol=mol,
                    partition=best_interp.partition,
                    disconnected_singleton_weight=(
                        self.disconnected_singleton_weight
                    )
                )
            )

            print(
                f"  Final raw score: "
                f"{best_interp.score:.4f}"
            )

            print(
                f"  Final connectivity penalty: "
                f"{final_penalty:.4f}"
            )

            print(
                f"  Final adjusted score: "
                f"{best_score:.4f}"
            )

            print(
                f"  Improvements: "
                f"{improvements}/"
                f"{iterations_completed}"
            )

            print(
                f"  Adjusted-score gain: "
                f"{best_score - initial_score:.4f}"
            )

        return best_interp




# =============================================================================
# PARTITIONING STRATEGIES
# =============================================================================

class ChemicalPartitioner:
    """
    Chemistry-aware partitioning strategies.
    
    """
    
    @staticmethod

    def distance_partition(
        mol,
        n_clusters: int = 3,
        seed: Optional[int] = 42
    ) -> Dict[int, int]:
        """
        Partition atoms by graph distance from randomly selected seed atoms.
        """
        rng = np.random.default_rng(seed)

        n_atoms = mol.GetNumAtoms()

        if n_atoms == 0:
            return {}

        n_clusters = max(
            1,
            min(int(n_clusters), n_atoms)
        )

        seeds = rng.choice(
            n_atoms,
            size=n_clusters,
            replace=False
        )

        dist_matrix = Chem.GetDistanceMatrix(mol)
        seed_distances = dist_matrix[:, seeds]
        assignments = np.argmin(seed_distances, axis=1)

        return {
            atom_idx: int(partition_id)
            for atom_idx, partition_id in enumerate(assignments)
        }
    
    @staticmethod
   
    def random_partition(
        mol,
        n_clusters: int = 3,
        seed: Optional[int] = 42
    ) -> Dict[int, int]:
        """
        Create a random partition with exactly n_clusters nonempty partitions.
        """
        rng = np.random.default_rng(seed)
        n_atoms = mol.GetNumAtoms()

        if n_atoms == 0:
            return {}

        n_clusters = max(
            1,
            min(int(n_clusters), n_atoms)
        )

        # Guarantee each partition appears at least once.
        assignments = np.empty(n_atoms, dtype=int)
        assignments[:n_clusters] = np.arange(n_clusters)

        if n_atoms > n_clusters:
            assignments[n_clusters:] = rng.integers(
                low=0,
                high=n_clusters,
                size=n_atoms - n_clusters
            )

        rng.shuffle(assignments)

        return {
            atom_idx: int(partition_id)
            for atom_idx, partition_id in enumerate(assignments)
        }


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_partition_comparison(mol, results: List[Dict], 
                                   figsize: Tuple[int, int] = (20, 12)) -> plt.Figure:
    """
    Create comprehensive visualization comparing multiple partition strategies.
    
    Args:
        mol: RDKit molecule object
        results: List of result dictionaries from partition comparison
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    n_strategies = len(results)
    fig = plt.figure(figsize=figsize)
    
    for idx, result in enumerate(results):
        ax = plt.subplot(((n_strategies + 2) // 3), 3, idx + 1)
        
        interp = result['Interpretation']
        breakdown = interp.get_contribution_breakdown()
        
        contributions = {
            'Within': breakdown['within_frac'],
            'Between': breakdown['between_frac'],
            'Higher': breakdown['higher_frac']
        }
        
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        ax.bar(contributions.keys(), contributions.values(), color=colors)
        ax.set_ylabel('Fraction')
        raw_score = result.get("Score", interp.score)
        adjusted_score = result.get("Adjusted Score")

        if adjusted_score is None:
            score_text = f"Raw={raw_score:.3f}"
        else:
            score_text = (
                f"Raw={raw_score:.3f}, "
                f"Adjusted={adjusted_score:.3f}"
            )

        ax.set_title(
            f"{result['Strategy']}\n"
            f"{score_text}, "
            f"N_partitions={result['N_partitions']}"
        )
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig


# =============================================================================
# VALIDATION FUNCTION
# =============================================================================

def validate_partition_decomposition(mol, interpreter: PartitionInterpreter,
                                     partition: Dict[int, int],
                                     tolerance: float = 1e-2) -> Dict[str, Any]:
    """
    Validate that Equation (2) is correctly implemented.
    
    Per Proposal_20260440ER-3.pdf: f(G) = Σ_p F_p + Σ_{p1,p2} F_{p1,p2} + ...
    The sum of all partition contributions should equal total prediction.
    
    Args:
        mol: RDKit molecule
        interpreter: PartitionInterpreter instance
        partition: Partition to test
        tolerance: Numerical tolerance for validation
        
    Returns:
        Dictionary with validation results
    """
    interp = interpreter.compute_partition_contributions(mol, partition)
    
    # Sum all contributions
    within_sum = sum(interp.within_partition.values())
    between_sum = sum(interp.between_partition.values())
    higher_sum = sum(interp.higher_order.values())
    total_from_parts = within_sum + between_sum + higher_sum
    
    # Check against total prediction
    diff = abs(total_from_parts - interp.total_prediction)
    passed = diff < tolerance
    
    return {
        'passed': passed,
        'total_prediction': interp.total_prediction,
        'sum_of_parts': total_from_parts,
        'difference': diff,
        'tolerance': tolerance,
        'relative_error': diff / (abs(interp.total_prediction) + 1e-10),
        'breakdown': {
            'within': within_sum,
            'between': between_sum,
            'higher': higher_sum
        }
    }
