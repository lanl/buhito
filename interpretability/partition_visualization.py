# =============================================================================
# partition_visualization.py
#
# Buhito-compatible partition visualization using MinervaChem GraphletDAG
# only for coefficient projection / plotting.
# =============================================================================

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np

from PIL import Image
from io import BytesIO

from typing import (
    Dict,
    Tuple,
    List,
    Optional,
    Union,
)

from copy import deepcopy

from buhito.graphlet_dags import GraphletDAG


# =============================================================================
# BUHITO -> GRAPHLETDAG BRIDGE
# =============================================================================

def build_graphlet_dag_from_buhito(
    mol: Chem.Mol,
    interpreter,
    layerwise: bool = True,
) -> GraphletDAG:
    """
    Construct a MinervaChem GraphletDAG using graphlet information
    produced by the fitted Buhito GraphletTransformer.

    Parameters
    ----------
    mol
        RDKit molecule.

    interpreter
        Buhito-based PartitionInterpreter.

        It must provide:
            interpreter.get_bitinfo(mol)
            interpreter.bit_ids
            interpreter.coefficients

    layerwise
        Passed directly to GraphletDAG.

    Returns
    -------
    GraphletDAG
        DAG containing the Buhito graphlet instances and coefficients.
    """

    bi = interpreter.get_bitinfo(
        mol
    )

    # -------------------------------------------------------------------------
    # Optional consistency checks
    # -------------------------------------------------------------------------

    if bi is None:
        raise RuntimeError(
            "No Buhito bit information was available for this molecule."
        )

    if not hasattr(
        interpreter,
        "bit_ids"
    ):
        raise AttributeError(
            "The interpreter does not expose fitted Buhito bit_ids."
        )

    if not hasattr(
        interpreter,
        "coefficients"
    ):
        raise AttributeError(
            "The interpreter does not expose fitted coefficients."
        )

    bit_ids = list(
        interpreter.bit_ids
    )

    coefficients = np.asarray(
        interpreter.coefficients,
        dtype=float
    ).reshape(-1)

    if len(bit_ids) != len(coefficients):
        raise ValueError(
            f"Dimension mismatch: "
            f"{len(bit_ids)} bit IDs vs "
            f"{len(coefficients)} coefficients."
        )

    # -------------------------------------------------------------------------
    # GraphletDAG accepts bi directly, so no MinervaChem fingerprinter
    # is needed here.
    # -------------------------------------------------------------------------

    dag = GraphletDAG(
        mol=mol,
        bi=bi,
        layerwise=layerwise,
        bit_ids=bit_ids,
        coef=coefficients,
    )

    return dag


# =============================================================================
# PARTITION COLORS
# =============================================================================

def get_partition_colormap(
    n_partitions: int,
    cmap_name: str = "distinct",
    partition_ids: Optional[List[int]] = None,
) -> Dict[int, Tuple[float, float, float]]:
    """
    Return distinct colors for partitions.
    """

    if partition_ids is None:
        partition_ids = list(
            range(n_partitions)
        )

    if cmap_name == "distinct":

        distinct_colors = [
            (0.12, 0.47, 0.71),  # blue
            (0.89, 0.10, 0.11),  # red
            (0.20, 0.63, 0.17),  # green
            (0.98, 0.60, 0.00),  # orange
            (0.42, 0.24, 0.60),  # purple
            (0.69, 0.35, 0.16),  # brown
            (0.99, 0.75, 0.85),  # pink
            (0.50, 0.50, 0.50),  # gray
        ]

    else:

        if n_partitions <= 9:
            cmap = plt.cm.Set1

        elif n_partitions <= 12:
            cmap = plt.cm.Paired

        else:
            cmap = plt.colormaps.get_cmap(
                cmap_name
            )

        distinct_colors = [
            cmap(
                i / max(
                    n_partitions - 1,
                    1
                )
            )[:3]
            for i
            in range(n_partitions)
        ]

    partition_colors = {}

    for i, partition_id in enumerate(
        partition_ids
    ):
        partition_colors[
            partition_id
        ] = distinct_colors[
            i % len(distinct_colors)
        ]

    return partition_colors


# =============================================================================
# COEFFICIENT ANNOTATIONS
# =============================================================================

def add_coefficient_annotations(
    mol: Chem.Mol,
    coefficients: Dict[int, float],
    level: int,
    precision: int = 2,
) -> None:
    """
    Add projected coefficient annotations to atoms or bonds.
    """

    if level == 2:

        iterator = mol.GetBonds
        property_name = "bondNote"

    elif level == 1:

        iterator = mol.GetAtoms
        property_name = "atomNote"

    else:

        raise ValueError(
            f"level must be 1 or 2, got {level}"
        )

    for index, component in enumerate(
        iterator()
    ):

        coefficient = float(
            coefficients.get(
                index,
                0.0
            )
        )

        component.SetProp(
            property_name,
            f"{coefficient:+.{precision}f}"
        )


# =============================================================================
# DRAW MOLECULE WITH PARTITION COLORS
# =============================================================================

def draw_mol_with_partition_colors(
    mol: Chem.Mol,
    partition: Dict[int, int],
    partition_colors: Dict[
        int,
        Tuple[float, float, float]
    ],
    level: int = 1,
    coefficients: Optional[
        Dict[int, float]
    ] = None,
    size: Tuple[int, int] = (
        800,
        800
    ),
    dpi: int = 150,
    highlight_radius: float = 0.5,
    show_coefficients: bool = True,
    show_bond_labels: bool = False,
    show_atom_labels: bool = False,
    label_precision: int = 2,
    show_atom_indices: bool = False,
) -> Image.Image:
    """
    Draw a molecule using partition colors and optionally modulate
    color intensity using GraphletDAG-projected coefficients.

    level = 1:
        atom projection

    level = 2:
        bond projection
    """

    mol = Chem.Mol(
        mol
    )

    # -------------------------------------------------------------------------
    # Validate partition.
    # -------------------------------------------------------------------------

    expected_atoms = set(
        range(
            mol.GetNumAtoms()
        )
    )

    provided_atoms = set(
        partition
    )

    if expected_atoms != provided_atoms:

        missing_atoms = (
            expected_atoms
            - provided_atoms
        )

        extra_atoms = (
            provided_atoms
            - expected_atoms
        )

        raise ValueError(
            "Partition does not match molecule. "
            f"Missing atoms: {sorted(missing_atoms)}; "
            f"extra atoms: {sorted(extra_atoms)}."
        )

    # -------------------------------------------------------------------------
    # Coordinates.
    # -------------------------------------------------------------------------

    if mol.GetNumConformers() == 0:

        AllChem.Compute2DCoords(
            mol
        )

    # -------------------------------------------------------------------------
    # Normalize coefficient representation.
    # -------------------------------------------------------------------------

    if coefficients is not None:

        if isinstance(
            coefficients,
            np.ndarray
        ):

            coefficients = {
                index: float(value)
                for index, value
                in enumerate(
                    coefficients
                )
            }

        elif not isinstance(
            coefficients,
            dict
        ):

            coefficients = dict(
                coefficients
            )

    # -------------------------------------------------------------------------
    # Optional labels.
    # -------------------------------------------------------------------------

    if (
        coefficients is not None
        and (
            (
                level == 1
                and show_atom_labels
            )
            or (
                level == 2
                and show_bond_labels
            )
        )
    ):

        add_coefficient_annotations(
            mol=mol,
            coefficients=coefficients,
            level=level,
            precision=label_precision,
        )

    # -------------------------------------------------------------------------
    # Drawing object.
    # -------------------------------------------------------------------------

    drawer = rdMolDraw2D.MolDraw2DCairo(
        size[0],
        size[1],
    )

    draw_options = (
        drawer.drawOptions()
    )

    draw_options.clearBackground = True

    draw_options.addAtomIndices = bool(
        show_atom_indices
    )

    if (
        show_bond_labels
        or show_atom_labels
    ):

        draw_options.annotationFontScale = 0.8

    # -------------------------------------------------------------------------
    # Determine global coefficient magnitude for this molecule.
    # -------------------------------------------------------------------------

    max_coefficient = 1.0

    if (
        show_coefficients
        and coefficients
    ):

        max_coefficient = max(
            abs(
                float(value)
            )
            for value
            in coefficients.values()
        )

        if max_coefficient <= 0:
            max_coefficient = 1.0

    # =========================================================================
    # ATOM-LEVEL PROJECTION
    # =========================================================================

    if level == 1:

        highlight_atoms = list(
            range(
                mol.GetNumAtoms()
            )
        )

        highlight_bonds = []

        highlight_radii = {
            atom_idx: (
                highlight_radius
            )
            for atom_idx
            in highlight_atoms
        }

        highlight_bond_colors = {}

        highlight_atom_colors = {}

        for atom_idx in (
            highlight_atoms
        ):

            partition_id = (
                partition[
                    atom_idx
                ]
            )

            base_color = (
                partition_colors[
                    partition_id
                ]
            )

            if (
                show_coefficients
                and coefficients is not None
            ):

                magnitude = abs(
                    float(
                        coefficients.get(
                            atom_idx,
                            0.0
                        )
                    )
                )

                relative_magnitude = (
                    magnitude
                    / max_coefficient
                )

                intensity = (
                    0.30
                    + 0.70
                    * relative_magnitude
                )

                highlight_atom_colors[
                    atom_idx
                ] = tuple(
                    channel
                    * intensity
                    for channel
                    in base_color
                )

            else:

                highlight_atom_colors[
                    atom_idx
                ] = base_color

    # =========================================================================
    # BOND-LEVEL PROJECTION
    # =========================================================================

    elif level == 2:

        highlight_atoms = []

        highlight_atom_colors = {}

        highlight_radii = {}

        highlight_bonds = list(
            range(
                mol.GetNumBonds()
            )
        )

        highlight_bond_colors = {}

        for bond_idx in (
            highlight_bonds
        ):

            bond = (
                mol.GetBondWithIdx(
                    bond_idx
                )
            )

            atom_a = int(
                bond.GetBeginAtomIdx()
            )

            atom_b = int(
                bond.GetEndAtomIdx()
            )

            partition_a = (
                partition[
                    atom_a
                ]
            )

            partition_b = (
                partition[
                    atom_b
                ]
            )

            if (
                partition_a
                == partition_b
            ):

                base_color = (
                    partition_colors[
                        partition_a
                    ]
                )

                if (
                    show_coefficients
                    and coefficients is not None
                ):

                    magnitude = abs(
                        float(
                            coefficients.get(
                                bond_idx,
                                0.0
                            )
                        )
                    )

                    relative_magnitude = (
                        magnitude
                        / max_coefficient
                    )

                    intensity = (
                        0.30
                        + 0.70
                        * relative_magnitude
                    )

                    highlight_bond_colors[
                        bond_idx
                    ] = tuple(
                        channel
                        * intensity
                        for channel
                        in base_color
                    )

                else:

                    highlight_bond_colors[
                        bond_idx
                    ] = base_color

            else:

                # Cross-partition bond.
                if (
                    show_coefficients
                    and coefficients is not None
                ):

                    magnitude = abs(
                        float(
                            coefficients.get(
                                bond_idx,
                                0.0
                            )
                        )
                    )

                    relative_magnitude = (
                        magnitude
                        / max_coefficient
                    )

                    gray = (
                        0.30
                        + 0.40
                        * relative_magnitude
                    )

                    highlight_bond_colors[
                        bond_idx
                    ] = (
                        gray,
                        gray,
                        gray
                    )

                else:

                    highlight_bond_colors[
                        bond_idx
                    ] = (
                        0.50,
                        0.50,
                        0.50
                    )

    else:

        raise ValueError(
            "level must be 1 for atoms "
            f"or 2 for bonds, got {level}."
        )

    # -------------------------------------------------------------------------
    # Draw.
    # -------------------------------------------------------------------------

    drawer.DrawMolecule(
        mol,
        highlightAtoms=(
            highlight_atoms
        ),
        highlightAtomColors=(
            highlight_atom_colors
        ),
        highlightAtomRadii=(
            highlight_radii
        ),
        highlightBonds=(
            highlight_bonds
        ),
        highlightBondColors=(
            highlight_bond_colors
        ),
    )

    drawer.FinishDrawing()

    image = Image.open(
        BytesIO(
            drawer.GetDrawingText()
        )
    )

    return image


# =============================================================================
# CREATE PARTITION LEGEND
# =============================================================================

def create_partition_legend(
    mol: Chem.Mol,
    partition: Dict[int, int],
    partition_colors: Dict[
        int,
        Tuple[float, float, float]
    ],
    within_partition_contribs: Optional[
        Dict[int, float]
    ] = None,
) -> List[mpatches.Patch]:
    """
    Create legend showing partition composition and optional F_p.
    """

    legend_elements = []

    unique_partitions = sorted(
        set(
            partition.values()
        )
    )

    for partition_id in (
        unique_partitions
    ):

        atoms_in_partition = [
            atom_idx
            for atom_idx, assigned_partition
            in partition.items()
            if (
                assigned_partition
                == partition_id
            )
        ]

        atom_symbols = [
            mol.GetAtomWithIdx(
                atom_idx
            ).GetSymbol()
            for atom_idx
            in atoms_in_partition
        ]

        symbol_counts = {}

        for symbol in (
            atom_symbols
        ):

            symbol_counts[
                symbol
            ] = (
                symbol_counts.get(
                    symbol,
                    0
                )
                + 1
            )

        composition = ", ".join(
            f"{count}{symbol}"
            for symbol, count
            in sorted(
                symbol_counts.items()
            )
        )

        label = (
            f"P{partition_id}: "
            f"{composition}"
        )

        if (
            within_partition_contribs
            is not None
        ):

            contribution = float(
                within_partition_contribs.get(
                    partition_id,
                    0.0
                )
            )

            label += (
                f" | "
                f"$F_{{{partition_id}}}$"
                f" = {contribution:+.2f}"
            )

        legend_elements.append(
            mpatches.Patch(
                facecolor=(
                    partition_colors[
                        partition_id
                    ]
                ),
                edgecolor="black",
                linewidth=2,
                label=label,
            )
        )

    return legend_elements


# =============================================================================
# DRAW EXISTING GRAPHLETDAG OBJECTS
# =============================================================================

def draw_projected_coefs_with_partitions(
    graphlet_dags: Union[
        GraphletDAG,
        List[GraphletDAG]
    ],
    partitions: Union[
        Dict[int, int],
        List[Dict[int, int]]
    ],
    within_partition_contribs: Optional[
        Union[
            Dict[int, float],
            List[Dict[int, float]]
        ]
    ] = None,
    level: int = 1,
    figsize: Optional[
        Tuple[float, float]
    ] = None,
    ncol: int = 3,
    titles: Optional[
        List[str]
    ] = None,
    show_coefficients: bool = True,
    show_bond_labels: bool = False,
    show_atom_labels: bool = False,
    label_precision: int = 2,
    cmap_name: str = "distinct",
    dpi: int = 300,
    img_size: Tuple[int, int] = (
        600,
        600
    ),
    highlight_radius: float = 0.5,
    show_atom_indices: bool = False,
):
    """
    Draw partition assignments together with GraphletDAG-projected
    atom or bond coefficients.
    """

    # -------------------------------------------------------------------------
    # Normalize input.
    # -------------------------------------------------------------------------

    if not isinstance(
        graphlet_dags,
        list
    ):

        graphlet_dags = [
            graphlet_dags
        ]

        partitions = [
            partitions
        ]

        if (
            within_partition_contribs
            is not None
        ):

            within_partition_contribs = [
                within_partition_contribs
            ]

    graphlet_dags = deepcopy(
        graphlet_dags
    )

    n_molecules = len(
        graphlet_dags
    )

    if len(partitions) != n_molecules:

        raise ValueError(
            "Number of partitions must equal "
            "number of GraphletDAG objects."
        )

    # -------------------------------------------------------------------------
    # Layout.
    # -------------------------------------------------------------------------

    ncol = min(
        ncol,
        n_molecules
    )

    nrow = int(
        np.ceil(
            n_molecules
            / ncol
        )
    )

    if figsize is None:

        figsize = (
            6 * ncol,
            6 * nrow
        )

    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
    )

    axes = axes.ravel()

    # -------------------------------------------------------------------------
    # Draw.
    # -------------------------------------------------------------------------

    for i, (
        dag,
        partition
    ) in enumerate(
        zip(
            graphlet_dags,
            partitions
        )
    ):

        mol = dag.mol

        # GraphletDAG projection.
        projected_coefs = (
            dag.project_to_layer(
                level
            )
        )

        projected_coefs_dict = {
            index: float(value)
            for index, value
            in enumerate(
                projected_coefs
            )
        }

        unique_partitions = sorted(
            set(
                partition.values()
            )
        )

        partition_colors = (
            get_partition_colormap(
                n_partitions=len(
                    unique_partitions
                ),
                cmap_name=cmap_name,
                partition_ids=(
                    unique_partitions
                ),
            )
        )

        image = (
            draw_mol_with_partition_colors(
                mol=mol,
                partition=partition,
                partition_colors=(
                    partition_colors
                ),
                level=level,
                coefficients=(
                    projected_coefs_dict
                    if show_coefficients
                    else None
                ),
                size=img_size,
                dpi=dpi,
                highlight_radius=(
                    highlight_radius
                ),
                show_coefficients=(
                    show_coefficients
                ),
                show_bond_labels=(
                    show_bond_labels
                ),
                show_atom_labels=(
                    show_atom_labels
                ),
                label_precision=(
                    label_precision
                ),
                show_atom_indices=(
                    show_atom_indices
                ),
            )
        )

        ax = axes[
            i
        ]

        ax.imshow(
            image
        )

        ax.axis(
            "off"
        )

        if (
            titles is not None
            and i < len(titles)
        ):

            ax.set_title(
                titles[i],
                fontsize=12,
                fontweight="bold",
            )

        # ---------------------------------------------------------------------
        # Legend.
        # ---------------------------------------------------------------------

        contributions = None

        if (
            within_partition_contribs
            is not None
            and i
            < len(
                within_partition_contribs
            )
        ):

            contributions = (
                within_partition_contribs[
                    i
                ]
            )

        legend_elements = (
            create_partition_legend(
                mol=mol,
                partition=partition,
                partition_colors=(
                    partition_colors
                ),
                within_partition_contribs=(
                    contributions
                ),
            )
        )

        ax.legend(
            handles=legend_elements,
            loc="lower center",
            fontsize=9,
            framealpha=0.95,
            bbox_to_anchor=(
                0.5,
                -0.02
            ),
            ncol=min(
                2,
                len(
                    unique_partitions
                )
            ),
        )

    for unused_idx in range(
        n_molecules,
        len(axes)
    ):

        axes[
            unused_idx
        ].set_visible(
            False
        )

    plt.tight_layout()

    return (
        fig,
        axes
    )


# =============================================================================
# BUHITO CONVENIENCE FUNCTION
# =============================================================================

def draw_buhito_partition_interpretations(
    mols,
    interpretations,
    interpreter,
    level: int = 1,
    figsize=None,
    ncol: int = 3,
    titles=None,
    show_coefficients: bool = True,
    show_bond_labels: bool = False,
    show_atom_labels: bool = False,
    label_precision: int = 2,
    cmap_name: str = "distinct",
    dpi: int = 300,
    img_size: Tuple[int, int] = (
        600,
        600
    ),
    highlight_radius: float = 0.5,
    show_atom_indices: bool = False,
):
    """
    Convenience wrapper for the Buhito PartitionInterpreter.

    This function:

        1. retrieves Buhito bi information,
        2. constructs GraphletDAG,
        3. projects graphlet coefficients,
        4. overlays partition colors,
        5. displays F_p values in the legend.
    """

    if not isinstance(
        mols,
        list
    ):
        mols = [
            mols
        ]

    if not isinstance(
        interpretations,
        list
    ):
        interpretations = [
            interpretations
        ]

    if (
        len(mols)
        != len(interpretations)
    ):

        raise ValueError(
            "mols and interpretations "
            "must have equal length."
        )

    graphlet_dags = []

    partitions = []

    within_contributions = []

    for (
        mol,
        interpretation
    ) in zip(
        mols,
        interpretations
    ):

        dag = (
            build_graphlet_dag_from_buhito(
                mol=mol,
                interpreter=interpreter,
                layerwise=True,
            )
        )

        graphlet_dags.append(
            dag
        )

        partitions.append(
            interpretation.partition
        )

        within_contributions.append(
            interpretation.within_partition
        )

    return (
        draw_projected_coefs_with_partitions(
            graphlet_dags=(
                graphlet_dags
            ),
            partitions=(
                partitions
            ),
            within_partition_contribs=(
                within_contributions
            ),
            level=level,
            figsize=figsize,
            ncol=ncol,
            titles=titles,
            show_coefficients=(
                show_coefficients
            ),
            show_bond_labels=(
                show_bond_labels
            ),
            show_atom_labels=(
                show_atom_labels
            ),
            label_precision=(
                label_precision
            ),
            cmap_name=cmap_name,
            dpi=dpi,
            img_size=img_size,
            highlight_radius=(
                highlight_radius
            ),
            show_atom_indices=(
                show_atom_indices
            ),
        )
    )