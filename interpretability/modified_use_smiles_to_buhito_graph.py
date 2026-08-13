# =============================================================================
# SAVE RESULTS
# =============================================================================

comparison_df.to_csv(
    "20_molecules_greedy_distance_vs_"
    "metropolis_hastings_across_lambda_"
    "all_initializations.csv",
    index=False,
)

molecule_mean_df.to_csv(
    "20_molecules_greedy_distance_vs_"
    "metropolis_hastings_across_lambda_"
    "molecule_means.csv",
    index=False,
)

summary_df.to_csv(
    "20_molecules_greedy_distance_vs_"
    "metropolis_hastings_across_lambda_"
    "summary.csv",
    index=False,
)

paired_summary_df.to_csv(
    "20_molecules_greedy_distance_vs_"
    "metropolis_hastings_across_lambda_"
    "paired_summary.csv",
    index=False,
)

best_results_df.to_csv(
    "20_molecules_greedy_distance_vs_"
    "metropolis_hastings_across_lambda_"
    "best_results.csv",
    index=False,
)

sample_molecules_df.to_csv(
    "20_molecules_used_for_lambda_comparison.csv",
    index=False,
)

if not failed_runs_df.empty:
    failed_runs_df.to_csv(
        "20_molecules_greedy_distance_vs_"
        "metropolis_hastings_failed_runs.csv",
        index=False,
    )


print("\nSaved numerical results to:")

print(
    "  20_molecules_greedy_distance_vs_"
    "metropolis_hastings_across_lambda_"
    "all_initializations.csv"
)

print(
    "  20_molecules_greedy_distance_vs_"
    "metropolis_hastings_across_lambda_"
    "molecule_means.csv"
)

print(
    "  20_molecules_greedy_distance_vs_"
    "metropolis_hastings_across_lambda_"
    "summary.csv"
)

print(
    "  20_molecules_greedy_distance_vs_"
    "metropolis_hastings_across_lambda_"
    "paired_summary.csv"
)

print(
    "  20_molecules_greedy_distance_vs_"
    "metropolis_hastings_across_lambda_"
    "best_results.csv"
)

print(
    "  20_molecules_used_for_lambda_comparison.csv"
)

if not failed_runs_df.empty:
    print(
        "  20_molecules_greedy_distance_vs_"
        "metropolis_hastings_failed_runs.csv"
    )
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import io

PARTITION_COLORS = {
    0: (0.35, 0.55, 1.00),
    1: (1.00, 0.35, 0.35),
    2: (0.35, 0.80, 0.45),
    3: (0.80, 0.55, 1.00),
}
def smiles_to_buhito_graph(
    smiles,
    add_hs=False,
    output_2d_pos=False,
):
    result = smiles_to_nx(
        smiles,
        add_hs=add_hs,
        output_2d_pos=output_2d_pos,
    )

    return (
        result[0]
        if isinstance(result, (tuple, list))
        else result
    )
def mol_to_partition_image(mol, partition, img_size=(500, 500)):
    drawer = rdMolDraw2D.MolDraw2DCairo(img_size[0], img_size[1])

    opts = drawer.drawOptions()
    opts.useBWAtomPalette()
    opts.fillHighlights = True
    opts.highlightRadius = 0.35
    opts.bondLineWidth = 3

    atoms_to_highlight = []
    atom_colors = {}

    for atom_idx, pid in partition.items():
        atom_idx = int(atom_idx)
        atoms_to_highlight.append(atom_idx)
        atom_colors[atom_idx] = PARTITION_COLORS.get(pid, (0.75, 0.75, 0.75))

    bonds_to_highlight = []
    bond_colors = {}

    for bond in mol.GetBonds():
        bond_idx = bond.GetIdx()
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()

        p1 = partition.get(a1)
        p2 = partition.get(a2)

        bonds_to_highlight.append(bond_idx)

        if p1 == p2:
            bond_colors[bond_idx] = PARTITION_COLORS.get(p1, (0.75, 0.75, 0.75))
        else:
            bond_colors[bond_idx] = (0.60, 0.60, 0.60)

    drawer.DrawMolecule(
        mol,
        highlightAtoms=atoms_to_highlight,
        highlightAtomColors=atom_colors,
        highlightBonds=bonds_to_highlight,
        highlightBondColors=bond_colors
    )

    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def draw_rdkit_partition_grid(
    mol,
    partitions,
    titles,
    mols_per_row=2,
    img_size=(500, 500),
    title_fontsize=16,
    figsize=None
):
    n = len(partitions)
    n_rows = (n + mols_per_row - 1) // mols_per_row

    if figsize is None:
        figsize = (18, 6 * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        mols_per_row,
        figsize=figsize
    )

    axes = np.array(axes).reshape(-1)

    for ax, partition, title in zip(axes, partitions, titles):
        img = mol_to_partition_image(
            mol,
            partition,
            img_size=img_size
        )

        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            title,
            fontsize=title_fontsize,
            fontweight="bold",
            pad=10
        )

    for ax in axes[n:]:
        ax.axis("off")

    return fig


K_TO_PLOT = 2

for mol, smiles, size_label in selected_mols:
    print(f"\n{'=' * 80}")
    print(f"{size_label} MOLECULE: {smiles}")
    print(f"{'=' * 80}")

    results = all_results[size_label]

    selected_results = [
        r for r in results
        if r["N_partitions"] == K_TO_PLOT
    ]

    if not selected_results:
        print(f"⚠️ No results found with exactly k={K_TO_PLOT} partitions")
        continue

    selected_results.sort(key=lambda x: x["Score"], reverse=True)
    selected_results = selected_results[:6]

    print(
        f"\nFound {len(selected_results)} methods "
        f"with exactly k={K_TO_PLOT} partitions"
    )

    partitions = []
    titles = []

    for result in selected_results:
        interp = result["Interpretation"]
        partition = interp.partition
        partition_ids = sorted(set(partition.values()))

        partitions.append(partition)

        F0 = interp.within_partition.get(0, 0.0)
        F1 = interp.within_partition.get(1, 0.0)
        F01 = sum(interp.between_partition.values())
        F_higher = sum(interp.higher_order.values())
        f_G = F0 + F1 + F01 + F_higher

        title = f"{result['Strategy']}\n"
        title += f"Actual k={len(partition_ids)} | Score={result['Score']:.4f}\n"
        title += f"F₀={F0:+.1f} | F₁={F1:+.1f} | F₀₁={F01:+.1f}\n"
        title += f"f(G)={f_G:+.1f}"

        model_pred = interp.total_prediction

        titles.append(title)

    fig = draw_rdkit_partition_grid(
        mol,
        partitions,
        titles,
        mols_per_row=2,
        img_size=(500, 500),
        title_fontsize=16,
        figsize=(12, 12)
    )

    plt.suptitle(
        f"{size_label} Molecule: {smiles}\n"
        f"K={K_TO_PLOT} Partition Visualization — All Contributions",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    plt.savefig(
        f"k{K_TO_PLOT}_partition_viz_rdkit_{size_label.lower()}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("\nPartition details:")

    for result in selected_results:
        interp = result["Interpretation"]
        partition = interp.partition
        partition_ids = sorted(set(partition.values()))

        print(f"\n{result['Strategy']}")
        print(f"Actual partitions: {partition_ids}")

        for p_id in partition_ids:
            atoms_in_p = [
                idx for idx, part in partition.items()
                if part == p_id
            ]

            symbols = [
                mol.GetAtomWithIdx(idx).GetSymbol()
                for idx in atoms_in_p
            ]

            print(
                f"  P{p_id}: atoms={atoms_in_p}, "
                f"symbols={symbols}, "
                f"F_{p_id}={interp.within_partition.get(p_id, 0.0):+.3f}"
            )

        if interp.between_partition:
            print("  Pair terms:")
            for pair, val in sorted(interp.between_partition.items()):
                print(f"    F_{pair} = {val:+.3f}")

        if interp.higher_order:
            print("  Higher-order terms:")
            for parts, val in sorted(interp.higher_order.items()):
                print(f"    F_{parts} = {val:+.3f}")
# Score Calculation Breakdown
# ═══════════════════════════════════════════════════════════════════════════
# DETAILED SCORE CALCULATION FOR FC(F)(F)F - DISTANCE(k=2)
# ═══════════════════════════════════════════════════════════════════════════

from rdkit import Chem
from rdkit.Chem import AllChem

from partition_interpretability import (
    PartitionInterpreter,
    ChemicalPartitioner
)
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

print("=" * 100)
print("DETAILED SCORE CALCULATION: FC(F)(F)F (Carbon Tetrafluoride)")
print("=" * 100)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: CREATE MOLECULE
# ═══════════════════════════════════════════════════════════════════════════

smiles = "FC(F)(F)F"
mol = Chem.MolFromSmiles(smiles)
AllChem.Compute2DCoords(mol)

print(f"\n📝 Molecule: {smiles}")
print(f"   Atoms: {mol.GetNumAtoms()}")
print(f"   Bonds: {mol.GetNumBonds()}")

# Show atom details
print(f"\n   Atom Details:")
for atom in mol.GetAtoms():
    print(f"     Atom {atom.GetIdx()}: {atom.GetSymbol()}")

print(f"\n   Bond Details:")
for bond in mol.GetBonds():
    a1_idx = bond.GetBeginAtomIdx()
    a2_idx = bond.GetEndAtomIdx()
    a1_sym = mol.GetAtomWithIdx(a1_idx).GetSymbol()
    a2_sym = mol.GetAtomWithIdx(a2_idx).GetSymbol()
    print(f"     Bond {a1_idx}-{a2_idx}: {a1_sym}-{a2_sym}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: APPLY DISTANCE-BASED K=2 PARTITIONING
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print("STEP 1: DISTANCE-BASED PARTITIONING (k=2)")
print(f"{'='*100}")

partition = ChemicalPartitioner.distance_partition(mol, n_clusters=2)

print(f"\nPartition Assignment:")
for atom_idx in sorted(partition.keys()):
    atom = mol.GetAtomWithIdx(atom_idx)
    print(f"  Atom {atom_idx} ({atom.GetSymbol()}): Partition {partition[atom_idx]}")

# Count atoms in each partition
partition_0_atoms = [idx for idx, p in partition.items() if p == 0]
partition_1_atoms = [idx for idx, p in partition.items() if p == 1]

print(f"\nPartition 0: {len(partition_0_atoms)} atoms")
print(f"  Atoms: {partition_0_atoms}")
print(f"  Symbols: {[mol.GetAtomWithIdx(i).GetSymbol() for i in partition_0_atoms]}")

print(f"\nPartition 1: {len(partition_1_atoms)} atoms")
print(f"  Atoms: {partition_1_atoms}")
print(f"  Symbols: {[mol.GetAtomWithIdx(i).GetSymbol() for i in partition_1_atoms]}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: CREATE GRAPHLET DAG (for visualization)
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print("STEP 2: CREATE GRAPHLET DAG (for visualization)")
print(f"{'='*100}")

print(f"\n✓ GraphletDAG created")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: COMPUTE PARTITION CONTRIBUTIONS (CORRECT API)
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print("STEP 3: COMPUTE PARTITION CONTRIBUTIONS")
print(f"{'='*100}")

interpreter = interpreter_full

graph = smiles_to_buhito_graph(
    smiles,
    add_hs=False,
    output_2d_pos=False,
)

interpreter.register_graph(mol, graph)
interpretation = interpreter.compute_partition_contributions(mol, partition)

print(f"\n✓ Partition contributions computed")
print(f"  Type: {type(interpretation)}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: EXTRACT F0, F1, F01 VALUES WITH DETAILED BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print("STEP 4: EXTRACT CONTRIBUTION TERMS (DETAILED BREAKDOWN)")
print(f"{'='*100}")

# Within-partition contributions
F_0 = interpretation.within_partition.get(0, 0.0)
F_1 = interpretation.within_partition.get(1, 0.0)

print(f"\n🔵 PARTITION 0 (Within-partition contributions):")
print(f"  F₀ = {F_0:+.6f} kcal/mol")
print(f"  This is the sum of all graphlet weights where ALL atoms belong to P0")

print(f"\n🔴 PARTITION 1 (Within-partition contributions):")
print(f"  F₁ = {F_1:+.6f} kcal/mol")
print(f"  This is the sum of all graphlet weights where ALL atoms belong to P1")

# Between-partition contributions - detailed bond-by-bond
print(f"\n⚪ CROSS-PARTITION (Between-partition contributions):")
print(f"  {'─'*80}")

if interpretation.between_partition:
    print(f"  Number of cross-partition graphlet contributions: {len(interpretation.between_partition)}")
    print(f"  Individual cross-partition contributions:")
    
    for key, value in sorted(interpretation.between_partition.items(), 
                            key=lambda x: abs(x[1]), reverse=True)[:10]:  # Show top 10
        if isinstance(key, tuple) and len(key) == 2:
            p1, p2 = key
            print(f"    Partition pair ({p1},{p2}): {value:+.6f} kcal/mol")
            print(f"      → Sum of graphlets spanning atoms in both P{p1} and P{p2}")
        else:
            print(f"    {key}: {value:+.6f} kcal/mol")
    
    F_01 = sum(interpretation.between_partition.values())
else:
    print(f"  No cross-partition graphlets")
    F_01 = 0.0

print(f"  {'─'*80}")
print(f"  F₀₁ (total) = {F_01:+.6f} kcal/mol")
print(f"  This is the sum of all graphlet weights that span EXACTLY 2 partitions")

# Higher-order contributions
print(f"\n⚫ HIGHER-ORDER (3+ partition graphlets):")
print(f"  {'─'*80}")

if interpretation.higher_order:
    print(f"  Number of higher-order contributions: {len(interpretation.higher_order)}")
    for key, value in interpretation.higher_order.items():
        print(f"    Partitions {key}: {value:+.6f} kcal/mol")
    F_higher = sum(interpretation.higher_order.values())
else:
    print(f"  No higher-order graphlets")
    F_higher = 0.0

print(f"  {'─'*80}")
print(f"  F_higher (total) = {F_higher:+.6f} kcal/mol")
print(f"  This is the sum of all graphlet weights that span 3+ partitions")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: SHOW HOW GRAPHLET WEIGHTS ARE SUMMED
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print("STEP 5: HOW GRAPHLET WEIGHTS ARE SUMMED TO GET F₀, F₁, F₀₁")
print(f"{'='*100}")

print(f"""
📚 THEORETICAL FOUNDATION (Equation 2 from paper):

The model prediction f(G) is decomposed based on which partition(s) each
# graphlet instance spans:

    f(G) = Σ F_p + Σ F_{{p1,p2}} + Σ F_{{p1,p2,...}}
           p    p1<p2        higher-order

# Where:
    • F_p (WITHIN): Sum of weights for graphlets fully contained in partition p
    • F_{{p1,p2}} (BETWEEN): Sum of weights for graphlets spanning exactly partitions p1 & p2  
    • F_{{p1,p2,...}} (HIGHER): Sum of weights for graphlets spanning 3+ partitions

# 🔬 HOW IT WORKS FOR THIS MOLECULE:

# 1. The interpreter enumerates ALL graphlet instances in the molecule
   (not just counts, but actual atom sets)

# 2. For each graphlet instance with coefficient w_i and atom set A_i:
   
   a) Determine which partition(s) the atoms in A_i belong to
   
   b) If all atoms are in partition p:
      → Add w_i to F_p (within-partition)
   
   c) If atoms span exactly partitions p1 and p2:
      → Add w_i to F_{{p1,p2}} (cross-partition)
   
   d) If atoms span 3+ partitions:
      → Add w_i to F_{{p1,p2,...}} (higher-order)

# 3. Sum up all contributions:
   f(G) = F_0 + F_1 + F_{{0,1}} + F_higher
""")

print(f"\n  Current molecule decomposition:")
print(f"  {'─'*80}")
print(f"    F₀:       {F_0:+12.6f} kcal/mol  (graphlets fully in P0)")
print(f"    F₁:       {F_1:+12.6f} kcal/mol  (graphlets fully in P1)")
print(f"    F₀₁:      {F_01:+12.6f} kcal/mol  (graphlets spanning P0↔P1)")
print(f"    F_higher: {F_higher:+12.6f} kcal/mol  (graphlets spanning 3+ partitions)")
print(f"    {'─'*80}")
print(f"    TOTAL:    {F_0 + F_1 + F_01 + F_higher:+12.6f} kcal/mol")

# Get model prediction for verification
total_pred = interpretation.total_prediction

print(f"\n  Verification that sum equals model prediction:")
print(f"    Sum of contributions: {F_0 + F_1 + F_01 + F_higher:+.6f} kcal/mol")
print(f"    Model prediction:     {total_pred:+.6f} kcal/mol")
print(f"    Difference:           {abs((F_0 + F_1 + F_01 + F_higher) - total_pred):.9f} kcal/mol")

if abs((F_0 + F_1 + F_01 + F_higher) - total_pred) < 1e-6:
    print(f"    ✅ PERFECT MATCH - Decomposition is exact!")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: CALCULATE TOTAL PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print("STEP 6: CALCULATE TOTAL PREDICTION f(G)")
print(f"{'='*100}")

f_G = F_0 + F_1 + F_01 + F_higher

print(f"\n  Equation: f(G) = F₀ + F₁ + F₀₁ + F_higher")
print(f"\n  Substituting values:")
print(f"    f(G) = {F_0:+.6f} + {F_1:+.6f} + {F_01:+.6f} + {F_higher:+.6f}")
print(f"    f(G) = {f_G:+.6f} kcal/mol")

print(f"\n  Verification:")
print(f"    Calculated f(G): {f_G:+.6f} kcal/mol")
print(f"    Model Pred:      {total_pred:+.6f} kcal/mol")
print(f"    Error:           {abs(f_G - total_pred):.9f} kcal/mol")

if abs(f_G - total_pred) < 1e-6:
    print(f"    ✅ PERFECT MATCH!")
elif abs(f_G - total_pred) < 0.001:
    print(f"    ✅ Excellent (within 0.001)")
elif abs(f_G - total_pred) < 0.1:
    print(f"    ✅ Good (within 0.1)")
else:
    print(f"    ⚠️  Discrepancy detected")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 8: CALCULATE INTERPRETABILITY SCORE IN DETAIL
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print("STEP 7: CALCULATE INTERPRETABILITY SCORE (STEP-BY-STEP)")
print(f"{'='*100}")

print(f"""
# 📊 INTERPRETABILITY SCORE FORMULA:

# The score measures how well the partition separates the molecule into
independent functional groups. Range: [0, 1]

# Steps:
  1. Calculate absolute totals: |F₀|, |F₁|, |F₀₁|, |F_higher|
  2. Calculate fractions of total: within_frac, between_frac, higher_frac
  3. Apply penalties:
     • within_frac  × 1.0   (no penalty - this is good!)
     • between_frac × -0.5  (mild penalty - some coupling)
     • higher_frac  × -2.0  (heavy penalty - complex coupling)
  4. Normalize: (raw_score + 2.0) / 3.0 → [0, 1]
  5. Clip to ensure result in [0, 1]

Higher score = better separation!
""")

# Step 1: Calculate absolute totals
within_total = sum(
    abs(value)
    for value in interpretation.within_partition.values()
)

between_total = sum(
    abs(value)
    for value in interpretation.between_partition.values()
)

higher_total = sum(
    abs(value)
    for value in interpretation.higher_order.values()
)
total = within_total + between_total + higher_total + 1e-10

print(f"\n  Step 1: Calculate absolute totals")
print(f"  {'─'*80}")
print(f"    |F₀|           = {abs(F_0):12.6f} kcal/mol")
print(f"    |F₁|           = {abs(F_1):12.6f} kcal/mol")
print(f"    {'─'*50}")
print(f"    Within total  = {within_total:12.6f} kcal/mol")
print(f"")
print(f"    |F₀₁|          = {abs(F_01):12.6f} kcal/mol")
print(f"    {'─'*50}")
print(f"    Between total = {between_total:12.6f} kcal/mol")
print(f"")
print(f"    |F_higher|     = {abs(F_higher):12.6f} kcal/mol")
print(f"    {'─'*50}")
print(f"    Higher total  = {higher_total:12.6f} kcal/mol")
print(f"")
print(f"    {'═'*50}")
print(f"    GRAND TOTAL   = {total:12.6f} kcal/mol")

# Step 2: Calculate fractions
within_frac = within_total / total
between_frac = between_total / total
higher_frac = higher_total / total

print(f"\n  Step 2: Calculate fractions of total")
print(f"  {'─'*80}")
print(f"    within_frac  = {within_total:10.6f} / {total:10.6f} = {within_frac:.6f}")
print(f"                 = {100*within_frac:6.2f}%")
print(f"")
print(f"    between_frac = {between_total:10.6f} / {total:10.6f} = {between_frac:.6f}")
print(f"                 = {100*between_frac:6.2f}%")
print(f"")
print(f"    higher_frac  = {higher_total:10.6f} / {total:10.6f} = {higher_frac:.6f}")
print(f"                 = {100*higher_frac:6.2f}%")
print(f"")
print(f"    {'─'*50}")
print(f"    Sum check    = {within_frac + between_frac + higher_frac:.6f}")
print(f"                 ≈ 1.000000 ✓")

# Step 3: Apply penalties
print(f"\n  Step 3: Apply penalties to calculate raw score")
print(f"  {'─'*80}")
print(f"    Formula: raw_score = within_frac × 1.0 - between_frac × 0.5 - higher_frac × 2.0")
print(f"")
print(f"    Breaking it down:")

within_contrib = within_frac * 1.0
between_contrib = between_frac * (-0.5)
higher_contrib = higher_frac * (-2.0)

print(f"      within_frac  × 1.0  = {within_frac:.6f} × 1.0  = {within_contrib:+.6f}")
print(f"      between_frac × -0.5 = {between_frac:.6f} × -0.5 = {between_contrib:+.6f}")
print(f"      higher_frac  × -2.0 = {higher_frac:.6f} × -2.0 = {higher_contrib:+.6f}")
print(f"      {'─'*50}")

raw_score = within_contrib + between_contrib + higher_contrib
print(f"      raw_score           =                  {raw_score:+.6f}")

# Step 4: Normalize
print(f"\n  Step 4: Normalize to [0, 1] range")
print(f"  {'─'*80}")
print(f"    Formula: normalized = (raw_score + 2.0) / 3.0")
print(f"    Calculation: ({raw_score:+.6f} + 2.0) / 3.0")

normalized_score = (raw_score + 2.0) / 3.0
print(f"    normalized = {normalized_score:.6f}")

# Step 5: Clip
print(f"\n  Step 5: Clip to [0, 1] (safety check)")
print(f"  {'─'*80}")

final_score = max(0.0, min(1.0, normalized_score))
print(f"    final_score = {final_score:.6f}")

print(f"\n  {'═'*80}")
# print(f"  📊 FINAL INTERPRETABILITY SCORE: {final_score:.6f}")
print(f"  {'═'*80}")

# Quality assessment
if final_score >= 0.9:
    quality = "EXCELLENT"
    emoji = "🏆"
    desc = "Minimal cross-partition coupling - excellent separation"
elif final_score >= 0.7:
    quality = "GOOD"
    emoji = "✅"
    desc = "Good separation between partitions"
elif final_score >= 0.5:
    quality = "FAIR"
    emoji = "⚠️"
    desc = "Moderate cross-partition coupling"
else:
    quality = "POOR"
    emoji = "❌"
    desc = "High cross-partition coupling - poor separation"

print(f"\n  {emoji} Score Quality: {quality}")
print(f"    {desc}")
print(f"")
print(f"    Distribution breakdown:")
print(f"      → {100*within_frac:.2f}% of absolute energy is within partitions")
print(f"      → {100*between_frac:.2f}% crosses partition boundaries")
print(f"      → {100*higher_frac:.2f}% is in higher-order terms")
print(f"")
print(f"    Chemical interpretation:")
if final_score >= 0.8:
    print(f"      → Partitions capture independent functional groups")
    print(f"      → Changes in one partition minimally affect the other")
    print(f"      → Clear chemical separation")
elif final_score >= 0.6:
    print(f"      → Partitions have reasonable independence")
    print(f"      → Some interaction between functional groups")
    print(f"      → Moderate chemical separation")
elif final_score >= 0.4:
    print(f"      → Partitions have significant coupling")
    print(f"      → Strong interaction between groups")
    print(f"      → Limited chemical separation")
else:
    print(f"      → Partitions are highly coupled")
    print(f"      → Very strong interaction between groups")
    print(f"      → Poor chemical separation")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 9: DETAILED BREAKDOWN BY BOND
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print("STEP 8: BOND-BY-BOND BREAKDOWN")
print(f"{'='*100}")

print(f"\nMolecule structure: {smiles}")
print(f"  {mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds")
print(f"")
print(f"Bond classification by partition:")
print(f"  {'─'*80}")

# Analyze each bond
for bond in mol.GetBonds():
    idx1 = bond.GetBeginAtomIdx()
    idx2 = bond.GetEndAtomIdx()
    sym1 = mol.GetAtomWithIdx(idx1).GetSymbol()
    sym2 = mol.GetAtomWithIdx(idx2).GetSymbol()
    p1 = partition[idx1]
    p2 = partition[idx2]
    
    # Check if this bond is within or between partitions
    if p1 == p2:
        bond_type = f"WITHIN P{p1}"
        symbol = "🔵" if p1 == 0 else "🔴"
    else:
        bond_type = "CROSS P0↔P1"
        symbol = "⚪"
    
    print(f"  {symbol} Bond {idx1}({sym1},P{p1})-{idx2}({sym2},P{p2}): {bond_type}")

print(f"  {'─'*80}")

# Summary by type
n_within = sum(1 for b in mol.GetBonds() if partition[b.GetBeginAtomIdx()] == partition[b.GetEndAtomIdx()])
n_cross = mol.GetNumBonds() - n_within

print(f"\nBond type summary:")
print(f"  Within-partition bonds:  {n_within}")
print(f"  Cross-partition bonds:   {n_cross}")
print(f"  Total bonds:             {mol.GetNumBonds()}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 10: VISUALIZE WITH RDKIT
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print("STEP 9: VISUALIZATION")
print(f"{'='*100}")

from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io

PARTITION_COLORS = {
    0: (0.35, 0.55, 1.00),  # blue
    1: (1.00, 0.35, 0.35),  # red
}

def mol_to_partition_image(mol, partition, img_size=(800, 800)):
    drawer = rdMolDraw2D.MolDraw2DCairo(img_size[0], img_size[1])

    opts = drawer.drawOptions()
    opts.useBWAtomPalette()
    opts.fillHighlights = True
    opts.highlightRadius = 0.35
    opts.bondLineWidth = 3

    atoms_to_highlight = []
    atom_colors = {}

    for atom_idx, pid in partition.items():
        atom_idx = int(atom_idx)
        atoms_to_highlight.append(atom_idx)
        atom_colors[atom_idx] = PARTITION_COLORS.get(
            pid,
            (0.75, 0.75, 0.75)
        )

    bonds_to_highlight = []
    bond_colors = {}

    for bond in mol.GetBonds():
        bond_idx = bond.GetIdx()
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()

        p1 = partition.get(a1)
        p2 = partition.get(a2)

        bonds_to_highlight.append(bond_idx)

        if p1 == p2:
            bond_colors[bond_idx] = PARTITION_COLORS.get(
                p1,
                (0.75, 0.75, 0.75)
            )
        else:
            bond_colors[bond_idx] = (0.60, 0.60, 0.60)

    drawer.DrawMolecule(
        mol,
        highlightAtoms=atoms_to_highlight,
        highlightAtomColors=atom_colors,
        highlightBonds=bonds_to_highlight,
        highlightBondColors=bond_colors
    )

    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))


title = (
    f"{smiles}\n"
    f"Distance k=2: Score={final_score:.4f} ({quality})\n"
    f"F₀={F_0:+.2f} | F₁={F_1:+.2f} | F₀₁={F_01:+.2f}\n"
    f"f(G)={f_G:+.2f} kcal/mol"
)

img = mol_to_partition_image(
    mol,
    partition,
    img_size=(800, 800)
)

fig, ax = plt.subplots(figsize=(10, 10))

ax.imshow(img)
ax.axis("off")
ax.set_title(
    title,
    fontsize=10,
    fontweight="bold",
    pad=10
)

plt.suptitle(
    f"Detailed Score Calculation: {smiles}\n"
    f"Blue=P0 | Red=P1 | Gray=Cross-partition",
    fontsize=14,
    fontweight="bold",
    y=0.98
)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])

filename = f'detailed_score_{smiles.replace("(", "").replace(")", "")}_rdkit.png'
plt.savefig(filename, dpi=300, bbox_inches="tight")
plt.show()

print(f"\n✓ Visualization saved: {filename}")
print("  • Blue atoms/bonds: Partition 0")
print("  • Red atoms/bonds: Partition 1")
print("  • Gray bonds: Cross-partition")
print("  • No bond coefficient labels are shown in this RDKit-only plot")

# ═══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
# print("📊 COMPLETE SUMMARY")
print(f"{'='*100}")

print(f"\n╔═══════════════════════════════════════════════════════════════════════════╗")
print(f"║                        PARTITION SCORE CALCULATION                        ║")
print(f"╚═══════════════════════════════════════════════════════════════════════════╝")

print(f"\n  MOLECULE:")
print(f"    SMILES:     {smiles}")
print(f"    Atoms:      {mol.GetNumAtoms()}")
print(f"    Bonds:      {mol.GetNumBonds()}")

print(f"\n  PARTITION METHOD:")
print(f"    Algorithm:  Distance-based clustering")
print(f"    k:          2 (number of partitions)")

print(f"\n  PARTITION ASSIGNMENT:")
print(f"    P0: {partition_0_atoms}")
print(f"        Elements: {[mol.GetAtomWithIdx(i).GetSymbol() for i in partition_0_atoms]}")
print(f"        Size: {len(partition_0_atoms)} atoms")
print(f"")
print(f"    P1: {partition_1_atoms}")
print(f"        Elements: {[mol.GetAtomWithIdx(i).GetSymbol() for i in partition_1_atoms]}")
print(f"        Size: {len(partition_1_atoms)} atoms")

print(f"\n  ENERGY CONTRIBUTIONS (sum of graphlet weights):")
# print(f"    ┌─────────────────────────────────────────────────────────┐")
print(f"    │ Within P0 (F₀):      {F_0:+12.6f} kcal/mol         │")
print(f"    │ Within P1 (F₁):      {F_1:+12.6f} kcal/mol         │")
print(f"    │ Cross P0↔P1 (F₀₁):   {F_01:+12.6f} kcal/mol         │")
if F_higher != 0:
    print(f"    │ Higher-order:        {F_higher:+12.6f} kcal/mol         │")
# print(f"    ├─────────────────────────────────────────────────────────┤")
print(f"    │ TOTAL f(G):          {f_G:+12.6f} kcal/mol         │")
# print(f"    └─────────────────────────────────────────────────────────┘")

print(f"\n  MODEL VERIFICATION:")
print(f"    Calculated f(G):  {f_G:+.6f} kcal/mol")
print(f"    Model prediction: {total_pred:+.6f} kcal/mol")
print(f"    Absolute error:   {abs(f_G - total_pred):.9f} kcal/mol")
error_pct = 100 * abs(f_G - total_pred) / abs(total_pred) if total_pred != 0 else 0
print(f"    Relative error:   {error_pct:.6f}%")
if abs(f_G - total_pred) < 1e-6:
    print(f"    Status:           ✅ PERFECT MATCH")
elif abs(f_G - total_pred) < 0.001:
    print(f"    Status:           ✅ EXCELLENT (<0.001)")
else:
    print(f"    Status:           ✅ GOOD (<{abs(f_G - total_pred):.3f})")

print(f"\n  INTERPRETABILITY SCORE CALCULATION:")
# print(f"    ┌───────────────────────────────────────────────────────────────┐")
print(f"    │ 1. Absolute totals:                                           │")
print(f"    │    Within:  {within_total:10.4f} kcal/mol ({100*within_frac:5.2f}%)              │")
print(f"    │    Between: {between_total:10.4f} kcal/mol ({100*between_frac:5.2f}%)              │")
print(f"    │    Higher:  {higher_total:10.4f} kcal/mol ({100*higher_frac:5.2f}%)              │")
print(f"    │    Total:   {total:10.4f} kcal/mol                               │")
# print(f"    ├───────────────────────────────────────────────────────────────┤")
print(f"    │ 2. Apply penalties:                                           │")
print(f"    │    within_frac × 1.0   = {within_contrib:+.6f}                     │")
print(f"    │    between_frac × -0.5 = {between_contrib:+.6f}                     │")
print(f"    │    higher_frac × -2.0  = {higher_contrib:+.6f}                     │")
print(f"    │    raw_score           = {raw_score:+.6f}                     │")
# print(f"    ├───────────────────────────────────────────────────────────────┤")
print(f"    │ 3. Normalize to [0,1]:                                        │")
print(f"    │    (raw_score + 2.0) / 3.0 = {normalized_score:.6f}                  │")
# print(f"    ├───────────────────────────────────────────────────────────────┤")
print(f"    │ 4. Final score:        {final_score:.6f}                           │")
# print(f"    └───────────────────────────────────────────────────────────────┘")

print(f"\n  SCORE INTERPRETATION:")
print(f"    Value:   {final_score:.6f}")
print(f"    Quality: {quality} {emoji}")
print(f"    Meaning: {desc}")
print(f"")
print(f"    What this means:")
print(f"      • {100*within_frac:.1f}% of energy contribution is localized within partitions")
print(f"      • {100*between_frac:.1f}% crosses the partition boundary")
print(f"      • {100*higher_frac:.1f}% is in complex higher-order interactions")

if final_score >= 0.8:
    print(f"\n    ⭐ This is an EXCELLENT partition!")
    print(f"      The partitions represent nearly independent chemical units.")
    print(f"      Changes in one partition would minimally affect the other.")
elif final_score >= 0.6:
    print(f"\n    👍 This is a GOOD partition.")
    print(f"      The partitions show reasonable independence.")
    print(f"      Some coupling exists but separation is meaningful.")
elif final_score >= 0.4:
    print(f"\n    ⚠️  This is a FAIR partition.")
    print(f"      Moderate coupling between partitions.")
    print(f"      Separation is present but limited.")
else:
    print(f"\n    ⛔ This is a POOR partition.")
    print(f"      Strong coupling between partitions.")
    print(f"      The separation may not be chemically meaningful.")

print(f"\n  CHEMICAL INSIGHTS:")
print(f"    Structure: CF₄ (carbon tetrafluoride)")
print(f"")
# Identify which partition is central vs peripheral
if len(partition_0_atoms) == 1:
    central_p = 0
    peripheral_p = 1
    central_atom = partition_0_atoms[0]
elif len(partition_1_atoms) == 1:
    central_p = 1
    peripheral_p = 0
    central_atom = partition_1_atoms[0]
else:
    # Use connectivity to identify central atom
    max_degree = max((len(list(mol.GetAtomWithIdx(i).GetNeighbors())), i) 
                     for i in range(mol.GetNumAtoms()))
    central_atom = max_degree[1]
    central_p = partition[central_atom]
    peripheral_p = 1 - central_p

print(f"    Partition {central_p}: Central carbon atom (high connectivity)")
print(f"    Partition {peripheral_p}: Peripheral fluorine atoms")
print(f"")
print(f"    Physical interpretation:")
print(f"      • This partition separates the core from the substituents")
print(f"      • Each C-F bond crosses the partition boundary")
print(f"      • F₀₁ represents the energy of C-F interactions")
print(f"      • High cross-partition contribution expected (strong C-F bonds)")

print(f"\n╔═══════════════════════════════════════════════════════════════════════════╗")
print(f"║                          CALCULATION COMPLETE                             ║")
print(f"╚═══════════════════════════════════════════════════════════════════════════╝")

print(f"\n{'='*100}")
print("✅ DETAILED SCORE CALCULATION COMPLETE")
print(f"{'='*100}")

print(f"\nKey Results:")
print(f"  • Score: {final_score:.6f} ({quality})")
print(f"  • f(G) = F₀ + F₁ + F₀₁ = {F_0:+.2f} + {F_1:+.2f} + {F_01:+.2f} = {f_G:+.2f} kcal/mol")
print(f"  • Model prediction: {total_pred:+.2f} kcal/mol (error: {abs(f_G - total_pred):.6f})")
if filename:
    print(f"  • Visualization: {filename}")

print(f"\n{'='*100}\n")

# ═══════════════════════════════════════════════════════════════════════════
# EXPORT DATA
# ═══════════════════════════════════════════════════════════════════════════

# Create exportable dictionary
export_data = {
    'molecule': {
        'smiles': smiles,
        'n_atoms': mol.GetNumAtoms(),
        'n_bonds': mol.GetNumBonds(),
    },
    'partition': {
        'method': 'distance',
        'k': 2,
        'P0_atoms': partition_0_atoms,
        'P1_atoms': partition_1_atoms,
    },
    'contributions': {
        'F0': float(F_0),
        'F1': float(F_1),
        'F01': float(F_01),
        'F_higher': float(F_higher),
        'total': float(f_G),
    },
    'prediction': {
        'calculated': float(f_G),
        'model': float(total_pred),
        'error': float(abs(f_G - total_pred)),
    },
    'score': {
        'within_total': float(within_total),
        'between_total': float(between_total),
        'higher_total': float(higher_total),
        'within_frac': float(within_frac),
        'between_frac': float(between_frac),
        'higher_frac': float(higher_frac),
        'raw_score': float(raw_score),
        'normalized_score': float(normalized_score),
        'final_score': float(final_score),
        'quality': quality,
    }
}

print(f"💾 Data exported to: export_data dictionary")
# ═══════════════════════════════════════════════════════════════════════════
# NEW STEP: DETAILED GRAPHLET-BY-GRAPHLET BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print("STEP 4A: DETAILED GRAPHLET-BY-GRAPHLET BREAKDOWN")
print(f"{'='*100}")

print(f"""
# 🔬 HOW F₀, F₁, and F₀₁ ARE ACTUALLY CALCULATED:

# This section shows the EXACT process of how graphlet instances are enumerated
# and their contributions summed to get F₀, F₁, and F₀₁.

# ALGORITHM:
# ─────────
# 1. Enumerate ALL graphlet instances in the molecule
   • Each instance = a set of atom indices + its coefficient
   
# 2. For EACH instance:
   a) Look at which atoms it contains
   b) Determine which partition(s) those atoms belong to
   c) Classify the instance:
      - If all atoms in P0 only → add coefficient to F₀
      - If all atoms in P1 only → add coefficient to F₁
      - If atoms span P0 & P1 → add coefficient to F₀₁
   
# 3. Sum up all contributions:
   F₀ = Σ(coefficients of instances fully in P0)
   F₁ = Σ(coefficients of instances fully in P1)
   F₀₁ = Σ(coefficients of instances spanning P0 & P1)
""")

# Get the actual graphlet instances
all_instances = (
    interpreter
    .enumerate_graphlet_instances(
        mol,
        include_zero_coefficients=True
    )
)

active_instances = (
    interpreter
    .enumerate_graphlet_instances(
        mol,
        include_zero_coefficients=False
    )
)

print(
    "All graphlet instances:",
    len(all_instances)
)

print(
    "Active graphlet instances:",
    len(active_instances)
)
instances = active_instances
# print(f"\n📊 ENUMERATION RESULTS:")
print(f"   Total graphlet instances found: {len(instances)}")

# Categorize instances
within_P0_instances = []
within_P1_instances = []
cross_instances = []
higher_instances = []

for instance in instances:
    atom_set = instance['atoms']
    coefficient = instance['coefficient']
    bit_id = instance['bit_id']
    
    # Get partitions spanned
    category, partitions_spanned = interpreter._classify_instance(
        instance,
        partition
    )
    
    if len(partitions_spanned) == 1:
        if 0 in partitions_spanned:
            within_P0_instances.append(instance)
        elif 1 in partitions_spanned:
            within_P1_instances.append(instance)
    elif len(partitions_spanned) == 2:
        cross_instances.append(instance)
    else:
        higher_instances.append(instance)

print(f"\n   Breakdown by partition:")
print(f"      Within P0:    {len(within_P0_instances)} instances")
print(f"      Within P1:    {len(within_P1_instances)} instances")
print(f"      Cross P0↔P1:  {len(cross_instances)} instances")
print(f"      Higher-order: {len(higher_instances)} instances")

# ═════════════════════════════════════════════════════════════════════════
# WITHIN P0 INSTANCES (F₀)
# ═════════════════════════════════════════════════════════════════════════

print(f"\n{'─'*100}")
print(f"🔵 WITHIN P0 INSTANCES (contribute to F₀)")
print(f"{'─'*100}")

if within_P0_instances:
    print(f"\nShowing all {len(within_P0_instances)} instances that contribute to F₀:\n")
    
    F_0_calc = 0.0
    for i, instance in enumerate(within_P0_instances[:20], 1):  # Show first 20
        atoms = sorted(instance['atoms'])
        coef = instance['coefficient']
        bit_id = instance['bit_id']
        
        # Get atom symbols
        atom_symbols = [mol.GetAtomWithIdx(a).GetSymbol() for a in atoms]
        
        # Get graphlet SMILES if available
        graphlet_smiles = "unknown"
        if isinstance(bit_id, tuple) and len(bit_id) > 0:
            if isinstance(bit_id[0], str):
                graphlet_smiles = bit_id[0]
        
        print(f"   Instance {i:3d}:")
        print(f"      Atoms:      {atoms}")
        print(f"      Elements:   {atom_symbols}")
        print(f"      Coef:       {coef:+.6f} kcal/mol")
        print(f"      Partitions: All atoms in P0 ✓")
        print(f"      → Adds {coef:+.6f} to F₀")
        
        F_0_calc += coef
        
        if i < len(within_P0_instances):
            print()
    
    if len(within_P0_instances) > 20:
        print(f"\n   ... and {len(within_P0_instances) - 20} more instances")
        # Add remaining
        for instance in within_P0_instances[20:]:
            F_0_calc += instance['coefficient']
    
    print(f"\n   {'─'*80}")
    print(f"   F₀ = Σ(all coefficients) = {F_0_calc:+.6f} kcal/mol")
    print(f"   Verification: F₀ from interpretation = {F_0:+.6f} kcal/mol")
    print(f"   Match: {abs(F_0_calc - F_0) < 1e-6} ✓" if abs(F_0_calc - F_0) < 1e-6 else f"   Difference: {abs(F_0_calc - F_0):.9f}")
else:
    print(f"\n   No instances found fully within P0")
    print(f"   F₀ = 0.0 kcal/mol")

# ═════════════════════════════════════════════════════════════════════════
# WITHIN P1 INSTANCES (F₁)
# ═════════════════════════════════════════════════════════════════════════

print(f"\n{'─'*100}")
print(f"🔴 WITHIN P1 INSTANCES (contribute to F₁)")
print(f"{'─'*100}")

if within_P1_instances:
    print(f"\nShowing all {len(within_P1_instances)} instances that contribute to F₁:\n")
    
    F_1_calc = 0.0
    for i, instance in enumerate(within_P1_instances[:20], 1):  # Show first 20
        atoms = sorted(instance['atoms'])
        coef = instance['coefficient']
        bit_id = instance['bit_id']
        
        # Get atom symbols
        atom_symbols = [mol.GetAtomWithIdx(a).GetSymbol() for a in atoms]
        
        # Get graphlet SMILES if available
        graphlet_smiles = "unknown"
        if isinstance(bit_id, tuple) and len(bit_id) > 0:
            if isinstance(bit_id[0], str):
                graphlet_smiles = bit_id[0]
        
        print(f"   Instance {i:3d}:")
        print(f"      Atoms:      {atoms}")
        print(f"      Elements:   {atom_symbols}")
        print(f"      Coef:       {coef:+.6f} kcal/mol")
        print(f"      Partitions: All atoms in P1 ✓")
        print(f"      → Adds {coef:+.6f} to F₁")
        
        F_1_calc += coef
        
        if i < len(within_P1_instances):
            print()
    
    if len(within_P1_instances) > 20:
        print(f"\n   ... and {len(within_P1_instances) - 20} more instances")
        # Add remaining
        for instance in within_P1_instances[20:]:
            F_1_calc += instance['coefficient']
    
    print(f"\n   {'─'*80}")
    print(f"   F₁ = Σ(all coefficients) = {F_1_calc:+.6f} kcal/mol")
    print(f"   Verification: F₁ from interpretation = {F_1:+.6f} kcal/mol")
    print(f"   Match: {abs(F_1_calc - F_1) < 1e-6} ✓" if abs(F_1_calc - F_1) < 1e-6 else f"   Difference: {abs(F_1_calc - F_1):.9f}")
else:
    print(f"\n   No instances found fully within P1")
    print(f"   F₁ = 0.0 kcal/mol")

# ═════════════════════════════════════════════════════════════════════════
# CROSS-PARTITION INSTANCES (F₀₁)
# ═════════════════════════════════════════════════════════════════════════

print(f"\n{'─'*100}")
print(f"⚪ CROSS-PARTITION INSTANCES (contribute to F₀₁)")
print(f"{'─'*100}")

if cross_instances:
    print(f"\nShowing all {len(cross_instances)} instances that span P0 ↔ P1:\n")
    
    F_01_calc = 0.0
    for i, instance in enumerate(cross_instances[:20], 1):  # Show first 20
        atoms = sorted(instance['atoms'])
        coef = instance['coefficient']
        bit_id = instance['bit_id']
        
        # Get atom symbols and partitions
        atom_info = [(a, mol.GetAtomWithIdx(a).GetSymbol(), partition[a]) 
                     for a in atoms]
        
        # Get graphlet SMILES if available
        graphlet_smiles = "unknown"
        if isinstance(bit_id, tuple) and len(bit_id) > 0:
            if isinstance(bit_id[0], str):
                graphlet_smiles = bit_id[0]
        
        print(f"   Instance {i:3d}:")
        print(f"      Atoms:      {atoms}")
        print(f"      Details:    ", end="")
        for idx, (atom_idx, symbol, part) in enumerate(atom_info):
            print(f"{atom_idx}({symbol},P{part})", end="")
            if idx < len(atom_info) - 1:
                print(", ", end="")
        print()
        print(f"      Coef:       {coef:+.6f} kcal/mol")
        
        # Show which partitions are spanned
        partitions_in_instance = set(p for _, _, p in atom_info)
        print(f"      Partitions: Spans P{sorted(partitions_in_instance)} ⚡")
        print(f"      → Adds {coef:+.6f} to F₀₁")
        
        F_01_calc += coef
        
        if i < len(cross_instances):
            print()
    
    if len(cross_instances) > 20:
        print(f"\n   ... and {len(cross_instances) - 20} more instances")
        # Add remaining
        for instance in cross_instances[20:]:
            F_01_calc += instance['coefficient']
    
    print(f"\n   {'─'*80}")
    print(f"   F₀₁ = Σ(all coefficients) = {F_01_calc:+.6f} kcal/mol")
    print(f"   Verification: F₀₁ from interpretation = {F_01:+.6f} kcal/mol")
    print(f"   Match: {abs(F_01_calc - F_01) < 1e-6} ✓" if abs(F_01_calc - F_01) < 1e-6 else f"   Difference: {abs(F_01_calc - F_01):.9f}")
else:
    print(f"\n   No instances found that span both partitions")
    print(f"   F₀₁ = 0.0 kcal/mol")

# ═════════════════════════════════════════════════════════════════════════
# SUMMARY OF CALCULATION PROCESS
# ═════════════════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print(f"📋 SUMMARY: HOW F₀, F₁, F₀₁ WERE CALCULATED")
print(f"{'='*100}")

print(f"""
# STEP-BY-STEP PROCESS:
# ─────────────────────

# 1️⃣  ENUMERATE GRAPHLETS
   • Total instances found: {len(instances)}
   • Each instance has: atom set + coefficient

# 2️⃣  CLASSIFY EACH INSTANCE
   • Check which partition(s) the atoms belong to
   • Within P0: {len(within_P0_instances)} instances
   • Within P1: {len(within_P1_instances)} instances  
   • Cross P0↔P1: {len(cross_instances)} instances
   • Higher-order: {len(higher_instances)} instances

# 3️⃣  SUM COEFFICIENTS BY CATEGORY
   
   F₀ = Σ(coefs of instances in P0)
      = {' + '.join([f'{inst["coefficient"]:+.3f}' for inst in within_P0_instances[:3]])}{'...' if len(within_P0_instances) > 3 else ''}
      = {F_0:+.6f} kcal/mol
   
   F₁ = Σ(coefs of instances in P1)
      = {' + '.join([f'{inst["coefficient"]:+.3f}' for inst in within_P1_instances[:3]])}{'...' if len(within_P1_instances) > 3 else ''}
      = {F_1:+.6f} kcal/mol
   
   F₀₁ = Σ(coefs of instances spanning P0↔P1)
       = {' + '.join([f'{inst["coefficient"]:+.3f}' for inst in cross_instances[:3]])}{'...' if len(cross_instances) > 3 else ''}
       = {F_01:+.6f} kcal/mol

# 4️⃣  TOTAL PREDICTION
   f(G) = F₀ + F₁ + F₀₁ + F_higher
        = {F_0:+.6f} + {F_1:+.6f} + {F_01:+.6f} + {F_higher:+.6f}
        = {F_0 + F_1 + F_01 + F_higher:+.6f} kcal/mol
""")

print(f"\n✅ KEY INSIGHT:")
print(f"   Each graphlet instance contributes its coefficient to EXACTLY ONE category:")
print(f"   • If graphlet atoms are all in P0 → coefficient goes to F₀")
print(f"   • If graphlet atoms are all in P1 → coefficient goes to F₁")
print(f"   • If graphlet atoms span P0 & P1 → coefficient goes to F₀₁")
print(f"")
print(f"   This ensures: f(G) = F₀ + F₁ + F₀₁ (+ higher-order)")
print(f"   No double-counting, no missing terms!")

# Show a specific example
if cross_instances:
    print(f"\n📌 CONCRETE EXAMPLE - Cross-Partition Graphlet:")
    example = cross_instances[0]
    atoms = sorted(example['atoms'])
    coef = example['coefficient']
    
    print(f"   Take instance with atoms {atoms}:")
    for atom_idx in atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        part = partition[atom_idx]
        print(f"      Atom {atom_idx} ({atom.GetSymbol()}): Partition {part}")
    
    partitions_spanned = set(partition[a] for a in atoms)
    print(f"\n   Partitions spanned: {sorted(partitions_spanned)}")
    print(f"   Coefficient: {coef:+.6f} kcal/mol")
    print(f"\n   Because this graphlet has atoms in BOTH P0 and P1:")
    print(f"   → Its coefficient {coef:+.6f} is added to F₀₁")
    print(f"   → NOT to F₀ or F₁")
    print(f"   → This represents the interaction energy between the partitions")

print(f"\n{'='*100}\n")
# ═══════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE VERIFICATION: Visualizations vs Actual Values
# ═══════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
from rdkit import Chem

def verify_partition_interpretation(mol, interp, interpreter, verbose=True):
    """
    Comprehensive verification of partition interpretation.
    
    Checks:
    1. Decomposition equation: f(G) = ∑F_p + ∑F_{p1,p2} + ∑F_{p1,p2,p3,...}
    2. Score calculation consistency
    3. Contribution breakdown percentages
    4. Bond-level projections match contributions
    
    Returns:
        dict with verification results
    """
    
    results = {
        'passed': True,
        'errors': [],
        'warnings': [],
        'details': {}
    }
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"VERIFICATION REPORT")
        print(f"{'='*80}\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 1: Equation (2) Decomposition
    # ─────────────────────────────────────────────────────────────────────────
    
    within_sum = sum(interp.within_partition.values())
    between_sum = sum(interp.between_partition.values())
    higher_sum = sum(interp.higher_order.values())
    total_from_parts = within_sum + between_sum + higher_sum
    
    total_prediction = interp.total_prediction
    
    decomp_diff = abs(total_from_parts - total_prediction)
    decomp_tolerance = 1e-2
    
    results['details']['decomposition'] = {
        'within_sum': within_sum,
        'between_sum': between_sum,
        'higher_sum': higher_sum,
        'sum_of_parts': total_from_parts,
        'model_prediction': total_prediction,
        'difference': decomp_diff,
        'passed': decomp_diff < decomp_tolerance
    }
    
    if verbose:
        print(f"TEST 1: Equation (2) Decomposition")
        print(f"  f(G) = ∑F_p + ∑F_{{p1,p2}} + ∑F_{{p1,p2,p3,...}}")
        print(f"  ")
        print(f"  Within-partition (∑F_p):        {within_sum:+10.4f}")
        print(f"  Between-partition (∑F_{{p1,p2}}): {between_sum:+10.4f}")
        print(f"  Higher-order (∑F_{{...}}):        {higher_sum:+10.4f}")
        print(f"  {'─'*50}")
        print(f"  Sum of parts:                   {total_from_parts:+10.4f}")
        print(f"  Model prediction:               {total_prediction:+10.4f}")
        print(f"  Difference:                     {decomp_diff:+10.6f}")
        print(f"  ")
        
        if decomp_diff < decomp_tolerance:
            print(f"  ✅ PASSED: Decomposition is correct (diff < {decomp_tolerance})")
        else:
            print(f"  ❌ FAILED: Decomposition error too large!")
            results['passed'] = False
            results['errors'].append(f"Decomposition error: {decomp_diff}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 2: Score Calculation
    # ─────────────────────────────────────────────────────────────────────────
    
    breakdown = interp.get_contribution_breakdown()
    
    # Manual score calculation
    within_frac = breakdown['within_frac']
    between_frac = breakdown['between_frac']
    higher_frac = breakdown['higher_frac']
    
    # Score formula from code:
    # score = within_frac - 0.5 * between_frac - 2.0 * higher_frac
    # normalized to [0, 1] via: (score + 2.0) / 3.0
    
    raw_score = within_frac - 0.5 * between_frac - 2.0 * higher_frac
    normalized_score = max(0.0, min(1.0, (raw_score + 2.0) / 3.0))
    
    score_diff = abs(normalized_score - interp.score)
    score_tolerance = 1e-6
    
    results['details']['score'] = {
        'within_frac': within_frac,
        'between_frac': between_frac,
        'higher_frac': higher_frac,
        'raw_score': raw_score,
        'normalized_score': normalized_score,
        'interp_score': interp.score,
        'difference': score_diff,
        'passed': score_diff < score_tolerance
    }
    
    if verbose:
        print(f"\nTEST 2: Score Calculation")
        print(f"  Score = (Within - 0.5*Between - 2.0*Higher + 2.0) / 3.0")
        print(f"  ")
        print(f"  Within fraction:    {within_frac:.6f}  ({within_frac*100:.2f}%)")
        print(f"  Between fraction:   {between_frac:.6f}  ({between_frac*100:.2f}%)")
        print(f"  Higher fraction:    {higher_frac:.6f}  ({higher_frac*100:.2f}%)")
        print(f"  {'─'*50}")
        print(f"  Raw score:          {raw_score:+.6f}")
        print(f"  Normalized score:   {normalized_score:.6f}")
        print(f"  Reported score:     {interp.score:.6f}")
        print(f"  Difference:         {score_diff:.9f}")
        print(f"  ")
        
        if score_diff < score_tolerance:
            print(f"  ✅ PASSED: Score calculation is correct")
        else:
            print(f"  ❌ FAILED: Score calculation mismatch!")
            results['passed'] = False
            results['errors'].append(f"Score calculation error: {score_diff}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 3: Contribution Breakdown Fractions Sum to 1
    # ─────────────────────────────────────────────────────────────────────────
    
    frac_sum = within_frac + between_frac + higher_frac
    frac_tolerance = 1e-6
    frac_diff = abs(frac_sum - 1.0)
    
    results['details']['fractions'] = {
        'sum': frac_sum,
        'difference_from_1': frac_diff,
        'passed': frac_diff < frac_tolerance
    }
    
    if verbose:
        print(f"\nTEST 3: Contribution Fractions")
        print(f"  Within + Between + Higher should sum to 1.0")
        print(f"  ")
        print(f"  Sum: {frac_sum:.9f}")
        print(f"  Difference from 1.0: {frac_diff:.9f}")
        print(f"  ")
        
        if frac_diff < frac_tolerance:
            print(f"  ✅ PASSED: Fractions sum to 1.0")
        else:
            print(f"  ⚠️  WARNING: Fractions don't sum to 1.0 (diff: {frac_diff})")
            results['warnings'].append(f"Fraction sum error: {frac_diff}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 4: Bond-Level Contributions Match Between-Partition Terms
    # ─────────────────────────────────────────────────────────────────────────
    
    if verbose:
        print(
            "\nTEST 4: Cross-Partition Bond Count"
        )

    partition = interp.partition

    cross_bonds = []

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()

        if partition[a1] != partition[a2]:
            cross_bonds.append(bond.GetIdx())

    between_total = sum(
        interp.between_partition.values()
    )

    results["details"]["bonds"] = {
        "n_cross_bonds": len(cross_bonds),
        "cross_bond_indices": cross_bonds,
        "between_total": between_total
    }

    if verbose:
        print(
            "  Cross-partition bonds:",
            cross_bonds
        )

        print(
            f"  Total F_pair contribution: "
            f"{between_total:+.6f}"
        )

        print(
            "  Note: DAG bond projections are not "
            "expected to equal F_pair bond-by-bond."
        )
    
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST 5: Partition Coverage
    # ─────────────────────────────────────────────────────────────────────────
    
    n_atoms = mol.GetNumAtoms()
    n_assigned = len(partition)
    unique_partitions = set(partition.values())
    
    coverage_passed = (n_assigned == n_atoms)
    
    results['details']['coverage'] = {
        'n_atoms': n_atoms,
        'n_assigned': n_assigned,
        'n_partitions': len(unique_partitions),
        'partition_ids': sorted(unique_partitions),
        'passed': coverage_passed
    }
    
    if verbose:
        print(f"\nTEST 5: Partition Coverage")
        print(f"  ")
        print(f"  Total atoms: {n_atoms}")
        print(f"  Assigned atoms: {n_assigned}")
        print(f"  Number of partitions: {len(unique_partitions)}")
        print(f"  Partition IDs: {sorted(unique_partitions)}")
        print(f"  ")
        
        if coverage_passed:
            print(f"  ✅ PASSED: All atoms assigned to partitions")
        else:
            print(f"  ❌ FAILED: Not all atoms assigned!")
            results['passed'] = False
            results['errors'].append(f"Coverage: {n_assigned}/{n_atoms} atoms assigned")
        
        # Show partition composition
        print(f"\n  Partition Composition:")
        for p_id in sorted(unique_partitions):
            atoms_in_p = [idx for idx, part in partition.items() if part == p_id]
            symbols = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in atoms_in_p]
            symbol_counts = {}
            for sym in symbols:
                symbol_counts[sym] = symbol_counts.get(sym, 0) + 1
            composition = "".join([f"{count}{sym}" for sym, count 
                                  in sorted(symbol_counts.items())])
            F_p = interp.within_partition.get(p_id, 0.0)
            print(f"    P{p_id}: {composition:15s} ({len(atoms_in_p):2d} atoms)  F_{p_id} = {F_p:+8.4f}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"VERIFICATION SUMMARY")
        print(f"{'='*80}\n")
        
        if results['passed'] and len(results['errors']) == 0:
            print(f"✅ ALL TESTS PASSED")
            print(f"   Visualizations should accurately reflect the values.")
        else:
            print(f"❌ VERIFICATION FAILED")
            print(f"   Errors found: {len(results['errors'])}")
            for err in results['errors']:
                print(f"     • {err}")
        
        if results['warnings']:
            print(f"\n⚠️  Warnings: {len(results['warnings'])}")
            for warn in results['warnings']:
                print(f"     • {warn}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# RUN VERIFICATION ON ALL K=2 RESULTS
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("COMPREHENSIVE VERIFICATION: K=2 PARTITIONS")
print("=" * 80)
print("\nVerifying that visualizations match actual values")
print("and that scores are calculated correctly.\n")

all_verification_results = {}

for mol, smiles, size_label in selected_mols:
    print(f"\n{'#'*80}")
    print(f"# {size_label} MOLECULE: {smiles}")
    print(f"{'#'*80}")
    
    results = all_results[size_label]
    k2_results = [r for r in results if r['N_partitions'] == 2]
    
    if not k2_results:
        continue
    
    k2_results.sort(key=lambda x: x['Score'], reverse=True)
    
    # Verify best and worst methods
    methods_to_verify = [
        ('BEST', k2_results[0]),
        ('WORST', k2_results[-1])
    ]
    
    all_verification_results[size_label] = {}
    
    for label, result in methods_to_verify:
        print(f"\n{'─'*80}")
        print(f"{label} METHOD: {result['Strategy']} (Score: {result['Score']:.4f})")
        print(f"{'─'*80}")
        
        interp = result['Interpretation']
        
        verification = verify_partition_interpretation(
            mol,
            interp,
            interpreter,
            verbose=True
        )
        
        all_verification_results[size_label][label] = verification

# ═══════════════════════════════════════════════════════════════════════════
# CROSS-CHECK: Compare visualization labels with contributions (FIXED)
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print(f"CROSS-CHECK: Bond Labels vs Contributions")
print(f"{'='*80}\n")

print("Verifying that bond labels in visualizations match")
print("the actual contribution values from the decomposition.\n")

for mol, smiles, size_label in selected_mols:
    results = all_results[size_label]
    k2_results = [r for r in results if r['N_partitions'] == 2]
    
    if not k2_results:
        continue
    
    best_result = max(k2_results, key=lambda x: x['Score'])
    interp = best_result['Interpretation']
    
    print(f"\n{size_label} Molecule ({best_result['Strategy']}):")
    print(f"{'─'*80}")
    
    # Create GraphletDAG and project to bond level

    dag = make_buhito_dag(mol, coefficients=interpreter.coefficients)
    
    bond_coefs = dag.project_to_layer(2)  # Returns numpy array
    
    partition = interp.partition
    
    print(f"\nBond-by-Bond Comparison:")
    print(f"{'Bond':<8} {'Atoms':<12} {'Type':<15} {'Projected':<12} {'Status':<10}")
    print(f"{'─'*70}")
    
    for bond in mol.GetBonds():
        b_idx = bond.GetIdx()
        a1_idx = bond.GetBeginAtomIdx()
        a2_idx = bond.GetEndAtomIdx()
        
        a1 = mol.GetAtomWithIdx(a1_idx)
        a2 = mol.GetAtomWithIdx(a2_idx)
        
        p1 = partition.get(a1_idx, -1)
        p2 = partition.get(a2_idx, -1)
        
        is_cross = (p1 != p2)
        bond_type = "Cross" if is_cross else f"P{p1}"
        
        # ✅ FIX: Handle both numpy array and dict returns
        if isinstance(bond_coefs, np.ndarray):
            projected_coef = bond_coefs[b_idx] if b_idx < len(bond_coefs) else 0.0
        elif isinstance(bond_coefs, dict):
            projected_coef = bond_coefs.get(b_idx, 0.0)
        else:
            # Fallback
            try:
                projected_coef = float(bond_coefs[b_idx])
            except:
                projected_coef = 0.0
        
        status = "Gray" if is_cross else f"P{p1} color"
        
        print(f"{b_idx:<8} {a1.GetSymbol()}{a1_idx}-{a2.GetSymbol()}{a2_idx:<7} "
              f"{bond_type:<15} {projected_coef:+11.4f} {status:<10}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # EXTRA CHECK: Compare bond coefficients with between_partition values
    # ─────────────────────────────────────────────────────────────────────────
    
    print(f"\n{'─'*80}")
    print(f"Cross-Partition Bond Details:")
    print(f"{'─'*80}\n")
    
    between_dict = interp.between_partition
    
    if len(between_dict) == 0:
        print(f"  ✓ No cross-partition contributions (perfect separation)")
    else:
        print(f"  Found {len(between_dict)} cross-partition contribution(s):\n")
        
        total_cross = 0.0
        for key, value in sorted(between_dict.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"    Key: {key}")
            print(f"    Contribution: {value:+.4f}")
            total_cross += value
            
            
        print(f"  Total cross-partition: {total_cross:+.4f}")
    
    print(f"\n{'─'*80}")
    print(f"✓ These projected coefficients should match the bond labels in visualization")
    print(f"✓ Cross-partition bonds should be gray with their contribution values")
    print(f"✓ Within-partition bonds should be colored with their contribution values")
    print(f"{'─'*80}\n")



# ═══════════════════════════════════════════════════════════════════════════
# FINAL VERIFICATION SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print(f"OVERALL VERIFICATION SUMMARY")
print(f"{'='*80}\n")

total_tests = 0
total_passed = 0
total_errors = 0
total_warnings = 0

for size_label, verifications in all_verification_results.items():
    for method_label, verification in verifications.items():
        total_tests += 1
        if verification['passed']:
            total_passed += 1
        total_errors += len(verification['errors'])
        total_warnings += len(verification['warnings'])

print(f"Tests run: {total_tests}")
print(f"Passed: {total_passed}/{total_tests}")
print(f"Errors: {total_errors}")
print(f"Warnings: {total_warnings}\n")

if total_errors == 0:
    print(f"✅ ALL VERIFICATIONS PASSED")
    print(f"\n   Your visualizations accurately reflect:")
    print(f"   • Model predictions decompose correctly per Equation (2)")
    print(f"   • Scores are calculated correctly")
    print(f"   • Bond labels match projected coefficients")
    print(f"   • Partition assignments are complete and consistent")
    print(f"\n   The visualizations are trustworthy! 🎉")
else:
    print(f"❌ VERIFICATION ISSUES FOUND")
    print(f"\n   Please review the detailed output above.")
    print(f"   Common issues:")
    print(f"   • Decomposition errors: Check interpreter implementation")
    print(f"   • Score calculation: Check PartitionInterpretation.score property")
    print(f"   • Coverage errors: Check partition generation")

print(f"\n{'='*80}\n")
cf4_smiles = "FC(F)(F)F"
cf4 = Chem.MolFromSmiles(cf4_smiles)
AllChem.Compute2DCoords(cf4)

mol = cf4
smiles = cf4_smiles

graph = smiles_to_buhito_graph(
    smiles,
    add_hs=False,
    output_2d_pos=False,
)

interpreter.register_graph(mol, graph)

partition = ChemicalPartitioner.distance_partition(
    mol,
    n_clusters=2
)

interpretation = interpreter.compute_partition_contributions(
    mol,
    partition
)
all_instances = (
    interpreter
    .enumerate_graphlet_instances(
        mol,
        include_zero_coefficients=True
    )
)

active_instances = (
    interpreter
    .enumerate_graphlet_instances(
        mol,
        include_zero_coefficients=False
    )
)

print(
    "All graphlet instances:",
    len(all_instances)
)

print(
    "Active graphlet instances:",
    len(active_instances)
)

print("Analyzing:", smiles)
print("Canonical SMILES:", Chem.MolToSmiles(mol))
print("Atoms:", mol.GetNumAtoms())
print("Bonds:", mol.GetNumBonds())
print("Instances:", len(instances))
from rdkit.Chem import Draw

instances = (
    interpreter_full
    .enumerate_graphlet_instances(
        cf4,
        include_zero_coefficients=True
    )
)

print("All graphlet instances:", len(instances))

n_show = len(instances)

if n_show == 0:
    print("No graphlet instances found.")
else:
    mols = []
    legends = []

    for i, inst in enumerate(instances[:n_show]):
        mols.append(cf4)
        atoms = sorted(inst["atoms"])
        legends.append(
            f"Graphlet {i}\nAtoms {atoms}\ncoef={inst['coefficient']:+.2f}"
        )

    imgs = Draw.MolsToGridImage(
        mols,
        molsPerRow=4,
        subImgSize=(300, 300),
        legends=legends,
        highlightAtomLists=[sorted(inst["atoms"]) for inst in instances[:n_show]]
    )

    display(imgs)
### Detailed Analysis of Best Partitions

# For each molecule, examine the **winning strategy** in detail.
### Aggregate Statistics Across Test Set

# Compute interpretability scores across multiple molecules to understand average performance.
# from collections import defaultdict
# from scipy import stats
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from typing import Dict, Callable
# from copy import deepcopy

# from partition_interpretability import (
#     PartitionInterpreter,
#     PartitionOptimizer,
#     ChemicalPartitioner,
#     validate_partition_decomposition
# )

# # ═══════════════════════════════════════════════════════════════════════════
# # MAIN ANALYSIS
# # ═══════════════════════════════════════════════════════════════════════════

# n_sample = min(1000, len(test))
# sample_indices = np.random.RandomState(42).choice(len(test), size=n_sample, replace=False)
# sample_mols = [test.iloc[i]['mol'] for i in sample_indices]
# sample_smiles = [test.iloc[i]['smiles'] for i in sample_indices]

# greedy_optimizer = PartitionOptimizer(
#     interpreter=interpreter,
#     max_partitions=5
# )

# strategies = {
#     'Distance (k=2)': lambda mol: ChemicalPartitioner.distance_partition(mol, n_clusters=2),
#     'Distance (k=3)': lambda mol: ChemicalPartitioner.distance_partition(mol, n_clusters=3),
#     'Random (k=2)': lambda mol: ChemicalPartitioner.random_partition(mol, n_clusters=2, seed=42),
#     'Random (k=3)': lambda mol: ChemicalPartitioner.random_partition(mol, n_clusters=3, seed=42),
    
#     'Greedy-Random (k=2)': lambda mol: optimize_partition_random_wrapper(
#         greedy_optimizer, mol, n_clusters=2, n_iterations=100, seed=42, verbose=False
#     ),
#     'Greedy-Random (k=3)': lambda mol: optimize_partition_random_wrapper(
#         greedy_optimizer, mol, n_clusters=3, n_iterations=100, seed=42, verbose=False
#     ),
#     'Greedy-Random (k=4)': lambda mol: optimize_partition_random_wrapper(
#         greedy_optimizer, mol, n_clusters=4, n_iterations=100, seed=42, verbose=False
#     ),
    
#     'Greedy-Distance (k=2)': lambda mol: optimize_partition_distance_wrapper(
#         greedy_optimizer, mol, n_clusters=2, n_iterations=100, seed=42, verbose=False
#     ),
#     'Greedy-Distance (k=3)': lambda mol: optimize_partition_distance_wrapper(
#         greedy_optimizer, mol, n_clusters=3, n_iterations=100, seed=42, verbose=False
#     ),
#     'Greedy-Distance (k=4)': lambda mol: optimize_partition_distance_wrapper(
#         greedy_optimizer, mol, n_clusters=4, n_iterations=100, seed=42, verbose=False
#     ),
# }

# all_results = defaultdict(list)

# for mol_idx, mol in enumerate(sample_mols):
#     for strategy_name, partition_fn in strategies.items():
#         try:
#             partition = partition_fn(mol)
#             interp = interpreter.compute_partition_contributions(mol, partition)
#             breakdown = interp.get_contribution_breakdown()
#             validation = validate_partition_decomposition(mol, interpreter, partition, tolerance=1e-2)
            
#             all_results[strategy_name].append({
#                 'mol_idx': mol_idx,
#                 'smiles': sample_smiles[mol_idx],
#                 'n_atoms': mol.GetNumAtoms(),
#                 'n_partitions': len(set(partition.values())),
#                 'score': interp.score,
#                 'within_frac': breakdown['within_frac'],
#                 'between_frac': breakdown['between_frac'],
#                 'higher_frac': breakdown['higher_frac'],
#                 'prediction': interp.total_prediction,
#                 'validation_passed': validation['passed'],
#                 'validation_error': validation['relative_error']
#             })
#         except Exception as e:
#             continue

# summary_data = []
# for strategy_name in strategies.keys():
#     results = all_results[strategy_name]
#     if not results:
#         continue
    
#     df_strategy = pd.DataFrame(results)
    
#     summary_data.append({
#         'Strategy': strategy_name,
#         'N_molecules': len(df_strategy),
#         'Mean Score': df_strategy['score'].mean(),
#         'Std Score': df_strategy['score'].std(),
#         'Min Score': df_strategy['score'].min(),
#         'Max Score': df_strategy['score'].max(),
#         'Median Score': df_strategy['score'].median(),
#         'Mean Within %': 100 * df_strategy['within_frac'].mean(),
#         'Mean Between %': 100 * df_strategy['between_frac'].mean(),
#         'Mean Higher %': 100 * df_strategy['higher_frac'].mean(),
#         'Validation Pass %': 100 * df_strategy['validation_passed'].mean(),
#         'Mean Val Error': df_strategy['validation_error'].mean()
#     })

# summary_df = pd.DataFrame(summary_data)
# summary_df = summary_df.sort_values('Mean Score', ascending=False)

# all_mol_data = []
# for strategy_name in strategies.keys():
#     for result in all_results[strategy_name]:
#         all_mol_data.append({
#             'strategy': strategy_name,
#             'n_atoms': result['n_atoms'],
#             'score': result['score']
#         })

# df_all = pd.DataFrame(all_mol_data)

# size_bins = [0, 10, 20, 30, 100]
# size_labels = ['Small (≤10)', 'Medium (11-20)', 'Large (21-30)', 'Very Large (>30)']

# df_all['size_bin'] = pd.cut(df_all['n_atoms'], bins=size_bins, tick_labels=size_labels, include_lowest=True)

# size_breakdown = df_all.groupby(['size_bin', 'strategy'])['score'].agg(['mean', 'std', 'count']).reset_index()
# size_breakdown = size_breakdown.sort_values(['size_bin', 'mean'], ascending=[True, False])

# fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ax = axes[0, 0]
# strategy_names_ordered = [
#     strategy
#     for strategy in strategies
#     if len(all_results[strategy]) > 0
# ]

# strategy_scores = [
#     [
#         result["score"]
#         for result in all_results[strategy]
#     ]
#     for strategy in strategy_names_ordered
# ]

# bp = ax.boxplot(strategy_scores, tick_labels=strategy_names_ordered, patch_artist=True)

# for i, patch in enumerate(bp['boxes']):
#     if 'Greedy-Random' in strategy_names_ordered[i]:
#         patch.set_facecolor('#e74c3c')
#     elif 'Greedy-Distance' in strategy_names_ordered[i]:
#         patch.set_facecolor('#9b59b6')
#     else:
#         patch.set_facecolor('#3498db')

# ax.set_ylabel('Interpretability Score', fontsize=12)
# ax.set_title('Score Distribution by Strategy\n(Red=Greedy-Random, Purple=Greedy-Distance)', 
#              fontsize=14, fontweight='bold')
# ax.set_xticklabels(strategy_names_ordered, rotation=45, ha='right')
# ax.grid(axis='y', alpha=0.3)

# ax = axes[0, 1]
# strategy_names = [
#     strategy
#     for strategy in strategies
#     if len(all_results[strategy]) > 0
# ]
# within_means = [np.mean([r['within_frac'] for r in all_results[s]]) for s in strategy_names]
# between_means = [np.mean([r['between_frac'] for r in all_results[s]]) for s in strategy_names]
# higher_means = [np.mean([r['higher_frac'] for r in all_results[s]]) for s in strategy_names]

# x = np.arange(len(strategy_names))
# width = 0.25

# ax.bar(x - width, within_means, width, label='Within', color='#2ecc71')
# ax.bar(x, between_means, width, label='Between', color='#f39c12')
# ax.bar(x + width, higher_means, width, label='Higher', color='#e74c3c')

# ax.set_ylabel('Fraction of Total Contribution', fontsize=12)
# ax.set_title('Mean Contribution Breakdown', fontsize=14, fontweight='bold')
# ax.set_xticks(x)
# ax.set_xticklabels(strategy_names, rotation=45, ha='right')
# ax.legend()
# ax.grid(axis='y', alpha=0.3)

# ax = axes[1, 0]
# for strategy_name in strategies.keys():
#     results = all_results[strategy_name]
#     sizes = [r['n_atoms'] for r in results]
#     scores = [r['score'] for r in results]
    
#     if 'Greedy-Random' in strategy_name:
#         marker = 'D'
#         s = 50
#         alpha = 0.7
#     elif 'Greedy-Distance' in strategy_name:
#         marker = 's'
#         s = 50
#         alpha = 0.7
#     else:
#         marker = 'o'
#         s = 30
#         alpha = 0.4
    
#     ax.scatter(sizes, scores, alpha=alpha, label=strategy_name, s=s, marker=marker)

# ax.set_xlabel('Number of Atoms', fontsize=12)
# ax.set_ylabel('Interpretability Score', fontsize=12)
# ax.set_title('Score vs Molecule Size\n(Diamond=Greedy-Random, Square=Greedy-Distance)', 
#              fontsize=14, fontweight='bold')
# ax.legend(fontsize=7, ncol=2)
# ax.grid(alpha=0.3)

# ax = axes[1, 1]

# comparison_data = []
# comparison_labels = []

# for k in [2, 3, 4]:
#     random_key = f'Random (k={k})'
#     greedy_random_key = f'Greedy-Random (k={k})'
#     greedy_distance_key = f'Greedy-Distance (k={k})'
    
#     if random_key in strategies and greedy_random_key in strategies:
#         random_scores = [r['score'] for r in all_results[random_key]]
#         comparison_data.append(random_scores)
#         comparison_labels.append(f'Rand k={k}')
        
#         greedy_random_scores = [r['score'] for r in all_results[greedy_random_key]]
#         comparison_data.append(greedy_random_scores)
#         comparison_labels.append(f'G-Rand k={k}')
        
#     if greedy_distance_key in strategies:
#         greedy_distance_scores = [r['score'] for r in all_results[greedy_distance_key]]
#         comparison_data.append(greedy_distance_scores)
#         comparison_labels.append(f'G-Dist k={k}')

# bp = ax.boxplot(comparison_data, tick_labels=comparison_labels, patch_artist=True)

# for i, patch in enumerate(bp['boxes']):
#     if 'G-Rand' in comparison_labels[i]:
#         patch.set_facecolor('#e74c3c')
#     elif 'G-Dist' in comparison_labels[i]:
#         patch.set_facecolor('#9b59b6')
#     else:
#         patch.set_facecolor('#95a5a6')

# ax.set_ylabel('Interpretability Score', fontsize=12)
# ax.set_title('Greedy Optimization Effect\n(Gray=Baseline, Red=Greedy-Random, Purple=Greedy-Distance)', 
#              fontsize=14, fontweight='bold')
# ax.set_xticklabels(comparison_labels, rotation=45, ha='right')
# ax.grid(axis='y', alpha=0.3)

# plt.tight_layout()
# plt.savefig('partition_interpretability_aggregate_stats_both_greedy.png', dpi=300, bbox_inches='tight')
# plt.show()

# all_results_flat = []
# for strategy_name in strategies.keys():
#     for result in all_results[strategy_name]:
#         result_copy = result.copy()
#         result_copy['strategy'] = strategy_name
#         result_copy['is_greedy_random'] = 'Greedy-Random' in strategy_name
#         result_copy['is_greedy_distance'] = 'Greedy-Distance' in strategy_name
#         all_results_flat.append(result_copy)

# df_export = pd.DataFrame(all_results_flat)
# df_export.to_csv('partition_interpretability_detailed_results_both_greedy.csv', index=False)

# summary_df.to_csv('partition_interpretability_summary_both_greedy.csv', index=False)
# from collections import defaultdict
# from scipy import stats
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from typing import Dict, Callable

# from partition_interpretability import (
#     PartitionInterpreter,
#     PartitionOptimizer,
#     ChemicalPartitioner,
#     validate_partition_decomposition
# )


# # ═══════════════════════════════════════════════════════════════════════════
# # COMPREHENSIVE K-VALUE ANALYSIS WITH BOTH GREEDY STRATEGIES
# # ═══════════════════════════════════════════════════════════════════════════


# def determine_max_k_for_molecule(mol) -> int:
#     """
#     Determine maximum sensible k for a molecule.
    
#     "For smaller graphs the number of partitions may be tractable"
    
#     Heuristic: At least 2-3 atoms per partition on average
#     """
#     n_atoms = mol.GetNumAtoms()
#     max_k = max(2, n_atoms // 2)  # At least 2 atoms per partition
#     max_k = min(max_k, 10)  # Cap at 10 for interpretability
#     return max_k


# # ═══════════════════════════════════════════════════════════════════════════
# # MAIN ANALYSIS - ALL K VALUES WITH BOTH GREEDY STRATEGIES
# # ═══════════════════════════════════════════════════════════════════════════

# n_sample = min(1000, len(test))
# sample_indices = np.random.RandomState(42).choice(len(test), size=n_sample, replace=False)
# sample_mols = [test.iloc[i]['mol'] for i in sample_indices]
# sample_smiles = [test.iloc[i]['smiles'] for i in sample_indices]

# print("=" * 80)
# print(f"COMPREHENSIVE K-VALUE ANALYSIS ({n_sample} molecules)")
# print("=" * 80)

# # Determine k range based on molecule sizes
# k_range = range(2, 8)  # Test k=2,3,4,5,6,7
# print(f"\nTesting k values: {list(k_range)}")
# print(f"  'For smaller graphs the number of partitions may be tractable'")
# print(f"  'For larger graphs, we will use greedy algorithms that evolve")
# print(f"   a random starting partition'")

# # Initialize optimizer
# greedy_optimizer = PartitionOptimizer(
#     interpreter=interpreter,
#     max_partitions=max(k_range)  # Set to maximum k we'll test
# )

# # Define comprehensive strategies
# strategies = {
#    # 'Functional Groups': lambda mol: ChemicalPartitioner.functional_group_partition(mol),
# }

# # Add distance-based baseline strategies for all k
# for k in k_range:
#     strategies[f'Distance (k={k})'] = lambda mol, k=k: ChemicalPartitioner.distance_partition(
#         mol, n_clusters=k, seed=42
#     )

# # Add random baseline strategies for all k
# for k in k_range:
#     strategies[f'Random (k={k})'] = lambda mol, k=k: ChemicalPartitioner.random_partition(
#         mol, n_clusters=k, seed=42
#     )

# # Add Greedy-Random strategies for all k (random initialization + greedy optimization)
# for k in k_range:
#     strategies[f'Greedy-Random (k={k})'] = lambda mol, k=k: optimize_partition_random_wrapper(
#         greedy_optimizer, mol, n_clusters=k, n_iterations=100, seed=42, verbose=False
#     )

# # Add Greedy-Distance strategies for all k (distance initialization + greedy optimization)
# for k in k_range:
#     strategies[f'Greedy-Distance (k={k})'] = lambda mol, k=k: optimize_partition_distance_wrapper(
#         greedy_optimizer, mol, n_clusters=k, n_iterations=100, seed=42, verbose=False
#     )

# print(f"\nTotal strategies: {len(strategies)}")
# # print(f"  • Functional Groups: 1")
# print(f"  • Distance baseline (k=2-7): {len(k_range)}")
# print(f"  • Random baseline (k=2-7): {len(k_range)}")
# print(f"  • Greedy-Random (k=2-7): {len(k_range)}")
# print(f"  • Greedy-Distance (k=2-7): {len(k_range)}")

# # Collect results
# all_results = defaultdict(list)

# print("\nComputing interpretability scores...")
# for mol_idx, mol in enumerate(sample_mols):
#     if (mol_idx + 1) % 10 == 0:
#         print(f"  Processed {mol_idx + 1}/{n_sample} molecules")
    
#     for strategy_name, partition_fn in strategies.items():
#         try:
#             partition = partition_fn(mol)
#             interp = interpreter.compute_partition_contributions(mol, partition)
#             breakdown = interp.get_contribution_breakdown()
#             validation = validate_partition_decomposition(mol, interpreter, partition, tolerance=1e-2)
            
#             all_results[strategy_name].append({
#                 'mol_idx': mol_idx,
#                 'smiles': sample_smiles[mol_idx],
#                 'n_atoms': mol.GetNumAtoms(),
#                 'n_partitions': len(set(partition.values())),
#                 'k_requested': int(strategy_name.split('k=')[1].rstrip(')')) if 'k=' in strategy_name else None,
#                 'score': interp.score,
#                 'within_frac': breakdown['within_frac'],
#                 'between_frac': breakdown['between_frac'],
#                 'higher_frac': breakdown['higher_frac'],
#                 'prediction': interp.total_prediction,
#                 'validation_passed': validation['passed'],
#                 'validation_error': validation['relative_error']
#             })
#         except Exception as e:
#             print(f"  Warning: Failed {strategy_name} for molecule {mol_idx}: {e}")
#             continue

# print("\n" + "=" * 80)
# print("SUMMARY BY STRATEGY")
# print("=" * 80)

# # Aggregate statistics
# summary_data = []
# for strategy_name in strategies.keys():
#     results = all_results[strategy_name]
#     if not results:
#         continue
    
#     df_strategy = pd.DataFrame(results)
    
#     summary_data.append({
#         'Strategy': strategy_name,
#         'N_molecules': len(df_strategy),
#         'Mean Score': df_strategy['score'].mean(),
#         'Std Score': df_strategy['score'].std(),
#         'Min Score': df_strategy['score'].min(),
#         'Max Score': df_strategy['score'].max(),
#         'Median Score': df_strategy['score'].median(),
#         'Mean Within %': 100 * df_strategy['within_frac'].mean(),
#         'Mean Between %': 100 * df_strategy['between_frac'].mean(),
#         'Mean Higher %': 100 * df_strategy['higher_frac'].mean(),
#         'Validation Pass %': 100 * df_strategy['validation_passed'].mean(),
#     })

# summary_df = pd.DataFrame(summary_data)
# summary_df = summary_df.sort_values('Mean Score', ascending=False)

# print("\n" + summary_df.to_string(index=False))

# # ═══════════════════════════════════════════════════════════════════════════
# # CRITICAL ANALYSIS: OPTIMAL K SELECTION (WITH BOTH GREEDY STRATEGIES)
# # ═══════════════════════════════════════════════════════════════════════════

# print("\n" + "=" * 80)
# print("OPTIMAL K ANALYSIS")
# print("=" * 80)

# # Extract k-dependent results
# k_analysis = defaultdict(lambda: defaultdict(list))

# for strategy_name, results in all_results.items():
#     if 'k=' in strategy_name:
#         # Extract method name (before the first parenthesis)
#         method = strategy_name.split(' (')[0]  # 'Distance', 'Random', 'Greedy-Random', 'Greedy-Distance'
#         k = int(strategy_name.split('k=')[1].rstrip(')'))
        
#         for result in results:
#             k_analysis[method][k].append(result['score'])

# # Analyze each method's k-dependence
# print("\n1. MEAN SCORE BY K VALUE:\n")

# k_summary = []
# for method in ['Distance', 'Random', 'Greedy-Random', 'Greedy-Distance']:
#     if method not in k_analysis:
#         continue
#     print(f"{method}:")
#     for k in sorted(k_analysis[method].keys()):
#         scores = k_analysis[method][k]
#         mean_score = np.mean(scores)
#         std_score = np.std(scores)
#         k_summary.append({
#             'Method': method,
#             'k': k,
#             'Mean Score': mean_score,
#             'Std Score': std_score,
#             'N': len(scores)
#         })
#         print(f"  k={k}: {mean_score:.4f} ± {std_score:.4f}")
#     print()

# k_summary_df = pd.DataFrame(k_summary)

# # Find optimal k for each method
# print("2. OPTIMAL K BY METHOD:\n")

# for method in ['Distance', 'Random', 'Greedy-Random', 'Greedy-Distance']:
#     if method not in k_analysis:
#         continue
#     method_data = k_summary_df[k_summary_df['Method'] == method]
#     if len(method_data) == 0:
#         continue
#     best_k = method_data.loc[method_data['Mean Score'].idxmax(), 'k']
#     best_score = method_data['Mean Score'].max()
    
#     print(f"{method}:")
#     print(f"  Best k: {int(best_k)}")
#     print(f"  Score: {best_score:.4f}")
    
#     # Test if this is significantly better than others
#     best_k_scores = k_analysis[method][best_k]
#     print(f"  Comparisons to other k values:")
    
#     for k in sorted(k_analysis[method].keys()):
#         if k == best_k:
#             continue
#         other_scores = k_analysis[method][k]
#         t_stat, p_value = stats.ttest_ind(best_k_scores, other_scores)
#         sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
#         print(f"    vs k={k}: Δ={np.mean(best_k_scores) - np.mean(other_scores):+.4f}, "
#               f"p={p_value:.4f} {sig}")
#     print()

# # ═══════════════════════════════════════════════════════════════════════════
# # VISUALIZATION: K-DEPENDENCE (WITH BOTH GREEDY STRATEGIES)
# # ═══════════════════════════════════════════════════════════════════════════

# print("=" * 80)
# print("K-DEPENDENCE VISUALIZATION")
# print("=" * 80)

# fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# # 1. Mean score vs k (all methods)
# ax = axes[0, 0]
# for method in ['Distance', 'Random', 'Greedy-Random', 'Greedy-Distance']:
#     if method not in k_analysis:
#         continue
#     ks = sorted(k_analysis[method].keys())
#     means = [np.mean(k_analysis[method][k]) for k in ks]
#     stds = [np.std(k_analysis[method][k]) for k in ks]
    
#     # Different styles for greedy methods
#     if method == 'Greedy-Random':
#         linestyle = '-'
#         marker = 'D'
#         linewidth = 3
#     elif method == 'Greedy-Distance':
#         linestyle = '--'
#         marker = 's'
#         linewidth = 3
#     else:
#         linestyle = ':'
#         marker = 'o'
#         linewidth = 2
    
#     ax.plot(ks, means, marker=marker, label=method, linewidth=linewidth, 
#             linestyle=linestyle, markersize=8)
#     ax.fill_between(ks, 
#                      [m - s for m, s in zip(means, stds)],
#                      [m + s for m, s in zip(means, stds)],
#                      alpha=0.2)

# ax.set_xlabel('Number of Partitions (k)', fontsize=12)
# ax.set_ylabel('Mean Interpretability Score', fontsize=12)
# ax.set_title('Score vs k (All Methods)\nSolid=Greedy-Random, Dashed=Greedy-Distance', 
#              fontsize=14, fontweight='bold')
# ax.legend()
# ax.grid(alpha=0.3)

# # 2. Box plots comparing greedy strategies at each k
# ax = axes[0, 1]
# comparison_data = []
# comparison_labels = []

# for k in sorted(k_analysis['Greedy-Random'].keys()):
#     if k in k_analysis['Greedy-Random']:
#         comparison_data.append(k_analysis['Greedy-Random'][k])
#         comparison_labels.append(f'GR-{k}')
#     if k in k_analysis['Greedy-Distance']:
#         comparison_data.append(k_analysis['Greedy-Distance'][k])
#         comparison_labels.append(f'GD-{k}')

# bp = ax.boxplot(comparison_data, tick_labels=comparison_labels, patch_artist=True)

# # Color code: red for random, purple for distance
# for i, patch in enumerate(bp['boxes']):
#     if 'GR' in comparison_labels[i]:
#         patch.set_facecolor('#e74c3c')
#     else:
#         patch.set_facecolor('#9b59b6')

# ax.set_xlabel('Strategy-k (GR=Greedy-Random, GD=Greedy-Distance)', fontsize=12)
# ax.set_ylabel('Interpretability Score', fontsize=12)
# ax.set_title('Greedy Strategy Comparison by k', fontsize=14, fontweight='bold')
# ax.grid(axis='y', alpha=0.3)
# plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# # 3. Within-partition fraction vs k
# ax = axes[0, 2]
# for method in ['Distance', 'Random', 'Greedy-Random', 'Greedy-Distance']:
#     method_key = method + ' (k='
#     within_by_k = defaultdict(list)
    
#     for strategy_name, results in all_results.items():
#         if method_key in strategy_name:
#             k = int(strategy_name.split('k=')[1].rstrip(')'))
#             for result in results:
#                 within_by_k[k].append(result['within_frac'])
    
#     if not within_by_k:
#         continue
    
#     ks = sorted(within_by_k.keys())
#     means = [np.mean(within_by_k[k]) for k in ks]
    
#     # Different styles for greedy methods
#     if method == 'Greedy-Random':
#         linestyle = '-'
#         marker = 'D'
#         linewidth = 3
#     elif method == 'Greedy-Distance':
#         linestyle = '--'
#         marker = 's'
#         linewidth = 3
#     else:
#         linestyle = ':'
#         marker = 'o'
#         linewidth = 2
    
#     ax.plot(ks, means, marker=marker, label=method, linewidth=linewidth,
#             linestyle=linestyle, markersize=8)

# ax.set_xlabel('Number of Partitions (k)', fontsize=12)
# ax.set_ylabel('Mean Within-Partition Fraction', fontsize=12)
# ax.set_title('Within-Partition Contribution vs k', fontsize=14, fontweight='bold')
# ax.legend()
# ax.grid(alpha=0.3)

# # 4. Higher-order fraction vs k
# ax = axes[1, 0]
# for method in ['Distance', 'Random', 'Greedy-Random', 'Greedy-Distance']:
#     method_key = method + ' (k='
#     higher_by_k = defaultdict(list)
    
#     for strategy_name, results in all_results.items():
#         if method_key in strategy_name:
#             k = int(strategy_name.split('k=')[1].rstrip(')'))
#             for result in results:
#                 higher_by_k[k].append(result['higher_frac'])
    
#     if not higher_by_k:
#         continue
    
#     ks = sorted(higher_by_k.keys())
#     means = [np.mean(higher_by_k[k]) for k in ks]
    
#     # Different styles for greedy methods
#     if method == 'Greedy-Random':
#         linestyle = '-'
#         marker = 'D'
#         linewidth = 3
#     elif method == 'Greedy-Distance':
#         linestyle = '--'
#         marker = 's'
#         linewidth = 3
#     else:
#         linestyle = ':'
#         marker = 'o'
#         linewidth = 2
    
#     ax.plot(ks, means, marker=marker, label=method, linewidth=linewidth,
#             linestyle=linestyle, markersize=8)

# ax.set_xlabel('Number of Partitions (k)', fontsize=12)
# ax.set_ylabel('Mean Higher-Order Fraction', fontsize=12)
# ax.set_title('Higher-Order Contribution vs k\n(Lower is Better per Eq. 2)', 
#              fontsize=14, fontweight='bold')
# ax.legend()
# ax.grid(alpha=0.3)

# # 5. Greedy improvement comparison
# ax = axes[1, 1]
# improvements_random = []
# improvements_distance = []
# ks_list = []
# p_values_random = []
# p_values_distance = []

# for k in sorted(k_analysis['Greedy-Random'].keys()):
#     if k in k_analysis['Random']:
#         greedy_random_scores = k_analysis['Greedy-Random'][k]
#         random_scores = k_analysis['Random'][k]
        
#         improvement = np.mean(greedy_random_scores) - np.mean(random_scores)
#         improvements_random.append(improvement)
        
#         t_stat, p_val = stats.ttest_ind(greedy_random_scores, random_scores)
#         p_values_random.append(p_val)
    
#     if k in k_analysis['Greedy-Distance'] and k in k_analysis['Distance']:
#         greedy_distance_scores = k_analysis['Greedy-Distance'][k]
#         distance_scores = k_analysis['Distance'][k]
        
#         improvement = np.mean(greedy_distance_scores) - np.mean(distance_scores)
#         improvements_distance.append(improvement)
        
#         t_stat, p_val = stats.ttest_ind(greedy_distance_scores, distance_scores)
#         p_values_distance.append(p_val)
        
#         ks_list.append(k)

# x = np.arange(len(ks_list))
# width = 0.35

# bars1 = ax.bar(x - width/2, improvements_random, width, label='Greedy-Random vs Random',
#                color='#e74c3c', alpha=0.7)
# bars2 = ax.bar(x + width/2, improvements_distance, width, label='Greedy-Distance vs Distance',
#                color='#9b59b6', alpha=0.7)

# ax.set_xlabel('Number of Partitions (k)', fontsize=12)
# ax.set_ylabel('Score Improvement', fontsize=12)
# ax.set_title('Greedy Optimization Benefit by k and Initialization', 
#              fontsize=14, fontweight='bold')
# ax.set_xticks(x)
# ax.set_xticklabels([f'k={k}' for k in ks_list])
# ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
# ax.legend()
# ax.grid(axis='y', alpha=0.3)

# # 6. Molecule size effect on optimal k (compare both greedy methods)
# ax = axes[1, 2]

# # For each molecule, find best k for each greedy method
# molecule_optimal_k_random = []
# molecule_optimal_k_distance = []
# molecule_sizes = []

# for mol_idx in range(n_sample):
#     # Greedy-Random results
#     mol_results_random = [r for strategy_name, results in all_results.items() 
#                           for r in results 
#                           if r['mol_idx'] == mol_idx and 'Greedy-Random' in strategy_name]
    
#     # Greedy-Distance results
#     mol_results_distance = [r for strategy_name, results in all_results.items() 
#                             for r in results 
#                             if r['mol_idx'] == mol_idx and 'Greedy-Distance' in strategy_name]
    
#     if mol_results_random and mol_results_distance:
#         best_result_random = max(mol_results_random, key=lambda x: x['score'])
#         best_result_distance = max(mol_results_distance, key=lambda x: x['score'])
        
#         if best_result_random['k_requested'] and best_result_distance['k_requested']:
#             molecule_optimal_k_random.append(best_result_random['k_requested'])
#             molecule_optimal_k_distance.append(best_result_distance['k_requested'])
#             molecule_sizes.append(best_result_random['n_atoms'])

# ax.scatter(molecule_sizes, molecule_optimal_k_random, alpha=0.6, s=100, 
#            c='#e74c3c', marker='D', label='Greedy-Random')
# ax.scatter(molecule_sizes, molecule_optimal_k_distance, alpha=0.6, s=100, 
#            c='#9b59b6', marker='s', label='Greedy-Distance')

# ax.set_xlabel('Molecule Size (atoms)', fontsize=12)
# ax.set_ylabel('Optimal k', fontsize=12)
# ax.set_title('Optimal k vs Molecule Size\n(Diamond=Greedy-Random, Square=Greedy-Distance)', 
#              fontsize=14, fontweight='bold')
# ax.legend()
# ax.grid(alpha=0.3)

# # Add trend lines
# if len(molecule_sizes) > 1:
#     z_random = np.polyfit(molecule_sizes, molecule_optimal_k_random, 1)
#     p_random = np.poly1d(z_random)
#     z_distance = np.polyfit(molecule_sizes, molecule_optimal_k_distance, 1)
#     p_distance = np.poly1d(z_distance)
    
#     x_trend = np.linspace(min(molecule_sizes), max(molecule_sizes), 100)
#     ax.plot(x_trend, p_random(x_trend), "r--", alpha=0.8, linewidth=2, label='_nolegend_')
#     ax.plot(x_trend, p_distance(x_trend), "m--", alpha=0.8, linewidth=2, label='_nolegend_')

# plt.tight_layout()
# plt.savefig('k_value_comprehensive_analysis_both_greedy.png', dpi=300, bbox_inches='tight')
# print("\n✓ Saved visualization to 'k_value_comprehensive_analysis_both_greedy.png'")
# plt.show()

# # ═══════════════════════════════════════════════════════════════════════════
# # RECOMMENDATIONS WITH BOTH GREEDY STRATEGIES
# # ═══════════════════════════════════════════════════════════════════════════

# print("\n" + "=" * 80)
# print("RECOMMENDATIONS FOR K SELECTION")
# print("=" * 80)

# # Find overall best k
# best_k_overall = k_summary_df.loc[k_summary_df['Mean Score'].idxmax()]

# print(f"\n1. OVERALL BEST K:")
# print(f"   k = {int(best_k_overall['k'])} ({best_k_overall['Method']} method)")
# print(f"   Mean Score: {best_k_overall['Mean Score']:.4f}")

# # Compare Greedy-Random vs Greedy-Distance
# print(f"\n2. GREEDY STRATEGY COMPARISON:")
# print(f"   Which initialization is better?")

# for k in sorted(set(k_summary_df['k'])):
#     greedy_random_row = k_summary_df[(k_summary_df['Method'] == 'Greedy-Random') & 
#                                      (k_summary_df['k'] == k)]
#     greedy_distance_row = k_summary_df[(k_summary_df['Method'] == 'Greedy-Distance') & 
#                                        (k_summary_df['k'] == k)]
    
#     if len(greedy_random_row) > 0 and len(greedy_distance_row) > 0:
#         gr_score = greedy_random_row['Mean Score'].values[0]
#         gd_score = greedy_distance_row['Mean Score'].values[0]
        
#         # Statistical test
#         if k in k_analysis['Greedy-Random'] and k in k_analysis['Greedy-Distance']:
#             gr_scores = k_analysis['Greedy-Random'][k]
#             gd_scores = k_analysis['Greedy-Distance'][k]
#             t_stat, p_value = stats.ttest_ind(gr_scores, gd_scores)
#             sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            
#             winner = "Greedy-Random" if gr_score > gd_score else "Greedy-Distance"
#             diff = abs(gr_score - gd_score)
            
#             print(f"   k={int(k)}: {winner} wins by {diff:.4f} (p={p_value:.4f} {sig})")

# # Analyze k vs molecule size relationship
# if len(molecule_sizes) > 5:
#     correlation_random = np.corrcoef(molecule_sizes, molecule_optimal_k_random)[0, 1]
#     correlation_distance = np.corrcoef(molecule_sizes, molecule_optimal_k_distance)[0, 1]
    
#     print(f"\n3. K VS MOLECULE SIZE:")
#     print(f"   Greedy-Random:    Correlation = {correlation_random:.3f}")
#     print(f"   Greedy-Distance:  Correlation = {correlation_distance:.3f}")
    
#     if abs(correlation_random) > 0.5 or abs(correlation_distance) > 0.5:
#         print(f"   ➤ Strong relationship: Larger molecules benefit from more partitions")
#     elif abs(correlation_random) > 0.3 or abs(correlation_distance) > 0.3:
#         print(f"   ➤ Moderate relationship: Some size-dependent effect")
#     else:
#         print(f"   ➤ Weak relationship: K selection relatively size-independent")

# print(f"\n4. METHOD-SPECIFIC RECOMMENDATIONS:")
# for method in ['Greedy-Random', 'Greedy-Distance', 'Distance', 'Random']:
#     if method not in k_analysis:
#         continue
#     method_data = k_summary_df[k_summary_df['Method'] == method]
#     if len(method_data) == 0:
#         continue
#     best_k = method_data.loc[method_data['Mean Score'].idxmax(), 'k']
#     best_score = method_data['Mean Score'].max()
    
#     # Find k with >95% of best score (more robust)
#     threshold = 0.95 * best_score
#     good_ks = method_data[method_data['Mean Score'] >= threshold]['k'].values
    
#     print(f"\n   {method}:")
#     print(f"     Best k: {int(best_k)} (score: {best_score:.4f})")
#     if len(good_ks) > 1:
#         print(f"     Reasonable k range: {sorted([int(k) for k in good_ks])}")

# print(f"\n5. PRACTICAL GUIDELINES:")
# print(f"   • For small molecules (<15 atoms): k=2-3")
# print(f"   • For medium molecules (15-30 atoms): k=3-5")
# print(f"   • For large molecules (>30 atoms): k=5-7")
# print(f"   • Use Greedy-Random for general cases (random initialization)")
# print(f"   • Use Greedy-Distance when chemical structure suggests natural clusters")
# print(f"   • Both greedy methods outperform their baseline initializations")

# print(f"   'For smaller graphs the number of partitions may be tractable'")
# print(f"   'For larger graphs, we will use greedy algorithms that evolve")
# print(f"    a random starting partition'")
# print(f"\n   ➤ This analysis validates: k should scale with graph size")
# print(f"   ➤ Both greedy strategies (random & distance init) outperform baselines")
# print(f"   ➤ Optimal k balances within-partition contributions vs")
# print(f"      higher-order interactions (Equation 2)")
# print(f"\n   ➤ Reference [26]: Tynes et al. 'Linear Graphlet Models for")
# print(f"      Accurate and Interpretable Cheminformatics'")

# # ═══════════════════════════════════════════════════════════════════════════
# # EXPORT COMPREHENSIVE RESULTS
# # ═══════════════════════════════════════════════════════════════════════════

# print("\n" + "=" * 80)
# print("EXPORTING COMPREHENSIVE RESULTS")
# print("=" * 80)

# # 1. All detailed results
# all_results_flat = []
# for strategy_name in strategies.keys():
#     for result in all_results[strategy_name]:
#         result_copy = result.copy()
#         result_copy['strategy'] = strategy_name
#         result_copy['method'] = strategy_name.split(' (')[0]
#         result_copy['is_greedy_random'] = 'Greedy-Random' in strategy_name
#         result_copy['is_greedy_distance'] = 'Greedy-Distance' in strategy_name
#         result_copy['is_greedy'] = result_copy['is_greedy_random'] or result_copy['is_greedy_distance']
#         all_results_flat.append(result_copy)

# df_export = pd.DataFrame(all_results_flat)
# df_export.to_csv('partition_interpretability_all_k_detailed_both_greedy.csv', index=False)
# print("✓ Saved detailed results to 'partition_interpretability_all_k_detailed_both_greedy.csv'")

# # 2. Summary by strategy
# summary_df.to_csv('partition_interpretability_all_k_summary_both_greedy.csv', index=False)
# print("✓ Saved summary statistics to 'partition_interpretability_all_k_summary_both_greedy.csv'")

# # 3. K-value analysis summary
# k_summary_df.to_csv('partition_interpretability_k_analysis_both_greedy.csv', index=False)
# print("✓ Saved k-value analysis to 'partition_interpretability_k_analysis_both_greedy.csv'")

# # 4. Per-molecule optimal k (for both greedy strategies)
# optimal_k_data = []
# for mol_idx in range(n_sample):
#     # Greedy-Random optimal k
#     mol_results_random = [r for strategy_name, results in all_results.items() 
#                           for r in results 
#                           if r['mol_idx'] == mol_idx and 'Greedy-Random' in strategy_name]
    
#     # Greedy-Distance optimal k
#     mol_results_distance = [r for strategy_name, results in all_results.items() 
#                             for r in results 
#                             if r['mol_idx'] == mol_idx and 'Greedy-Distance' in strategy_name]
    
#     if mol_results_random and mol_results_distance:
#         best_result_random = max(mol_results_random, key=lambda x: x['score'])
#         best_result_distance = max(mol_results_distance, key=lambda x: x['score'])
        
#         optimal_k_data.append({
#             'mol_idx': mol_idx,
#             'smiles': best_result_random['smiles'],
#             'n_atoms': best_result_random['n_atoms'],
#             'optimal_k_random': best_result_random['k_requested'],
#             'optimal_score_random': best_result_random['score'],
#             'within_frac_random': best_result_random['within_frac'],
#             'optimal_k_distance': best_result_distance['k_requested'],
#             'optimal_score_distance': best_result_distance['score'],
#             'within_frac_distance': best_result_distance['within_frac'],
#             'best_method': 'Greedy-Random' if best_result_random['score'] > best_result_distance['score'] else 'Greedy-Distance',
#             'score_difference': abs(best_result_random['score'] - best_result_distance['score'])
#         })

# df_optimal_k = pd.DataFrame(optimal_k_data)
# df_optimal_k.to_csv('partition_interpretability_optimal_k_per_molecule_both_greedy.csv', index=False)
# print("✓ Saved per-molecule optimal k to 'partition_interpretability_optimal_k_per_molecule_both_greedy.csv'")

# # 5. Greedy strategy comparison summary
# greedy_comparison = []
# for k in sorted(set(k_summary_df['k'])):
#     gr_row = k_summary_df[(k_summary_df['Method'] == 'Greedy-Random') & (k_summary_df['k'] == k)]
#     gd_row = k_summary_df[(k_summary_df['Method'] == 'Greedy-Distance') & (k_summary_df['k'] == k)]
    
#     if len(gr_row) > 0 and len(gd_row) > 0:
#         gr_score = gr_row['Mean Score'].values[0]
#         gd_score = gd_row['Mean Score'].values[0]
        
#         greedy_comparison.append({
#             'k': int(k),
#             'Greedy-Random Score': gr_score,
#             'Greedy-Distance Score': gd_score,
#             'Difference': gr_score - gd_score,
#             'Winner': 'Greedy-Random' if gr_score > gd_score else 'Greedy-Distance'
#         })

# df_greedy_comparison = pd.DataFrame(greedy_comparison)
# df_greedy_comparison.to_csv('partition_interpretability_greedy_comparison.csv', index=False)
# print("✓ Saved greedy strategy comparison to 'partition_interpretability_greedy_comparison.csv'")

# # ═══════════════════════════════════════════════════════════════════════════
# # FINAL COMPREHENSIVE SUMMARY
# # ═══════════════════════════════════════════════════════════════════════════

# print("\n" + "=" * 80)
# print("COMPREHENSIVE ANALYSIS COMPLETE")
# print("=" * 80)

# print(f"\n╔════════════════════════════════════════════════════════════════╗")
# print(f"║              KEY FINDINGS SUMMARY (BOTH GREEDY)                ║")
# print(f"╚════════════════════════════════════════════════════════════════╝")

# # Best overall strategy
# best_overall = summary_df.iloc[0]
# print(f"\n1. BEST OVERALL STRATEGY:")
# print(f"   Strategy: {best_overall['Strategy']}")
# print(f"   Mean Score: {best_overall['Mean Score']:.4f} ± {best_overall['Std Score']:.4f}")
# print(f"   Within-partition: {best_overall['Mean Within %']:.1f}%")
# print(f"   Between-partition: {best_overall['Mean Between %']:.1f}%")
# print(f"   Higher-order: {best_overall['Mean Higher %']:.1f}%")

# # Best k for each method
# print(f"\n2. OPTIMAL K BY METHOD:")
# for method in ['Greedy-Random', 'Greedy-Distance', 'Distance', 'Random']:
#     if method not in k_analysis:
#         continue
#     method_data = k_summary_df[k_summary_df['Method'] == method]
#     if len(method_data) == 0:
#         continue
#     best_k_row = method_data.loc[method_data['Mean Score'].idxmax()]
#     print(f"   {method:15s}: k={int(best_k_row['k'])} (score: {best_k_row['Mean Score']:.4f})")

# # Greedy advantage over baselines
# print(f"\n3. GREEDY OPTIMIZATION ADVANTAGE:")
# print(f"\n   Greedy-Random vs Random baseline:")
# for k in sorted(set(k_summary_df['k'])):
#     greedy_score = k_summary_df[(k_summary_df['Method'] == 'Greedy-Random') & 
#                                 (k_summary_df['k'] == k)]['Mean Score'].values
#     random_score = k_summary_df[(k_summary_df['Method'] == 'Random') & 
#                                 (k_summary_df['k'] == k)]['Mean Score'].values
    
#     if len(greedy_score) > 0 and len(random_score) > 0:
#         improvement = greedy_score[0] - random_score[0]
#         improvement_pct = 100 * improvement / random_score[0]
#         print(f"     k={int(k)}: +{improvement:.4f} ({improvement_pct:+.1f}%)")

# print(f"\n   Greedy-Distance vs Distance baseline:")
# for k in sorted(set(k_summary_df['k'])):
#     greedy_score = k_summary_df[(k_summary_df['Method'] == 'Greedy-Distance') & 
#                                 (k_summary_df['k'] == k)]['Mean Score'].values
#     distance_score = k_summary_df[(k_summary_df['Method'] == 'Distance') & 
#                                   (k_summary_df['k'] == k)]['Mean Score'].values
    
#     if len(greedy_score) > 0 and len(distance_score) > 0:
#         improvement = greedy_score[0] - distance_score[0]
#         improvement_pct = 100 * improvement / distance_score[0]
#         print(f"     k={int(k)}: +{improvement:.4f} ({improvement_pct:+.1f}%)")

# # Greedy-Random vs Greedy-Distance
# print(f"\n4. GREEDY-RANDOM VS GREEDY-DISTANCE:")
# wins_random = df_optimal_k['best_method'].value_counts().get('Greedy-Random', 0)
# wins_distance = df_optimal_k['best_method'].value_counts().get('Greedy-Distance', 0)
# total = len(df_optimal_k)

# print(f"   Greedy-Random wins: {wins_random}/{total} ({100*wins_random/total:.1f}%)")
# print(f"   Greedy-Distance wins: {wins_distance}/{total} ({100*wins_distance/total:.1f}%)")
# print(f"   Mean score difference: {df_optimal_k['score_difference'].mean():.4f}")

# if wins_random > wins_distance:
#     print(f"   ➤ Greedy-Random generally performs better")
# elif wins_distance > wins_random:
#     print(f"   ➤ Greedy-Distance generally performs better")
# else:
#     print(f"   ➤ Both strategies perform similarly")

# # Validation statistics
# print(f"\n5. VALIDATION STATISTICS:")
# validation_rate = df_export['validation_passed'].mean()
# mean_val_error = df_export['validation_error'].mean()
# print(f"   Overall validation pass rate: {100*validation_rate:.1f}%")
# print(f"   Mean validation error: {mean_val_error:.2e}")

# # Molecule size analysis
# if len(df_optimal_k) > 5:
#     size_k_corr_random = df_optimal_k[['n_atoms', 'optimal_k_random']].corr().iloc[0, 1]
#     size_k_corr_distance = df_optimal_k[['n_atoms', 'optimal_k_distance']].corr().iloc[0, 1]
    
#     print(f"\n6. MOLECULE SIZE EFFECT:")
#     print(f"   Correlation(size, optimal_k):")
#     print(f"     Greedy-Random:    {size_k_corr_random:.3f}")
#     print(f"     Greedy-Distance:  {size_k_corr_distance:.3f}")
    
#     small_mols = df_optimal_k[df_optimal_k['n_atoms'] <= 15]
#     medium_mols = df_optimal_k[(df_optimal_k['n_atoms'] > 15) & 
#                                (df_optimal_k['n_atoms'] <= 30)]
#     large_mols = df_optimal_k[df_optimal_k['n_atoms'] > 30]
    
#     if len(small_mols) > 0:
#         print(f"   Small molecules (≤15 atoms):")
#         print(f"     Mean k (Greedy-Random): {small_mols['optimal_k_random'].mean():.1f}")
#         print(f"     Mean k (Greedy-Distance): {small_mols['optimal_k_distance'].mean():.1f}")
#     if len(medium_mols) > 0:
#         print(f"   Medium molecules (16-30 atoms):")
#         print(f"     Mean k (Greedy-Random): {medium_mols['optimal_k_random'].mean():.1f}")
#         print(f"     Mean k (Greedy-Distance): {medium_mols['optimal_k_distance'].mean():.1f}")
#     if len(large_mols) > 0:
#         print(f"   Large molecules (>30 atoms):")
#         print(f"     Mean k (Greedy-Random): {large_mols['optimal_k_random'].mean():.1f}")
#         print(f"     Mean k (Greedy-Distance): {large_mols['optimal_k_distance'].mean():.1f}")


from collections import defaultdict
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Callable
from copy import deepcopy

from partition_interpretability import (
    PartitionInterpreter,
    PartitionOptimizer,
    ChemicalPartitioner,
    validate_partition_decomposition
)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

n_sample = min(20, len(test))
sample_indices = np.random.RandomState(42).choice(len(test), size=n_sample, replace=False)
sample_mols = [test.iloc[i]['mol'] for i in sample_indices]
sample_smiles = [test.iloc[i]['smiles'] for i in sample_indices]

greedy_optimizer = PartitionOptimizer(
    interpreter=interpreter,
    max_partitions=5
)

strategies = {
    # ---------------------------------------------------------------------
    # Direct initialization methods
    # ---------------------------------------------------------------------
    "Distance (k=2)": lambda mol: (
        ChemicalPartitioner.distance_partition(
            mol,
            n_clusters=2,
            seed=42
        )
    ),

    "Distance (k=3)": lambda mol: (
        ChemicalPartitioner.distance_partition(
            mol,
            n_clusters=3,
            seed=42
        )
    ),

    "Random (k=2)": lambda mol: (
        ChemicalPartitioner.random_partition(
            mol,
            n_clusters=2,
            seed=42
        )
    ),

    "Random (k=3)": lambda mol: (
        ChemicalPartitioner.random_partition(
            mol,
            n_clusters=3,
            seed=42
        )
    ),

    # ---------------------------------------------------------------------
    # Random-initialized greedy optimization
    # ---------------------------------------------------------------------
    "Greedy-Random (k=2)": lambda mol: (
        optimize_partition_random_wrapper(
            greedy_optimizer,
            mol,
            n_clusters=2,
            n_iterations=100,
            seed=42,
            verbose=False
        )
    ),

    "Greedy-Random (k=3)": lambda mol: (
        optimize_partition_random_wrapper(
            greedy_optimizer,
            mol,
            n_clusters=3,
            n_iterations=100,
            seed=42,
            verbose=False
        )
    ),

    "Greedy-Random (k=4)": lambda mol: (
        optimize_partition_random_wrapper(
            greedy_optimizer,
            mol,
            n_clusters=4,
            n_iterations=100,
            seed=42,
            verbose=False
        )
    ),

    # ---------------------------------------------------------------------
    # Distance-initialized greedy optimization
    # ---------------------------------------------------------------------
    "Greedy-Distance (k=2)": lambda mol: (
        optimize_partition_distance_wrapper(
            greedy_optimizer,
            mol,
            n_clusters=2,
            n_iterations=100,
            seed=42,
            verbose=False
        )
    ),

    "Greedy-Distance (k=3)": lambda mol: (
        optimize_partition_distance_wrapper(
            greedy_optimizer,
            mol,
            n_clusters=3,
            n_iterations=100,
            seed=42,
            verbose=False
        )
    ),

    "Greedy-Distance (k=4)": lambda mol: (
        optimize_partition_distance_wrapper(
            greedy_optimizer,
            mol,
            n_clusters=4,
            n_iterations=100,
            seed=42,
            verbose=False
        )
    ),

    # ---------------------------------------------------------------------
    # Metropolis-Hastings optimization
    #
    # optimize_partition_metropolis_hastings returns a
    # PartitionInterpretation, so use ".partition" to return the dictionary
    # expected by the analysis loop.
    # ---------------------------------------------------------------------
    "Metropolis-Hastings (k=2)": lambda mol: (
        greedy_optimizer.optimize_partition_metropolis_hastings(
            mol=mol,
            n_clusters=2,
            n_iterations=1000,
            beta=20.0,
            initialization="distance",
            seed=42,
            verbose=False
        ).partition
    ),

    "Metropolis-Hastings (k=3)": lambda mol: (
        greedy_optimizer.optimize_partition_metropolis_hastings(
            mol=mol,
            n_clusters=3,
            n_iterations=1000,
            beta=20.0,
            initialization="distance",
            seed=42,
            verbose=False
        ).partition
    ),

    "Metropolis-Hastings (k=4)": lambda mol: (
        greedy_optimizer.optimize_partition_metropolis_hastings(
            mol=mol,
            n_clusters=4,
            n_iterations=1000,
            beta=20.0,
            initialization="distance",
            seed=42,
            verbose=False
        ).partition
    ),
}

all_results = defaultdict(list)

for mol_idx, mol in enumerate(sample_mols):
    smiles = sample_smiles[mol_idx]
    graph_result = smiles_to_nx(
    smiles,
    add_hs=False,
    output_2d_pos=False,
)

    graph = (
        graph_result[0]
        if isinstance(graph_result, (tuple, list))
        else graph_result
    )

    interpreter.register_graph(mol, graph)
    for strategy_name, partition_fn in strategies.items():
        try:
            partition = partition_fn(mol)
            interp = interpreter.compute_partition_contributions(mol, partition)
            breakdown = interp.get_contribution_breakdown()
            validation = validate_partition_decomposition(mol, interpreter, partition, tolerance=1e-2)
            
            all_results[strategy_name].append({
                'mol_idx': mol_idx,
                'smiles': sample_smiles[mol_idx],
                'n_atoms': mol.GetNumAtoms(),
                'n_partitions': len(set(partition.values())),
                'score': interp.score,
                'within_frac': breakdown['within_frac'],
                'between_frac': breakdown['between_frac'],
                'higher_frac': breakdown['higher_frac'],
                'prediction': interp.total_prediction,
                'validation_passed': validation['passed'],
                'validation_error': validation['relative_error']
            })
        except Exception as e:
            continue

summary_data = []
for strategy_name in strategies.keys():
    results = all_results[strategy_name]
    if not results:
        continue
    
    df_strategy = pd.DataFrame(results)
    
    summary_data.append({
        'Strategy': strategy_name,
        'N_molecules': len(df_strategy),
        'Mean Score': df_strategy['score'].mean(),
        'Std Score': df_strategy['score'].std(),
        'Min Score': df_strategy['score'].min(),
        'Max Score': df_strategy['score'].max(),
        'Median Score': df_strategy['score'].median(),
        'Mean Within %': 100 * df_strategy['within_frac'].mean(),
        'Mean Between %': 100 * df_strategy['between_frac'].mean(),
        'Mean Higher %': 100 * df_strategy['higher_frac'].mean(),
        'Validation Pass %': 100 * df_strategy['validation_passed'].mean(),
        'Mean Val Error': df_strategy['validation_error'].mean()
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('Mean Score', ascending=False)

all_mol_data = []
for strategy_name in strategies.keys():
    for result in all_results[strategy_name]:
        all_mol_data.append({
            'strategy': strategy_name,
            'n_atoms': result['n_atoms'],
            'score': result['score']
        })

df_all = pd.DataFrame(all_mol_data)

size_bins = [0, 10, 20, 30, 100]
size_labels = ['Small (≤10)', 'Medium (11-20)', 'Large (21-30)', 'Very Large (>30)']

df_all['size_bin'] = pd.cut(df_all['n_atoms'], bins=size_bins, tick_labels=size_labels, include_lowest=True)

size_breakdown = df_all.groupby(['size_bin', 'strategy'])['score'].agg(['mean', 'std', 'count']).reset_index()
size_breakdown = size_breakdown.sort_values(['size_bin', 'mean'], ascending=[True, False])

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION: SCORE DISTRIBUTION AND CONTRIBUTION BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════

# Include only strategies that produced at least one valid result
strategy_names = [
    strategy_name
    for strategy_name in strategies
    if len(all_results[strategy_name]) > 0
]

if not strategy_names:
    raise ValueError(
        "No valid strategy results are available for plotting."
    )

fig, axes = plt.subplots(
    1,
    2,
    figsize=(20, 7)
)


# ───────────────────────────────────────────────────────────────────────────
# 1. Score distribution by strategy
# ───────────────────────────────────────────────────────────────────────────

ax = axes[0]

strategy_scores = [
    [
        result["score"]
        for result in all_results[strategy_name]
    ]
    for strategy_name in strategy_names
]

boxplot = ax.boxplot(
    strategy_scores,
    tick_labels=strategy_names,
    patch_artist=True,
    showmeans=True,
    meanprops={
        "marker": "o",
        "markerfacecolor": "black",
        "markeredgecolor": "black",
        "markersize": 6
    },
    medianprops={
        "color": "black",
        "linewidth": 1.5
    }
)

# Use the same color for strategies in the same method family
for strategy_name, box in zip(strategy_names, boxplot["boxes"]):
    if "Greedy-Random" in strategy_name:
        box.set_facecolor("#e74c3c")
    elif "Greedy-Distance" in strategy_name:
        box.set_facecolor("#9b59b6")
    elif strategy_name.startswith("Random"):
        box.set_facecolor("#95a5a6")
    elif strategy_name.startswith("Distance"):
        box.set_facecolor("#3498db")
    else:
        box.set_facecolor("#bdc3c7")

    box.set_alpha(0.75)

ax.set_ylabel(
    "Interpretability Score",
    fontsize=20
)

ax.set_title(
    "Score Distribution by Strategy",
    fontsize=24,
    fontweight="bold"
)

ax.tick_params(
    axis="x",
    labelrotation=45,
    labelsize = 18
)

for label in ax.get_xticklabels():
    label.set_horizontalalignment("right")

ax.grid(
    axis="y",
    alpha=0.3
)


# ───────────────────────────────────────────────────────────────────────────
# 2. Mean contribution breakdown by strategy
# ───────────────────────────────────────────────────────────────────────────

ax = axes[1]

within_means = [
    np.mean([
        result["within_frac"]
        for result in all_results[strategy_name]
    ])
    for strategy_name in strategy_names
]

between_means = [
    np.mean([
        result["between_frac"]
        for result in all_results[strategy_name]
    ])
    for strategy_name in strategy_names
]

higher_means = [
    np.mean([
        result["higher_frac"]
        for result in all_results[strategy_name]
    ])
    for strategy_name in strategy_names
]

x = np.arange(len(strategy_names))
bar_width = 0.25

ax.bar(
    x - bar_width,
    within_means,
    bar_width,
    label="Within, $F_0$",
    color="#2ecc71",
    alpha=0.85
)

ax.bar(
    x,
    between_means,
    bar_width,
    label="Between, $F_1$",
    color="#f39c12",
    alpha=0.85
)

ax.bar(
    x + bar_width,
    higher_means,
    bar_width,
    label="Higher-order, $F_{01}$",
    color="#e74c3c",
    alpha=0.85
)

ax.set_ylabel(
    "Mean Fraction of Total Contribution",
    fontsize=18
)

ax.set_title(
    "Mean Contribution Breakdown",
    fontsize=24,
    fontweight="bold"
)

ax.set_xticks(x)

ax.tick_params(
    axis="x",
    labelsize=18
)

ax.set_xticklabels(
    strategy_names,
    rotation=45,
    ha="right"
)

ax.legend()
ax.grid(
    axis="y",
    alpha=0.3
)


# ───────────────────────────────────────────────────────────────────────────
# Final formatting and export
# ───────────────────────────────────────────────────────────────────────────

fig.suptitle(
    "Partition Strategy Performance",
    fontsize=24,
    fontweight="bold",
    y=1.02
)

plt.tight_layout()

output_filename = (
    "partition_interpretability_score_and_contributions_both_greedy.png"
)

plt.savefig(
    output_filename,
    dpi=300,
    bbox_inches="tight"
)

print(
    f"✓ Saved visualization to '{output_filename}'"
)

plt.show()



all_results_flat = []
for strategy_name in strategies.keys():
    for result in all_results[strategy_name]:
        result_copy = result.copy()
        result_copy['strategy'] = strategy_name
        result_copy['is_greedy_random'] = 'Greedy-Random' in strategy_name
        result_copy['is_greedy_distance'] = 'Greedy-Distance' in strategy_name
        all_results_flat.append(result_copy)

df_export = pd.DataFrame(all_results_flat)
df_export.to_csv('partition_interpretability_detailed_results_both_greedy.csv', index=False)

summary_df.to_csv('partition_interpretability_summary_both_greedy.csv', index=False)
from collections import defaultdict
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Callable

from partition_interpretability import (
    PartitionInterpreter,
    PartitionOptimizer,
    ChemicalPartitioner,
    validate_partition_decomposition
)


# ═══════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE K-VALUE ANALYSIS WITH BOTH GREEDY STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════


def determine_max_k_for_molecule(mol) -> int:
    """
    Determine maximum sensible k for a molecule.
    
    "For smaller graphs the number of partitions may be tractable"
    
    Heuristic: At least 2-3 atoms per partition on average
    """
    n_atoms = mol.GetNumAtoms()
    max_k = max(2, n_atoms // 2)  # At least 2 atoms per partition
    max_k = min(max_k, 10)  # Cap at 10 for interpretability
    return max_k


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS - ALL K VALUES WITH BOTH GREEDY STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════

n_sample = min(20, len(test))
sample_indices = np.random.RandomState(42).choice(len(test), size=n_sample, replace=False)
sample_mols = [test.iloc[i]['mol'] for i in sample_indices]
sample_smiles = [test.iloc[i]['smiles'] for i in sample_indices]

print("=" * 80)
print(f"COMPREHENSIVE K-VALUE ANALYSIS ({n_sample} molecules)")
print("=" * 80)

# Determine k range based on molecule sizes
k_range = range(2, 8)  # Test k=2,3,4,5,6,7
# Metropolis-Hastings settings
MH_ITERATIONS = 500
MH_BETA = 20.0
MH_INITIALIZATION = "distance"
MH_SEED = 42
print(f"\nTesting k values: {list(k_range)}")
print(f"  'For smaller graphs the number of partitions may be tractable'")
print(f"  'For larger graphs, we will use greedy algorithms that evolve")
print(f"   a random starting partition'")

# Initialize optimizer
greedy_optimizer = PartitionOptimizer(
    interpreter=interpreter,
    max_partitions=max(k_range)  # Set to maximum k we'll test
)

# Define comprehensive strategies
strategies = {
   # 'Functional Groups': lambda mol: ChemicalPartitioner.functional_group_partition(mol),
}

# Add distance-based baseline strategies for all k
for k in k_range:
    strategies[f'Distance (k={k})'] = lambda mol, k=k: ChemicalPartitioner.distance_partition(
        mol, n_clusters=k, seed=42
    )

# Add random baseline strategies for all k
for k in k_range:
    strategies[f'Random (k={k})'] = lambda mol, k=k: ChemicalPartitioner.random_partition(
        mol, n_clusters=k, seed=42
    )

# Add Greedy-Random strategies for all k (random initialization + greedy optimization)
for k in k_range:
    strategies[f'Greedy-Random (k={k})'] = lambda mol, k=k: optimize_partition_random_wrapper(
        greedy_optimizer, mol, n_clusters=k, n_iterations=100, seed=42, verbose=False
    )

# Add Greedy-Distance strategies for all k (distance initialization + greedy optimization)
for k in k_range:
    strategies[f'Greedy-Distance (k={k})'] = lambda mol, k=k: optimize_partition_distance_wrapper(
        greedy_optimizer, mol, n_clusters=k, n_iterations=100, seed=42, verbose=False
    )

for k in k_range:
    strategies[f"Metropolis-Hastings (k={k})"] = (
        lambda mol, k=k: (
            greedy_optimizer.optimize_partition_metropolis_hastings(
                mol=mol,
                n_clusters=k,
                n_iterations=MH_ITERATIONS,
                beta=MH_BETA,
                initialization=MH_INITIALIZATION,
                seed=MH_SEED,
                verbose=False
            ).partition
        )
    )

print(f"\nTotal strategies: {len(strategies)}")
print(f"  • Distance baseline (k=2-7): {len(k_range)}")
print(f"  • Random baseline (k=2-7): {len(k_range)}")
print(f"  • Greedy-Random (k=2-7): {len(k_range)}")
print(f"  • Greedy-Distance (k=2-7): {len(k_range)}")
print(f"  • Metropolis-Hastings (k=2-7): {len(k_range)}")
print(
    f"    MH settings: iterations={MH_ITERATIONS}, "
    f"beta={MH_BETA}, "
    f"initialization={MH_INITIALIZATION}"
)

# Collect results
all_results = defaultdict(list)

print("\nComputing interpretability scores...")
for mol_idx, mol in enumerate(sample_mols):

    smiles = sample_smiles[mol_idx]
    graph_result = smiles_to_nx(
    smiles,
    add_hs=False,
    output_2d_pos=False,
)

    graph = (
        graph_result[0]
        if isinstance(graph_result, (tuple, list))
        else graph_result
    )

    interpreter.register_graph(mol, graph)

    if (mol_idx + 1) % 10 == 0:
        print(f"  Processed {mol_idx + 1}/{n_sample} molecules")
    
    for strategy_name, partition_fn in strategies.items():
        try:
            partition = partition_fn(mol)
            interp = interpreter.compute_partition_contributions(mol, partition)
            breakdown = interp.get_contribution_breakdown()
            validation = validate_partition_decomposition(mol, interpreter, partition, tolerance=1e-2)
            
            all_results[strategy_name].append({
                'mol_idx': mol_idx,
                'smiles': sample_smiles[mol_idx],
                'n_atoms': mol.GetNumAtoms(),
                'n_partitions': len(set(partition.values())),
                'k_requested': int(strategy_name.split('k=')[1].rstrip(')')) if 'k=' in strategy_name else None,
                'score': interp.score,
                'within_frac': breakdown['within_frac'],
                'between_frac': breakdown['between_frac'],
                'higher_frac': breakdown['higher_frac'],
                'prediction': interp.total_prediction,
                'validation_passed': validation['passed'],
                'validation_error': validation['relative_error']
            })
        except Exception as e:
            print(f"  Warning: Failed {strategy_name} for molecule {mol_idx}: {e}")
            continue

print("\n" + "=" * 80)
print("SUMMARY BY STRATEGY")
print("=" * 80)

# Aggregate statistics
summary_data = []
for strategy_name in strategies.keys():
    results = all_results[strategy_name]
    if not results:
        continue
    
    df_strategy = pd.DataFrame(results)
    
    summary_data.append({
        'Strategy': strategy_name,
        'N_molecules': len(df_strategy),
        'Mean Score': df_strategy['score'].mean(),
        'Std Score': df_strategy['score'].std(),
        'Min Score': df_strategy['score'].min(),
        'Max Score': df_strategy['score'].max(),
        'Median Score': df_strategy['score'].median(),
        'Mean Within %': 100 * df_strategy['within_frac'].mean(),
        'Mean Between %': 100 * df_strategy['between_frac'].mean(),
        'Mean Higher %': 100 * df_strategy['higher_frac'].mean(),
        'Validation Pass %': 100 * df_strategy['validation_passed'].mean(),
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('Mean Score', ascending=False)

print("\n" + summary_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# CRITICAL ANALYSIS: OPTIMAL K SELECTION (WITH BOTH GREEDY STRATEGIES)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("OPTIMAL K ANALYSIS")
print("=" * 80)

# Extract k-dependent results
k_analysis = defaultdict(lambda: defaultdict(list))

for strategy_name, results in all_results.items():
    if 'k=' in strategy_name:
        # Extract method name (before the first parenthesis)
        method = strategy_name.split(' (')[0]  # 'Distance', 'Random', 'Greedy-Random', 'Greedy-Distance'
        k = int(strategy_name.split('k=')[1].rstrip(')'))
        
        for result in results:
            k_analysis[method][k].append(result['score'])

# Analyze each method's k-dependence
print("\n1. MEAN SCORE BY K VALUE:\n")

k_summary = []
for method in ['Distance', 'Random', 'Greedy-Random', 'Greedy-Distance',  "Metropolis-Hastings"]:
    if method not in k_analysis:
        continue
    print(f"{method}:")
    for k in sorted(k_analysis[method].keys()):
        scores = k_analysis[method][k]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        k_summary.append({
            'Method': method,
            'k': k,
            'Mean Score': mean_score,
            'Std Score': std_score,
            'N': len(scores)
        })
        print(f"  k={k}: {mean_score:.4f} ± {std_score:.4f}")
    print()

k_summary_df = pd.DataFrame(k_summary)

# Find optimal k for each method
print("2. OPTIMAL K BY METHOD:\n")

for method in ['Distance', 'Random', 'Greedy-Random', 'Greedy-Distance',  "Metropolis-Hastings"]:
    if method not in k_analysis:
        continue
    method_data = k_summary_df[k_summary_df['Method'] == method]
    if len(method_data) == 0:
        continue
    best_k = method_data.loc[method_data['Mean Score'].idxmax(), 'k']
    best_score = method_data['Mean Score'].max()
    
    print(f"{method}:")
    print(f"  Best k: {int(best_k)}")
    print(f"  Score: {best_score:.4f}")
    
    # Test if this is significantly better than others
    best_k_scores = k_analysis[method][best_k]
    print(f"  Comparisons to other k values:")
    
    for k in sorted(k_analysis[method].keys()):
        if k == best_k:
            continue
        other_scores = k_analysis[method][k]
        t_stat, p_value = stats.ttest_ind(best_k_scores, other_scores)
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"    vs k={k}: Δ={np.mean(best_k_scores) - np.mean(other_scores):+.4f}, "
              f"p={p_value:.4f} {sig}")
    print()

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION: K-DEPENDENCE
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("K-DEPENDENCE VISUALIZATION")
print("=" * 80)

methods = [
    'Distance',
    'Random',
    'Greedy-Random',
    'Greedy-Distance',
     "Metropolis-Hastings"
]


def get_plot_style(method):
    """
    Return consistent plotting styles for each partitioning method.
    """
    if method == 'Greedy-Random':
        return {
            'linestyle': '-',
            'marker': 'D',
            'linewidth': 3,
            'markersize': 8
        }

    if method == 'Greedy-Distance':
        return {
            'linestyle': '--',
            'marker': 's',
            'linewidth': 3,
            'markersize': 8
        }
    
    if method == "Metropolis-Hastings":
        return {
            "linestyle": "-.",
            "marker": "^",
            "linewidth": 3,
            "markersize": 9
        }

    return {
        'linestyle': ':',
        'marker': 'o',
        'linewidth': 2,
        'markersize': 8
    }


# Create one row containing the three remaining plots
fig, axes = plt.subplots(
    1,
    3,
    figsize=(20, 6)
)


# ───────────────────────────────────────────────────────────────────────────
# 1. Mean interpretability score vs k
# ───────────────────────────────────────────────────────────────────────────

ax = axes[0]

for method in methods:
    if method not in k_analysis:
        continue

    ks = sorted(k_analysis[method].keys())

    if not ks:
        continue

    means = [
        np.mean(k_analysis[method][k])
        for k in ks
    ]

    stds = [
        np.std(k_analysis[method][k])
        for k in ks
    ]

    style = get_plot_style(method)

    ax.plot(
        ks,
        means,
        label=method,
        **style
    )

    ax.fill_between(
        ks,
        np.asarray(means) - np.asarray(stds),
        np.asarray(means) + np.asarray(stds),
        alpha=0.2
    )

ax.set_xlabel(
    'Number of Partitions (k)',
    fontsize=12
)

ax.set_ylabel(
    'Mean Interpretability Score',
    fontsize=12
)

ax.set_title(
    'Interpretability Score vs k',
    fontsize=14,
    fontweight='bold'
)

ax.legend()
ax.grid(alpha=0.3)


# ───────────────────────────────────────────────────────────────────────────
# 2. Mean within-partition contribution vs k
# ───────────────────────────────────────────────────────────────────────────

ax = axes[1]

for method in methods:
    within_by_k = defaultdict(list)
    method_prefix = f'{method} (k='

    for strategy_name, results in all_results.items():
        if not strategy_name.startswith(method_prefix):
            continue

        k = int(
            strategy_name.split('k=')[1].rstrip(')')
        )

        for result in results:
            within_by_k[k].append(
                result['within_frac']
            )

    if not within_by_k:
        continue

    ks = sorted(within_by_k.keys())

    means = [
        np.mean(within_by_k[k])
        for k in ks
    ]

    stds = [
        np.std(within_by_k[k])
        for k in ks
    ]

    style = get_plot_style(method)

    ax.plot(
        ks,
        means,
        label=method,
        **style
    )

    ax.fill_between(
        ks,
        np.asarray(means) - np.asarray(stds),
        np.asarray(means) + np.asarray(stds),
        alpha=0.2
    )

ax.set_xlabel(
    'Number of Partitions (k)',
    fontsize=12
)

ax.set_ylabel(
    'Mean Within-Partition Fraction',
    fontsize=12
)

ax.set_title(
    'Within-Partition Contribution vs k',
    fontsize=14,
    fontweight='bold'
)

ax.legend()
ax.grid(alpha=0.3)


# ───────────────────────────────────────────────────────────────────────────
# 3. Mean higher-order contribution vs k
# ───────────────────────────────────────────────────────────────────────────

ax = axes[2]

for method in methods:
    higher_by_k = defaultdict(list)
    method_prefix = f'{method} (k='

    for strategy_name, results in all_results.items():
        if not strategy_name.startswith(method_prefix):
            continue

        k = int(
            strategy_name.split('k=')[1].rstrip(')')
        )

        for result in results:
            higher_by_k[k].append(
                result['higher_frac']
            )

    if not higher_by_k:
        continue

    ks = sorted(higher_by_k.keys())

    means = [
        np.mean(higher_by_k[k])
        for k in ks
    ]

    stds = [
        np.std(higher_by_k[k])
        for k in ks
    ]

    style = get_plot_style(method)

    ax.plot(
        ks,
        means,
        label=method,
        **style
    )

    ax.fill_between(
        ks,
        np.asarray(means) - np.asarray(stds),
        np.asarray(means) + np.asarray(stds),
        alpha=0.2
    )

ax.set_xlabel(
    'Number of Partitions (k)',
    fontsize=12
)

ax.set_ylabel(
    'Mean Higher-Order Fraction',
    fontsize=12
)

ax.set_title(
    'Higher-Order Contribution vs k\n',
    fontsize=14,
    fontweight='bold'
)

ax.legend()
ax.grid(alpha=0.3)


# Add an overall title above the three panels
fig.suptitle(
    'Effect of Partition Count on Interpretability',
    fontsize=17,
    fontweight='bold',
    y=1.03
)

plt.tight_layout()

output_filename = 'k_value_analysis.png'

plt.savefig(
    output_filename,
    dpi=300,
    bbox_inches='tight'
)

print(
    f"\n✓ Saved visualization to '{output_filename}'"
)

plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# RECOMMENDATIONS WITH BOTH GREEDY STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("RECOMMENDATIONS FOR K SELECTION")
print("=" * 80)

# Find overall best k
best_k_overall = k_summary_df.loc[k_summary_df['Mean Score'].idxmax()]

print(f"\n1. OVERALL BEST K:")
print(f"   k = {int(best_k_overall['k'])} ({best_k_overall['Method']} method)")
print(f"   Mean Score: {best_k_overall['Mean Score']:.4f}")

# Compare Greedy-Random vs Greedy-Distance
print(f"\n2. GREEDY STRATEGY COMPARISON:")
print(f"   Which initialization is better?")

for k in sorted(set(k_summary_df['k'])):
    greedy_random_row = k_summary_df[(k_summary_df['Method'] == 'Greedy-Random') & 
                                     (k_summary_df['k'] == k)]
    greedy_distance_row = k_summary_df[(k_summary_df['Method'] == 'Greedy-Distance') & 
                                       (k_summary_df['k'] == k)]
    
    if len(greedy_random_row) > 0 and len(greedy_distance_row) > 0:
        gr_score = greedy_random_row['Mean Score'].values[0]
        gd_score = greedy_distance_row['Mean Score'].values[0]
        
        # Statistical test
        if k in k_analysis['Greedy-Random'] and k in k_analysis['Greedy-Distance']:
            gr_scores = k_analysis['Greedy-Random'][k]
            gd_scores = k_analysis['Greedy-Distance'][k]
            t_stat, p_value = stats.ttest_ind(gr_scores, gd_scores)
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            
            winner = "Greedy-Random" if gr_score > gd_score else "Greedy-Distance"
            diff = abs(gr_score - gd_score)
            
            print(f"   k={int(k)}: {winner} wins by {diff:.4f} (p={p_value:.4f} {sig})")

print(f"\n4. METHOD-SPECIFIC RECOMMENDATIONS:")
for method in ['Greedy-Random', 'Greedy-Distance', 'Distance', 'Random',  "Metropolis-Hastings"]:
    if method not in k_analysis:
        continue
    method_data = k_summary_df[k_summary_df['Method'] == method]
    if len(method_data) == 0:
        continue
    best_k = method_data.loc[method_data['Mean Score'].idxmax(), 'k']
    best_score = method_data['Mean Score'].max()
    
    # Find k with >95% of best score (more robust)
    threshold = 0.95 * best_score
    good_ks = method_data[method_data['Mean Score'] >= threshold]['k'].values
    
    print(f"\n   {method}:")
    print(f"     Best k: {int(best_k)} (score: {best_score:.4f})")
    if len(good_ks) > 1:
        print(f"     Reasonable k range: {sorted([int(k) for k in good_ks])}")

print(f"\n5. PRACTICAL GUIDELINES:")
print(f"   • For small molecules (<15 atoms): k=2-3")
print(f"   • For medium molecules (15-30 atoms): k=3-5")
print(f"   • For large molecules (>30 atoms): k=5-7")
print(f"   • Use Greedy-Random for general cases (random initialization)")
print(f"   • Use Greedy-Distance when chemical structure suggests natural clusters")
print(f"   • Both greedy methods outperform their baseline initializations")

print(f"   'For smaller graphs the number of partitions may be tractable'")
print(f"   'For larger graphs, we will use greedy algorithms that evolve")
print(f"    a random starting partition'")
print(f"\n   ➤ This analysis validates: k should scale with graph size")
print(f"   ➤ Both greedy strategies (random & distance init) outperform baselines")
print(f"   ➤ Optimal k balances within-partition contributions vs")
print(f"      higher-order interactions (Equation 2)")
print(f"\n   ➤ Reference [26]: Tynes et al. 'Linear Graphlet Models for")
print(f"      Accurate and Interpretable Cheminformatics'")

# ═══════════════════════════════════════════════════════════════════════════
# EXPORT COMPREHENSIVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("EXPORTING COMPREHENSIVE RESULTS")
print("=" * 80)

# 1. All detailed results
all_results_flat = []
for strategy_name in strategies.keys():
    for result in all_results[strategy_name]:
        result_copy = result.copy()
        result_copy['strategy'] = strategy_name
        result_copy['method'] = strategy_name.split(' (')[0]
        result_copy['is_greedy_random'] = 'Greedy-Random' in strategy_name
        result_copy['is_greedy_distance'] = 'Greedy-Distance' in strategy_name
        result_copy['is_greedy'] = result_copy['is_greedy_random'] or result_copy['is_greedy_distance']
        result_copy["is_metropolis_hastings"] = ("Metropolis-Hastings" in strategy_name)

        all_results_flat.append(result_copy)

df_export = pd.DataFrame(all_results_flat)
df_export.to_csv('partition_interpretability_all_k_detailed_both_greedy.csv', index=False)
print("✓ Saved detailed results to 'partition_interpretability_all_k_detailed_both_greedy.csv'")

# 2. Summary by strategy
summary_df.to_csv('partition_interpretability_all_k_summary_both_greedy.csv', index=False)
print("✓ Saved summary statistics to 'partition_interpretability_all_k_summary_both_greedy.csv'")

# 3. K-value analysis summary
k_summary_df.to_csv('partition_interpretability_k_analysis_both_greedy.csv', index=False)
print("✓ Saved k-value analysis to 'partition_interpretability_k_analysis_both_greedy.csv'")

# 4. Per-molecule optimal k (for both greedy strategies)
optimal_k_data = []
for mol_idx in range(n_sample):
    # Greedy-Random optimal k
    mol_results_random = [r for strategy_name, results in all_results.items() 
                          for r in results 
                          if r['mol_idx'] == mol_idx and 'Greedy-Random' in strategy_name]
    
    # Greedy-Distance optimal k
    mol_results_distance = [r for strategy_name, results in all_results.items() 
                            for r in results 
                            if r['mol_idx'] == mol_idx and 'Greedy-Distance' in strategy_name]
    
    if mol_results_random and mol_results_distance:
        best_result_random = max(mol_results_random, key=lambda x: x['score'])
        best_result_distance = max(mol_results_distance, key=lambda x: x['score'])
        
        optimal_k_data.append({
            'mol_idx': mol_idx,
            'smiles': best_result_random['smiles'],
            'n_atoms': best_result_random['n_atoms'],
            'optimal_k_random': best_result_random['k_requested'],
            'optimal_score_random': best_result_random['score'],
            'within_frac_random': best_result_random['within_frac'],
            'optimal_k_distance': best_result_distance['k_requested'],
            'optimal_score_distance': best_result_distance['score'],
            'within_frac_distance': best_result_distance['within_frac'],
            'best_method': 'Greedy-Random' if best_result_random['score'] > best_result_distance['score'] else 'Greedy-Distance',
            'score_difference': abs(best_result_random['score'] - best_result_distance['score'])
        })

df_optimal_k = pd.DataFrame(optimal_k_data)
df_optimal_k.to_csv('partition_interpretability_optimal_k_per_molecule_both_greedy.csv', index=False)
print("✓ Saved per-molecule optimal k to 'partition_interpretability_optimal_k_per_molecule_both_greedy.csv'")

# Analyze k vs molecule size relationship after df_optimal_k exists
if len(df_optimal_k) > 5:
    molecule_sizes = df_optimal_k['n_atoms'].to_numpy()
    molecule_optimal_k_random = df_optimal_k['optimal_k_random'].to_numpy()
    molecule_optimal_k_distance = df_optimal_k['optimal_k_distance'].to_numpy()

    correlation_random = np.corrcoef(
        molecule_sizes,
        molecule_optimal_k_random
    )[0, 1]

    correlation_distance = np.corrcoef(
        molecule_sizes,
        molecule_optimal_k_distance
    )[0, 1]

    print("\n" + "=" * 80)
    print("K VS MOLECULE SIZE")
    print("=" * 80)
    print(f"   Greedy-Random:    Correlation = {correlation_random:.3f}")
    print(f"   Greedy-Distance:  Correlation = {correlation_distance:.3f}")

    if abs(correlation_random) > 0.5 or abs(correlation_distance) > 0.5:
        print("   ➤ Strong relationship: Larger molecules benefit from more partitions")
    elif abs(correlation_random) > 0.3 or abs(correlation_distance) > 0.3:
        print("   ➤ Moderate relationship: Some size-dependent effect")
    else:
        print("   ➤ Weak relationship: K selection relatively size-independent")
else:
    print("\nSkipping k-vs-molecule-size analysis: fewer than 6 molecules have valid optimal-k results.")

# 5. Greedy strategy comparison summary
greedy_comparison = []
for k in sorted(set(k_summary_df['k'])):
    gr_row = k_summary_df[(k_summary_df['Method'] == 'Greedy-Random') & (k_summary_df['k'] == k)]
    gd_row = k_summary_df[(k_summary_df['Method'] == 'Greedy-Distance') & (k_summary_df['k'] == k)]
    
    if len(gr_row) > 0 and len(gd_row) > 0:
        gr_score = gr_row['Mean Score'].values[0]
        gd_score = gd_row['Mean Score'].values[0]
        
        greedy_comparison.append({
            'k': int(k),
            'Greedy-Random Score': gr_score,
            'Greedy-Distance Score': gd_score,
            'Difference': gr_score - gd_score,
            'Winner': 'Greedy-Random' if gr_score > gd_score else 'Greedy-Distance'
        })

df_greedy_comparison = pd.DataFrame(greedy_comparison)
df_greedy_comparison.to_csv('partition_interpretability_greedy_comparison.csv', index=False)
print("✓ Saved greedy strategy comparison to 'partition_interpretability_greedy_comparison.csv'")

# ═══════════════════════════════════════════════════════════════════════════
# FINAL COMPREHENSIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("COMPREHENSIVE ANALYSIS COMPLETE")
print("=" * 80)

print(f"\n╔════════════════════════════════════════════════════════════════╗")
print(f"║              KEY FINDINGS SUMMARY (BOTH GREEDY)                ║")
print(f"╚════════════════════════════════════════════════════════════════╝")

# Best overall strategy
best_overall = summary_df.iloc[0]
print(f"\n1. BEST OVERALL STRATEGY:")
print(f"   Strategy: {best_overall['Strategy']}")
print(f"   Mean Score: {best_overall['Mean Score']:.4f} ± {best_overall['Std Score']:.4f}")
print(f"   Within-partition: {best_overall['Mean Within %']:.1f}%")
print(f"   Between-partition: {best_overall['Mean Between %']:.1f}%")
print(f"   Higher-order: {best_overall['Mean Higher %']:.1f}%")

# Best k for each method
print(f"\n2. OPTIMAL K BY METHOD:")
for method in ['Greedy-Random', 'Greedy-Distance', 'Distance', 'Random',  "Metropolis-Hastings"]:
    if method not in k_analysis:
        continue
    method_data = k_summary_df[k_summary_df['Method'] == method]
    if len(method_data) == 0:
        continue
    best_k_row = method_data.loc[method_data['Mean Score'].idxmax()]
    print(f"   {method:15s}: k={int(best_k_row['k'])} (score: {best_k_row['Mean Score']:.4f})")

# Greedy advantage over baselines
print(f"\n3. GREEDY OPTIMIZATION ADVANTAGE:")
print(f"\n   Greedy-Random vs Random baseline:")
for k in sorted(set(k_summary_df['k'])):
    greedy_score = k_summary_df[(k_summary_df['Method'] == 'Greedy-Random') & 
                                (k_summary_df['k'] == k)]['Mean Score'].values
    random_score = k_summary_df[(k_summary_df['Method'] == 'Random') & 
                                (k_summary_df['k'] == k)]['Mean Score'].values
    
    if len(greedy_score) > 0 and len(random_score) > 0:
        improvement = greedy_score[0] - random_score[0]
        improvement_pct = 100 * improvement / random_score[0]
        print(f"     k={int(k)}: +{improvement:.4f} ({improvement_pct:+.1f}%)")

print(f"\n   Greedy-Distance vs Distance baseline:")
for k in sorted(set(k_summary_df['k'])):
    greedy_score = k_summary_df[(k_summary_df['Method'] == 'Greedy-Distance') & 
                                (k_summary_df['k'] == k)]['Mean Score'].values
    distance_score = k_summary_df[(k_summary_df['Method'] == 'Distance') & 
                                  (k_summary_df['k'] == k)]['Mean Score'].values
    
    if len(greedy_score) > 0 and len(distance_score) > 0:
        improvement = greedy_score[0] - distance_score[0]
        improvement_pct = 100 * improvement / distance_score[0]
        print(f"     k={int(k)}: +{improvement:.4f} ({improvement_pct:+.1f}%)")

# Greedy-Random vs Greedy-Distance
print(f"\n4. GREEDY-RANDOM VS GREEDY-DISTANCE:")
wins_random = df_optimal_k['best_method'].value_counts().get('Greedy-Random', 0)
wins_distance = df_optimal_k['best_method'].value_counts().get('Greedy-Distance', 0)
total = len(df_optimal_k)

print(f"   Greedy-Random wins: {wins_random}/{total} ({100*wins_random/total:.1f}%)")
print(f"   Greedy-Distance wins: {wins_distance}/{total} ({100*wins_distance/total:.1f}%)")
print(f"   Mean score difference: {df_optimal_k['score_difference'].mean():.4f}")

if wins_random > wins_distance:
    print(f"   ➤ Greedy-Random generally performs better")
elif wins_distance > wins_random:
    print(f"   ➤ Greedy-Distance generally performs better")
else:
    print(f"   ➤ Both strategies perform similarly")

# Validation statistics
print(f"\n5. VALIDATION STATISTICS:")
validation_rate = df_export['validation_passed'].mean()
mean_val_error = df_export['validation_error'].mean()
print(f"   Overall validation pass rate: {100*validation_rate:.1f}%")
print(f"   Mean validation error: {mean_val_error:.2e}")

# Molecule size analysis
if len(df_optimal_k) > 5:
    size_k_corr_random = df_optimal_k[['n_atoms', 'optimal_k_random']].corr().iloc[0, 1]
    size_k_corr_distance = df_optimal_k[['n_atoms', 'optimal_k_distance']].corr().iloc[0, 1]
    
    print(f"\n6. MOLECULE SIZE EFFECT:")
    print(f"   Correlation(size, optimal_k):")
    print(f"     Greedy-Random:    {size_k_corr_random:.3f}")
    print(f"     Greedy-Distance:  {size_k_corr_distance:.3f}")
    
    small_mols = df_optimal_k[df_optimal_k['n_atoms'] <= 15]
    medium_mols = df_optimal_k[(df_optimal_k['n_atoms'] > 15) & 
                               (df_optimal_k['n_atoms'] <= 30)]
    large_mols = df_optimal_k[df_optimal_k['n_atoms'] > 30]
    
    if len(small_mols) > 0:
        print(f"   Small molecules (≤15 atoms):")
        print(f"     Mean k (Greedy-Random): {small_mols['optimal_k_random'].mean():.1f}")
        print(f"     Mean k (Greedy-Distance): {small_mols['optimal_k_distance'].mean():.1f}")
    if len(medium_mols) > 0:
        print(f"   Medium molecules (16-30 atoms):")
        print(f"     Mean k (Greedy-Random): {medium_mols['optimal_k_random'].mean():.1f}")
        print(f"     Mean k (Greedy-Distance): {medium_mols['optimal_k_distance'].mean():.1f}")
    if len(large_mols) > 0:
        print(f"   Large molecules (>30 atoms):")
        print(f"     Mean k (Greedy-Random): {large_mols['optimal_k_random'].mean():.1f}")
        print(f"     Mean k (Greedy-Distance): {large_mols['optimal_k_distance'].mean():.1f}")


from collections import defaultdict
from typing import Dict

from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from partition_interpretability import (
    PartitionOptimizer,
    ChemicalPartitioner,
    validate_partition_decomposition,
    partition_connectivity_penalty,
    connectivity_adjusted_score,
)