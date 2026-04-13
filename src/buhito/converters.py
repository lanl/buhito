# collection of functions to convert from other formats to networkx graphs
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
import numpy as np  
import pandas as pd
from rdkit.Chem import rdDepictor

def smiles_to_nx(smiles, add_hs=False, output_2d_pos=False):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles)) if add_hs else Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    G = nx.Graph()
    if output_2d_pos:
        rdDepictor.Compute2DCoords(mol)
        conf = mol.GetConformer()
        pos={}

    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        if output_2d_pos:
            p = conf.GetAtomPosition(i)
            pos[i] = (float(p.x), float(p.y))
        G.add_node(
            i,
            atom_symbol=atom.GetSymbol(),
            atom_key=(atom.GetAtomicNum(),atom.GetFormalCharge()))
            # atom_key=str(atom.GetAtomicNum()))

    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        G.add_edge(
            u, v,
            bond_key=str(bond.GetBondType()),
            is_aromatic=bond.GetIsAromatic()
        )
    # nx.set_node_attributes(G, pos, "pos")
    if output_2d_pos:
        return G, pos
    else:
        return G, None
