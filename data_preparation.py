"""
MDM2 GNN - Data Preparation
============================
Converts SMILES strings into molecular graphs for Graph Attention Network.
Each molecule becomes a graph where:
- Nodes = atoms (with features like element, charge, aromaticity)
- Edges = bonds (with features like bond type)
"""

import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data
import os

# ── Configuration ──────────────────────────────────────────────────────────
DATA_PATH = r'C:\Users\danny\Downloads\MDM2_QSAR\data\mdm2_cleaned.csv'
SAVE_PATH = 'data'
os.makedirs(SAVE_PATH, exist_ok=True)

# ── Atom feature extraction ─────────────────────────────────────────────────
def get_atom_features(atom):
    """
    Extract numerical features for a single atom.
    These become the node features in the molecular graph.
    
    Features:
    - Atomic number (element identity e.g. C=6, N=7, O=8)
    - Degree (number of bonds)
    - Formal charge
    - Number of hydrogens
    - Is aromatic (True/False as 1/0)
    - Hybridisation (sp, sp2, sp3 etc)
    """
    hybridisation_map = {
        rdchem.HybridizationType.SP:    0,
        rdchem.HybridizationType.SP2:   1,
        rdchem.HybridizationType.SP3:   2,
        rdchem.HybridizationType.SP3D:  3,
        rdchem.HybridizationType.SP3D2: 4,
        rdchem.HybridizationType.OTHER: 5
    }

    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        int(atom.GetIsAromatic()),
        hybridisation_map.get(atom.GetHybridization(), 5)
    ]

# ── Bond feature extraction ─────────────────────────────────────────────────
def get_bond_features(bond):
    """
    Extract numerical features for a single bond.
    These become the edge features in the molecular graph.
    
    Features:
    - Bond type (single=0, double=1, triple=2, aromatic=3)
    - Is in ring
    - Is conjugated
    """
    bond_type_map = {
        rdchem.BondType.SINGLE:    0,
        rdchem.BondType.DOUBLE:    1,
        rdchem.BondType.TRIPLE:    2,
        rdchem.BondType.AROMATIC:  3
    }

    return [
        bond_type_map.get(bond.GetBondType(), 0),
        int(bond.IsInRing()),
        int(bond.GetIsConjugated())
    ]

# ── SMILES to graph conversion ──────────────────────────────────────────────
def smiles_to_graph(smiles, label):
    """
    Convert a SMILES string into a PyTorch Geometric Data object.
    
    A Data object contains:
    - x: node feature matrix (one row per atom)
    - edge_index: connectivity matrix (which atoms are bonded)
    - edge_attr: edge feature matrix (one row per bond)
    - y: target value (pIC50)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features — one row per atom
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    # Edge index and edge features — bonds go both directions in undirected graph
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        features = get_bond_features(bond)

        # Add both directions (undirected graph)
        edge_index += [[i, j], [j, i]]
        edge_attr += [features, features]

    if len(edge_index) == 0:
        return None

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Target value
    y = torch.tensor([label], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.smiles = smiles
    return data

# ── Main pipeline ───────────────────────────────────────────────────────────
def main():
    print("Loading MDM2 dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} compounds")

    print("Converting SMILES to molecular graphs...")
    graphs = []
    failed = 0

    for _, row in df.iterrows():
        graph = smiles_to_graph(row['canonical_smiles'], row['pchembl_value'])
        if graph is not None:
            graphs.append(graph)
        else:
            failed += 1

    print(f"Successfully converted: {len(graphs)} graphs")
    print(f"Failed conversions:     {failed}")

    # Save the graph dataset
    save_file = os.path.join(SAVE_PATH, 'mdm2_graphs.pt')
    torch.save(graphs, save_file)
    print(f"Saved to {save_file}")

    # Quick summary of what we built
    example = graphs[0]
    print(f"\nExample molecule graph:")
    print(f"  Atoms (nodes):     {example.x.shape[0]}")
    print(f"  Atom features:     {example.x.shape[1]}")
    print(f"  Bonds (edges):     {example.edge_index.shape[1] // 2}")
    print(f"  Bond features:     {example.edge_attr.shape[1]}")
    print(f"  Target pIC50:      {example.y.item():.2f}")

if __name__ == "__main__":
    main()