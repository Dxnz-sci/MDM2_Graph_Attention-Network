"""
BRD4 GNN - Data Preparation
============================
Converts BRD4 SMILES strings into molecular graphs, reusing the same
atom/bond featurisation as data_preparation.py (MDM2), so the GAT
architecture and training pipeline generalise unchanged to a second target.
"""

import os
import torch
import pandas as pd

from data_preparation import smiles_to_graph

DATA_PATH = os.path.join('data', 'brd4_cleaned.csv')
SAVE_PATH = 'data'
os.makedirs(SAVE_PATH, exist_ok=True)


def main():
    print("Loading BRD4 dataset...")
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

    save_file = os.path.join(SAVE_PATH, 'brd4_graphs.pt')
    torch.save(graphs, save_file)
    print(f"Saved to {save_file}")

    example = graphs[0]
    print(f"\nExample molecule graph:")
    print(f"  Atoms (nodes):     {example.x.shape[0]}")
    print(f"  Atom features:     {example.x.shape[1]}")
    print(f"  Bonds (edges):     {example.edge_index.shape[1] // 2}")
    print(f"  Target pIC50:      {example.y.item():.2f}")


if __name__ == "__main__":
    main()
