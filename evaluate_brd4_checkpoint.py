"""
Evaluates whatever is currently saved at checkpoints/brd4_best_model.pt on the
BRD4 test split, independent of train_brd4.py's own run loop — lets the
long-running training process be checked/stopped without losing final metrics.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import random

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from model import MDM2_GAT

DATA_FILE = os.path.join('data', 'brd4_graphs.pt')
CHECKPOINT_PATH = os.path.join('checkpoints', 'brd4_best_model.pt')
SEED = 42
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def split_dataset(graphs, train_frac, val_frac):
    indices = list(range(len(graphs)))
    random.shuffle(indices)
    n_train = int(len(indices) * train_frac)
    n_val = int(len(indices) * val_frac)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return [graphs[i] for i in train_idx], [graphs[i] for i in val_idx], [graphs[i] for i in test_idx]


def main():
    random.seed(SEED)
    graphs = torch.load(DATA_FILE, weights_only=False)
    _, _, test_set = split_dataset(graphs, TRAIN_FRAC, VAL_FRAC)
    loader = DataLoader(test_set, batch_size=32, shuffle=False)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model = MDM2_GAT(
        num_node_features=checkpoint['num_node_features'],
        hidden_channels=checkpoint['hidden_channels'],
        num_heads=checkpoint['num_heads'],
        dropout=checkpoint['dropout'],
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Checkpoint from epoch {checkpoint['epoch']}, val_r2={checkpoint['val_r2']:.4f}")

    criterion = nn.MSELoss()
    all_preds, all_targets = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.batch).view(-1)
            loss = criterion(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
            all_preds.append(out.cpu())
            all_targets.append(batch.y.cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae = float(np.mean(np.abs(preds - targets)))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = float(1 - ss_res / ss_tot)

    print(f"\nBRD4 test set ({len(test_set)} compounds):")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R2:   {r2:.4f}")


if __name__ == "__main__":
    main()
