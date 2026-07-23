"""
BRD4 GAT - Training Script
============================
Trains the same MDM2_GAT architecture on BRD4 bioactivity graphs
(data/brd4_graphs.pt from brd4_data_preparation.py), to demonstrate the
pipeline generalises beyond the original MDM2 dataset.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import random

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from model import MDM2_GAT

# ── Configuration ────────────────────────────────────────────────────────────
SEED = 42
DATA_FILE = os.path.join('data', 'brd4_graphs.pt')
CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'brd4_best_model.pt')
PLOT_PATH = os.path.join(CHECKPOINT_DIR, 'brd4_loss_curve.png')

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1

BATCH_SIZE = 32
HIDDEN_CHANNELS = 64
NUM_HEADS = 4
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 300
PATIENCE = 30

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_dataset(graphs, train_frac, val_frac):
    indices = list(range(len(graphs)))
    random.shuffle(indices)

    n_train = int(len(indices) * train_frac)
    n_val = int(len(indices) * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return (
        [graphs[i] for i in train_idx],
        [graphs[i] for i in val_idx],
        [graphs[i] for i in test_idx],
    )


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(DEVICE)

        with torch.set_grad_enabled(is_train):
            out = model(batch.x, batch.edge_index, batch.batch).view(-1)
            loss = criterion(out, batch.y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        all_preds.append(out.detach().cpu())
        all_targets.append(batch.y.detach().cpu())

    avg_loss = total_loss / len(loader.dataset)
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae = float(np.mean(np.abs(preds - targets)))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

    return avg_loss, rmse, mae, r2


def main():
    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")

    print(f"Loading graphs from {DATA_FILE}...")
    graphs = torch.load(DATA_FILE, weights_only=False)
    print(f"Loaded {len(graphs)} molecular graphs")

    train_set, val_set, test_set = split_dataset(graphs, TRAIN_FRAC, VAL_FRAC)
    print(f"Split: {len(train_set)} train / {len(val_set)} val / {len(test_set)} test")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    num_node_features = train_set[0].x.shape[1]
    model = MDM2_GAT(
        num_node_features=num_node_features,
        hidden_channels=HIDDEN_CHANNELS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    ).to(DEVICE)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    print("\nStarting training...\n")
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_rmse, train_mae, train_r2 = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_rmse, val_mae, val_r2 = run_epoch(model, val_loader, criterion)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_rmse': val_rmse,
                'val_r2': val_r2,
                'num_node_features': num_node_features,
                'hidden_channels': HIDDEN_CHANNELS,
                'num_heads': NUM_HEADS,
                'dropout': DROPOUT,
            }, CHECKPOINT_PATH)
        else:
            epochs_without_improvement += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d} | "
                f"train loss {train_loss:.4f} (rmse {train_rmse:.4f}, r2 {train_r2:.3f}) | "
                f"val loss {val_loss:.4f} (rmse {val_rmse:.4f}, r2 {val_r2:.3f})"
            )

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs)")
            break

    print(f"\nBest val loss: {best_val_loss:.4f}, saved to {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_rmse, test_mae, test_r2 = run_epoch(model, test_loader, criterion)
    print("\nBRD4 test set performance:")
    print(f"  Loss (MSE): {test_loss:.4f}")
    print(f"  RMSE:       {test_rmse:.4f}")
    print(f"  MAE:        {test_mae:.4f}")
    print(f"  R2:         {test_r2:.4f}")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label='Train loss')
        plt.plot(val_losses, label='Val loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('BRD4 GAT Training Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_PATH)
        print(f"\nSaved loss curve to {PLOT_PATH}")
    except ImportError:
        print("\nmatplotlib not installed — skipping loss curve plot")


if __name__ == "__main__":
    main()
