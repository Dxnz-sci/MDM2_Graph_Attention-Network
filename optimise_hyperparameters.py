"""
MDM2 GAT - Hyperparameter Optimisation
=========================================
Uses Optuna to search hidden_channels, learning rate, dropout and number of
attention heads, instead of relying on the hand-picked defaults in train.py.

Each trial trains a fresh MDM2_GAT with early stopping on validation loss
(fewer max epochs than the final training run, since this only needs to
rank configurations, not fully converge each one) and reports val RMSE.
Optuna's median pruner stops clearly unpromising trials early.

At the end, the best hyperparameters are printed and saved to
checkpoints/best_hyperparameters.json — train.py's config constants can be
updated with these values for the final full training run.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import json
import random

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from model import MDM2_GAT

# ── Configuration ────────────────────────────────────────────────────────────
SEED = 42
DATA_FILE = os.path.join('data', 'mdm2_graphs.pt')
CHECKPOINT_DIR = 'checkpoints'
RESULTS_PATH = os.path.join(CHECKPOINT_DIR, 'best_hyperparameters.json')

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1

N_TRIALS = 8
MAX_EPOCHS_PER_TRIAL = 15   # shorter than train.py's 300 — just enough to rank configs
PATIENCE = 5

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
    return avg_loss, rmse


def objective(trial, train_set, val_set, num_node_features):
    hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128])
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = MDM2_GAT(
        num_node_features=num_node_features,
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        dropout=dropout,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS_PER_TRIAL + 1):
        run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_rmse = run_epoch(model, val_loader, criterion)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if epochs_without_improvement >= PATIENCE:
            break

    return best_val_loss


def main():
    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Loading graphs from {DATA_FILE}...")
    graphs = torch.load(DATA_FILE, weights_only=False)
    print(f"Loaded {len(graphs)} molecular graphs")

    train_set, val_set, _ = split_dataset(graphs, TRAIN_FRAC, VAL_FRAC)
    num_node_features = train_set[0].x.shape[1]

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    print(f"\nRunning {N_TRIALS} Optuna trials...\n")
    study.optimize(
        lambda trial: objective(trial, train_set, val_set, num_node_features),
        n_trials=N_TRIALS,
    )

    print("\nBest trial:")
    print(f"  Val MSE loss: {study.best_value:.4f}")
    print("  Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    with open(RESULTS_PATH, 'w') as f:
        json.dump({
            'best_val_loss': study.best_value,
            'best_params': study.best_params,
            'n_trials': N_TRIALS,
        }, f, indent=2)
    print(f"\nSaved best hyperparameters to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
