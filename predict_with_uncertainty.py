"""
MDM2 GAT - Uncertainty Quantification (MC Dropout)
=====================================================
The trained GAT gives a single point estimate per compound (e.g. pIC50 8.2),
with no indication of how confident that estimate is. This script uses Monte
Carlo Dropout: dropout layers are normally switched off at inference, but
here they are kept active and the model runs N stochastic forward passes per
molecule. The spread across those passes approximates a confidence interval,
turning "predicted pIC50 8.2" into "predicted pIC50 8.2 +/- 0.3" — directly
usable when prioritising compounds for synthesis.

Reference: Gal & Ghahramani (2016), "Dropout as a Bayesian Approximation".
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from model import MDM2_GAT

# ── Configuration ────────────────────────────────────────────────────────────
GRAPH_PATH = os.path.join('data', 'mdm2_graphs.pt')
MODEL_PATH = os.path.join('checkpoints', 'best_model.pt')
N_FORWARD_PASSES = 50   # number of stochastic MC Dropout samples per molecule
N_SHOW = 10             # how many example predictions to print

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def enable_mc_dropout(model):
    """MDM2_GAT applies dropout functionally (F.dropout(..., training=self.training))
    rather than via nn.Dropout submodules, so it's gated by the model's own
    self.training flag. model.train() turns that on, but would also let
    BatchNorm use live batch statistics instead of its running stats — so
    BatchNorm layers are forced back to eval mode afterwards."""
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()


def get_test_data():
    graphs = torch.load(GRAPH_PATH, weights_only=False)
    _, temp_data = train_test_split(graphs, test_size=0.3, random_state=42)
    _, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    return test_data


def mc_dropout_predict(model, graph, n_passes):
    """Run n_passes stochastic forward passes on a single molecule.
    Returns (mean_pred, std_pred)."""
    graph = graph.to(DEVICE)
    preds = []

    with torch.no_grad():
        for _ in range(n_passes):
            out = model(graph.x, graph.edge_index,
                         torch.zeros(graph.x.shape[0], dtype=torch.long, device=DEVICE))
            preds.append(out.item())

    preds = np.array(preds)
    return preds.mean(), preds.std()


def main():
    print(f"Using device: {DEVICE}")

    print(f"Loading test set from {GRAPH_PATH}...")
    test_data = get_test_data()
    print(f"Test set size: {len(test_data)}")

    print(f"Loading model from {MODEL_PATH}...")
    sample = test_data[0]
    model = MDM2_GAT(num_node_features=sample.x.shape[1])
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    enable_mc_dropout(model)
    print(f"Model loaded - running {N_FORWARD_PASSES} MC Dropout passes per molecule.\n")

    means, stds, trues = [], [], []
    for graph in test_data:
        mean_pred, std_pred = mc_dropout_predict(model, graph, N_FORWARD_PASSES)
        means.append(mean_pred)
        stds.append(std_pred)
        trues.append(graph.y.item())

    means = np.array(means)
    stds = np.array(stds)
    trues = np.array(trues)

    r2 = r2_score(trues, means)
    rmse = float(np.sqrt(np.mean((means - trues) ** 2)))

    # 95% CI = mean +/- 1.96 * std (assumes an approximately Gaussian
    # posterior over predictions, which MC Dropout does not guarantee, but is
    # the standard convention used to report these intervals)
    within_95ci = np.mean(np.abs(means - trues) <= 1.96 * stds)

    print("MC Dropout results on test set:")
    print(f"  R2 (mean prediction):        {r2:.4f}")
    print(f"  RMSE (mean prediction):      {rmse:.4f}")
    print(f"  Mean predicted std:          {stds.mean():.4f}")
    print(f"  True values within 95% CI:   {within_95ci * 100:.1f}%")

    print(f"\nExample predictions (first {N_SHOW} test compounds):")
    print(f"{'True pIC50':>12} {'Pred pIC50':>14} {'95% CI':>20}")
    for i in range(min(N_SHOW, len(test_data))):
        ci_low = means[i] - 1.96 * stds[i]
        ci_high = means[i] + 1.96 * stds[i]
        print(f"{trues[i]:>12.2f} {means[i]:>9.2f} +/- {stds[i]:<5.2f} [{ci_low:>5.2f}, {ci_high:>5.2f}]")


if __name__ == "__main__":
    main()
