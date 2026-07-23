"""
MDM2 GAT - Evaluation and Comparison
======================================
Evaluates the trained GAT model on the test set and compares
directly against the Random Forest QSAR baseline.
Generates publication-quality plots.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os

from model import MDM2_GAT

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ── Configuration ───────────────────────────────────────────────────────────
GRAPH_PATH      = 'data/mdm2_graphs.pt'
DESCRIPTOR_PATH = r'C:\Users\danny\Downloads\MDM2_QSAR\data\mdm2_descriptors.csv'
MODEL_PATH      = 'checkpoints/best_model.pt'
PLOT_PATH       = 'plots'
os.makedirs(PLOT_PATH, exist_ok=True)

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
HIDDEN     = 64
HEADS      = 4
DROPOUT    = 0.2

# ── Load and split data ──────────────────────────────────────────────────────
def get_test_data():
    graphs = torch.load(GRAPH_PATH, weights_only=False)
    _, temp_data = train_test_split(graphs, test_size=0.3, random_state=42)
    _, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    return test_data

# ── GAT predictions ──────────────────────────────────────────────────────────
def get_gat_predictions(test_data):
    model = MDM2_GAT(num_node_features=6, hidden_channels=HIDDEN,
                     num_heads=HEADS, dropout=DROPOUT).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    preds, labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.batch)
            preds.extend(out.squeeze().cpu().numpy())
            labels.extend(batch.y.cpu().numpy())

    return np.array(labels), np.array(preds)

# ── Random Forest baseline ───────────────────────────────────────────────────
def get_rf_predictions():
    print("Training Random Forest baseline...")
    df = pd.read_csv(DESCRIPTOR_PATH)
    X = df.drop(columns=['molecule_chembl_id', 'pchembl_value'])
    y = df['pchembl_value']

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    X = X.loc[:, X.max() < 1e10]

    from sklearn.model_selection import train_test_split as tts
    X_train, X_temp, y_train, y_temp = tts(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = tts(X_temp, y_temp, test_size=0.5, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    return y_test.values, rf_preds

# ── Metrics ──────────────────────────────────────────────────────────────────
def print_metrics(name, labels, preds):
    r2   = r2_score(labels, preds)
    rmse = np.sqrt(mean_squared_error(labels, preds))
    mae  = mean_absolute_error(labels, preds)
    print(f"{name}:")
    print(f"  R²:   {r2:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE:  {mae:.3f}")
    return r2, rmse, mae

# ── Plots ────────────────────────────────────────────────────────────────────
def plot_comparison(gat_labels, gat_preds, rf_labels, rf_preds):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # GAT plot
    axes[0].scatter(gat_labels, gat_preds, alpha=0.4, color='steelblue', s=15)
    axes[0].plot([4, 11], [4, 11], 'r--', label='Perfect prediction')
    r2_gat = r2_score(gat_labels, gat_preds)
    axes[0].set_title(f'Graph Attention Network\nR² = {r2_gat:.3f}')
    axes[0].set_xlabel('Actual pIC50')
    axes[0].set_ylabel('Predicted pIC50')
    axes[0].legend()

    # RF plot
    axes[1].scatter(rf_labels, rf_preds, alpha=0.4, color='darkorange', s=15)
    axes[1].plot([4, 11], [4, 11], 'r--', label='Perfect prediction')
    r2_rf = r2_score(rf_labels, rf_preds)
    axes[1].set_title(f'Random Forest (Baseline)\nR² = {r2_rf:.3f}')
    axes[1].set_xlabel('Actual pIC50')
    axes[1].set_ylabel('Predicted pIC50')
    axes[1].legend()

    plt.suptitle('MDM2 Inhibitor Potency Prediction: GAT vs Random Forest',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_PATH}/gat_vs_rf_comparison.png', dpi=300)
    plt.show()
    print(f"Saved: {PLOT_PATH}/gat_vs_rf_comparison.png")

def plot_bar_comparison(metrics):
    models = ['GAT', 'Random Forest']
    r2_scores = [metrics['gat']['r2'], metrics['rf']['r2']]
    rmse_scores = [metrics['gat']['rmse'], metrics['rf']['rmse']]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # R² comparison
    bars1 = axes[0].bar(models, r2_scores,
                        color=['steelblue', 'darkorange'], edgecolor='black')
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('R²')
    axes[0].set_title('R² Score Comparison')
    for bar, val in zip(bars1, r2_scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', fontweight='bold')

    # RMSE comparison
    bars2 = axes[1].bar(models, rmse_scores,
                        color=['steelblue', 'darkorange'], edgecolor='black')
    axes[1].set_ylabel('RMSE (pIC50 units)')
    axes[1].set_title('RMSE Comparison')
    for bar, val in zip(bars2, rmse_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', fontweight='bold')

    plt.suptitle('MDM2 Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_PATH}/model_comparison_bars.png', dpi=300)
    plt.show()
    print(f"Saved: {PLOT_PATH}/model_comparison_bars.png")

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("="*55)
    print("MDM2 GAT vs Random Forest Evaluation")
    print("="*55)

    # GAT evaluation
    print("\nEvaluating GAT model...")
    test_data = get_test_data()
    gat_labels, gat_preds = get_gat_predictions(test_data)
    gat_r2, gat_rmse, gat_mae = print_metrics("\nGAT", gat_labels, gat_preds)

    # Random Forest evaluation
    print("\nEvaluating Random Forest baseline...")
    rf_labels, rf_preds = get_rf_predictions()
    rf_r2, rf_rmse, rf_mae = print_metrics("\nRandom Forest", rf_labels, rf_preds)

    # Summary
    print(f"\n{'='*55}")
    print(f"SUMMARY")
    print(f"{'='*55}")
    print(f"{'Model':<20} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
    print(f"{'-'*44}")
    print(f"{'GAT':<20} {gat_r2:>8.3f} {gat_rmse:>8.3f} {gat_mae:>8.3f}")
    print(f"{'Random Forest':<20} {rf_r2:>8.3f} {rf_rmse:>8.3f} {rf_mae:>8.3f}")

    metrics = {
        'gat': {'r2': gat_r2, 'rmse': gat_rmse},
        'rf':  {'r2': rf_r2,  'rmse': rf_rmse}
    }

    # Plots
    print("\nGenerating comparison plots...")
    plot_comparison(gat_labels, gat_preds, rf_labels, rf_preds)
    plot_bar_comparison(metrics)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()