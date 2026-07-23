# MDM2 Graph Attention Network (GAT)

A Graph Attention Network for predicting MDM2 inhibitor potency (pIC50) with atom-level attention visualisation.

Companion project to the MDM2-QSAR pipeline — compares classical ML (Random Forest) against deep learning (GAT) on the same dataset.

## Results

| Model | R² | RMSE |
|-------|-----|------|
| Graph Attention Network | 0.655 | 0.776 |
| Random Forest (baseline) | 0.700 | 0.724 |

Random Forest outperforms GAT on this dataset — consistent with published literature showing classical ML often outperforms deep learning on small molecular datasets (<10k compounds).

## Attention Visualisation

The GAT model produces atom-level attention weights showing which atoms drive the potency prediction. Red atoms received high attention, blue atoms low attention — effectively highlighting the model's learned pharmacophore.

![Attention visualisation](visualisations/attention_map.png)

## Pipeline

1. `data_preparation.py` — converts SMILES to molecular graphs (nodes=atoms, edges=bonds)
2. `model.py` — 3-layer GAT architecture, 4 attention heads, 137k parameters
3. `train.py` — training with early stopping and learning rate scheduling
4. `evaluate.py` — GAT vs Random Forest comparison
5. `visualise.py` — attention weight extraction and pharmacophore visualisation

## Repo structure

```
data_preparation.py     SMILES -> molecular graphs (data/mdm2_graphs.pt)
model.py                MDM2_GAT architecture
train.py                training loop, saves checkpoints/best_model.pt
evaluate.py             GAT vs Random Forest metrics + comparison plots
visualise.py            attention heatmaps -> visualisations/attention_map.png
checkpoints/            saved model weights
data/                   processed graph dataset
visualisations/         generated attention maps
```

## Dataset

MDM2 bioactivity data from ChEMBL (4,146 compounds, IC50 values converted to pIC50)

## Environment

```bash
conda create -n qsar python=3.10
conda activate qsar
conda install -c conda-forge rdkit cairo
pip install torch torch-geometric chembl-webresource-client scikit-learn pandas numpy matplotlib scipy cairosvg
```

`cairo` (conda) + `cairosvg` (pip) are required for `visualise.py` to render actual molecule structures rather than placeholder text.

## Key Finding

On the MDM2 dataset (4,146 compounds), Random Forest with RDKit descriptors slightly outperformed the GAT (R²=0.700 vs 0.655). The GAT's advantage lies in its interpretability — attention weights provide atom-level insight into which molecular features drive binding affinity predictions, with no feature engineering required.

## Author

Daniel Szolc | MSci Pharmaceutical Science | University of Salford
