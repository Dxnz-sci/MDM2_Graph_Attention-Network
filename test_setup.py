import torch
import torch_geometric
from rdkit import Chem
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch Geometric version: {torch_geometric.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("RDKit imported successfully")
print("\nAll packages ready for GNN project!")