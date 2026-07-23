"""
MDM2 GAT - Model Architecture
==============================
Graph Attention Network for predicting MDM2 inhibitor potency (pIC50).

Architecture:
- 3 Graph Attention layers (message passing between atoms)
- Global mean pooling (combines atom features into molecule representation)
- 2 fully connected layers (final prediction)
- Attention weights available for visualisation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class MDM2_GAT(nn.Module):
    """
    Graph Attention Network for MDM2 inhibitor potency prediction.
    
    Parameters
    ----------
    num_node_features : int
        Number of atom features (6 in our case)
    hidden_channels : int
        Size of hidden layers (controls model capacity)
    num_heads : int
        Number of attention heads (multiple perspectives on each atom)
    dropout : float
        Dropout rate for regularisation
    """

    def __init__(self, num_node_features=6, hidden_channels=64, num_heads=4, dropout=0.2):
        super(MDM2_GAT, self).__init__()

        self.dropout = dropout

        # Graph Attention Layer 1
        # Takes raw atom features, outputs attended representations
        self.conv1 = GATConv(
            in_channels=num_node_features,
            out_channels=hidden_channels,
            heads=num_heads,
            dropout=dropout,
            concat=True  # concatenate all attention heads
        )

        # Graph Attention Layer 2
        # Input size is hidden_channels * num_heads because of concat=True
        self.conv2 = GATConv(
            in_channels=hidden_channels * num_heads,
            out_channels=hidden_channels,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )

        # Graph Attention Layer 3 — final graph layer
        # Average the attention heads rather than concatenating
        self.conv3 = GATConv(
            in_channels=hidden_channels * num_heads,
            out_channels=hidden_channels,
            heads=num_heads,
            dropout=dropout,
            concat=False  # average heads at final layer
        )

        # Fully connected layers for final prediction
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, 1)  # output single pIC50 value

        # Batch normalisation for stable training
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.bn2 = nn.BatchNorm1d(hidden_channels * num_heads)

    def forward(self, x, edge_index, batch, return_attention=False):
        """
        Forward pass through the network.
        
        Parameters
        ----------
        x : tensor
            Node features (atom features)
        edge_index : tensor
            Graph connectivity (which atoms are bonded)
        batch : tensor
            Batch vector (which atoms belong to which molecule)
        return_attention : bool
            If True, also return attention weights for visualisation
        
        Returns
        -------
        out : tensor
            Predicted pIC50 values
        attention_weights : list (only if return_attention=True)
            Attention weights from each layer
        """
        attention_weights = []

        # Layer 1 — first round of message passing
        x, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = self.bn1(x)
        x = F.elu(x)  # ELU activation — smooth version of ReLU
        x = F.dropout(x, p=self.dropout, training=self.training)
        attention_weights.append(attn1)

        # Layer 2 — second round of message passing
        x, attn2 = self.conv2(x, edge_index, return_attention_weights=True)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        attention_weights.append(attn2)

        # Layer 3 — final graph layer
        x, attn3 = self.conv3(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        attention_weights.append(attn3)

        # Global mean pooling — collapses all atom representations into one molecule vector
        # Like averaging all atom "opinions" into a single molecule fingerprint
        x = global_mean_pool(x, batch)

        # Fully connected layers for final prediction
        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.fc2(x)

        if return_attention:
            return out, attention_weights
        return out


def test_model():
    """Quick test to verify the model runs correctly."""
    from torch_geometric.data import Data, Batch

    # Create a fake molecule with 10 atoms and 12 bonds
    num_atoms = 10
    num_bonds = 12

    x = torch.randn(num_atoms, 6)          # 10 atoms, 6 features each
    edge_index = torch.randint(0, num_atoms, (2, num_bonds * 2))  # bonds both directions
    batch = torch.zeros(num_atoms, dtype=torch.long)  # all atoms in same molecule
    y = torch.tensor([7.5])                # target pIC50

    model = MDM2_GAT()
    out = model(x, edge_index, batch)

    print(f"Model architecture:\n{model}")
    print(f"\nInput:  {num_atoms} atoms, {num_bonds} bonds")
    print(f"Output: {out.shape} — predicted pIC50: {out.item():.3f}")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nModel test passed!")


if __name__ == "__main__":
    test_model()