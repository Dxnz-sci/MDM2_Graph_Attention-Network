"""
visualise.py — MDM2 GAT Attention Weight Visualisation
Extracts attention weights from trained GAT model and highlights
pharmacophore hotspots on real MDM2 inhibitor molecules.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.colorbar as colorbar
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io

from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from model import MDM2_GAT

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH   = "checkpoints/best_model.pt"
DATA_PATH    = "data/mdm2_graphs.pt"
OUTPUT_DIR   = "visualisations"
N_MOLECULES  = 6      # how many molecules to visualise
IMG_SIZE     = (600, 500)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
graphs = torch.load(DATA_PATH, weights_only=False)

# Same 70/15/15 split as evaluate.py, so "test" here matches the held-out test set
_, temp_data = train_test_split(graphs, test_size=0.3, random_state=42)
_, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
print(f"Test set size: {len(test_data)}")

# ── Load model ───────────────────────────────────────────────────────────────
print("Loading model...")
# Infer input dimension from first graph
sample = test_data[0]
in_channels  = sample.x.shape[1]
out_channels = 64
heads        = 4

model = MDM2_GAT(num_node_features=in_channels, hidden_channels=out_channels, num_heads=heads, dropout=0.2)
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
print("Model loaded.")

# ── Attention extraction hook ─────────────────────────────────────────────────
attention_weights_store = {}

def make_hook(layer_name):
    """Returns a forward hook that captures attention weights."""
    def hook(module, input, output):
        # PyG GAT returns (out, (edge_index, alpha)) when return_attention_weights=True
        # We capture alpha here via the hook on the conv layer output
        attention_weights_store[layer_name] = output
    return hook

# ── Run model and extract attention ──────────────────────────────────────────
def get_attention_weights(graph_data):
    """
    Run a single molecule through the GAT and extract per-atom attention scores.
    Returns: atom_scores (np.array, shape [num_atoms])
    """
    graph_data = graph_data.to(device)

    with torch.no_grad():
        # We need to modify the forward pass to return attention weights
        # Use PyG's built-in return_attention_weights on the conv layers
        x = graph_data.x
        edge_index = graph_data.edge_index

        # Access the GAT conv layers directly
        # Assumes model has .conv1, .conv2, .conv3 — adjust if your model differs
        layer_attentions = []

        # Layer 1
        out1, (ei1, alpha1) = model.conv1(x, edge_index, return_attention_weights=True)
        out1 = torch.relu(out1)
        layer_attentions.append((ei1, alpha1))

        # Layer 2
        out2, (ei2, alpha2) = model.conv2(out1, edge_index, return_attention_weights=True)
        out2 = torch.relu(out2)
        layer_attentions.append((ei2, alpha2))

        # Layer 3 (final conv)
        out3, (ei3, alpha3) = model.conv3(out2, edge_index, return_attention_weights=True)
        layer_attentions.append((ei3, alpha3))

    # Aggregate attention: for each atom, sum attention weights from all incoming edges
    # across all layers and all heads — gives scalar importance per atom
    num_atoms = graph_data.num_nodes
    atom_scores = np.zeros(num_atoms)

    for edge_index_l, alpha_l in layer_attentions:
        # alpha_l shape: [num_edges, num_heads]
        # NOTE: alpha is softmax-normalised per destination node, so grouping by
        # destination always sums to 1 per head — no information there.
        # Group by source node instead: how much attention each atom receives
        # from its neighbours as they aggregate messages.
        alpha_mean = alpha_l.abs().mean(dim=1).cpu().numpy()  # mean across heads
        source_nodes = edge_index_l[0].cpu().numpy()
        for i, src in enumerate(source_nodes):
            atom_scores[src] += alpha_mean[i]

    # Normalise to [0, 1]
    if atom_scores.max() > 0:
        atom_scores = atom_scores / atom_scores.max()

    return atom_scores


# ── Colour atoms by attention score ──────────────────────────────────────────
def scores_to_colours(atom_scores):
    """Map attention scores to RGB colours using a hot colourmap."""
    cmap = cm.get_cmap("RdYlBu_r")  # blue=low, red=high attention
    norm = Normalize(vmin=0, vmax=1)
    colours = {}
    for i, score in enumerate(atom_scores):
        r, g, b, _ = cmap(norm(score))
        colours[i] = (r, g, b)
    return colours


# ── Draw molecule with attention overlay ─────────────────────────────────────
def draw_molecule_attention(smiles, atom_scores, title="", predicted_pIC50=None, true_pIC50=None):
    """
    Draw molecule coloured by attention weights.
    Returns PIL Image.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Could not parse SMILES: {smiles}")
        return None

    colours = scores_to_colours(atom_scores)

    # Atom highlight colours and radii
    atom_highlight = {i: colours[i] for i in range(mol.GetNumAtoms())}
    atom_radii     = {i: 0.4 for i in range(mol.GetNumAtoms())}

    drawer = rdMolDraw2D.MolDraw2DSVG(IMG_SIZE[0], IMG_SIZE[1])
    drawer.drawOptions().addStereoAnnotation = False

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=list(atom_highlight.keys()),
        highlightAtomColors=atom_highlight,
        highlightAtomRadii=atom_radii,
        highlightBonds=[],
        highlightBondColors={}
    )
    drawer.FinishDrawing()

    svg = drawer.GetDrawingText()
    # Convert SVG → PNG via cairosvg if available, else save as SVG
    try:
        import cairosvg
        png_data = cairosvg.svg2png(bytestring=svg.encode(), output_width=IMG_SIZE[0], output_height=IMG_SIZE[1])
        img = Image.open(io.BytesIO(png_data))
    except ImportError:
        # Fallback: save SVG string directly
        return svg, "svg"

    return img, "png"


# ── Build figure with colorbar ────────────────────────────────────────────────
def make_figure(molecules_data):
    """
    molecules_data: list of dicts with keys: smiles, atom_scores, pred_pIC50, true_pIC50
    Saves a grid figure to OUTPUT_DIR.
    """
    n = len(molecules_data)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5.5))
    axes = np.array(axes).flatten()

    for ax in axes:
        ax.axis("off")

    for idx, mol_data in enumerate(molecules_data):
        smiles      = mol_data["smiles"]
        atom_scores = mol_data["atom_scores"]
        pred        = mol_data["pred_pIC50"]
        true        = mol_data["true_pIC50"]

        result = draw_molecule_attention(smiles, atom_scores)
        if result is None:
            continue
        img, fmt = result

        if fmt == "png":
            axes[idx].imshow(img)
        else:
            axes[idx].text(0.5, 0.5, "SVG (install cairosvg for PNG)", ha="center", va="center")

        axes[idx].set_title(
            f"Pred pIC50: {pred:.2f}  |  True pIC50: {true:.2f}",
            fontsize=11, pad=6
        )
        axes[idx].axis("off")

    # Colorbar
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm  = cm.ScalarMappable(cmap="RdYlBu_r", norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_label("Attention weight (normalised)", fontsize=12)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["Low", "", "Medium", "", "High"])

    fig.suptitle(
        "MDM2 GAT — Atom-Level Attention Weights\n(pharmacophore hotspot visualisation)",
        fontsize=14, fontweight="bold", y=1.01
    )

    out_path = os.path.join(OUTPUT_DIR, "attention_map.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Pick N_MOLECULES from the test set
    # Sort by highest predicted pIC50 to get the most interesting compounds
    loader = DataLoader(test_data, batch_size=1, shuffle=False)

    molecules_data = []

    for graph in loader:
        if len(molecules_data) >= N_MOLECULES:
            break

        # Each graph needs a .smiles attribute
        if not hasattr(graph, "smiles"):
            print("Warning: graph missing .smiles — add SMILES to Data object in data_preparation.py")
            continue

        smiles = graph.smiles[0] if isinstance(graph.smiles, list) else graph.smiles

        atom_scores = get_attention_weights(graph)

        # Get prediction
        graph_device = graph.to(device)
        with torch.no_grad():
            pred = model(graph_device.x, graph_device.edge_index, graph_device.batch).item()

        true = graph.y.item()

        molecules_data.append({
            "smiles":      smiles,
            "atom_scores": atom_scores,
            "pred_pIC50":  pred,
            "true_pIC50":  true,
        })
        print(f"Processed molecule {len(molecules_data)}: pred={pred:.3f}, true={true:.3f}")

    if not molecules_data:
        print("No molecules processed. Check that .smiles is stored in your Data objects.")
        return

    make_figure(molecules_data)
    print(f"\nDone. Open visualisations/attention_map.png")


if __name__ == "__main__":
    main()