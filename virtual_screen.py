"""
virtual_screen.py — MDM2 GAT Virtual Screening
================================================
Downloads a subset of ZINC compounds sized/lipophilicity-matched to the MDM2
training set, converts them to molecular graphs, runs them through the
trained GAT model, ranks by predicted pIC50, and visualises attention maps
for the top in-domain hits.

Pipeline:
1. Download ~50k SMILES from ZINC15 (tranches matched to training MW/LogP)
2. Convert to molecular graphs, filtering to the training-like property window
3. Predict pIC50 with trained GAT
4. Score applicability domain (similarity to training compounds) and flag hits
   too dissimilar from anything the model was trained on to be trustworthy
5. Rank and export top in-domain candidates
6. Visualise attention maps for top 6 in-domain hits
"""

import torch
import numpy as np
import pandas as pd
import requests
import os
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from PIL import Image
import io

from model import MDM2_GAT
from data_preparation import smiles_to_graph

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "checkpoints/best_model.pt"
OUTPUT_DIR   = "virtual_screening"
N_COMPOUNDS  = 50000   # compounds to screen
TOP_N        = 50      # top candidates to save
TOP_VISUAL   = 6       # top compounds to visualise with attention maps
MIN_TRAIN_SIMILARITY = 0.35  # applicability-domain cutoff (Tanimoto, Morgan r=2)
BATCH_SIZE   = 256
IMG_SIZE     = (600, 500)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    print("Loading trained GAT model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    model = MDM2_GAT(
        num_node_features=checkpoint["num_node_features"],
        hidden_channels=checkpoint["hidden_channels"],
        num_heads=checkpoint["num_heads"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print("Model loaded.")
    return model

# ── Download ZINC compounds ───────────────────────────────────────────────────
def download_zinc_smiles(n_compounds=50000):
    """
    Download drug-like SMILES from ZINC15 using their free tranches API.
    Drug-like = MW 250-500, LogP 0-5 (Goldilocks zone for oral drugs).
    """
    zinc_file = os.path.join(OUTPUT_DIR, "zinc_compounds.smi")

    if os.path.exists(zinc_file):
        print(f"ZINC file already exists, loading from disk...")
        df = pd.read_csv(zinc_file, sep="\t", header=None, names=["smiles", "zinc_id"])
        print(f"Loaded {len(df)} compounds from disk.")
        return df

    print("Downloading ZINC15 drug-like compounds...")
    print("This may take a minute...")

    # ZINC15 flat-file mirror (files.docking.org) — serves tranche .smi files
    # directly, unlike the interactive zinc15.docking.org site which now sits
    # behind a bot-verification challenge page.
    # Tranche codes are <MW letter><LogP letter> (A=smallest/least lipophilic,
    # K=largest/most lipophilic). The MDM2 training set (data/mdm2_graphs.pt)
    # has median MW ~558 (p10-p90: 482-657) and median LogP ~5.7 (p10-p90:
    # 4.3-7.0) — these are large, lipophilic PPI-inhibitor-like compounds,
    # not classic Lipinski-compliant oral drugs. Empirically: HH~(410,3.8),
    # II~(437,4.3), JJ~(484,4.7), KK~(538,5.9) — bins H-K are the closest
    # available match to the training distribution.
    mw_letters   = "HIJK"
    logp_letters = "GHIJK"
    zinc_urls = [
        f"https://files.docking.org/2D/{mw}{logp}/{mw}{logp}AA.smi"
        for mw in mw_letters
        for logp in logp_letters
    ]

    all_smiles = []
    headers = {"User-Agent": "Mozilla/5.0 (academic research)"}

    for url in zinc_urls:
        if len(all_smiles) >= n_compounds:
            break
        try:
            response = requests.get(url, headers=headers, timeout=60)
            if response.status_code == 200:
                lines = response.text.strip().split("\n")
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    smiles = parts[0]
                    zinc_id = parts[1] if len(parts) >= 2 else "ZINC_unknown"
                    if Chem.MolFromSmiles(smiles) is None:
                        continue  # skip non-SMILES junk (e.g. HTML/JS bot-check pages)
                    all_smiles.append({"smiles": smiles, "zinc_id": zinc_id})
                print(f"  Downloaded {len(all_smiles)} compounds so far...")
            else:
                print(f"  Warning: HTTP {response.status_code} for URL, trying next...")
        except Exception as e:
            print(f"  Warning: {e}, trying next URL...")
        time.sleep(1)  # be polite to ZINC servers

    # Fallback: if ZINC download fails, use a curated set of known drug-like scaffolds
    if len(all_smiles) < 100:
        print("ZINC download returned few results, using fallback compound set...")
        all_smiles = get_fallback_compounds()

    df = pd.DataFrame(all_smiles[:n_compounds])
    df.to_csv(zinc_file, sep="\t", index=False, header=False)
    print(f"Saved {len(df)} compounds to {zinc_file}")
    return df

def get_fallback_compounds():
    """
    Curated set of drug-like SMILES for testing if ZINC download fails.
    Includes known MDM2 inhibitor scaffolds and diverse drug-like molecules.
    """
    smiles_list = [
        # Known MDM2 inhibitor-like scaffolds
        "CC1=CC=C(C=C1)C2=CC(=O)NC(=O)N2",
        "O=C1NC(=O)C(=C1)C2=CC=CC=C2",
        "CC(C)(C)C1=CC=C(C=C1)C2=CC=CC=N2",
        "ClC1=CC=C(C=C1)C2=CC=NC=C2",
        "FC1=CC=C(C=C1)C2=CN=CC=C2",
        "CC1=CN=CC=C1C2=CC=C(Cl)C=C2",
        "O=C(O)C1=CC=C(C=C1)NC2=NC=CS2",
        "CC1=CC=C(C=C1)NC2=NC=CO2",
        "O=C(O)C1=CC=CC=C1NC2=NN=CS2",
        "CC(=O)NC1=CC=C(C=C1)C2=CC=CN=C2",
        # Diverse drug-like molecules
        "CC1=CC2=C(C=C1)N=C(S2)NC3=CC=CC=C3",
        "O=C1NC2=CC=CC=C2C(=O)N1CC3=CC=CO3",
        "CC1=CC=C(C=C1)S(=O)(=O)N2CCCC2",
        "O=C(NC1=CC=CC=C1)C2=CC=C(O)C=C2",
        "CC1=CC=C(C=C1)C(=O)NC2=CC=CC=N2",
        "FC1=CC=C(C=C1)C(=O)NCC2=CC=CC=C2",
        "ClC1=CC=C(C=C1)NC(=O)C2=CC=CO2",
        "O=C(O)C1=CN=CC=C1NC2=CC=CC=C2",
        "CC(C)NC1=NC=CS1",
        "O=C1NC(=S)NC1=O",
        "CC1=CC=C(C=C1)NC(=O)CN2CCOCC2",
        "O=C(NC1CCCCC1)C2=CC=CO2",
        "CC1CCN(CC1)C(=O)C2=CC=CS2",
        "O=C(NCC1=CC=CO1)C2=CC=CC=C2",
        "CC(=O)N1CCN(CC1)C2=CC=CC=C2",
        "FC1=CC=C(C=C1)C(=O)N2CCCC2",
        "ClC1=CC=C(C=C1)C(=O)N2CCOCC2",
        "O=C(N1CCCC1)C2=CC=CC=N2",
        "CC1=CC=NC=C1C(=O)NCC2=CC=CO2",
        "O=C(NC1=CC=CO1)C2=CC=NC=C2",
    ]
    return [{"smiles": s, "zinc_id": f"FALLBACK_{i:04d}"} for i, s in enumerate(smiles_list)]

# ── Property filter ───────────────────────────────────────────────────────────
# Classic Lipinski Ro5 (MW<=500, LogP<=5) is the wrong filter for this model:
# the MDM2 training set is dominated by large, lipophilic PPI-inhibitor-like
# compounds (p10-p90 MW 482-657, LogP 4.3-7.0) that mostly *violate* Ro5.
# Screening Ro5-compliant candidates pushes the model into chemical space it
# never saw, producing unreliable, spuriously high extrapolated predictions.
# These bounds instead track the training distribution's ~p5-p95 range.
MW_MIN, MW_MAX     = 400.0, 700.0
LOGP_MIN, LOGP_MAX = 3.0, 7.5

def passes_property_filter(smiles):
    """Keep only compounds whose MW/LogP resemble the MDM2 training set."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)

    return MW_MIN <= mw <= MW_MAX and LOGP_MIN <= logp <= LOGP_MAX

# ── Convert ZINC SMILES to graphs ─────────────────────────────────────────────
def prepare_screening_graphs(df):
    """Convert SMILES to PyG graphs, filtering invalid and out-of-range compounds."""
    print(f"Converting {len(df)} SMILES to molecular graphs...")
    print(f"Applying property filter (MW {MW_MIN}-{MW_MAX}, LogP {LOGP_MIN}-{LOGP_MAX})...")

    graphs      = []
    failed      = 0
    off_profile = 0
    zinc_ids    = []

    for _, row in df.iterrows():
        smiles   = str(row["smiles"]).strip()
        zinc_id  = str(row["zinc_id"]).strip()

        if not passes_property_filter(smiles):
            off_profile += 1
            continue

        graph = smiles_to_graph(smiles, label=0.0)  # dummy label for screening
        if graph is not None:
            graphs.append(graph)
            zinc_ids.append(zinc_id)
        else:
            failed += 1

    print(f"  Total input:          {len(df)}")
    print(f"  Failed property filter: {off_profile}")
    print(f"  Failed graph build:   {failed}")
    print(f"  Screened:             {len(graphs)}")
    return graphs, zinc_ids

# ── Applicability domain ──────────────────────────────────────────────────────
def compute_train_fingerprints():
    """Morgan fingerprints for the MDM2 training set, used as an applicability-
    domain reference: predictions on candidates far (in fingerprint space) from
    every training compound are extrapolations and should be treated with
    suspicion regardless of how confident the model looks."""
    from rdkit.Chem import AllChem

    graphs = torch.load("data/mdm2_graphs.pt", weights_only=False)
    fps = []
    for g in graphs:
        if not hasattr(g, "smiles"):
            continue
        mol = Chem.MolFromSmiles(g.smiles)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
    return fps

def max_train_similarity(smiles, train_fps):
    """Highest Tanimoto similarity between this compound and any training compound."""
    from rdkit.Chem import AllChem
    from rdkit import DataStructs

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
    return max(sims) if sims else 0.0

# ── Run virtual screen ────────────────────────────────────────────────────────
def run_screening(model, graphs, zinc_ids):
    """Run all graphs through the model and collect predictions."""
    print(f"\nRunning virtual screen on {len(graphs)} compounds...")

    loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch.x, batch.edge_index, batch.batch)
            all_preds.extend(preds.cpu().numpy().flatten().tolist())

    results = pd.DataFrame({
        "zinc_id":        zinc_ids[:len(all_preds)],
        "smiles":         [g.smiles for g in graphs[:len(all_preds)]],
        "predicted_pIC50": all_preds
    })

    print("Scoring applicability domain (similarity to training compounds)...")
    train_fps = compute_train_fingerprints()
    results["max_train_similarity"] = [
        max_train_similarity(s, train_fps) for s in results["smiles"]
    ]
    results["in_domain"] = results["max_train_similarity"] >= MIN_TRAIN_SIMILARITY

    results = results.sort_values("predicted_pIC50", ascending=False).reset_index(drop=True)
    results["rank"] = results.index + 1

    n_in_domain = int(results["in_domain"].sum())
    print(f"Screening complete.")
    print(f"Top predicted pIC50: {results['predicted_pIC50'].iloc[0]:.3f}")
    print(f"Mean predicted pIC50: {results['predicted_pIC50'].mean():.3f}")
    print(f"In applicability domain (similarity >= {MIN_TRAIN_SIMILARITY}): {n_in_domain}/{len(results)}")

    return results

# ── Attention extraction ──────────────────────────────────────────────────────
def get_attention_weights(model, graph_data):
    """Extract per-atom attention scores from all GAT layers."""
    graph_data = graph_data.to(device)

    with torch.no_grad():
        x          = graph_data.x
        edge_index = graph_data.edge_index

        layer_attentions = []

        out1, (ei1, alpha1) = model.conv1(x, edge_index, return_attention_weights=True)
        out1 = torch.relu(out1)
        layer_attentions.append((ei1, alpha1))

        out2, (ei2, alpha2) = model.conv2(out1, edge_index, return_attention_weights=True)
        out2 = torch.relu(out2)
        layer_attentions.append((ei2, alpha2))

        out3, (ei3, alpha3) = model.conv3(out2, edge_index, return_attention_weights=True)
        layer_attentions.append((ei3, alpha3))

    num_atoms   = graph_data.num_nodes
    atom_scores = np.zeros(num_atoms)

    for edge_index_l, alpha_l in layer_attentions:
        # alpha is softmax-normalised per destination node, so grouping by
        # destination always sums to 1 per head. Group by source node instead:
        # how much attention each atom receives from its neighbours.
        alpha_mean = alpha_l.abs().mean(dim=1).cpu().numpy()
        source_nodes = edge_index_l[0].cpu().numpy()
        for i, src in enumerate(source_nodes):
            atom_scores[src] += alpha_mean[i]

    if atom_scores.max() > 0:
        atom_scores = atom_scores / atom_scores.max()

    return atom_scores

# ── Draw molecule with attention ──────────────────────────────────────────────
def draw_molecule_attention(smiles, atom_scores):
    """Draw molecule coloured by attention weights. Returns PIL Image."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    cmap = cm.get_cmap("RdYlBu_r")
    norm = Normalize(vmin=0, vmax=1)

    atom_highlight = {}
    atom_radii     = {}
    for i, score in enumerate(atom_scores):
        r, g, b, _ = cmap(norm(score))
        atom_highlight[i] = (r, g, b)
        atom_radii[i]     = 0.4

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

    try:
        import cairosvg
        png_data = cairosvg.svg2png(bytestring=svg.encode(),
                                     output_width=IMG_SIZE[0],
                                     output_height=IMG_SIZE[1])
        return Image.open(io.BytesIO(png_data)), "png"
    except ImportError:
        return svg, "svg"

# ── Visualise top hits ────────────────────────────────────────────────────────
def visualise_top_hits(model, results, graphs, top_n=6):
    """Generate attention map figure for top N predicted hits."""
    print(f"\nVisualising top {top_n} hits...")

    # Build smiles → graph lookup
    graph_lookup = {g.smiles: g for g in graphs}

    top_results  = results[results["in_domain"]].head(top_n)
    cols         = 3
    rows         = (top_n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5.5))
    axes      = np.array(axes).flatten()

    for ax in axes:
        ax.axis("off")

    for idx, (_, row) in enumerate(top_results.iterrows()):
        smiles  = row["smiles"]
        pred    = row["predicted_pIC50"]
        rank    = row["rank"]
        zinc_id = row["zinc_id"]

        graph = graph_lookup.get(smiles)
        if graph is None:
            continue

        atom_scores = get_attention_weights(model, graph)
        result      = draw_molecule_attention(smiles, atom_scores)
        if result is None:
            continue

        img, fmt = result

        if fmt == "png":
            axes[idx].imshow(img)
        else:
            axes[idx].text(0.5, 0.5, "SVG output\n(install cairosvg for PNG)",
                           ha="center", va="center", fontsize=10)

        axes[idx].set_title(
            f"Rank #{rank} | Pred pIC50: {pred:.2f}\n{zinc_id}",
            fontsize=10, pad=6
        )
        axes[idx].axis("off")

    fig.suptitle(
        "MDM2 GAT Virtual Screen — Top Predicted Inhibitors\n"
        "(atom-level attention weights showing predicted pharmacophore hotspots)",
        fontsize=13, fontweight="bold", y=1.01
    )
    # Reserve space on the right for the colorbar before laying out the molecule grid,
    # so the rightmost column doesn't get squeezed under it.
    plt.tight_layout(rect=[0, 0, 0.87, 1])

    # Colorbar
    cax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    sm  = cm.ScalarMappable(cmap="RdYlBu_r", norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_label("Attention weight (normalised)", fontsize=12)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["Low", "", "Medium", "", "High"])

    out_path = os.path.join(OUTPUT_DIR, "top_hits_attention.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

# ── Save results ──────────────────────────────────────────────────────────────
def save_results(results, top_n=50):
    """Save full results (all candidates) and top hits (in-domain only) to CSV.

    Ranking purely by predicted_pIC50 lets the model's own out-of-domain
    extrapolation artifacts dominate the top of the list. top_hits.csv is
    restricted to compounds similar enough to a real training compound
    (in_domain) that the prediction is worth trusting; screening_results_full.csv
    keeps everything, uncensored, for transparency.
    """
    full_path = os.path.join(OUTPUT_DIR, "screening_results_full.csv")
    top_path  = os.path.join(OUTPUT_DIR, "top_hits.csv")

    results.to_csv(full_path, index=False)

    in_domain_results = results[results["in_domain"]]
    top_hits = in_domain_results.head(top_n)
    top_hits.to_csv(top_path, index=False)

    print(f"\nResults saved:")
    print(f"  Full results: {full_path}")
    print(f"  Top {top_n} in-domain hits: {top_path}")
    print(f"\nTop 10 predicted MDM2 inhibitors (in applicability domain):")
    print(top_hits[["rank", "zinc_id", "predicted_pIC50", "max_train_similarity", "smiles"]]
          .head(10).to_string(index=False))

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("MDM2 GAT Virtual Screening Pipeline")
    print("=" * 60)

    # 1. Load model
    model = load_model()

    # 2. Download ZINC compounds
    df = download_zinc_smiles(n_compounds=N_COMPOUNDS)

    # 3. Convert to graphs + property filter
    graphs, zinc_ids = prepare_screening_graphs(df)

    if len(graphs) == 0:
        print("No valid compounds after filtering. Exiting.")
        return

    # 4. Run virtual screen
    results = run_screening(model, graphs, zinc_ids)

    # 5. Save results
    save_results(results, top_n=TOP_N)

    # 6. Visualise top hits
    visualise_top_hits(model, results, graphs, top_n=TOP_VISUAL)

    print("\n" + "=" * 60)
    print("Virtual screening complete.")
    print(f"Outputs saved to: {OUTPUT_DIR}/")
    print("  screening_results_full.csv — all screened compounds ranked")
    print("  top_hits.csv               — top 50 in-domain candidates")
    print("  top_hits_attention.png     — attention maps for top 6 in-domain hits")
    print("=" * 60)

if __name__ == "__main__":
    main()