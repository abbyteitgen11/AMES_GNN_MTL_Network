"""
visualize_graphs.py

Visualize molecular graphs from the graph database side-by-side with their
2D chemical structures (from SMILES) for sanity-checking graph construction.

Each page/image shows:
  Left : 2D chemical structure rendered by RDKit from the SMILES string
  Right: Molecular graph from the .pkl file (node colours = element, edge
         colour = first edge-attribute value, typically bond distance)

H atoms are hidden by default (pass --show_H to include them).
Graph layout uses the x,y projection of the 3D atomic positions stored in
graph.pos, giving a chemically meaningful orientation.

Usage examples:
    # 20 test-set molecules → PDF (default)
    python visualize_graphs.py --input_file train_sample.yml

    # First 100 training molecules → individual PNGs
    python visualize_graphs.py --input_file train_sample.yml \\
        --partition train --n_graphs 100 --output_format png

    # Explicit paths, show H atoms
    python visualize_graphs.py \\
        --database_dir /path/to/GraphDataBase_AMES_NEW \\
        --data_file    /path/to/data_new_with_split.csv \\
        --n_graphs 50 --partition test --show_H
"""

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                          # non-interactive backend
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# CPK colour palette for every element in the 42-species list
# ---------------------------------------------------------------------------
CPK_COLORS = {
    "H":  "#DDDDDD", "C":  "#505050", "N":  "#3050F8", "O":  "#FF0D0D",
    "S":  "#E0E000", "Cl": "#1FF01F", "Be": "#C2FF00", "Br": "#A62929",
    "Pt": "#D0D0E0", "P":  "#FF8000", "F":  "#90E050", "As": "#BD80E3",
    "Hg": "#B8B8D0", "Zn": "#7D80B0", "Si": "#F0C8A0", "V":  "#A6A6AB",
    "I":  "#940094", "B":  "#FFB5B5", "Sn": "#668080", "Ge": "#668F8F",
    "Ag": "#C0C0C0", "Sb": "#9E63B5", "Cu": "#C88033", "Cr": "#8A99C7",
    "Pb": "#575961", "Mo": "#54B5B5", "Se": "#FFA100", "Al": "#BFA6A6",
    "Cd": "#FFD98F", "Mn": "#9C7AC7", "Fe": "#E06633", "Ga": "#C28F8F",
    "Pd": "#006985", "Na": "#AB5CF2", "Ti": "#BFC2C7", "Bi": "#9E4FB5",
    "Co": "#F090A0", "Ni": "#50D050", "Ce": "#FFFFC7", "Ba": "#00C900",
    "Zr": "#94DBFF", "Rh": "#0A7D8C",
}

FALLBACK_COLOR = "#AAAAAA"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_numpy(t):
    """Convert a torch.Tensor, list, or numpy array to a numpy array."""
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    if hasattr(t, "numpy"):
        return t.numpy()
    return np.array(t)


def load_species_list(database_dir):
    """Read species list from graph_description.yml; fall back to AMES defaults."""
    yml_path = os.path.join(database_dir, "graph_description.yml")
    if os.path.exists(yml_path):
        with open(yml_path) as f:
            desc = yaml.safe_load(f) or {}
        species = desc.get("species")
        if species:
            return list(species)
    # Hard-coded fallback matching AMES_NEW database
    return [
        "N", "C", "H", "O", "S", "Cl", "Be", "Br", "Pt", "P", "F", "As",
        "Hg", "Zn", "Si", "V", "I", "B", "Sn", "Ge", "Ag", "Sb", "Cu",
        "Cr", "Pb", "Mo", "Se", "Al", "Cd", "Mn", "Fe", "Ga", "Pd", "Na",
        "Ti", "Bi", "Co", "Ni", "Ce", "Ba", "Zr", "Rh",
    ]


def collect_pkl_files(database_dir, partition, n_graphs):
    """Return sorted list of up to n_graphs .pkl paths for the given partition(s)."""
    subdir_map = {"train": "train", "validate": "validate", "test": "test"}
    subdirs = list(subdir_map.values()) if partition == "all" else [subdir_map[partition]]
    files = []
    for sd in subdirs:
        files.extend(sorted(glob.glob(os.path.join(database_dir, sd, "*.pkl"))))
    return files[:n_graphs]


def mol_id_from_path(pkl_path):
    """Extract integer molecule ID from the pkl filename prefix (before first '_')."""
    return int(Path(pkl_path).stem.split("_")[0])


# ---------------------------------------------------------------------------
# Left panel: 2D chemical structure from SMILES
# ---------------------------------------------------------------------------

def draw_structure_2d(smiles, ax, mol_id):
    """Render 2D RDKit structure on *ax*; show placeholder if SMILES is missing/invalid."""
    ax.set_title("2D Structure (SMILES)", fontsize=9, pad=4)
    ax.axis("off")

    if not isinstance(smiles, str) or not smiles.strip():
        ax.text(0.5, 0.5, f"No SMILES\n(ID {mol_id})",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="gray")
        return

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        ax.text(0.5, 0.5, f"Invalid SMILES\n(ID {mol_id})\n{smiles[:40]}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color="red")
        return

    img = Draw.MolToImage(mol, size=(440, 340))
    ax.imshow(img)


# ---------------------------------------------------------------------------
# Right panel: molecular graph from the .pkl file
# ---------------------------------------------------------------------------

def draw_graph_panel(graph, species_list, show_H, ax):
    """Draw the PyG graph on *ax* using x,y projection of 3D positions."""
    try:
        spec_ids   = _to_numpy(graph.spec_id).astype(int).flatten()
        pos3d      = _to_numpy(graph.pos)          # [n_atoms, 3]
        edge_index = _to_numpy(graph.edge_index)   # [2, n_edges]
        edge_attr  = _to_numpy(graph.edge_attr)    # [n_edges, n_feat]
    except AttributeError as e:
        ax.text(0.5, 0.5, f"Missing attribute:\n{e}",
                ha="center", va="center", transform=ax.transAxes, color="red")
        ax.axis("off")
        return

    if edge_attr.ndim == 1:
        edge_attr = edge_attr[:, None]

    n_atoms = len(spec_ids)

    # ---- decide which atoms to show ----
    if show_H:
        keep_mask = np.ones(n_atoms, dtype=bool)
    else:
        keep_mask = np.array(
            [species_list[s] != "H" for s in spec_ids], dtype=bool
        )

    kept = np.where(keep_mask)[0]
    old_to_new = {int(old): new for new, old in enumerate(kept)}

    if len(kept) == 0:
        ax.text(0.5, 0.5, "No atoms to display",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    # ---- build NetworkX graph ----
    G = nx.Graph()
    for new_i, old_i in enumerate(kept):
        G.add_node(new_i, element=species_list[spec_ids[old_i]])

    for e in range(edge_index.shape[1]):
        i_old, j_old = int(edge_index[0, e]), int(edge_index[1, e])
        # de-duplicate: keep only one direction (i < j)
        if i_old < j_old and i_old in old_to_new and j_old in old_to_new:
            i_new, j_new = old_to_new[i_old], old_to_new[j_old]
            dist = float(edge_attr[e, 0])
            G.add_edge(i_new, j_new, dist=dist)

    # ---- positions: x,y projection of 3D coordinates ----
    pos_2d   = pos3d[kept, :2]
    pos_dict = {i: pos_2d[i] for i in range(len(kept))}

    # ---- node styling ----
    node_colors = [CPK_COLORS.get(G.nodes[n]["element"], FALLBACK_COLOR)
                   for n in G.nodes()]
    node_labels = {n: G.nodes[n]["element"] for n in G.nodes()}

    nx.draw_networkx_nodes(
        G, pos=pos_dict, node_color=node_colors,
        node_size=220, linewidths=0.5, edgecolors="#444444", ax=ax
    )
    nx.draw_networkx_labels(
        G, pos=pos_dict, labels=node_labels,
        font_size=5, font_color="#000000", ax=ax
    )

    # ---- edge styling: colour by first edge-attribute ----
    if G.number_of_edges() > 0:
        edge_list  = list(G.edges())
        edge_vals  = [G[u][v]["dist"] for u, v in edge_list]
        vmin, vmax = min(edge_vals), max(edge_vals)

        lc = nx.draw_networkx_edges(
            G, pos=pos_dict, edgelist=edge_list,
            edge_color=edge_vals, edge_cmap=plt.cm.viridis,
            edge_vmin=vmin, edge_vmax=vmax,
            width=1.5, ax=ax
        )
        if lc is not None:
            sm = ScalarMappable(cmap=plt.cm.viridis,
                                norm=Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cb = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.04)
            cb.set_label("edge_attr[0]  (bond distance, Å)", fontsize=6)
            cb.ax.tick_params(labelsize=6)

    h_note = "shown" if show_H else "hidden"
    ax.set_title(
        f"Graph — {G.number_of_nodes()} atoms (H {h_note}), "
        f"{G.number_of_edges()} edges",
        fontsize=9, pad=4
    )
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

def add_element_legend(fig, species_list):
    """Append a compact CPK colour legend at the bottom of the figure."""
    unique = sorted({el for el in species_list if el in CPK_COLORS})
    patches = [
        mpatches.Patch(
            facecolor=CPK_COLORS[el], edgecolor="#444444",
            linewidth=0.5, label=el
        )
        for el in unique
    ]
    fig.legend(
        handles=patches, loc="lower center",
        ncol=min(len(patches), 14),
        fontsize=6, framealpha=0.85,
        title="Element (CPK colours)", title_fontsize=7,
        bbox_to_anchor=(0.5, -0.01),
    )


# ---------------------------------------------------------------------------
# Per-molecule figure
# ---------------------------------------------------------------------------

def make_figure(pkl_path, species_list, id_to_smiles, id_to_labels, show_H):
    """Return a matplotlib Figure for one molecule (2D structure + graph)."""
    mol_id = mol_id_from_path(pkl_path)

    with open(pkl_path, "rb") as f:
        graph = pickle.load(f)

    smiles     = id_to_smiles.get(mol_id, "")
    labels_row = id_to_labels.get(mol_id, {})
    strain_names = ["TA98", "TA100", "TA102", "TA1535", "TA1537", "Overall"]
    label_str  = "  ".join(
        f"{s}={'?' if labels_row.get(s, -1) == -1 else int(labels_row.get(s, -1))}"
        for s in strain_names
        if s in labels_row
    )

    smiles_display = (smiles[:80] + "…") if len(smiles) > 80 else smiles
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        f"Mol ID: {mol_id}   {label_str}\n{smiles_display}",
        fontsize=8.5, y=1.01
    )

    draw_structure_2d(smiles, ax_left, mol_id)
    draw_graph_panel(graph, species_list, show_H, ax_right)
    add_element_legend(fig, species_list)

    fig.tight_layout(rect=[0, 0.05, 1, 1.0])
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize molecular graphs from the database next to 2D chemical structures."
    )
    parser.add_argument(
        "--input_file", default="train_sample.yml",
        help="YAML config to read default database/data_file paths (default: train_sample.yml)"
    )
    parser.add_argument("--database_dir", default=None,
                        help="Path to graph database root directory (overrides YAML 'database' key)")
    parser.add_argument("--data_file", default=None,
                        help="Path to CSV with SMILES and labels (overrides YAML 'data_file' key)")
    parser.add_argument("--n_graphs", type=int, default=20,
                        help="Number of graphs to visualize (default: 20)")
    parser.add_argument("--partition", default="test",
                        choices=["train", "validate", "test", "all"],
                        help="Partition to sample from (default: test)")
    parser.add_argument("--output_dir", default="./graph_viz",
                        help="Output directory (default: ./graph_viz)")
    parser.add_argument("--output_format", default="pdf",
                        choices=["pdf", "png"],
                        help="'pdf' = all graphs in one file; 'png' = one file per graph (default: pdf)")
    parser.add_argument("--show_H", action="store_true",
                        help="Show hydrogen atoms in graph panel (default: hidden)")
    args = parser.parse_args()

    # ---- resolve paths (CLI > YAML > default) ----
    yaml_data = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(args.input_file):
        with open(args.input_file) as f:
            yaml_data = yaml.safe_load(f) or {}
    elif not args.database_dir:
        print(f"WARNING: --input_file '{args.input_file}' not found. "
              "Pass --database_dir and --data_file explicitly.")

    database_dir = (args.database_dir
                    or yaml_data.get("database", "").strip()
                    or "")
    data_file    = (args.data_file
                    or yaml_data.get("data_file", "").strip()
                    or os.path.join(script_dir, "data", "data_new_with_split.csv"))

    if not os.path.isdir(database_dir):
        sys.exit(f"ERROR: database_dir not found: {database_dir!r}\n"
                 "Pass --database_dir or set 'database' in the YAML config.")
    if not os.path.isfile(data_file):
        sys.exit(f"ERROR: data_file not found: {data_file!r}\n"
                 "Pass --data_file or set 'data_file' in the YAML config.")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- load ancillary data ----
    species_list = load_species_list(database_dir)
    print(f"Species list ({len(species_list)} elements): {species_list}")

    df = pd.read_csv(data_file)
    if "Id" not in df.columns:
        sys.exit("ERROR: data_file must contain an 'Id' column.")
    id_to_smiles = df.set_index("Id")["SMILES"].to_dict()
    label_cols   = [c for c in ["TA98","TA100","TA102","TA1535","TA1537","Overall"]
                    if c in df.columns]
    id_to_labels = df.set_index("Id")[label_cols].to_dict("index")

    # ---- collect pkl files ----
    pkl_files = collect_pkl_files(database_dir, args.partition, args.n_graphs)
    if not pkl_files:
        sys.exit(f"ERROR: No .pkl files found under {database_dir}/"
                 f"{'[train|validate|test]' if args.partition == 'all' else args.partition}/")

    n = len(pkl_files)
    print(f"Visualizing {n} graph(s) from partition='{args.partition}'  "
          f"(H atoms: {'shown' if args.show_H else 'hidden'})")

    # ---- render ----
    if args.output_format == "pdf":
        out_path = os.path.join(args.output_dir,
                                f"graphs_{args.partition}.pdf")
        with PdfPages(out_path) as pdf:
            for i, pkl_path in enumerate(pkl_files):
                mol_id = mol_id_from_path(pkl_path)
                print(f"  [{i+1}/{n}]  mol {mol_id}  ({os.path.basename(pkl_path)})")
                try:
                    fig = make_figure(pkl_path, species_list,
                                      id_to_smiles, id_to_labels, args.show_H)
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                except Exception as exc:
                    print(f"    WARNING: skipped mol {mol_id}: {exc}")
                    plt.close("all")
        print(f"\nSaved: {out_path}")

    else:  # png
        for i, pkl_path in enumerate(pkl_files):
            mol_id = mol_id_from_path(pkl_path)
            print(f"  [{i+1}/{n}]  mol {mol_id}  ({os.path.basename(pkl_path)})")
            try:
                fig = make_figure(pkl_path, species_list,
                                  id_to_smiles, id_to_labels, args.show_H)
                out_path = os.path.join(args.output_dir, f"{mol_id}.png")
                fig.savefig(out_path, bbox_inches="tight", dpi=150)
                plt.close(fig)
            except Exception as exc:
                print(f"    WARNING: skipped mol {mol_id}: {exc}")
                plt.close("all")
        print(f"\nSaved {n} PNG file(s) to {args.output_dir}/")


if __name__ == "__main__":
    main()
