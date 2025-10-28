import os
import re
import pickle
import torch
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

from graph_dataset import GraphDataSet

# Define dataset and directories
database_path = "/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/GraphDataBase_AMES_additional_data"
xyz_dir = "/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/Additional_data/FILES_XYZ_new"
csv_file = "/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/Additional_data/final_additional_dataset.csv"

testDataset = GraphDataSet(database_path, nMaxEntries=None, seed=45, transform=None)

df = pd.read_csv(csv_file)

for idx, graph in enumerate(testDataset):
    filepath = testDataset.filenames[idx]
    base_name = os.path.basename(filepath).replace(".pkl", "")  # e.g. "1_additional_data_1"

    try:
        # Load the graph
        with open(filepath, "rb") as f:
            graph = pickle.load(f)

        nx_graph = to_networkx(graph, to_undirected=True)

        # Load matching XYZ file
        xyz_path = os.path.join(xyz_dir, f"{base_name}.xyz")

        coords = []
        with open(xyz_path, "r") as f:
            lines = f.readlines()
            for line in lines[2:]:  # skip atom count + comment
                parts = line.strip().split()
                if len(parts) >= 4:
                    coords.append((float(parts[1]), float(parts[2]), float(parts[3])))

        if len(coords) != graph.num_nodes:
            print(f"⚠️ Mismatch in {base_name}: {len(coords)} coords vs {graph.num_nodes} nodes")
            continue

        # 2D projection for plotting (XY plane)
        pos = {i: coords[i][:2] for i in range(len(coords))}

        # Map element types (optional)
        element_mapping = {
            0: "N", 1: "C", 2: "H", 3: "O", 4: "S", 5: "Cl", 6: "Be",
            7: "Br", 8: "Pt", 9: "P", 10: "F", 11: "As", 12: "Hg",
            13: "Zn", 14: "Si", 15: "V", 16: "I", 17: "B", 18: "Sn",
            19: "Ge", 20: "Ag", 21: "Sb", 22: "Cu", 23: "Cr", 24: "Pb",
            25: "Mo", 26: "Se", 27: "Al", 28: "Cd", 29: "Mn", 30: "Fe",
            31: "Ga", 32: "Pd", 33: "Na", 34: "Ti", 35: "Bi", 36: "Co",
            37: "Ni", 38: "Ce", 39: "Ba", 40: "Zr", 41: "Rh"
        }

        element_types = [
            element_mapping.get(int(spec_id.item()), "X") for spec_id in graph.spec_id
        ]

        # Plot molecule and graph side by side
        fig, ax = plt.subplots(1, 2, figsize=(15, 7))

        # Plot chemical structure from SMILES for comparison
        molecule_index = int(re.search(r'(\d+)_', base_name).group(1))
        smiles_string = df.iloc[molecule_index - 1, 0]
        molecule = Chem.MolFromSmiles(smiles_string)
        molecule = Chem.AddHs(molecule)
        Chem.rdDepictor.Compute2DCoords(molecule)
        img = Draw.MolToImage(molecule, size=(300, 300))
        ax[0].imshow(img)
        ax[0].axis("off")
        ax[0].set_title(base_name)

        # Plot networkx graph
        nx.draw(
            nx_graph,
            pos,
            with_labels=True,
            node_size=700,
            font_size=10,
            font_weight="bold",
            ax=ax[1],
            node_color="lightblue",
        )

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Error in {base_name}: {type(e).__name__} - {e}")
        continue

    if idx > 200:
        break


    #if num_nodes_in_graph < num_atoms_in_smiles:
    #    mismatched_molecules.append({
    #        'filepath': filepath,
    #        'num_atoms': num_atoms_in_smiles,
    #        'num_nodes': num_nodes_in_graph,
    #        'smiles': smiles_string
    #    })
#
#print(f"\nFound {len(mismatched_molecules)} mismatched molecules.")
#for m in mismatched_molecules:
#   print(f"{m['filepath']} | Atoms: {m['num_atoms']} | Nodes: {m['num_nodes']}")

"""
def check_for_nans(dataset):
    for idx, graph in enumerate(dataset):
        if torch.isnan(graph.x).any():
            print(f"NaN found in node features (x) at index {idx}")
            print(dataset.filenames[idx])
        if torch.isnan(graph.edge_attr).any():
            print(f"NaN found in edge attributes (edge_attr) at index {idx}")
            print(dataset.filenames[idx])
        if torch.isnan(graph.y).any():
            print(f"NaN found in target values (y) at index {idx}")
            print(dataset.filenames[idx])


def check_for_infs(dataset):
    for idx, graph in enumerate(dataset):
        if torch.isinf(graph.x).any():
            print(f"Inf found in node features (x) at index {idx}")
            print(dataset.filenames[idx])
        if torch.isinf(graph.edge_attr).any():
            print(f"Inf found in edge attributes (edge_attr) at index {idx}")
            print(dataset.filenames[idx])
        if torch.isinf(graph.y).any():
            print(f"Inf found in target values (y) at index {idx}")
            print(dataset.filenames[idx])


def check_value_ranges(dataset):
    for idx, graph in enumerate(dataset):
        print(f"Graph {idx}:")
        print(dataset.filenames[idx])
        print(f"  x min/max: {graph.x.min().item():.4f} / {graph.x.max().item():.4f}")
        print(f"  edge_attr min/max: {graph.edge_attr.min().item():.4f} / {graph.edge_attr.max().item():.4f}")
        print(f"  y min/max: {graph.y.min().item():.4f} / {graph.y.max().item():.4f}")


def check_empty_graphs(dataset):
    for idx, graph in enumerate(dataset):
        if graph.x.numel() == 0:
            print(f"Graph {idx} has no nodes!")
            print(dataset.filenames[idx])
        if graph.edge_index.numel() == 0:
            print(f"Graph {idx} has no edges!")
            print(dataset.filenames[idx])


#check_for_nans(trainDataset)
#check_for_infs(trainDataset)
#check_value_ranges(trainDataset)
check_empty_graphs(trainDataset)

"""