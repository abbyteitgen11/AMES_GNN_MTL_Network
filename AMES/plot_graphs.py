import pickle
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
from networkx.drawing import nx_agraph
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import re
from graph_dataset import GraphDataSet

database_path = "/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/GraphDataBase_AMES"
trainDir = database_path + '/train/'
valDir = database_path + '/validate/'
testDir = database_path + '/test/'

# Read in graph data
trainDataset = GraphDataSet(trainDir, nMaxEntries=None, seed=42, transform=None)

valDataset = GraphDataSet(valDir, nMaxEntries=None, seed=42, transform=None)

testDataset = GraphDataSet(testDir, nMaxEntries=None, seed=42, transform=None)


for idx, graph in enumerate(trainDataset): #3870
    idx = idx + 400
    filepath = trainDataset.filenames[idx]

#filepath = '/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL/GraphDataBase_AMES/train/2440_ames_mutagenicity_data_69.pkl'

# Load the graph from pickle file
    with open(filepath, 'rb') as f:
        graph = pickle.load(f)

    # Convert to networkx for visualization
    nx_graph = to_networkx(graph, to_undirected=True)

    # Elements
    element_mapping = {0: "N", 1: "C", 2: "H", 3: "O", 4: "S", 5: "Cl", 6: "Be",
                       7: "Br", 8: "Pt", 9: "P", 10: "F", 11: "As", 12: "Hg",
                       13: "Zn", 14: "Si", 15: "V", 16: "I", 17: "B", 18: "Sn",
                       19: "Ge", 20: "Ag", 21: "Sb", 22: "Cu", 23: "Cr", 24: "Pb",
                       25: "Mo", 26: "Se", 27: "Al", 28: "Cd", 29: "Mn", 30: "Fe",
                       31: "Ga", 32: "Pd", 33: "Na", 34: "Ti", 35: "Bi", 36: "Co",
                       37: "Ni", 38: "Ce", 39: "Ba", 40: "Zr", 41: "Rh"}
    element_types = [element_mapping[spec_id.item()] for spec_id in graph.spec_id]

    # CSV file with structure data
    csv_file = '/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL/DataBase_AMES/FILES/ames_mutagenicity_data.csv'

    df = pd.read_csv(csv_file)

    molecule_index = molecule_index = int(re.search(r'(\d+)_', filepath).group(1)) # get molecule number from input file name
    smiles_column_index = 3

    # Extract the SMILES string from the specific row and column
    smiles_string = df.iloc[molecule_index-1, smiles_column_index]

    # Convert the SMILES string to an RDKit molecule
    molecule = Chem.MolFromSmiles(smiles_string)

    # Add hydrogens
    molecule = Chem.AddHs(molecule)

    # Convert graph to networkx for visualization
    nx_graph = to_networkx(graph, to_undirected=True)

    # Generate the molecule's 2D coordinates (needed for drawing in a graph)
    AllChem.Compute2DCoords(molecule)

    # Plot chemical structure with graph
    # Plot chemical structure
    pos = {i: (molecule.GetConformer().GetAtomPosition(i).x, molecule.GetConformer().GetAtomPosition(i).y)
           for i in range(molecule.GetNumAtoms())}

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # Draw the chemical structure using RDKit
    img = Draw.MolToImage(molecule, size=(300, 300))
    ax[0].imshow(img)
    ax[0].axis('off')  # Hide axes
    #ax[0].set_title('Chemical Structure')
    ax[0].set_title(filepath)

    # Plot graph (using RDKit for node positions)
    node_labels = nx.get_node_attributes(nx_graph, 'label')
    nx.draw(nx_graph, pos, with_labels=True, labels=node_labels, node_size=700, font_size=10, font_weight='bold', ax=ax[1], node_color='lightblue')
    #ax[1].set_title('Graph Representation')

    #plt.title(filepath)
    plt.tight_layout()
    plt.show()

    if idx > 500:
        break

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
#check_empty_graphs(trainDataset)

"""