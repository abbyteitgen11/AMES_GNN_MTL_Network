from datetime import datetime
import os
import pathlib
import pdb
import re
import sys
from typing import Dict, List

import numpy as np
import pickle
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import yaml

from generate_graphs import generate_graphs
from set_up_atomic_structure_graphs import set_up_atomic_structure_graphs
from load_data import load_data

def write_node_features(features: List[str]) -> str:
    log_text = "---\n"
    log_text += "##        Node Features: \n"
    log_text += "---\n"
    
    for feature in features:
        if re.match("^gro|^per", feature):
           n_features = " 25 (one-hot encoded)\n" # 7 period, 18 group = 25

        else:
           n_features = "       1 float\n"
        log_text += "- " + feature + "    " + n_features

    return log_text

def write_parameters(head: str, parameters: Dict,
                      units: str = "Angstrom") -> str:

    log_text = "---\n"
    log_text += "##    parameters for: " + head + "\n"
    log_text += "---\n"
    log_text += "- nFeatures: " + repr(parameters["n_features"]) + "\n"
    log_text += "- r_min: " + repr(parameters["x_min"]) + " " + units + "\n"
    log_text += "- r_max: " + repr(parameters["x_max"]) + " " + units + "\n"
    log_text += "- sigma: " + repr(parameters["sigma"]) + " " + units + "\n"
    if parameters["norm"]:
        log_text += "- Normalised: " + "True\n"
    log_text += "---\n"

    return log_text

"""
A script to construct graph representations of atomic systems.

To execute: python graph_maker.py input-file

where input-file is a yaml file specifying how the graphs are to be
constructed and from which source file(s). The graphs will be stored
in the indicated directory tree in the form of json files. 

A sample input file is graph_maker_sample.yml

"""

input_file = sys.argv[1]  # input_file is a yaml compliant file

with open( input_file, 'r' ) as input_stream:
    input_data = yaml.load(input_stream, Loader=yaml.Loader)

data_path = '/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/data.csv'

source_directory = input_data.get("DataBaseDirectory", "./DataBase_AMES/FILES_XYZ")
target_directory = input_data.get("TargetDirectory", "./GraphDataBase_AMES/")

# Check if the source directory already contains train, validate and test subdirectories
split_database = False

train = pathlib.Path(source_directory + "/train/")
validate = pathlib.Path(source_directory + "/validate/")
test = pathlib.Path(source_directory + "/test/")

if train.exists() and train.is_dir() and \
   validate.exists() and validate.is_dir() and \
   test.exists() and test.is_dir():

   split_database = True  # Database is already split, so no need to split it

if not split_database:
   data_base = pathlib.Path(source_directory)
   #split_fractions = input_data.get("SplitFractions", [0.8,0.1,0.1])

# If necessary, create the target directory and its subdirectories
target = pathlib.Path(target_directory)
target_train = pathlib.Path(target_directory + "/train" )
target_validate = pathlib.Path(target_directory + "/validate" )
target_test = pathlib.Path(target_directory + "/test" )

if not target.exists():
   target.mkdir()
   target_train.mkdir()
   target_validate.mkdir()
   target_test.mkdir()

else:
   if not target_train: target_train.mkdir()
   if not target_validate: target_validate.mkdir()
   if not target_test: target_test.mkdir()

# File extensions
input_file_ext = input_data.get("InputFileExtension", '.xyz')
output_file_ext = input_data.get("OutputFileExtension", '.pkl')

# Generate respective graph databases
pattern = '*' + input_file_ext

if split_database:
   train_files = train.glob(pattern)
   val_files = validate.glob(pattern)
   test_files = test.glob(pattern)

else:
   files_list = list(data_base.glob(pattern))
   #generator = torch.Generator().manual_seed(42)
   #train_list, val_list, test_list = random_split( files_list,
                    #split_fractions, generator = generator)

    # Load training, validation, and test sets from csv
   train_set, val_set, test_set = load_data(data_path, model="MTL")

   # Find corresponding files in files_list based on ID number
   file_dict = {int(f.stem.split("_")[0]): f for f in files_list}

   train_list = [file_dict[i] for i in train_set[0] if i in file_dict]
   val_list = [file_dict[i] for i in val_set[0] if i in file_dict]
   test_list = [file_dict[i] for i in test_set[0] if i in file_dict]

   # Convert these lists to generators
   train_files = (file for file in train_list)
   val_files = (file for file in val_list)
   test_files = (file for file in test_list)

graph_type = input_data.get("graphType", "XG")  # default is XG

log_text = "\n\n\n"
log_text += "------------------------------------------------\n"
log_text += "#              Graph Description                \n"
log_text += "------------------------------------------------\n"
log_text += "\n\n"

log_text += "- graph construction style: " + graph_type + "  \n"

# Specify the graph construction strategy
n_max_neighbours = input_data.get("nMaxNeighbours", 6)
node_features = input_data.get("nodeFeatures", [])
species = input_data.get("species", ["N", "C", "H", "O", "S", "Cl", "Be", "Br", "Pt", "P", 
          "F", "As", "Hg", "Zn", "Si", "V", "I", "B", "Sn", "Ge", 
          "Ag", "Sb", "Cu", "Cr", "Pb", "Mo", "Se", "Al", "Cd", 
          "Mn", "Fe", "Ga", "Pd", "Na", "Ti", "Bi", "Co", "Ni", 
          "Ce", "Ba", "Zr", "Rh"])

transformData = input_data.get("transformData", False)
# transform = SetUpDataTransform( transformData, directories )
transform = None

if transform:
    log_text += "- Using data transformation " + transformData + "   \n"

bond_angle_features = input_data.get('BondAngleFeatures', False)
dihedral_angle_features = input_data.get('DihedralAngleFeatures', False)

Graphs = set_up_atomic_structure_graphs(
    graph_type = graph_type,
    species = species,
    bond_angle_feature = bond_angle_features,
    dihedral_angle_feature = dihedral_angle_features,
    spec_features = node_features,
    n_max_neighbours = n_max_neighbours,
)


nNodeFeatures = Graphs.n_node_features()

write_node_features(node_features)

log_text += "- nNodeFeatures: " + repr(nNodeFeatures) + "  \n"

if bond_angle_features:
   log_text += "Using Bond Angle Features \n"
else:
   log_text += "NOT using Bond Angle Features \n"

if dihedral_angle_features:
   log_text += "Using Dihedral Features \n"
else:
   log_text += "NOT using Dihedral Features \n"
   

descriptionText = input_data.get("descriptionText", " ")

descriptionText += log_text

# Now proceed to generate the graphs for training, validation and test datasets
make_graphs = input_data.get("generate_graphs", True)

if make_graphs:
   generate_graphs(Graphs, train_files, target_train, train_set, output_file_ext)
   generate_graphs(Graphs, val_files, target_validate, val_set, output_file_ext)
   generate_graphs(Graphs, test_files, target_test, test_set, output_file_ext)

# Save the description text as a new key to the yaml file
log_text += "\n\n"
log_text += "------------------------------------------------\n"
log_text += "#           End of Graph Description            \n"
log_text += "------------------------------------------------\n"
log_text += "\n\n"

input_data['descriptionText'] = log_text

# finally, we will write a yaml file containing a description of the 
# graph construction; this will be later read and employed by the 
# graph fitting process
description_file = target_directory + '/' + 'graph_description.yml'

# we append to the input data information on the total number of node features
input_data['nNodeFeatures'] = nNodeFeatures

with open(description_file, 'w') as description:
    yaml.dump(input_data, description)
