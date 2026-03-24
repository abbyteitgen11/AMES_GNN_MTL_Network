from datetime import datetime
import faulthandler
import os
import pdb
import re
import sys
import csv
import h5py
import random
import markdown
import csv
import argparse
import logging
import optuna
from optuna.trial import TrialState
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import pickle
import matplotlib.pyplot as plt
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from callbacks import set_up_callbacks
from count_model_parameters import count_model_parameters
from device import device
from graph_dataset import GraphDataSet
from compute_metrics import *
from data import load_data
from BuildNN_GNN_MTL_HP import BuildNN_GNN_MTL
from masked_loss_function import masked_loss_function
from set_seed import set_seed
from MTLDataset import MTLDataset

# Set seed
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)


"""
A driver script to fit a Graph Convolutional Neural Network + MTL Neural Network model to
represent properties of molecular/condensed matter systems.

To execute: python GNN_MTL input-file

where input-file is a yaml file specifying different parameters of the
model and how the job is to be run. For an example see sample.yml

"""
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    return parser.parse_args()


def plot_fold_distribution(fold_num, counts, title):
    fig, ax = plt.subplots()
    categories = [-1, 0, 1]
    for task_idx, task_counts in enumerate(counts):
        values = [task_counts.get(c, 0) for c in categories]
        ax.bar([f"T{task_idx + 1}-{c}" for c in categories], values, label=f"Task {task_idx + 1}")

    ax.set_title(f"{title} Label Counts (Fold {fold_num + 1})")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.show()


def count_label_distribution(dataset, indices):
    # Initialize counts: one dict per task
    counts_per_task = [{-1: 0, 0: 0, 1: 0} for _ in range(5)]

    for i in indices:
        y = dataset[i].y.squeeze().tolist()  # shape: (5,)
        for task_idx, label in enumerate(y):
            counts_per_task[task_idx][int(label)] += 1

    return counts_per_task


def objective(trial):
    print(f"Starting trial {trial.number}")
    args = get_args()
    output_dir = ''
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Read in input data
    # input_file = sys.argv[1]  # input_file is a yaml compliant file
    input_file = args.input_file

    with open(input_file, 'r') as input_stream:
        input_data = yaml.load(input_stream, Loader=yaml.Loader)

    # Set database path
    database_path = input_data.get("database", "./GraphDataBase_AMES")

    # The database is described with its own yaml file; so read it
    database_file = database_path + '/graph_description.yml'

    with open(database_file, 'r') as database_stream:
        database_data = yaml.load(database_stream, Loader=yaml.Loader)

    # Model parameters
    n_graph_convolution_layers = trial.suggest_int("nGraphConvolutionalLayers", 1,
                                                   5)  # Number of graph convolutional layers
    n_node_neurons = trial.suggest_int("n_node_neurons", 1,
                                       300)  # Number of neurons in GNN
    n_edge_neurons = trial.suggest_int("n_edge_neurons", 1,
                                       300)  # Number of edges in GNN
    dropout_GNN = trial.suggest_float("DropoutGNN", 0.0, 0.5)
    momentum_batch_norm = trial.suggest_float("momentumBatchNorm", 0.0, 1.0)
    n_shared_layers = trial.suggest_int("nSharedLayers", 1, 4) #input_data.get("nSharedLayers", 4)  # Number of layers in shared core
    n_target_specific_layers = trial.suggest_int("nTargetSpecificLayers", 1, 3) #input_data.get("nTargetSpecificLayers", 2)  # Number of layers in target specific core
    n_shared = [
        trial.suggest_int(f"n_shared_{i}", 1, 300)
        for i in range(n_shared_layers)
    ]
    n_target = [
        trial.suggest_int(f"n_target_{i}", 1, 300)
        for i in range(n_target_specific_layers)
    ]
    dropout_shared = [
        trial.suggest_float(f"DropoutShared_{i}", 0.0, 0.5)
        for i in range(n_shared_layers)
    ]
    dropout_target = [
        trial.suggest_float(f"DropoutTarget_{i}", 0.0, 0.5)
        for i in range(n_target_specific_layers)
    ]

    activation = input_data.get("ActivationFunction", "ReLU")  # Activation function
    weighted_loss_function = input_data.get("weightedCostFunction", False)
    if weighted_loss_function:
        w1 = trial.suggest_float("w1", 1.0, 6.0)
        w2 = trial.suggest_float("w2", 1.0, 6.0)
        w3 = trial.suggest_float("w3", 1.0, 6.0)
        w4 = trial.suggest_float("w4", 1.0, 6.0)
        w5 = trial.suggest_float("w5", 1.0, 6.0)
        class_weights = {
            '98': {0: 1.0, 1: w1, -1: 0},
            '100': {0: 1.0, 1: w2, -1: 0},
            '102': {0: 1.0, 1: w3, -1: 0},
            '1535': {0: 1.0, 1: w4, -1: 0},
            '1537': {0: 1.0, 1: w5, -1: 0},
        }
    else:
        class_weights = {
            '98': {0: 1.0, 1: 1.0, -1: 0.0},
            '100': {0: 1.0, 1: 1.0, -1: 0.0},
            '102': {0: 1.0, 1: 1.0, -1: 0.0},
            '1535': {0: 1.0, 1: 1.0, -1: 0.0},
            '1537': {0: 1.0, 1: 1.0, -1: 0.0},
        }
    output_keys = ['98', '100', '102', '1535', '1537']

    # Graph information
    graph_type = database_data.get("graphType", "covalent")
    n_node_features = database_data.get("nNodeFeatures")
    edge_parameters = database_data.get("EdgeFeatures")
    bond_angle_features = database_data.get("BondAngleFeatures", True)
    dihedral_angle_features = database_data.get("DihedralFeatures", True)
    n_edge_features = 1  # 1 for distance features
    if bond_angle_features: n_edge_features += 1  # bond-angle feature
    if dihedral_angle_features: n_edge_features += 1  # dihedral-angle feature

    # Training parameters
    nEpochs = input_data.get("nEpochs", 10)  # Number of epochs
    nBatch = input_data.get("nBatch", 50)  # Batch size
    chkptFreq = input_data.get("nCheckpoint", 10)  # Checkpoint frequency
    seed = input_data.get("randomSeed", 42)  # Random seed
    nTrainMaxEntries = input_data.get("nTrainMaxEntries",
                                      None)  # Number of training examples to use (if not using whole dataset)
    nValMaxEntries = input_data.get("nValMaxEntries",
                                    None)  # Number of validation examples to use (if not using whole dataset)
    learningRate = trial.suggest_float("learningRate", 1e-5, 1e-1, log=True) #input_data.get("learningRate", 0.0001)  # Learning rate
    weightedCostFunction = input_data.get("weightedCostFunction", None)  # Use weighted  cost function
    L2Regularization = input_data.get("L2Regularization", 0.005)  # L2 regularization coefficient
    loadModel = input_data.get("loadModel", False)
    loadOptimizer = input_data.get("loadOptimizer", False)
    useMolecularDescriptors = input_data.get("useMolecularDescriptors",
                                             False)  # Use molecular descriptors instead of graphs for comparison to original MTL paper

    trainDir = database_path + '/train/'
    valDir = database_path + '/validate/'
    testDir = database_path + '/test/'
    directories = [trainDir, valDir, testDir]

    n_inputs = 0

    # Read in graph data
    trainDataset = GraphDataSet(
        trainDir, nMaxEntries=nTrainMaxEntries, seed=seed
    )

    if nTrainMaxEntries:
        nTrain = nTrainMaxEntries
    else:
        nTrain = len(trainDataset)

    valDataset = GraphDataSet(
        valDir, nMaxEntries=nValMaxEntries, seed=seed
    )

    if nValMaxEntries:
        nValidation = nValMaxEntries
    else:
        nValidation = len(valDataset)

    testDataset = GraphDataSet(
        testDir, nMaxEntries=nValMaxEntries, seed=seed
    )

    g = torch.Generator()
    g.manual_seed(seed)

    # K fold
    def get_multilabel_targets(dataset):
        multilabel_targets = []
        for i in range(len(dataset)):
            y = dataset[i].y.squeeze()
            #binary_y = (y != -1).int().tolist()  # e.g., [1, 0, 1, 1, 0]
            multilabel_targets.append(y)
        return np.array(multilabel_targets)

    X_indices = np.arange(len(trainDataset))
    #X_indices = np.arange(len(valDataset))
    y_multilabel = get_multilabel_targets(trainDataset)

    #total_counts = count_label_distribution(valDataset, X_indices)

    #fig, ax = plt.subplots()
    #categories = [-1, 0, 1]
    #for task_idx, task_counts in enumerate(total_counts):
    #    values = [task_counts.get(c, 0) for c in categories]
    #    ax.bar([f"T{task_idx + 1}-{c}" for c in categories], values, label=f"Task {task_idx + 1}")

    #ax.set_ylabel("Count")
    #ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    #plt.tight_layout()
    #plt.show()


    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    n_start = 0

    # Train model
    # factor = float(n_train) / float(n_validation)
    val_losses = []
    val_loss_log = []

    for fold, (train_idx, val_idx) in enumerate(mskf.split(X_indices, y_multilabel)):
        train_subset = Subset(trainDataset, train_idx)
        val_subset = Subset(trainDataset, val_idx)

        #train_counts = count_label_distribution(trainDataset, train_idx)
        #val_counts = count_label_distribution(trainDataset, val_idx)

        #for task_idx in range(5):
        #    print(f"  Task {task_idx + 1}:")
        #    print(f"    Train: {train_counts[task_idx]}")
        #    print(f"    Val:   {val_counts[task_idx]}")

        #plot_fold_distribution(fold, train_counts, "Train")
        #plot_fold_distribution(fold, val_counts, "Val")

        trainLoader = DataLoader(train_subset, batch_size=nBatch, generator=g)
        valLoader = DataLoader(val_subset, batch_size=nBatch, generator=g)

        #print(f"Fold {fold + 1}: {len(train_idx)} train / {len(val_idx)} val")

        model = BuildNN_GNN_MTL(trial, n_graph_convolution_layers, n_node_neurons, n_edge_neurons, n_node_features,
                                n_edge_features, dropout_GNN, momentum_batch_norm,
                                n_shared_layers, n_target_specific_layers, n_shared, n_target, dropout_shared,
                                dropout_target, activation, useMolecularDescriptors, n_inputs)

        # move to GPU
        model = model.to(device)

        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=L2Regularization)  # l2 reg

        for epoch in range(nEpochs):
            model.train()
            train_loss = 0
            for sample in trainLoader:
                # Compute prediction
                pred = model(sample.x.to(device), sample.edge_index.to(device), sample.edge_attr.to(device),
                             sample.batch.to(device), n_node_neurons, n_node_features, n_edge_neurons, n_edge_features,
                             n_graph_convolution_layers, n_shared_layers, n_target_specific_layers,
                             useMolecularDescriptors)
                losses = 0

                for i in range(5):
                    output_key = output_keys[i]
                    loss = masked_loss_function(sample.y[:,i], pred[i].squeeze(1), class_weights[output_key])
                    losses += loss
                loss_final = losses / 5  # Scalar loss

                # Backpropagation
                optimizer.zero_grad()
                loss_final.backward()
                optimizer.step()

                train_loss += loss_final.item()

            train_loss /= len(trainLoader)

            # Evaluate model
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for sample in valLoader:
                    pred = model(sample.x.to(device), sample.edge_index.to(device), sample.edge_attr.to(device),
                                 sample.batch.to(device), n_node_neurons, n_node_features, n_edge_neurons,
                                 n_edge_features, n_graph_convolution_layers, n_shared_layers, n_target_specific_layers,
                                 useMolecularDescriptors)
                    losses = 0
                    for i in range(5):
                        output_key = output_keys[i]
                        loss = masked_loss_function(sample.y[:, i], pred[i].squeeze(1), class_weights[output_key])
                        losses += loss
                    loss_final = losses / 5  # Scalar loss
                    val_loss += loss_final.item()
            val_loss /= len(valLoader)
            #print(f"Trial {trial.number} | Fold {fold + 1} | Epoch {epoch + 1} | Val Loss: {val_loss:.4f}")


            val_loss_log.append({
                "fold": fold,
                "epoch": epoch,
                "val_loss": val_loss
            })

        val_losses.append(val_loss)

        #trial.set_user_attr(f"fold_{fold}_val_loss", val_loss)
        trial.set_user_attr("val_loss_log", val_loss_log)

        if "best_fold_loss" not in trial.user_attrs or val_loss < trial.user_attrs["best_fold_loss"]:
            trial.set_user_attr("best_fold_loss", val_loss)
            trial.set_user_attr("best_fold", fold)

    avg_val_loss = sum(val_losses) / len(val_losses)

    trial.report(avg_val_loss, epoch)
    if trial.should_prune():
        raise optuna.TrialPruned()

    if trial.number % 10 == 0:
        save_study(study, '/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/optuna/study.pkl')

    return avg_val_loss


def save_study(study, path):
    with open(path, "wb") as f:
        pickle.dump(study, f)

def load_study(path):
    with open(path, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    try:
        study = load_study('/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/optuna/study.pkl')
        print("Loaded existing study.")
    except FileNotFoundError:
        study = optuna.create_study(direction="minimize")
        print("Created new study.")

    study.enqueue_trial({
        "nGraphConvolutionalLayers": 2,
        "n_node_neurons": 78,
        "n_edge_neurons": 107,
        "DropoutGNN": 0.001,
        "momentumBatchNorm": 0.1,
        "nSharedLayers": 4,
        "nTargetSpecificLayers": 2,
        "n_shared_0": 200,
        "n_shared_1": 100,
        "n_shared_2": 50,
        "n_shared_3": 10,
        "n_target_0": 50,
        "n_target_1": 10,
        "DropoutShared_0": 0.25,
        "DropoutShared_1": 0.15,
        "DropoutShared_2": 0.1,
        "DropoutShared_3": 0.0001,
        "DropoutTarget_0": 0.15,
        "DropoutTarget_1": 0.1,
        "w1": 1.90,
        "w2": 1.56,
        "w3": 3.31,
        "w4": 5.09,
        "w5": 5.11,
        "learningRate": 0.0001,
    })

    #study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=11, n_jobs=4)

    save_study(study, '/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/optuna/study.pkl')

    #pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    #print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))