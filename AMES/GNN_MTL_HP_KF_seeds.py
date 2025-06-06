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
from BuildNN_GNN_MTL import BuildNN_GNN_MTL
from masked_loss_function import masked_loss_function
from set_seed import set_seed
from MTLDataset import MTLDataset



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


def main():
    random_seeds = [3,7,15,24,42,45,62,77,79,88,90]

    val_losses = []
    avg_val_losses = []

    for seed in random_seeds:
        # Set seed
        print("Seed:", seed)

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)

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
        n_graph_convolution_layers = input_data.get("nGraphConvolutionLayers", 0)  # Number of graph convolutional layers
        n_node_neurons = input_data.get("nNodeNeurons", None)  # Number of neurons in GNN
        n_edge_neurons = input_data.get("nEdgeNeurons", None)  # Number of edges in GNN
        dropout_GNN = input_data.get("dropoutGNN", None)  # Dropout GNN
        momentum_batch_norm = input_data.get("momentumBatchNorm", None)  # Batch norm GNN

        n_shared_layers = input_data.get("nSharedLayers", 4)  # Number of layers in shared core
        n_target_specific_layers = input_data.get("nTargetSpecificLayers", 2)  # Number of layers in target specific core
        n_shared = input_data.get("nShared", None)  # Number of neurons in shared core
        n_target = input_data.get("nTarget", None)  # Number of neurons in target specific core
        dropout_shared = input_data.get("dropoutShared", None)  # Dropout in shared core
        dropout_target = input_data.get("dropoutTarget", None)  # Dropout in target specific core

        activation = input_data.get("ActivationFunction", "ReLU")  # Activation function
        weighted_loss_function = input_data.get("weightedCostFunction", False)
        w1 = input_data.get("w1", 1.0)
        w2 = input_data.get("w2", 1.0)
        w3 = input_data.get("w3", 1.0)
        w4 = input_data.get("w4", 1.0)
        w5 = input_data.get("w5", 1.0)
        if weighted_loss_function:
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
        learningRate = input_data.get("learningRate", 0.0001)  # Learning rate
        weightedCostFunction = input_data.get("weightedCostFunction", None)  # Use weighted  cost function
        L2Regularization = input_data.get("L2Regularization", 0.005)  # L2 regularization coefficient
        loadModel = input_data.get("loadModel", False)
        loadOptimizer = input_data.get("loadOptimizer", False)
        useMolecularDescriptors = input_data.get("useMolecularDescriptors",
                                                 False)  # Use molecular descriptors instead of graphs for comparison to original MTL paper
        GNN_explainer_analysis = input_data.get("GNNExplainerAnalysis", False)  # Run GNNExplainer analysis after training

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

        full_dataset = trainDataset + valDataset

        X_indices = np.arange(len(full_dataset))
        y_multilabel = get_multilabel_targets(full_dataset)

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


        mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        n_start = 0

        # Train model
        # factor = float(n_train) / float(n_validation)

        for fold, (train_idx, val_idx) in enumerate(mskf.split(X_indices, y_multilabel)):
            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)

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

            model = BuildNN_GNN_MTL(n_graph_convolution_layers, n_node_neurons, n_edge_neurons, n_node_features,
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
            val_losses.append(val_loss)
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_losses.append(avg_val_loss)

    print(val_losses)
    print(avg_val_losses)
if __name__ == "__main__":
    main()