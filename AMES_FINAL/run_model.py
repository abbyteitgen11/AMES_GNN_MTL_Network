"""
run_model.py — Consolidated GNN-MTL driver script for AMES mutagenicity prediction.

Usage:
    python run_model.py --mode <mode> --output_dir <dir> --input_file <yaml> [options]

Modes:
    train          Train with fixed HP from YAML; save checkpoints and TensorBoard logs.
    hp_opt         Hyperparameter optimization with Optuna (5-fold CV on train set).
    seeds_cfv      5-fold cross-validation across multiple random seeds (fixed HP from YAML).
    eval           Load a checkpoint; optimize thresholds on val set; evaluate on test set.
    top_seeds_eval Auto-pick top N seeds by avg val loss; evaluate on test; average metrics.
    analyze_cfv    Post-hoc analysis of seeds_cfv output CSVs (plots + statistics).
    viz_optuna     Visualize Optuna study: optimization history, param importances, val loss.
"""

# ==============================================================================
# Imports
# ==============================================================================
import argparse
import csv
import json
import logging
import os
import pickle
import random
import re
import sys
from datetime import datetime
from glob import glob

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import optuna.visualization as vis
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from optuna.trial import TrialState
from torch.utils.data import DataLoader, Subset, TensorDataset, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from BuildNN_GNN_MTL_GINEConv import BuildNN_GNN_MTL
from MTLDataset import MTLDataset
from callbacks import set_up_callbacks
from compute_metrics import filter_nan, get_metrics
from count_model_parameters import count_model_parameters
from data import load_data
from graph_dataset import GraphDataSet
from masked_loss_function import masked_loss_function
from set_seed import set_seed

# Default path to the AMES_FINAL directory (used as base for default output paths)
AMES_FINAL_DIR = os.path.dirname(os.path.abspath(__file__))


# ==============================================================================
# Argument Parsing
# ==============================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="GNN-MTL model driver for AMES mutagenicity prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Mode selection
    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "hp_opt", "seeds_cfv", "eval",
                                 "top_seeds_eval", "analyze_cfv", "viz_optuna"],
                        help="Operating mode.")

    # Core I/O
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for all output files (logs, metrics, plots).")
    parser.add_argument("--input_file", type=str, default=None,
                        help="YAML config file. Required for: train, hp_opt, seeds_cfv, eval, top_seeds_eval.")

    # Checkpoint and Optuna directories
    parser.add_argument("--checkpoints_dir", type=str,
                        default=os.path.join(AMES_FINAL_DIR, "checkpoints"),
                        help="Directory for model checkpoint .pt files.")
    parser.add_argument("--optuna_dir", type=str,
                        default=os.path.join(AMES_FINAL_DIR, "optuna"),
                        help="Directory for Optuna study .pkl files.")

    # Specific file overrides
    parser.add_argument("--checkpoint_file", type=str, default=None,
                        help="Specific checkpoint .pt file to load (required for eval mode).")
    parser.add_argument("--optuna_file", type=str, default=None,
                        help="Specific Optuna .pkl file. Used in hp_opt (resume) and viz_optuna (single study).")
    parser.add_argument("--data_file", type=str, default=None,
                        help="Path to data.csv. Overrides the value in the YAML config.")
    parser.add_argument("--metrics_dir", type=str, default=None,
                        help="Directory containing metrics_seed_*_fold_*.csv files. "
                             "Defaults to --output_dir if not set. Used in analyze_cfv and top_seeds_eval.")
    parser.add_argument("--val_loss_file", type=str, default=None,
                        help="Optional .xlsx file of validation losses for the heatmap plot in analyze_cfv.")

    # hp_opt options
    parser.add_argument("--n_trials", type=int, default=100,
                        help="Number of Optuna trials (hp_opt mode).")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Parallel jobs for Optuna optimization (hp_opt mode).")

    # seeds_cfv options
    parser.add_argument("--seeds", type=str, default="3 7 15 24 42 45 62 77 79 88 90",
                        help="Space-separated list of random seeds for seeds_cfv mode.")

    # top_seeds_eval options
    parser.add_argument("--n_top_seeds", type=int, default=5,
                        help="Number of top seeds to select in top_seeds_eval mode.")

    # Threshold options
    parser.add_argument("--use_thresholds", action="store_true",
                        help="If set, optimize per-task decision thresholds on the validation set "
                             "(cross-fitting) and apply them when evaluating the test set. "
                             "Used in eval and top_seeds_eval modes. "
                             "Default: use 0.5 for all tasks.")
    parser.add_argument("--threshold_metric", type=str, default="sn",
                        choices=["sn", "sp", "bal_acc", "ppv", "npv", "mcc", "f1", "h"],
                        help="Consensus metric to maximize when optimizing thresholds "
                             "(only applies when --use_thresholds is set). "
                             "sn=sensitivity, sp=specificity, bal_acc=balanced accuracy, "
                             "ppv=positive predictive value, npv=negative predictive value, "
                             "mcc=Matthews correlation coefficient, f1=F1 score, h=H1 score. "
                             "Default: sn (recommended for mutagenicity: prioritise catching mutagens).")

    return parser.parse_args()


# ==============================================================================
# Logging
# ==============================================================================

def setup_logging(log_file):
    """Configure logging to both a file and stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


# ==============================================================================
# Optuna study helpers
# ==============================================================================

def save_study(study, path):
    """Pickle an Optuna study to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(study, f)
    logging.info(f"Study saved to {path}")


def load_study(path):
    """Load an Optuna study from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ==============================================================================
# Shared config loading helpers
# ==============================================================================

def load_yaml_and_graph_info(input_file):
    """
    Load YAML training config and graph database description.
    Returns (input_data dict, n_node_features int, n_edge_features int, database_path str).
    """
    with open(input_file, "r") as f:
        input_data = yaml.load(f, Loader=yaml.Loader)

    database_path = input_data.get("database", "./GraphDataBase_AMES")
    database_file = os.path.join(database_path, "graph_description.yml")
    with open(database_file, "r") as f:
        database_data = yaml.load(f, Loader=yaml.Loader)

    n_node_features = database_data.get("nNodeFeatures")
    bond_angle_features = database_data.get("BondAngleFeatures", True)
    dihedral_angle_features = database_data.get("DihedralFeatures", True)
    n_edge_features = 1  # distance
    if bond_angle_features:
        n_edge_features += 1  # bond-angle
    if dihedral_angle_features:
        n_edge_features += 1  # dihedral-angle

    return input_data, n_node_features, n_edge_features, database_path


def get_model_params(input_data):
    """Extract model architecture hyperparameters from YAML config dict."""
    return {
        "n_graph_convolution_layers": input_data.get("nGraphConvolutionLayers", 0),
        "n_node_neurons": input_data.get("nNodeNeurons", None),
        "n_edge_neurons": input_data.get("nEdgeNeurons", None),
        "dropout_GNN": input_data.get("dropoutGNN", None),
        "momentum_batch_norm": input_data.get("momentumBatchNorm", None),
        "n_shared_layers": input_data.get("nSharedLayers", 4),
        "n_target_specific_layers": input_data.get("nTargetSpecificLayers", 2),
        "n_shared": input_data.get("nShared", None),
        "n_target": input_data.get("nTarget", None),
        "dropout_shared": input_data.get("dropoutShared", None),
        "dropout_target": input_data.get("dropoutTarget", None),
        "activation": input_data.get("ActivationFunction", "ReLU"),
    }


def get_class_weights(input_data, use_yaml_weights=True):
    """
    Build the class_weights dict for the masked loss function.
    If use_yaml_weights=True, reads w1-w5 from the YAML config.
    If False, uses unit weights (1.0 for all classes).
    """
    weighted = input_data.get("weightedCostFunction", False)
    if weighted and use_yaml_weights:
        w1 = input_data.get("w1", 1.0)
        w2 = input_data.get("w2", 1.0)
        w3 = input_data.get("w3", 1.0)
        w4 = input_data.get("w4", 1.0)
        w5 = input_data.get("w5", 1.0)
        return {
            "98":   {0: 1.0, 1: w1, -1: 0},
            "100":  {0: 1.0, 1: w2, -1: 0},
            "102":  {0: 1.0, 1: w3, -1: 0},
            "1535": {0: 1.0, 1: w4, -1: 0},
            "1537": {0: 1.0, 1: w5, -1: 0},
        }
    else:
        return {
            "98":   {0: 1.0, 1: 1.0, -1: 0.0},
            "100":  {0: 1.0, 1: 1.0, -1: 0.0},
            "102":  {0: 1.0, 1: 1.0, -1: 0.0},
            "1535": {0: 1.0, 1: 1.0, -1: 0.0},
            "1537": {0: 1.0, 1: 1.0, -1: 0.0},
        }


def build_model(params, n_node_features, n_edge_features, n_inputs=0):
    """Instantiate the GNN-MTL model from a parameter dict."""
    return BuildNN_GNN_MTL(
        params["n_graph_convolution_layers"],
        params["n_node_neurons"],
        params["n_edge_neurons"],
        n_node_features,
        n_edge_features,
        params["dropout_GNN"],
        params["momentum_batch_norm"],
        params["n_shared_layers"],
        params["n_target_specific_layers"],
        params["n_shared"],
        params["n_target"],
        params["dropout_shared"],
        params["dropout_target"],
        params["activation"],
        False,  # useMolecularDescriptors (graphs only)
        n_inputs,
    )


def load_graph_datasets(database_path, nTrainMaxEntries, nValMaxEntries, seed):
    """Load train, validation, and test GraphDataSet objects."""
    trainDir = os.path.join(database_path, "train/")
    valDir = os.path.join(database_path, "validate/")
    testDir = os.path.join(database_path, "test/")

    trainDataset = GraphDataSet(trainDir, nMaxEntries=nTrainMaxEntries, seed=seed)
    valDataset = GraphDataSet(valDir, nMaxEntries=nValMaxEntries, seed=seed)
    testDataset = GraphDataSet(testDir, nMaxEntries=nValMaxEntries, seed=seed)

    return trainDataset, valDataset, testDataset


def count_trainable_parameters(model):
    """Return total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==============================================================================
# Metrics writing helper
# ==============================================================================

def compute_npv_mcc(m1):
    """Compute NPV and MCC from counts list [TP, TN, FP, FN]. Returns (npv, mcc) rounded to 2 dp."""
    tp, tn, fp, fn = m1[0], m1[1], m1[2], m1[3]
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    denom = float(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    return round(float(npv), 2), round(float(mcc), 2)


METRICS_HEADERS = ["Strain", "TP", "TN", "FP", "FN", "Sp", "Sn", "PPV", "NPV", "Acc", "Bal acc", "MCC", "F1 score", "H score"]


def metrics_row(label, m1, m2):
    """Build a CSV row from a label, counts list m1, and rates list m2."""
    sp, sn, ppv, acc, balacc, f1, h = m2
    npv, mcc = compute_npv_mcc(m1)
    return [label] + m1 + [sp, sn, ppv, npv, acc, balacc, mcc, f1, h]


def write_metrics_csv(csv_path, y_true_cat, y_pred_cat, y_logit_cat):
    """
    Write per-strain classification metrics to a CSV file.
    Columns: Strain, TP, TN, FP, FN, Sp, Sn, PPV, NPV, Acc, Bal acc, MCC, F1 score, H score
    """
    strain_names = ["Strain TA98", "Strain TA100", "Strain TA102", "Strain TA1535", "Strain TA1537"]

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(METRICS_HEADERS)
        for i, strain in enumerate(strain_names):
            _, new_real, new_y_pred, _ = filter_nan(y_true_cat[:, i], y_pred_cat[:, i], y_logit_cat[:, i])
            m = get_metrics(new_real, new_y_pred)
            m1 = [int(x) for x in m[0]]
            m2 = [round(float(x), 2) for x in m[1]]
            writer.writerow(metrics_row(strain, m1, m2))


def get_multilabel_targets(dataset):
    """Extract multilabel target tensor (N x 5) from a graph dataset."""
    targets = []
    for i in range(len(dataset)):
        y = dataset[i].y.squeeze()
        targets.append(y)
    return np.array(targets)


# ==============================================================================
# GNN inference helper
# ==============================================================================

def run_inference(model, loader, device, params, thresholds=None):
    """
    Run model inference on a DataLoader.
    Returns (y_pred_logit_cat, y_pred_binary_cat, y_true_cat, file_names).

    thresholds: list of 5 per-task decision thresholds. Defaults to [0.5]*5 if None.
    """
    if thresholds is None:
        thresholds = [0.5] * 5
    p = params
    model.eval()
    y_pred_logit_batches = []
    y_pred_batches = []
    y_true_batches = []
    file_names = []

    with torch.no_grad():
        for sample in loader:
            pred = model(
                sample.x.to(device),
                sample.edge_index.to(device),
                sample.edge_attr.to(device),
                sample.batch.to(device),
                p["n_node_neurons"], p["n_node_features"],
                p["n_edge_neurons"], p["n_edge_features"],
                p["n_graph_convolution_layers"],
                p["n_shared_layers"],
                p["n_target_specific_layers"],
                False  # useMolecularDescriptors
            )
            y_pred_t = tuple(
                torch.where(tensor > thresholds[i], torch.tensor(1), torch.tensor(0))
                for i, tensor in enumerate(pred)
            )
            y_pred_batches.append(y_pred_t)
            y_pred_logit_batches.append(pred)
            y_true_batches.append(sample.y)

            if hasattr(sample, "to_data_list"):
                for data in sample.to_data_list():
                    file_names.append(data.file_name)

    y_logit_cat = [
        np.concatenate([t.cpu().numpy() for t in tensors], axis=0)
        for tensors in zip(*y_pred_logit_batches)
    ]
    y_logit_cat = np.hstack(y_logit_cat)

    y_pred_cat = [
        np.concatenate([t.cpu().numpy() for t in tensors], axis=0)
        for tensors in zip(*y_pred_batches)
    ]
    y_pred_cat = np.hstack(y_pred_cat)

    y_true_cat = torch.cat(y_true_batches).numpy()

    return y_logit_cat, y_pred_cat, y_true_cat, file_names


# ==============================================================================
# MODE: train
# ==============================================================================

def run_train(args):
    """
    Train the GNN-MTL model with fixed hyperparameters from the YAML config.
    Saves checkpoints every nCheckpoint epochs to --checkpoints_dir.
    Logs train/val loss to TensorBoard. Evaluates on test set after training.

    Ported from GNN_MTL_GPU.py.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, "training.log"))

    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load config
    input_data, n_node_features, n_edge_features, database_path = load_yaml_and_graph_info(args.input_file)
    params = get_model_params(input_data)
    params["n_node_features"] = n_node_features
    params["n_edge_features"] = n_edge_features
    class_weights = get_class_weights(input_data, use_yaml_weights=True)
    output_keys = ["98", "100", "102", "1535", "1537"]

    nEpochs = input_data.get("nEpochs", 10)
    nBatch = input_data.get("nBatch", 50)
    chkptFreq = input_data.get("nCheckpoint", 10)
    seed = input_data.get("randomSeed", 42)
    nTrainMaxEntries = input_data.get("nTrainMaxEntries", None)
    nValMaxEntries = input_data.get("nValMaxEntries", None)
    learningRate = input_data.get("learningRate", 0.0001)
    L2Regularization = input_data.get("L2Regularization", 0.005)
    loadModel = input_data.get("loadModel", False)
    loadOptimizer = input_data.get("loadOptimizer", False)
    useMolecularDescriptors = input_data.get("useMolecularDescriptors", False)
    data_path = args.data_file or input_data.get("data_file",
                os.path.join(AMES_FINAL_DIR, "data.csv"))

    # Build log text
    log_text = "\n# Model Description\n"
    log_text += f"- nGraphConvolutionLayers: {params['n_graph_convolution_layers']}\n"
    log_text += f"- nSharedLayers: {params['n_shared_layers']}\n"
    log_text += f"- nTargetSpecificLayers: {params['n_target_specific_layers']}\n"
    log_text += f"- nEpochs: {nEpochs}\n"
    log_text += f"- nBatch: {nBatch}\n"
    log_text += f"- learningRate: {learningRate}\n"
    log_text += f"- random seed: {seed}\n"

    g = torch.Generator()
    g.manual_seed(seed)

    if not useMolecularDescriptors:
        # --- Graph mode ---
        trainDataset, valDataset, testDataset = load_graph_datasets(
            database_path, nTrainMaxEntries, nValMaxEntries, seed
        )
        trainLoader = DataLoader(trainDataset, batch_size=nBatch, generator=g)
        valLoader = DataLoader(valDataset, batch_size=nBatch, generator=g)
        testLoader = DataLoader(testDataset, batch_size=nBatch, generator=g)

        model = build_model(params, n_node_features, n_edge_features)
        logging.info(f"Total trainable parameters: {count_trainable_parameters(model)}")

        optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=L2Regularization)

        if loadModel:
            loadModelFileName = input_data["StateDictFileName"]
            checkpoint = torch.load(loadModelFileName, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            if loadOptimizer:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            n_start = checkpoint["epoch"]
            log_text += f"- Starting from checkpoint: {loadModelFileName}\n"
        else:
            n_start = 0

        anyCallBacks = input_data.get("callbacks", None)
        callbacks = set_up_callbacks(anyCallBacks, optimizer)

        model = model.to(device)

        for epoch in range(n_start, nEpochs):
            # Training
            model.train()
            train_loss = 0
            for sample in trainLoader:
                pred = model(
                    sample.x.to(device), sample.edge_index.to(device),
                    sample.edge_attr.to(device), sample.batch.to(device),
                    params["n_node_neurons"], n_node_features,
                    params["n_edge_neurons"], n_edge_features,
                    params["n_graph_convolution_layers"],
                    params["n_shared_layers"], params["n_target_specific_layers"], False
                )
                losses = sum(
                    masked_loss_function(sample.y[:, i], pred[i].squeeze(1), class_weights[output_keys[i]])
                    for i in range(5)
                )
                loss_final = losses / 5
                optimizer.zero_grad()
                loss_final.backward()
                optimizer.step()
                train_loss += loss_final.item()
            train_loss /= len(trainLoader)

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for sample in valLoader:
                    pred = model(
                        sample.x.to(device), sample.edge_index.to(device),
                        sample.edge_attr.to(device), sample.batch.to(device),
                        params["n_node_neurons"], n_node_features,
                        params["n_edge_neurons"], n_edge_features,
                        params["n_graph_convolution_layers"],
                        params["n_shared_layers"], params["n_target_specific_layers"], False
                    )
                    losses = sum(
                        masked_loss_function(sample.y[:, i], pred[i].squeeze(1), class_weights[output_keys[i]])
                        for i in range(5)
                    )
                    val_loss += (losses / 5).item()
            val_loss /= len(valLoader)

            current_lr = optimizer.param_groups[0]["lr"]
            logging.info(
                f"Epoch {epoch + 1}/{nEpochs}  "
                f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  lr={current_lr:.2e}"
            )

            # Checkpoint
            if (epoch + 1) % chkptFreq == 0:
                ckpt_path = os.path.join(args.checkpoints_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }, ckpt_path)

            # Callbacks: pass val_loss (not train_loss) to both LRScheduler and EarlyStopping
            lr_before = optimizer.param_groups[0]["lr"]
            for callback in callbacks:
                callback(val_loss)
            lr_after = optimizer.param_groups[0]["lr"]
            if lr_after < lr_before:
                logging.info(f"  LR reduced: {lr_before:.2e} → {lr_after:.2e}")

            if any(cb.early_stop for cb in callbacks):
                logging.info(f"  Early stopping triggered at epoch {epoch + 1}.")
                ckpt_path = os.path.join(args.checkpoints_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }, ckpt_path)
                break

            # TensorBoard
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)

        writer.close()

        # Test set evaluation
        y_logit_cat, y_pred_cat, y_true_cat, _ = run_inference(model, testLoader, device, params)

        write_metrics_csv(
            os.path.join(args.output_dir, "metrics.csv"),
            y_true_cat, y_pred_cat, y_logit_cat
        )

        # Consensus prediction (OR rule across 5 heads)
        y_cons = np.where(np.any(y_pred_cat == 1, axis=1), 1,
                 np.where(np.all(y_pred_cat == 0, axis=1), 0, -1))
        y_cons_true = np.where(np.any(y_true_cat == 1, axis=1), 1,
                      np.where(np.all(y_true_cat == 0, axis=1), 0, -1))

        csv_file = os.path.join(args.output_dir, "metrics_cons.csv")
        with open(csv_file, mode="w", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(METRICS_HEADERS)
            _, new_real, new_y_pred, _ = filter_nan(y_cons_true, y_cons, y_cons)
            m = get_metrics(new_real, new_y_pred)
            m1 = [int(x) for x in m[0]]
            m2 = [round(float(x), 2) for x in m[1]]
            writer_csv.writerow(metrics_row("Cons", m1, m2))

    else:
        # --- Molecular descriptors mode ---
        train_data, internal_data, external_data = load_data(data_path, model="MTL", stage="GS")
        X_train, y_train = train_data
        X_internal, y_internal = internal_data
        X_external, y_external = external_data

        # Remove SMILES column and reformat
        X_train = np.array(X_train[:, 1:], dtype=np.float32)
        X_internal = np.array(X_internal[:, 1:], dtype=np.float32)
        X_external = np.array(X_external[:, 1:], dtype=np.float32)
        y_train = np.array(np.transpose(y_train), dtype=np.float32)
        y_internal = np.array(np.transpose(y_internal), dtype=np.float32)
        y_external = np.array(np.transpose(y_external), dtype=np.float32)

        if nTrainMaxEntries:
            X_train, y_train = X_train[:nTrainMaxEntries], y_train[:nTrainMaxEntries]
        if nValMaxEntries:
            X_internal, y_internal = X_internal[:nValMaxEntries], y_internal[:nValMaxEntries]

        train_dataset = MTLDataset(torch.tensor(X_train), torch.tensor(y_train))
        val_dataset = MTLDataset(torch.tensor(X_internal), torch.tensor(y_internal))
        test_dataset = MTLDataset(torch.tensor(X_external), torch.tensor(y_external))

        n_inputs = X_train.shape[1]
        trainLoader = DataLoader(train_dataset, batch_size=nBatch, shuffle=True, generator=g)
        valLoader = DataLoader(val_dataset, batch_size=nBatch, generator=g)
        testLoader = DataLoader(test_dataset, batch_size=nBatch, generator=g)

        model = BuildNN_GNN_MTL(
            params["n_graph_convolution_layers"], params["n_node_neurons"],
            params["n_edge_neurons"], n_node_features, n_edge_features,
            params["dropout_GNN"], params["momentum_batch_norm"],
            params["n_shared_layers"], params["n_target_specific_layers"],
            params["n_shared"], params["n_target"],
            params["dropout_shared"], params["dropout_target"],
            params["activation"], True, n_inputs
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=L2Regularization)

        if loadModel:
            loadModelFileName = input_data["StateDictFileName"]
            checkpoint = torch.load(loadModelFileName, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            if loadOptimizer:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            n_start = checkpoint["epoch"]
        else:
            n_start = 0

        anyCallBacks = input_data.get("callbacks", None)
        callbacks = set_up_callbacks(anyCallBacks, optimizer)

        model = model.to(device)

        for epoch in range(n_start, nEpochs):
            model.train()
            train_loss = 0
            for X, y in trainLoader:
                pred = model(X.to(device), 0, 0, 0,
                             params["n_node_neurons"], n_node_features,
                             params["n_edge_neurons"], n_edge_features,
                             params["n_graph_convolution_layers"],
                             params["n_shared_layers"], params["n_target_specific_layers"], True)
                losses = sum(
                    masked_loss_function(y[:, i], pred[i].squeeze(1), class_weights[output_keys[i]])
                    for i in range(5)
                )
                loss_final = losses / 5
                optimizer.zero_grad()
                loss_final.backward()
                optimizer.step()
                train_loss += loss_final.item()
            train_loss /= len(trainLoader)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X, y in valLoader:
                    pred = model(X.to(device), 0, 0, 0,
                                 params["n_node_neurons"], n_node_features,
                                 params["n_edge_neurons"], n_edge_features,
                                 params["n_graph_convolution_layers"],
                                 params["n_shared_layers"], params["n_target_specific_layers"], True)
                    losses = sum(
                        masked_loss_function(y[:, i], pred[i].squeeze(1), class_weights[output_keys[i]])
                        for i in range(5)
                    )
                    val_loss += (losses / 5).item()
            val_loss /= len(valLoader)

            current_lr = optimizer.param_groups[0]["lr"]
            logging.info(
                f"Epoch {epoch + 1}/{nEpochs}  "
                f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  lr={current_lr:.2e}"
            )

            if (epoch + 1) % chkptFreq == 0:
                ckpt_path = os.path.join(args.checkpoints_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }, ckpt_path)

            # Callbacks: pass val_loss (not train_loss) to both LRScheduler and EarlyStopping
            lr_before = optimizer.param_groups[0]["lr"]
            for callback in callbacks:
                callback(val_loss)
            lr_after = optimizer.param_groups[0]["lr"]
            if lr_after < lr_before:
                logging.info(f"  LR reduced: {lr_before:.2e} → {lr_after:.2e}")

            if any(cb.early_stop for cb in callbacks):
                logging.info(f"  Early stopping triggered at epoch {epoch + 1}.")
                ckpt_path = os.path.join(args.checkpoints_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }, ckpt_path)
                break

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)

        writer.close()

        # Test set evaluation (molecular descriptors)
        model.eval()
        X_external_tensor = torch.tensor(X_external, dtype=torch.float32)
        with torch.no_grad():
            y_pred = model(
                X_external_tensor.to(device), 0, 0, 0,
                params["n_node_neurons"], n_node_features,
                params["n_edge_neurons"], n_edge_features,
                params["n_graph_convolution_layers"],
                params["n_shared_layers"], params["n_target_specific_layers"], True
            )
        y_pred = [yp.detach().cpu().numpy() for yp in y_pred]

        y_pred_binary = [np.where(y > 0.5, 1, 0) for y in y_pred]

        csv_file = os.path.join(args.output_dir, "metrics.csv")
        strain_names_out = ["Strain TA98", "Strain TA100", "Strain TA102", "Strain TA1535", "Strain TA1537"]
        with open(csv_file, mode="w", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(METRICS_HEADERS)
            for i, strain in enumerate(strain_names_out):
                _, new_real, new_y_pred, new_prob = filter_nan(
                    y_internal[:, i], y_pred_binary[i], y_pred[i]
                )
                m = get_metrics(new_real, new_y_pred)
                m1 = [int(x) for x in m[0]]
                m2 = [round(float(x), 2) for x in m[1]]
                writer_csv.writerow(metrics_row(strain, m1, m2))

        # Consensus
        overall = load_data(data_path, model="Overall", stage="EVAL")
        y_overall = overall[1]
        y_cons = np.zeros(len(y_overall))
        for i in range(len(y_overall)):
            preds_i = [y_pred_binary[t][i] for t in range(5)]
            if 1 in preds_i:
                y_cons[i] = 1
            elif all(p == 0 for p in preds_i):
                y_cons[i] = 0
            else:
                y_cons[i] = -1

        csv_file = os.path.join(args.output_dir, "metrics_cons.csv")
        with open(csv_file, mode="w", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(METRICS_HEADERS)
            _, new_real, new_y_pred, _ = filter_nan(y_overall, y_cons, y_cons)
            m = get_metrics(new_real, new_y_pred)
            m1 = [int(x) for x in m[0]]
            m2 = [round(float(x), 2) for x in m[1]]
            writer_csv.writerow(metrics_row("Cons", m1, m2))

    logging.info(log_text)
    sys.stdout.flush()
    logging.info("train mode complete.")


# ==============================================================================
# MODE: hp_opt
# ==============================================================================

def run_hp_opt(args):
    """
    Hyperparameter optimization with Optuna.
    Runs 5-fold CV on the training set for each trial.
    Saves/resumes the study as a .pkl file in --optuna_dir.

    Ported from GNN_MTL_HP_KF.py (using BuildNN_GNN_MTL_GINEConv throughout).
    """
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.optuna_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, "hp_opt.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Determine study pkl path
    if args.optuna_file:
        study_pkl_path = args.optuna_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d")
        study_pkl_path = os.path.join(args.optuna_dir, f"study_{timestamp}.pkl")

    # Load or create Optuna study
    if os.path.exists(study_pkl_path):
        study = load_study(study_pkl_path)
        logging.info(f"Resumed study from {study_pkl_path}")
    else:
        study = optuna.create_study(direction="minimize")
        logging.info("Created new Optuna study.")

    # Load config (shared across all trials)
    input_data, n_node_features, n_edge_features, database_path = load_yaml_and_graph_info(args.input_file)

    seed_global = input_data.get("randomSeed", 42)
    nEpochs = input_data.get("nEpochs", 10)
    nBatch = input_data.get("nBatch", 50)
    L2Regularization = input_data.get("L2Regularization", 0.005)
    activation = input_data.get("ActivationFunction", "ReLU")
    weighted_loss_function = input_data.get("weightedCostFunction", False)
    nTrainMaxEntries = input_data.get("nTrainMaxEntries", None)

    # Load training data (5-fold CV on train set only)
    trainDataset = GraphDataSet(
        os.path.join(database_path, "train/"),
        nMaxEntries=nTrainMaxEntries, seed=seed_global
    )

    # Precompute multilabel targets for stratified splitting
    X_indices = np.arange(len(trainDataset))
    y_multilabel = get_multilabel_targets(trainDataset)

    def objective(trial):
        """Optuna objective: 5-fold CV with trial-suggested hyperparameters."""
        logging.info(f"Starting trial {trial.number}")

        # Suggest hyperparameters
        n_graph_convolution_layers = trial.suggest_int("nGraphConvolutionalLayers", 1, 5)
        n_node_neurons = trial.suggest_int("n_node_neurons", 1, 300)
        n_edge_neurons = trial.suggest_int("n_edge_neurons", 1, 300)
        dropout_GNN = trial.suggest_float("DropoutGNN", 0.0, 0.5)
        momentum_batch_norm = trial.suggest_float("momentumBatchNorm", 0.0, 1.0)
        n_shared_layers = trial.suggest_int("nSharedLayers", 1, 4)
        n_target_specific_layers = trial.suggest_int("nTargetSpecificLayers", 1, 3)
        n_shared = [trial.suggest_int(f"n_shared_{i}", 1, 300) for i in range(n_shared_layers)]
        n_target = [trial.suggest_int(f"n_target_{i}", 1, 300) for i in range(n_target_specific_layers)]
        dropout_shared = [trial.suggest_float(f"DropoutShared_{i}", 0.0, 0.5) for i in range(n_shared_layers)]
        dropout_target = [trial.suggest_float(f"DropoutTarget_{i}", 0.0, 0.5) for i in range(n_target_specific_layers)]
        learningRate = trial.suggest_float("learningRate", 1e-5, 1e-1, log=True)

        if weighted_loss_function:
            w1 = trial.suggest_float("w1", 1.0, 6.0)
            w2 = trial.suggest_float("w2", 1.0, 6.0)
            w3 = trial.suggest_float("w3", 1.0, 6.0)
            w4 = trial.suggest_float("w4", 1.0, 6.0)
            w5 = trial.suggest_float("w5", 1.0, 6.0)
            class_weights = {
                "98":   {0: 1.0, 1: w1, -1: 0},
                "100":  {0: 1.0, 1: w2, -1: 0},
                "102":  {0: 1.0, 1: w3, -1: 0},
                "1535": {0: 1.0, 1: w4, -1: 0},
                "1537": {0: 1.0, 1: w5, -1: 0},
            }
        else:
            class_weights = {k: {0: 1.0, 1: 1.0, -1: 0.0} for k in ["98", "100", "102", "1535", "1537"]}
        output_keys = ["98", "100", "102", "1535", "1537"]

        g = torch.Generator()
        g.manual_seed(seed_global)
        mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        val_losses = []
        val_loss_log = []

        for fold, (train_idx, val_idx) in enumerate(mskf.split(X_indices, y_multilabel)):
            train_subset = Subset(trainDataset, train_idx)
            val_subset = Subset(trainDataset, val_idx)
            trainLoader = DataLoader(train_subset, batch_size=nBatch, generator=g)
            valLoader = DataLoader(val_subset, batch_size=nBatch, generator=g)

            model = BuildNN_GNN_MTL(
                n_graph_convolution_layers, n_node_neurons, n_edge_neurons,
                n_node_features, n_edge_features, dropout_GNN, momentum_batch_norm,
                n_shared_layers, n_target_specific_layers, n_shared, n_target,
                dropout_shared, dropout_target, activation, False, 0
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=L2Regularization)

            for epoch in range(nEpochs):
                model.train()
                train_loss = 0
                for sample in trainLoader:
                    pred = model(
                        sample.x.to(device), sample.edge_index.to(device),
                        sample.edge_attr.to(device), sample.batch.to(device),
                        n_node_neurons, n_node_features, n_edge_neurons, n_edge_features,
                        n_graph_convolution_layers, n_shared_layers, n_target_specific_layers, False
                    )
                    losses = sum(
                        masked_loss_function(sample.y[:, i], pred[i].squeeze(1), class_weights[output_keys[i]])
                        for i in range(5)
                    )
                    loss_final = losses / 5
                    optimizer.zero_grad()
                    loss_final.backward()
                    optimizer.step()
                    train_loss += loss_final.item()
                train_loss /= len(trainLoader)

                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for sample in valLoader:
                        pred = model(
                            sample.x.to(device), sample.edge_index.to(device),
                            sample.edge_attr.to(device), sample.batch.to(device),
                            n_node_neurons, n_node_features, n_edge_neurons, n_edge_features,
                            n_graph_convolution_layers, n_shared_layers, n_target_specific_layers, False
                        )
                        losses = sum(
                            masked_loss_function(sample.y[:, i], pred[i].squeeze(1), class_weights[output_keys[i]])
                            for i in range(5)
                        )
                        val_loss += (losses / 5).item()
                val_loss /= len(valLoader)

                val_loss_log.append({"fold": fold, "epoch": epoch, "val_loss": val_loss})

            val_losses.append(val_loss)
            trial.set_user_attr("val_loss_log", val_loss_log)
            if "best_fold_loss" not in trial.user_attrs or val_loss < trial.user_attrs["best_fold_loss"]:
                trial.set_user_attr("best_fold_loss", val_loss)
                trial.set_user_attr("best_fold", fold)

        avg_val_loss = sum(val_losses) / len(val_losses)
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Periodically save study
        if trial.number % 10 == 0:
            save_study(study, study_pkl_path)

        return avg_val_loss

    # Run optimization
    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)
    save_study(study, study_pkl_path)

    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    logging.info(f"Optimization complete. Finished trials: {len(study.trials)}, "
                 f"Complete trials: {len(complete_trials)}")

    best = study.best_trial
    logging.info(f"Best trial: value={best.value:.6f}")
    for key, value in best.params.items():
        logging.info(f"  {key}: {value}")


# ==============================================================================
# MODE: seeds_cfv
# ==============================================================================

def run_seeds_cfv(args):
    """
    5-fold cross-validation across multiple random seeds with fixed hyperparameters.
    Saves per-fold checkpoints to --checkpoints_dir and per-fold metrics CSVs to --output_dir.
    Also saves avg_val_losses.csv and val_losses.csv summary files.

    Ported from GNN_MTL_HP_KF_seeds.py.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, "seeds_cfv.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    random_seeds = [int(s) for s in args.seeds.split()]
    logging.info(f"Running seeds_cfv with seeds: {random_seeds}")

    # Load config
    input_data, n_node_features, n_edge_features, database_path = load_yaml_and_graph_info(args.input_file)
    params = get_model_params(input_data)
    params["n_node_features"] = n_node_features
    params["n_edge_features"] = n_edge_features
    class_weights = get_class_weights(input_data, use_yaml_weights=True)
    output_keys = ["98", "100", "102", "1535", "1537"]

    nEpochs = input_data.get("nEpochs", 10)
    nBatch = input_data.get("nBatch", 50)
    learningRate = input_data.get("learningRate", 0.0001)
    L2Regularization = input_data.get("L2Regularization", 0.005)
    nTrainMaxEntries = input_data.get("nTrainMaxEntries", None)
    nValMaxEntries = input_data.get("nValMaxEntries", None)

    # Load train + val datasets (combined for 5-fold CV)
    trainDataset = GraphDataSet(
        os.path.join(database_path, "train/"), nMaxEntries=nTrainMaxEntries, seed=42
    )
    valDataset = GraphDataSet(
        os.path.join(database_path, "validate/"), nMaxEntries=nValMaxEntries, seed=42
    )
    full_dataset = trainDataset + valDataset

    X_indices = np.arange(len(full_dataset))
    y_multilabel = get_multilabel_targets(full_dataset)

    # Collect avg val losses per seed (for top_seeds_eval)
    avg_val_losses_per_seed = {}
    # Per-fold losses per seed: {seed: [fold0_loss, fold1_loss, ...]}
    all_fold_losses = {}

    for seed in random_seeds:
        logging.info(f"Seed: {seed}")
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)

        g = torch.Generator()
        g.manual_seed(seed)

        mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        val_losses_this_seed = []

        for fold, (train_idx, val_idx) in enumerate(mskf.split(X_indices, y_multilabel)):
            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)
            trainLoader = DataLoader(train_subset, batch_size=nBatch, generator=g)
            valLoader = DataLoader(val_subset, batch_size=nBatch, generator=g)

            model = build_model(params, n_node_features, n_edge_features).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=L2Regularization)

            for epoch in range(nEpochs):
                model.train()
                train_loss = 0
                for sample in trainLoader:
                    pred = model(
                        sample.x.to(device), sample.edge_index.to(device),
                        sample.edge_attr.to(device), sample.batch.to(device),
                        params["n_node_neurons"], n_node_features,
                        params["n_edge_neurons"], n_edge_features,
                        params["n_graph_convolution_layers"],
                        params["n_shared_layers"], params["n_target_specific_layers"], False
                    )
                    losses = sum(
                        masked_loss_function(sample.y[:, i], pred[i].squeeze(1), class_weights[output_keys[i]])
                        for i in range(5)
                    )
                    loss_final = losses / 5
                    optimizer.zero_grad()
                    loss_final.backward()
                    optimizer.step()
                    train_loss += loss_final.item()
                train_loss /= len(trainLoader)

                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for sample in valLoader:
                        pred = model(
                            sample.x.to(device), sample.edge_index.to(device),
                            sample.edge_attr.to(device), sample.batch.to(device),
                            params["n_node_neurons"], n_node_features,
                            params["n_edge_neurons"], n_edge_features,
                            params["n_graph_convolution_layers"],
                            params["n_shared_layers"], params["n_target_specific_layers"], False
                        )
                        losses = sum(
                            masked_loss_function(sample.y[:, i], pred[i].squeeze(1), class_weights[output_keys[i]])
                            for i in range(5)
                        )
                        val_loss += (losses / 5).item()
                val_loss /= len(valLoader)

            val_losses_this_seed.append(val_loss)

            # Save checkpoint
            ckpt_path = os.path.join(args.checkpoints_dir, f"metrics_{seed}_{fold}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": val_loss,
            }, ckpt_path)

            # Evaluate on validation fold and save per-fold metrics
            y_logit_cat, y_pred_cat, y_true_cat, _ = run_inference(model, valLoader, device, params)
            csv_file = os.path.join(args.output_dir, f"metrics_{seed}_{fold}.csv")
            write_metrics_csv(csv_file, y_true_cat, y_pred_cat, y_logit_cat)

        avg_val_loss = sum(val_losses_this_seed) / len(val_losses_this_seed)
        avg_val_losses_per_seed[seed] = avg_val_loss
        all_fold_losses[seed] = val_losses_this_seed
        logging.info(f"Seed {seed}: avg val loss = {avg_val_loss:.6f}")

    # Write avg_val_losses.csv — one column per seed, one data row
    seed_headers = [str(s) for s in random_seeds]
    with open(os.path.join(args.output_dir, "avg_val_losses.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(seed_headers)
        writer.writerow([avg_val_losses_per_seed[s] for s in random_seeds])

    # Write val_losses.csv matching the expected format:
    #   Row 0: ,,Seed,...
    #   Row 1: Fold,,seed1,seed2,...
    #   Rows 2-6: ,fold_idx,loss,...
    #   Empty row
    #   Last row: ,Average,avg_loss,...
    n_folds = 5
    n_seeds = len(random_seeds)
    val_losses_path = os.path.join(args.output_dir, "val_losses.csv")
    with open(val_losses_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", "", "Seed"] + [""] * (n_seeds - 1))
        writer.writerow(["Fold", ""] + seed_headers)
        for fold in range(n_folds):
            writer.writerow(["", fold] + [all_fold_losses[s][fold] for s in random_seeds])
        writer.writerow([""] * (2 + n_seeds))
        writer.writerow(["", "Average"] + [avg_val_losses_per_seed[s] for s in random_seeds])
    logging.info(f"Wrote val_losses.csv to {val_losses_path}")

    logging.info("seeds_cfv mode complete.")


# ==============================================================================
# Threshold optimization helpers (for eval and top_seeds_eval modes)
# ==============================================================================

def consensus_from_heads(y_pred_cat):
    """Apply OR rule across 5 task heads to produce consensus predictions."""
    N = y_pred_cat.shape[0]
    y_cons = np.zeros(N, dtype=int)
    for i in range(N):
        row = y_pred_cat[i]
        if np.any(row == 1):
            y_cons[i] = 1
        elif np.all(row == 0):
            y_cons[i] = 0
        else:
            y_cons[i] = -1
    return y_cons


def consensus_truth(y_true_cat):
    """Apply OR rule to ground-truth labels to produce consensus truth."""
    N = y_true_cat.shape[0]
    y_cons_true = np.zeros(N, dtype=int)
    for i in range(N):
        row = y_true_cat[i]
        if np.any(row == 1):
            y_cons_true[i] = 1
        elif np.all(row == 0):
            y_cons_true[i] = 0
        else:
            y_cons_true[i] = -1
    return y_cons_true


def eval_consensus_metric(y_true_cat, y_logit_cat, thresholds, metric="balanced_accuracy"):
    """Apply per-task thresholds, compute consensus, and return the specified metric."""
    y_pred_cat = (y_logit_cat >= np.array(thresholds)[None, :]).astype(int)
    y_cons_pred = consensus_from_heads(y_pred_cat)
    y_cons_true = consensus_truth(y_true_cat)
    _, new_real, new_y_pred, _ = filter_nan(y_cons_true, y_cons_pred, y_cons_pred)
    counts, scores = get_metrics(new_real, new_y_pred)
    sp, sn, prec, acc, balacc, f1, h = scores
    return (balacc if metric == "bal_acc" else h), (sp, sn, prec, acc, balacc, f1, h)


def one_se_choice(th_grid, scores):
    """Return the threshold within 1 standard error of the best score (1-SE rule)."""
    scores = np.array(scores, dtype=float)
    best = np.max(scores)
    se = np.std(scores, ddof=1) / np.sqrt(len(scores)) if len(scores) > 1 else 0.0
    eligible = [t for t, s in zip(th_grid, scores) if s >= best - se]
    return float(np.median(eligible)) if eligible else float(th_grid[int(np.argmax(scores))])


def coord_ascent_consensus(y_true_cat, y_prob_cat, init_th=None, metric="sn", rounds=3):
    """
    Coordinate ascent over the 5 per-task thresholds using the 1-SE rule.
    Returns (thresholds, best_metric_value, best_scores).
    """
    ths = [0.5] * 5 if init_th is None else list(init_th)
    grid = np.linspace(0.05, 0.95, 19)
    best_val, best_scores = eval_consensus_metric(y_true_cat, y_prob_cat, ths, metric)
    for _ in range(rounds):
        improved = False
        for h in range(5):
            scores = []
            for t in grid:
                trial_ths = ths.copy()
                trial_ths[h] = float(t)
                val, _ = eval_consensus_metric(y_true_cat, y_prob_cat, trial_ths, metric)
                scores.append(val)
            t_star = one_se_choice(grid, scores)
            trial_ths = ths.copy()
            trial_ths[h] = t_star
            val, scr = eval_consensus_metric(y_true_cat, y_prob_cat, trial_ths, metric)
            if val > best_val + 1e-9:
                ths[h] = float(t_star)
                best_val, best_scores = val, scr
                improved = True
        if not improved:
            break
    return ths, best_val, best_scores


def crossfit_thresholds_for_consensus(y_true_cat, y_prob_cat, K=5, metric="sn", seed=0):
    """
    K-fold cross-fitting on the validation set to find robust consensus thresholds.
    Learns thresholds on K-1 folds; aggregates via median across folds.
    Returns list of 5 thresholds.
    """
    N = len(y_true_cat)
    idx = np.arange(N)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, K)

    ths_per_fold = []
    for k in range(K):
        tr_idx = np.concatenate([folds[j] for j in range(K) if j != k])
        ths_k, _, _ = coord_ascent_consensus(y_true_cat[tr_idx], y_prob_cat[tr_idx], metric=metric)
        ths_per_fold.append(ths_k)

    ths_final = np.median(np.array(ths_per_fold), axis=0).tolist()
    return ths_final


# ==============================================================================
# MODE: eval
# ==============================================================================

def run_eval(args):
    """
    Load a saved checkpoint, optimize consensus thresholds on the validation set,
    evaluate on the test set, and save comprehensive metrics and raw outputs.

    Ported from GNN_MTL_eval.py.
    """
    if args.checkpoint_file is None:
        raise ValueError("--checkpoint_file is required for eval mode.")

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, "eval.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load config
    input_data, n_node_features, n_edge_features, database_path = load_yaml_and_graph_info(args.input_file)
    params = get_model_params(input_data)
    params["n_node_features"] = n_node_features
    params["n_edge_features"] = n_edge_features

    # Note: GNN_MTL_eval.py uses hardcoded class weights when weightedCostFunction=True;
    # we preserve that behavior here.
    weighted = input_data.get("weightedCostFunction", False)
    if weighted:
        class_weights = {
            "98":   {0: 1.0, 1: 1.330, -1: 0.0},
            "100":  {0: 1.0, 1: 1.149, -1: 0.0},
            "102":  {0: 1.0, 1: 1.799, -1: 0.0},
            "1535": {0: 1.0, 1: 2.907, -1: 0.0},
            "1537": {0: 1.0, 1: 2.941, -1: 0.0},
        }
    else:
        class_weights = {k: {0: 1.0, 1: 1.0, -1: 0.0} for k in ["98", "100", "102", "1535", "1537"]}

    seed = input_data.get("randomSeed", 42)
    nBatch = input_data.get("nBatch", 50)
    nTrainMaxEntries = input_data.get("nTrainMaxEntries", None)
    nValMaxEntries = input_data.get("nValMaxEntries", None)
    data_path = args.data_file or os.path.join(AMES_FINAL_DIR, "data.csv")

    trainDataset, valDataset, testDataset = load_graph_datasets(
        database_path, nTrainMaxEntries, nValMaxEntries, seed
    )
    g = torch.Generator()
    g.manual_seed(seed)
    valLoader = DataLoader(valDataset, batch_size=nBatch, generator=g)
    testLoader = DataLoader(testDataset, batch_size=nBatch, generator=g)

    # Build model and load checkpoint
    model = build_model(params, n_node_features, n_edge_features)
    checkpoint = torch.load(args.checkpoint_file, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # --- Threshold selection ---
    if args.use_thresholds:
        # Collect validation set logits for cross-fit threshold optimization
        y_pred_logit_val = []
        y_true_val = []
        with torch.no_grad():
            for sample in valLoader:
                pred = model(
                    sample.x.to(device), sample.edge_index.to(device),
                    sample.edge_attr.to(device), sample.batch.to(device),
                    params["n_node_neurons"], n_node_features,
                    params["n_edge_neurons"], n_edge_features,
                    params["n_graph_convolution_layers"],
                    params["n_shared_layers"], params["n_target_specific_layers"], False
                )
                y_pred_logit_val.append(pred)
                y_true_val.append(sample.y)

        y_logit_val = np.hstack([
            np.concatenate([t.cpu().numpy() for t in tensors], axis=0)
            for tensors in zip(*y_pred_logit_val)
        ])
        y_true_val = torch.cat(y_true_val).numpy()

        best_ths = crossfit_thresholds_for_consensus(
            y_true_cat=y_true_val, y_prob_cat=y_logit_val, K=5, metric="sp", seed=42
        )
        logging.info(f"Cross-fit consensus thresholds: {best_ths}")
        _, val_scores = eval_consensus_metric(y_true_val, y_logit_val, best_ths, metric="bal_acc")
        logging.info(f"Validation consensus scores (Sp, Sn, PPV, Acc, BalAcc, F1, H): {val_scores}")
    else:
        best_ths = [0.5] * 5
        logging.info("Using default thresholds [0.5] x5. Pass --use_thresholds to enable optimization.")

    # --- Test set evaluation ---
    logging.info(f"Evaluating test set with thresholds: {best_ths}")
    y_logit_cat, y_pred_cat, y_true_cat, file_names = run_inference(
        model, testLoader, device, params, thresholds=best_ths
    )

    # Per-strain metrics
    write_metrics_csv(
        os.path.join(args.output_dir, "metrics.csv"),
        y_true_cat, y_pred_cat, y_logit_cat
    )

    # Read overall labels from data.csv
    df_data = pd.read_csv(data_path)
    y_overall = df_data["Overall"].values
    index = []
    for file_path in file_names:
        filename = os.path.basename(file_path)
        match = re.match(r"^(\d+)_", filename)
        first_number = int(match.group(1))
        index.append(first_number - 1)
    y_labels_overall = y_overall[index]

    # Consensus prediction and metrics
    y_cons = np.where(np.any(y_pred_cat == 1, axis=1), 1,
             np.where(np.all(y_pred_cat == 0, axis=1), 0, -1))
    y_cons_true = np.where(np.any(y_true_cat == 1, axis=1), 1,
                  np.where(np.all(y_true_cat == 0, axis=1), 0, -1))

    # Misclassified molecules
    wrong_indices = np.where(y_cons != y_cons_true)[0]
    df_wrong = pd.DataFrame({
        "file_name": [file_names[i] for i in wrong_indices],
        "true_label": [y_cons_true[i] for i in wrong_indices],
        "pred_label": [y_cons[i] for i in wrong_indices],
    })
    df_wrong.to_csv(os.path.join(args.output_dir, "misclassified_files.csv"), index=False)

    csv_file = os.path.join(args.output_dir, "metrics_cons.csv")
    with open(csv_file, mode="w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(METRICS_HEADERS)
        _, new_real, new_y_pred, _ = filter_nan(y_cons_true, y_cons, y_cons)
        m = get_metrics(new_real, new_y_pred)
        m1 = [int(x) for x in m[0]]
        m2 = [round(float(x), 2) for x in m[1]]
        writer_csv.writerow(metrics_row("Consensus", m1, m2))

    # Raw model outputs
    external = load_data(data_path, model="MTL", stage="EVAL")
    x_test = external[0]

    csv_file = os.path.join(args.output_dir, "model_output_raw.csv")
    headers_raw = [
        "file", "logits_98", "logits_100", "logits_102", "logits_1535", "logits_1537",
        "y_true_98", "y_true_100", "y_true_102", "y_true_1535", "y_true_1537",
        "y_pred_98", "y_pred_100", "y_pred_102", "y_pred_1535", "y_pred_1537",
        "y_true_consensus", "y_pred_consensus"
    ]
    with open(csv_file, mode="w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(headers_raw)
        rows = zip(
            np.array(file_names).flatten(),
            y_logit_cat[:, 0], y_logit_cat[:, 1], y_logit_cat[:, 2], y_logit_cat[:, 3], y_logit_cat[:, 4],
            y_true_cat[:, 0], y_true_cat[:, 1], y_true_cat[:, 2], y_true_cat[:, 3], y_true_cat[:, 4],
            y_pred_cat[:, 0], y_pred_cat[:, 1], y_pred_cat[:, 2], y_pred_cat[:, 3], y_pred_cat[:, 4],
            np.array(y_labels_overall).flatten(),
            np.array(y_cons).flatten()
        )
        writer_csv.writerows(rows)

    logging.info("eval mode complete.")
    sys.stdout.flush()


# ==============================================================================
# MODE: top_seeds_eval
# ==============================================================================

def run_top_seeds_eval(args):
    """
    Automatically select the top N seeds by lowest average validation loss,
    load their checkpoints, evaluate on the test set, and average the metrics.

    Reads avg_val_losses.csv from --metrics_dir (produced by seeds_cfv mode).
    Loads checkpoints from --checkpoints_dir.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, "top_seeds_eval.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    metrics_dir = args.metrics_dir or args.output_dir

    # Read avg_val_losses.csv
    csv_path = os.path.join(metrics_dir, "avg_val_losses.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"avg_val_losses.csv not found in {metrics_dir}. "
            "Run seeds_cfv mode first to generate this file."
        )
    loss_df = pd.read_csv(csv_path)
    # Format: columns are seed values, one data row
    seed_losses = {int(col): float(loss_df[col].iloc[0]) for col in loss_df.columns}
    sorted_seeds = sorted(seed_losses.items(), key=lambda x: x[1])
    top_seeds = [s for s, _ in sorted_seeds[:args.n_top_seeds]]
    logging.info(f"Top {args.n_top_seeds} seeds (by lowest avg val loss): {top_seeds}")
    for s, l in sorted_seeds[:args.n_top_seeds]:
        logging.info(f"  Seed {s}: avg val loss = {l:.6f}")

    # Load config
    input_data, n_node_features, n_edge_features, database_path = load_yaml_and_graph_info(args.input_file)
    params = get_model_params(input_data)
    params["n_node_features"] = n_node_features
    params["n_edge_features"] = n_edge_features
    data_path = args.data_file or os.path.join(AMES_FINAL_DIR, "data.csv")

    seed_yaml = input_data.get("randomSeed", 42)
    nBatch = input_data.get("nBatch", 50)
    nValMaxEntries = input_data.get("nValMaxEntries", None)

    _, valDataset, testDataset = load_graph_datasets(database_path, None, nValMaxEntries, seed_yaml)
    g = torch.Generator()
    g.manual_seed(seed_yaml)
    valLoader = DataLoader(valDataset, batch_size=nBatch, generator=g)
    testLoader = DataLoader(testDataset, batch_size=nBatch, generator=g)

    # Collect per-seed, per-fold metrics
    all_metrics_rows = []
    strain_names = ["Strain TA98", "Strain TA100", "Strain TA102", "Strain TA1535", "Strain TA1537"]

    for seed in top_seeds:
        for fold in range(5):
            ckpt_path = os.path.join(args.checkpoints_dir, f"metrics_{seed}_{fold}.pt")
            if not os.path.exists(ckpt_path):
                logging.warning(f"Checkpoint not found: {ckpt_path} — skipping.")
                continue

            model = build_model(params, n_node_features, n_edge_features)
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device)
            model.eval()

            if args.use_thresholds:
                y_logit_val, _, y_true_val, _ = run_inference(model, valLoader, device, params)
                fold_ths = crossfit_thresholds_for_consensus(
                    y_true_cat=y_true_val, y_prob_cat=y_logit_val, K=5, metric="sp", seed=42
                )
                logging.info(f"  Seed {seed} Fold {fold} thresholds: {fold_ths}")
            else:
                fold_ths = None
            y_logit_cat, y_pred_cat, y_true_cat, _ = run_inference(
                model, testLoader, device, params, thresholds=fold_ths
            )

            for i, strain in enumerate(strain_names):
                _, new_real, new_y_pred, _ = filter_nan(
                    y_true_cat[:, i], y_pred_cat[:, i], y_logit_cat[:, i]
                )
                metrics = get_metrics(new_real, new_y_pred)
                m1 = list(metrics[0])
                m2 = list(metrics[1])
                npv, mcc = compute_npv_mcc(m1)
                all_metrics_rows.append({
                    "seed": seed, "fold": fold, "strain": strain,
                    "TP": m1[0], "TN": m1[1], "FP": m1[2], "FN": m1[3],
                    "Sp": m2[0], "Sn": m2[1], "PPV": m2[2], "NPV": npv,
                    "Acc": m2[3], "Bal acc": m2[4], "MCC": mcc,
                    "F1 score": m2[5], "H score": m2[6],
                })

    if not all_metrics_rows:
        logging.error("No checkpoints could be loaded. Exiting.")
        return

    df_all = pd.DataFrame(all_metrics_rows)
    df_all.to_csv(os.path.join(args.output_dir, "top_seeds_all_metrics.csv"), index=False)

    # Average metrics across top seeds and folds
    metric_cols = ["Sp", "Sn", "PPV", "NPV", "Acc", "Bal acc", "MCC", "F1 score", "H score"]
    df_avg = df_all.groupby("strain")[metric_cols].mean().reset_index()
    df_avg.to_csv(os.path.join(args.output_dir, "top_seeds_avg_metrics.csv"), index=False)

    logging.info(f"Average test metrics across top {args.n_top_seeds} seeds:")
    logging.info(df_avg.to_string())
    logging.info("top_seeds_eval mode complete.")


# ==============================================================================
# MODE: analyze_cfv
# ==============================================================================

def run_analyze_cfv(args):
    """
    Post-hoc analysis of cross-fold validation results.
    Reads metrics_seed_*_fold_*.csv files and produces summary plots and statistics.
    Optionally plots a validation loss heatmap if --val_loss_file is provided.

    Ported from analyze_crossfold_val.py (active code, lines 134+).
    """
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_dir = args.metrics_dir or args.output_dir

    # Load all metrics CSVs
    csv_files = glob(os.path.join(metrics_dir, "metrics_seed_*_fold_*.csv"))
    if not csv_files:
        # Also support the format produced by seeds_cfv: metrics_{seed}_{fold}.csv
        csv_files = glob(os.path.join(metrics_dir, "metrics_[0-9]*_[0-9]*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No metrics CSV files found in {metrics_dir}.")

    # Match both naming conventions
    pattern = re.compile(r"metrics_(?:seed_)?(\d+)_(?:fold_)?(\d+)\.csv")

    all_data = []
    for file in csv_files:
        match = pattern.search(os.path.basename(file))
        if not match:
            continue
        seed, fold = int(match.group(1)), int(match.group(2))
        df = pd.read_csv(file)

        # Standardize column names
        df = df.rename(columns={
            "Sp": "Specificity", "Sn": "Sensitivity",
            "Acc": "Accuracy", "Bal acc": "Balanced Accuracy",
            "MCC": "MCC", "F1 score": "F1 Score", "H score": "H1 Score"
        })
        df["Seed"] = seed
        df["Fold"] = fold
        all_data.append(df)

    full_df = pd.concat(all_data, ignore_index=True)

    metrics = [
        "Specificity", "Sensitivity", "Accuracy", "Balanced Accuracy",
        "PPV", "NPV", "MCC", "F1 Score", "H1 Score"
    ]

    # Average metrics by strain
    avg_metrics = full_df.groupby(["Strain"])[metrics].mean().reset_index()

    # --- Plot 1: Average metrics across all seeds and folds ---
    melted_avg = avg_metrics.melt(id_vars=["Strain"], value_vars=metrics,
                                  var_name="Metric", value_name="Value")
    plt.figure(figsize=(14, 6))
    sns.barplot(data=melted_avg, x="Metric", y="Value", hue="Strain", errorbar=None)
    plt.title("Average Metrics Across All Folds and Seeds by Strain")
    plt.xticks(rotation=45)
    plt.legend(title="Strain")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "avg_metrics_by_strain.png"), dpi=300)
    plt.close()

    # --- Plot 2: Boxplot of metric distributions ---
    melted_full = full_df.melt(id_vars=["Strain", "Seed", "Fold"],
                               value_vars=metrics, var_name="Metric", value_name="Value")
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=melted_full, x="Metric", y="Value", hue="Strain")
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Metric", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.legend(title="Strain", fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "metric_distribution_by_strain.png"), dpi=300)
    plt.close()

    # --- Plot 3: Error bars (mean ± std) ---
    agg = melted_full.groupby(["Metric", "Strain"])["Value"].agg(["mean", "std"]).reset_index()
    plt.figure(figsize=(12, 6))
    for key, grp in agg.groupby("Strain"):
        plt.errorbar(grp["Metric"], grp["mean"], yerr=grp["std"], label=key, fmt="-o")
    plt.title("Metric Means with Error Bars (Std Dev)")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.legend(title="Strain")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "metric_error_bars.png"), dpi=300)
    plt.close()

    # --- Plot 4: Per-seed variability ---
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=melted_full, x="Metric", y="Value", hue="Seed")
    plt.title("Per-Seed Variability")
    plt.xticks(rotation=45)
    plt.legend(title="Seed", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "per_seed_variability.png"), dpi=300)
    plt.close()

    # --- Plot 5: Per-fold variability ---
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=melted_full, x="Metric", y="Value", hue="Fold")
    plt.title("Per-Fold Variability")
    plt.xticks(rotation=45)
    plt.legend(title="Fold", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "per_fold_variability.png"), dpi=300)
    plt.close()

    # --- Top 5 seeds by lowest average validation loss ---
    val_losses_path = os.path.join(metrics_dir, "val_losses.csv")
    if not os.path.exists(val_losses_path):
        print(f"Warning: val_losses.csv not found in {metrics_dir}. Skipping top-seeds-by-loss analysis.")
    else:
        # File layout:
        #   Row 0: ,,Seed,...
        #   Row 1: Fold,,seed1,seed2,...
        #   Rows 2-6: ,fold_idx,loss,...
        #   Empty row
        #   Last row: ,Average,avg_loss,...
        vl_raw = pd.read_csv(val_losses_path, header=None, dtype=str)

        # Seed names are in row 1 (0-indexed), starting at column 2
        seed_cols = [s.strip() for s in vl_raw.iloc[1, 2:].tolist()]

        # Data rows: rows where column 1 is a digit (fold indices)
        data_rows = vl_raw[vl_raw.iloc[:, 1].str.strip().str.match(r'^\d+$', na=False)]
        fold_indices = data_rows.iloc[:, 1].astype(int).tolist()
        fold_data = data_rows.iloc[:, 2:].astype(float).values
        fold_df = pd.DataFrame(fold_data, index=fold_indices, columns=seed_cols)

        # Average row: row where column 1 == "Average"
        avg_mask = vl_raw.iloc[:, 1].str.strip().str.lower() == "average"
        avg_values = vl_raw[avg_mask].iloc[0, 2:].astype(float).values
        avg_series = pd.Series(avg_values, index=seed_cols)

        top_5_seeds_loss = avg_series.nsmallest(5)
        print("\nTop 5 Seeds by Lowest Average Validation Loss:")
        for seed_str, loss in top_5_seeds_loss.items():
            print(f"  Seed {seed_str}: avg val loss = {loss:.6f}")

        # Within each top seed, find the fold with the lowest validation loss
        best_folds = []
        for seed_str in top_5_seeds_loss.index:
            if seed_str not in fold_df.columns:
                continue
            fold_losses = fold_df[seed_str]
            best_fold = int(fold_losses.idxmin())
            best_folds.append({
                "Seed": seed_str,
                "Best Fold": best_fold,
                "Val Loss (best fold)": fold_losses[best_fold],
                "Avg Val Loss": float(top_5_seeds_loss[seed_str]),
            })

        best_folds_df = pd.DataFrame(best_folds)
        print("\nBest Fold per Top Seed (by lowest validation loss):")
        print(best_folds_df.to_string(index=False))
        best_folds_df.to_csv(os.path.join(args.output_dir, "top_seeds_by_val_loss.csv"), index=False)

        # --- Validation loss heatmap (folds × seeds) ---
        # Append average row for display
        heat_df = pd.concat([
            fold_df,
            avg_series.rename("Average").to_frame().T.astype(float)
        ])
        heat_df.index = [f"Fold {i}" for i in fold_df.index] + ["Average"]
        heat_df.columns = [f"Seed {s}" for s in heat_df.columns]

        plt.figure(figsize=(max(8, len(seed_cols) * 0.9), 5))
        sns.heatmap(
            heat_df,
            annot=True,
            fmt=".4f",
            cmap="YlGnBu",
            cbar_kws={"label": "Validation Loss"},
            linewidths=0.5,
            linecolor="white",
        )
        plt.xlabel("Random Seed", fontsize=14)
        plt.ylabel("Fold", fontsize=14)
        plt.xticks(fontsize=12, rotation=45, ha="right")
        plt.yticks(fontsize=12, rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "validation_loss_heatmap.png"), dpi=300)
        plt.close()

    # --- Optional: Validation loss heatmap (legacy .xlsx source) ---
    if args.val_loss_file and os.path.exists(args.val_loss_file):
        import matplotlib
        matplotlib.rcParams["savefig.transparent"] = True
        sns.set(font_scale=1.2)
        df_heat = pd.read_excel(args.val_loss_file, index_col=0)
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_heat, annot=True, fmt=".4f", cmap="viridis",
                    cbar_kws={"label": "Validation Loss"})
        plt.xlabel("Random Seed", fontsize=16)
        plt.ylabel("Fold", fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "validation_loss_heatmap.png"), dpi=300)
        plt.close()

    print(f"analyze_cfv complete. Plots saved to {args.output_dir}.")


# ==============================================================================
# MODE: viz_optuna
# ==============================================================================

def run_viz_optuna(args):
    """
    Visualize Optuna hyperparameter optimization results.
    If --optuna_file is given, analyzes that single study.
    Otherwise, loads all .pkl files from --optuna_dir for multi-study analysis.

    Ported from visualize_optuna.py.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    if args.optuna_file:
        # Single study analysis
        study = joblib.load(args.optuna_file)
        logging.info(f"Loaded study from {args.optuna_file}")

        # Optimization history and parameter importances
        vis.plot_optimization_history(study).show()
        vis.plot_param_importances(study).show()

        # Extract trial data
        data = []
        for trial in study.trials:
            if trial.state.name == "COMPLETE":
                entry = {"trial_number": trial.number, "value": trial.value}
                entry.update(trial.user_attrs)
                data.append(entry)
        df = pd.DataFrame(data)
        if not df.empty:
            print(df[["trial_number", "best_fold", "best_fold_loss"]].to_string())

            # Validation loss per epoch for the first trial
            trial_id = 0
            if trial_id < len(study.trials) and "val_loss_log" in study.trials[trial_id].user_attrs:
                val_log = study.trials[trial_id].user_attrs["val_loss_log"]
                epochs = [e["epoch"] for e in val_log]
                losses = [e["val_loss"] for e in val_log]
                plt.plot(epochs, losses)
                plt.title(f"Trial {trial_id} Validation Loss per Epoch")
                plt.xlabel("Epoch")
                plt.ylabel("Validation Loss")
                plt.grid(True)
                plt.savefig(os.path.join(args.output_dir, f"trial_{trial_id}_val_loss.png"), dpi=300)
                plt.close()

            # Best trial info
            best_idx = df["best_fold_loss"].idxmin()
            best_trial_number = df.loc[best_idx, "trial_number"]
            best_trial = study.trials[best_trial_number]
            print(f"\nBest trial number: {best_trial_number}")
            print(f"Best fold loss: {df.loc[best_idx, 'best_fold_loss']:.6f}")
            print(f"Best fold: {df.loc[best_idx, 'best_fold']}")
            print("Best parameters:")
            for k, v in best_trial.params.items():
                print(f"  {k}: {v}")

    else:
        # Multi-study analysis: load all .pkl files in --optuna_dir
        study_paths = sorted(glob(os.path.join(args.optuna_dir, "*.pkl")))
        if not study_paths:
            raise FileNotFoundError(f"No .pkl files found in {args.optuna_dir}.")
        logging.info(f"Loading {len(study_paths)} studies from {args.optuna_dir}")

        all_trials = []
        for path in study_paths:
            study = joblib.load(path)
            all_trials.extend([t for t in study.trials if t.state.name == "COMPLETE"])
            logging.info(f"  Loaded {path}")

        combined_data = []
        for trial in all_trials:
            entry = {
                "trial_number": trial.number,
                "value": trial.value,
                "study_source": path,
            }
            entry.update(trial.user_attrs)
            combined_data.append(entry)

        df_combined = pd.DataFrame(combined_data)

        if not df_combined.empty and "best_fold_loss" in df_combined.columns:
            sns.histplot(df_combined["best_fold_loss"], bins=20)
            plt.title("Best Fold Validation Loss Across All Studies")
            plt.xlabel("Loss")
            plt.savefig(os.path.join(args.output_dir, "combined_best_fold_loss_hist.png"), dpi=300)
            plt.close()

            df_sorted = df_combined.sort_values("value", ascending=False).reset_index(drop=True)
            plt.plot(df_sorted.index, df_sorted["value"])
            plt.title("Combined Optimization History")
            plt.xlabel("Global Trial Number")
            plt.ylabel("Objective Value")
            plt.grid(True)
            plt.savefig(os.path.join(args.output_dir, "combined_optimization_history.png"), dpi=300)
            plt.close()

            sns.boxplot(data=df_combined, x="study_source", y="best_fold_loss")
            plt.title("Validation Loss by Study")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "validation_loss_by_study.png"), dpi=300)
            plt.close()

    print(f"viz_optuna complete. Plots saved to {args.output_dir}.")


# ==============================================================================
# Main entry point
# ==============================================================================

def main():
    args = get_args()

    # Validate required flags per mode
    modes_needing_input = {"train", "hp_opt", "seeds_cfv", "eval", "top_seeds_eval"}
    if args.mode in modes_needing_input and args.input_file is None:
        raise ValueError(f"--input_file is required for mode '{args.mode}'.")

    if args.mode == "train":
        run_train(args)
    elif args.mode == "hp_opt":
        run_hp_opt(args)
    elif args.mode == "seeds_cfv":
        run_seeds_cfv(args)
    elif args.mode == "eval":
        run_eval(args)
    elif args.mode == "top_seeds_eval":
        run_top_seeds_eval(args)
    elif args.mode == "analyze_cfv":
        run_analyze_cfv(args)
    elif args.mode == "viz_optuna":
        run_viz_optuna(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
