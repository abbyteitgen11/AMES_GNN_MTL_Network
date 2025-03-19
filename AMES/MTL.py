import tensorflow as tf
from keras.src.backend.jax.nn import binary_crossentropy
from tensorflow import keras

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import utils
from tensorflow.keras import losses
from tensorflow.keras import backend as K

import numpy as np
import pandas as pd
import random
import csv

import h5py
import sys

from compute_metrics import *
from data import load_data

#tf.compat.v1.disable_eager_execution() # Eager execution runs operations immediately without converting to computational graph
tf.config.run_functions_eagerly(True)
# --------------------------------------------------------------------------------

def set_seed(s):
    K.clear_session()

    # Seed value
    # Apparently you may use different seed values at each stage
    seed_value = s

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    import os
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    import tensorflow as tf
    tf.random.set_seed(seed_value)


# --------------------------------------------------------------------------------
def masked_loss_function(y_true, y_pred):
    mask_value=-1
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    y_true_mask = y_true * mask
    y_pred_mask = y_pred * mask

    return K.binary_crossentropy(y_true * mask, y_pred * mask)

# --------------------------------------------------------------------------------
# Load data (fixed partitions for grid search)
data_path = 'data.csv'
train, internal = load_data(data_path, model="MTL", stage="GS")
X_train, y_train = train
X_internal, y_internal = internal
# --------------------------------------------------------------------------------
####Changed this#####
X_train = X_train[:, 1:]  # Remove SMILES
X_internal = X_internal[:, 1:]  # Remove SMILES

X_train = np.array(X_train, dtype=np.float32)
X_internal = np.array(X_internal, dtype=np.float32)

y_train = [np.array(y, dtype=np.float32) for y in y_train]
y_internal = [np.array(y, dtype=np.float32) for y in y_internal]

# adam optimizer
l_rate = 0.0001

# dropout
prob_h1 = 0.25
prob_h2 = 0.15
prob_h3 = 0.1
prob_h4 = 0.0001

# batch normalization
momentum_batch_norm = 0.9

# early stopping
min_delta_val = 0.0005
patience_val = 1000


# --------------------------------------------------------------------------------

# Target-specific core of the MTL_DNN architecture
# Two-layer option
def two_specific_layers(x, n2, n3):
    # STRAIN 1
    y1 = Dense(n2, activation=act)(x)
    y1 = BatchNormalization(momentum=momentum_batch_norm)(y1)
    y1 = Dropout(prob_h2)(y1)
    y1 = Dense(n3, activation=act)(y1)
    y1 = BatchNormalization(momentum=momentum_batch_norm)(y1)
    y1 = Dropout(prob_h3)(y1)

    # STRAIN 2
    y2 = Dense(n2, activation=act)(x)
    y2 = BatchNormalization(momentum=momentum_batch_norm)(y2)
    y2 = Dropout(prob_h2)(y2)
    y2 = Dense(n3, activation=act)(y2)
    y2 = BatchNormalization(momentum=momentum_batch_norm)(y2)
    y2 = Dropout(prob_h3)(y2)

    # STRAIN 3
    y3 = Dense(n2, activation=act)(x)
    y3 = BatchNormalization(momentum=momentum_batch_norm)(y3)
    y3 = Dropout(prob_h2)(y3)
    y3 = Dense(n3, activation=act)(y3)
    y3 = BatchNormalization(momentum=momentum_batch_norm)(y3)
    y3 = Dropout(prob_h3)(y3)

    # STRAIN 4
    y4 = Dense(n2, activation=act)(x)
    y4 = BatchNormalization(momentum=momentum_batch_norm)(y4)
    y4 = Dropout(prob_h2)(y4)
    y4 = Dense(n3, activation=act)(y4)
    y4 = BatchNormalization(momentum=momentum_batch_norm)(y4)
    y4 = Dropout(prob_h3)(y4)

    # STRAIN 5
    y5 = Dense(n2, activation=act)(x)
    y5 = BatchNormalization(momentum=momentum_batch_norm)(y5)
    y5 = Dropout(prob_h2)(y5)
    y5 = Dense(n3, activation=act)(y5)
    y5 = BatchNormalization(momentum=momentum_batch_norm)(y5)
    y5 = Dropout(prob_h3)(y5)

    return y1, y2, y3, y4, y5


# Target-specific core of the MTL_DNN architecture
# One-layer option
def one_specific_layers(x, n2):
    # STRAIN 1
    y1 = Dense(n2, activation=act)(x)
    y1 = BatchNormalization(momentum=momentum_batch_norm)(y1)
    y1 = Dropout(prob_h2)(y1)

    # STRAIN 2
    y2 = Dense(n2, activation=act)(x)
    y2 = BatchNormalization(momentum=momentum_batch_norm)(y2)
    y2 = Dropout(prob_h2)(y2)

    # STRAIN 3
    y3 = Dense(n2, activation=act)(x)
    y3 = BatchNormalization(momentum=momentum_batch_norm)(y3)
    y3 = Dropout(prob_h2)(y3)

    # STRAIN 4
    y4 = Dense(n2, activation=act)(x)
    y4 = BatchNormalization(momentum=momentum_batch_norm)(y4)
    y4 = Dropout(prob_h2)(y4)

    # STRAIN 5
    y5 = Dense(n2, activation=act)(x)
    y5 = BatchNormalization(momentum=momentum_batch_norm)(y5)
    y5 = Dropout(prob_h2)(y5)

    return y1, y2, y3, y4, y5


# --------------------------------------------------------------------------------

# General MTL_DNN architecture
def build_model(ni, n0, n1, n2, n3, ln, act, spec_layers):
    # Shared core
    model_input = Input(shape=(ni,))
    x = Dense(n0, activation=act)(model_input)
    x = BatchNormalization(momentum=momentum_batch_norm)(x)
    x = Dropout(prob_h1)(x)
    x = Dense(n1, activation=act)(x)
    x = BatchNormalization(momentum=momentum_batch_norm)(x)
    x = Dropout(prob_h2)(x)
    x = Dense(n2, activation=act)(x)
    x = BatchNormalization(momentum=momentum_batch_norm)(x)
    x = Dropout(prob_h3)(x)
    x = Dense(n3, activation=act)(x) # added
    x = BatchNormalization(momentum=momentum_batch_norm)(x)
    x = Dropout(prob_h4)(x)

    # Target-specific core
    if (spec_layers == 0):
        y1 = Dense(units=1, activation='sigmoid', name='output_1')(x)
        y2 = Dense(units=1, activation='sigmoid', name='output_2')(x)
        y3 = Dense(units=1, activation='sigmoid', name='output_3')(x)
        y4 = Dense(units=1, activation='sigmoid', name='output_4')(x)
        y5 = Dense(units=1, activation='sigmoid', name='output_5')(x)
    else:
        if (spec_layers == 1):
            y1, y2, y3, y4, y5 = one_specific_layers(x, n2)
        else:
            if (spec_layers == 2):
                y1, y2, y3, y4, y5 = two_specific_layers(x, n2, n3)

        # Outputs 1 to 5
        y1 = Dense(units=1, activation='sigmoid', name='output_1')(y1)
        y2 = Dense(units=1, activation='sigmoid', name='output_2')(y2)
        y3 = Dense(units=1, activation='sigmoid', name='output_3')(y3)
        y4 = Dense(units=1, activation='sigmoid', name='output_4')(y4)
        y5 = Dense(units=1, activation='sigmoid', name='output_5')(y5)

    model = Model(inputs=model_input, outputs=[y1, y2, y3, y4, y5])
    return model


# --------------------------------------------------------------------------------

# Hyperparameters to try at the grid search
act = 'relu'

# Size of input layer
n_inputs = X_train.shape[1]

# Num epochs
n_epochs = 500 #np.iinfo(np.int32).max

# Different random seeds for parameter initialization
#seed_lst = [0, 20, 25, 65, 92, 39, 7, 88, 23, 55, 29, 5, 15, 30, 44, 70, 10, 18, 69, 80]

# --------------------------------------------------------------------------------

# set of already tested param combinations, to carry out a random grid search
#tested_params = set()

# Set parameters
lb = 0.005 # L2 regularization coefficient

n0, n1, n2, n3 = (200, 100, 50, 10) # Number of neurons shared core

w = None # Weighted cost function

spec_lay = 2 # Number of layers in target specific core

params = 'lb_' + str(lb) + '_spec_lay_' + str(spec_lay) + '_n0_' + str(n0) + '_n1_' + str(n1) + '_n2_' + str(
            n2) + '_n3_' + str(n3) + '_w_' + str(w)

# for monitoring purposes
#print(params)
sys.stdout.flush()

# random seed
seed_r = 0
set_seed(seed_r)

# regularization
reg = l2(lb)

# optimizer
adam_opt = Adam(l_rate)

# early stopping
early_stop = EarlyStopping(monitor='val_loss',
                            min_delta=min_delta_val,
                            patience=patience_val,
                            verbose=0,
                            mode='min',
                            restore_best_weights=True)

# build MTL_DNN model
model = build_model(n_inputs, n0, n1, n2, n3, reg, act, spec_lay)


metrics = [[tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.TruePositives(),
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.FalseNegatives()]
           for _ in range(5)]  # Repeat for each output *changed


# compile model
model.compile(loss=masked_loss_function, optimizer=adam_opt, metrics=metrics)

# train model

class PrintPredictionsCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, csv_filename="predictions_TF.csv"):
        super().__init__()
        self.validation_data = validation_data  # Store validation data manually
        self.csv_filename = csv_filename

        # Create CSV file and write header
        with open(self.csv_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Task", "True Value", "Predicted Value", "Training Loss", "Validation Loss"])  # Header row

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        # Retrieve loss values
        train_loss = logs.get("train_loss", "N/A")
        val_loss = logs.get("val_loss", "N/A")

        print(f"\nEpoch {epoch + 1}:")
        print(f"  Training Loss: {train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")

        # Get validation predictions
        X_val, y_val = self.validation_data
        y_pred = self.model.predict(X_val, verbose=0)

        rows = []

        # Ensure predictions are a list (for multitask outputs)
        if not isinstance(y_pred, list):
            y_pred = [y_pred]

        # Print and log sample predictions
        for i in range(len(y_pred)):  # Iterate over tasks
            true_values = np.array(y_val[i][:10]).flatten()  # Extract first 10 values
            predicted_values = np.array(y_pred[i][:10]).flatten()

            #print(f"  Task {i + 1} - True: {true_values}, Pred: {predicted_values}")

            # Store data in CSV (convert to scalar values)
            for true_val, pred_val in zip(true_values, predicted_values):
                rows.append([epoch + 1, i + 1, float(true_val), float(pred_val), float(train_loss), float(val_loss)])

        # Append predictions to CSV file
        with open(self.csv_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(rows)


# Use the callback when training
print_callback = PrintPredictionsCallback((X_internal, y_internal))

log_dir = "logs/MTL_tensorflow"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

import numpy as np
import tensorflow as tf


class TensorBoardMetricsLogger(tf.keras.callbacks.Callback):
    def __init__(self, X_internal, y_internal, log_dir="logs/MTL_tensorflow"):
        super(TensorBoardMetricsLogger, self).__init__()
        self.X_internal = X_internal
        self.y_internal = np.array(y_internal)  # Convert to NumPy array to support slicing
        self.writer = tf.summary.create_file_writer(log_dir)
        self.strains = ["TA98", "TA100", "TA102", "TA1535", "TA1537"]
        self.metric_names = ["TP", "TN", "FP", "FN", "Specificity", "Sensitivity",
                             "Precision", "Accuracy", "Balanced Accuracy", "F1 Score", "H Score"]

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_internal, verbose=0)

        # Ensure y_pred_out is a list of numpy arrays, not a list of lists
        y_pred_out = [np.where(yp > 0.5, 1, 0) for yp in y_pred]  # Convert to numpy arrays

        with self.writer.as_default():
            for i, strain in enumerate(self.strains):
                # Ensure y_internal and y_pred_out[i] are numpy arrays
                new_real = self.y_internal[:, i]  # Already a numpy array
                new_y_pred = y_pred_out[i]  # Already a numpy array

                # Call filter_nan on numpy arrays
                _, new_real, new_y_pred, new_prob = filter_nan(new_real, new_y_pred, y_pred[i])

                # Compute metrics (adjust depending on your implementation)
                metrics = get_metrics(new_real, new_y_pred)
                metrics_cat = np.concatenate(metrics)

                for j, metric_name in enumerate(self.metric_names):
                    tf.summary.scalar(f"Metrics/{strain}/{metric_name}", metrics_cat[j], step=epoch)

            # Log weights and biases
            for layer in self.model.layers:
                if hasattr(layer, 'kernel'):
                    tf.summary.histogram(f"Weights/{layer.name}", layer.kernel, step=epoch)
                if hasattr(layer, 'bias'):
                    tf.summary.histogram(f"Biases/{layer.name}", layer.bias, step=epoch)

            # Log loss values
            tf.summary.scalar('Loss/train', logs["loss"], step=epoch)
            tf.summary.scalar('Loss/val', logs["val_loss"], step=epoch)

        #self.writer.flush()

# Create instance of the custom callback
tensorboard_metrics_callback = TensorBoardMetricsLogger(X_internal, y_internal)

# Pass the callback when training
callbacks = [early_stop]
#callbacks = [early_stop, tensorboard_callback, tensorboard_metrics_callback]
#callbacks = [early_stop, print_callback]

learning_data = model.fit(X_train, y_train,
                          epochs=n_epochs,
                          validation_data=(X_internal, y_internal),
                          callbacks=callbacks,
                          # class_weight=w,
                          verbose=0)

# make predictions
y_pred = model.predict(X_internal)

# evaluate and save for future model selection
# change path names accordingly
y_pred_98 = np.where(y_pred[0] > 0.5, 1, 0)
y_pred_100 = np.where(y_pred[1] > 0.5, 1, 0)
y_pred_102 = np.where(y_pred[2] > 0.5, 1, 0)
y_pred_1535 = np.where(y_pred[3] > 0.5, 1, 0)
y_pred_1537 = np.where(y_pred[4] > 0.5, 1, 0)

# for monitoring purposes. should be dumped in a log file
print("Strain TA98")
print(
    'internal validation: (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score --- AUC)')
_, new_real, new_y_pred, new_prob = filter_nan(y_internal[0], y_pred_98, y_pred[0])
print(get_metrics(new_real, new_y_pred))

print("Strain TA100")
print(
    'internal validation: (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score --- AUC)')
_, new_real, new_y_pred, new_prob = filter_nan(y_internal[1], y_pred_100, y_pred[1])
print(get_metrics(new_real, new_y_pred))

print("Strain TA102")
print(
    'internal validation: (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score --- AUC)')
_, new_real, new_y_pred, new_prob = filter_nan(y_internal[2], y_pred_102, y_pred[2])
print(get_metrics(new_real, new_y_pred))

print("Strain TA1535")
print(
    'internal validation: (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score --- AUC)')
_, new_real, new_y_pred, new_prob = filter_nan(y_internal[3], y_pred_1535, y_pred[3])
print(get_metrics(new_real, new_y_pred))

print("Strain TA1537")
print(
    'internal validation: (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score --- AUC)')
_, new_real, new_y_pred, new_prob = filter_nan(y_internal[4], y_pred_1537, y_pred[4])
print(get_metrics(new_real, new_y_pred))

sys.stdout.flush()

# limpiar memoria
K.clear_session()