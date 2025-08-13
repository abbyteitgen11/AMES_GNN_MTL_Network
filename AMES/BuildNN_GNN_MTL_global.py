import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import BatchNorm


class BuildNN_GNN_MTL(nn.Module):
    def __init__(self,
                 n_gc_layers,
                 n_node_neurons,
                 n_edge_neurons,
                 n_node_features,
                 n_edge_features,
                 dropout_GNN,
                 momentum_batch_norm,
                 n_s_layers,
                 n_ts_layers,
                 n_shared,
                 n_target,
                 dropout_shared,
                 dropout_target,
                 act,
                 use_molecular_descriptors,
                 n_inputs):

        super(BuildNN_GNN_MTL, self).__init__()

        """
            A class that creates a model with the desired number of shared core
            layers (fully connected) + target specific core layers (fully connected)
            + Graph-Convolutional (n_gc_layers) + fully-connected (n_fc_layers)
            linear layers for GNN, using the specified non-linear activation layers
            interspaced between them.

            Args:
            :param int ni: number of input features; default: 4
            :param int n0: number of neurons in first layer of shared core; default: 200
            :param int n1: number of neurons in second layer of shared core; default: 100
            :param int n2: number of neurons in third layer of shared core; default: 50
            :param int n3: number of neurons in fourth layer of shared core; default: 10
            :param str act: activation function; can be any activation available in torch; default: relu
            :param float momentum_batch_norm: momentum for batch normalization; default: 0.9
            :param int prob_h1: dropout in first layer of shared core; default: 0.25
            :param int prob_h2: dropout in first layer of shared core; default: 0.15
            :param int prob_h3: dropout in first layer of shared core; default: 0.1
            :param int prob_h4: dropout in first layer of shared core; default: 0.0001
            :param int n_node_features: length of node feature vectors for GNN; default: 4
            :param int n_edge_features: length of edge feature vectors for GNN; default: 3
            :param int n_node_neurons: number of neurons in deep layers (if they exist)
                    in the densely-connected network if set to None, it is internally
                    set to n_node_features
            :param int n_edge_neurons: edges might have on input relatively few features (2,3);
                    this parameter allows the user to use a linear layer to expand the number
                    of edge features in the graph-convolution part.
            :param int n_gc_layers: number of Graph-Convolution layers (CGConv); default: 2
            :param int n_s_layers: number of layers in shared core; default: 4
            :param int n_ts_layers: number of layers in target specific core; default: 2

            """
        self.activation_layer = eval("nn." + act + "()")
        # self.activation_layer2 = nn.Tanh()

        # GNN
        if not use_molecular_descriptors:
            if n_node_neurons > n_node_features:  # Expand node features if desired
                self.GNNlinear1 = nn.Linear(n_node_features, n_node_neurons)
            if n_edge_neurons > n_edge_features:  # Expand edge features if desired
                self.GNNlinear2 = nn.Linear(n_edge_features, n_edge_neurons)

            if n_node_neurons > n_node_features:
                ni = n_node_neurons
            else:
                ni = n_node_features

            if n_edge_neurons > n_edge_features:
                ne = n_edge_neurons
            else:
                ne = n_edge_features

            for i in range(n_gc_layers):
                setattr(self, f"conv_GNN{i + 1}", CGConv(ni, dim=ne))
                setattr(self, f"dropout_GNN{i + 1}", nn.Dropout(dropout_GNN))
                setattr(self, f"bn_GNN{i + 1}", BatchNorm(ni, momentum=momentum_batch_norm))

        else:
            ni = n_inputs

        # Shared core
        if n_s_layers > 0:
            prev_dim = ni + 6
            for i, (n_units, dropout) in enumerate(zip(n_shared, dropout_shared)):
                setattr(self, f"linear_shared{i + 1}", nn.Linear(prev_dim, n_units))
                setattr(self, f"bn_shared{i + 1}", nn.BatchNorm1d(n_units, momentum=momentum_batch_norm))
                setattr(self, f"dropout_shared{i + 1}", nn.Dropout(dropout))
                prev_dim = n_units

            output_n = prev_dim

        else:
            output_n = ni

        # Target specific core
        if n_ts_layers > 0:
            for i in range(5):
                prev_dim = output_n
                for j, (n_units, dropout) in enumerate(zip(n_target, dropout_target)):
                    setattr(self, f"ts{i + 1}_linear_target{j + 1}", nn.Linear(prev_dim, n_units))
                    setattr(self, f"ts{i + 1}_bn_target{j + 1}", nn.BatchNorm1d(n_units, momentum=momentum_batch_norm))
                    setattr(self, f"ts{i + 1}_dropout_target{j + 1}", nn.Dropout(dropout))
                    prev_dim = n_units
                setattr(self, f"ts{i + 1}_sig", nn.Linear(prev_dim, 1))

        else:
            for i in range(5):
                setattr(self, f"ts{i + 1}_sig", nn.Linear(output_n, 1))




    def forward(self, x, edge_index, edge_attr, batch, n_node_neurons, n_node_features, n_edge_neurons, n_edge_features, n_gc_layers, n_s_layers, n_ts_layers, use_molecular_descriptors, global_feats, global_mean, global_std):
        # GNN
        if not use_molecular_descriptors:
            if n_node_neurons > n_node_features:
                x = self.GNNlinear1(x)

            if n_edge_neurons > n_edge_features:
                edge_attr = self.GNNlinear2(edge_attr)

            for i in range(n_gc_layers):
                dropout_layer = getattr(self, f"dropout_GNN{i + 1}")
                bn_layer = getattr(self, f"bn_GNN{i + 1}")
                conv_layer = getattr(self, f"conv_GNN{i + 1}")
                x = conv_layer(x, edge_index, edge_attr)
                x = bn_layer(x)
                x = self.activation_layer(x)
                x = dropout_layer(x)


            # Pooling layer
            x = global_add_pool(x, batch)

            batch_size = batch.max().item() + 1
            n_global_features = global_feats.shape[0] // batch_size
            global_feats = global_feats.view(batch_size, n_global_features)

            global_feats = (global_feats - global_mean) / (global_std + 1e-8)

            if global_feats is not None:
                x = torch.cat([x, global_feats], dim=1)

        #Shared core
        for i in range(n_s_layers):
            dropout_layer = getattr(self, f"dropout_shared{i + 1}")
            #x = dropout_layer(x)
            linear_layer = getattr(self, f"linear_shared{i + 1}")
            #x = linear_layer(x)
            bn_layer = getattr(self, f"bn_shared{i + 1}")
            #x = self.activation_layer(bn_layer(x))

            x = linear_layer(x)
            x = bn_layer(x)
            x = self.activation_layer(x)
            x = dropout_layer(x)

        #Target specific core
        y_outputs = []
        if n_ts_layers > 0:
            for i in range(5):
                y = x
                for j in range(n_ts_layers):
                    dropout_layer = getattr(self, f"ts{i + 1}_dropout_target{j + 1}")
                    #y = dropout_layer(y)
                    linear_layer = getattr(self, f"ts{i + 1}_linear_target{j + 1}")
                    #y = linear_layer(y)
                    bn_layer = getattr(self, f"ts{i + 1}_bn_target{j + 1}")
                    #y = self.activation_layer(bn_layer(y))

                    y = linear_layer(y)
                    y = bn_layer(y)
                    y = self.activation_layer(y)
                    y = dropout_layer(y)

                sig_layer = getattr(self, f"ts{i + 1}_sig")
                y = sig_layer(y)
                y = y.sigmoid()
                y_outputs.append(y)
        else:
            for i in range(5):
                y = x
                sig_layer = getattr(self, f"ts{i + 1}_sig")
                y = sig_layer(y)
                y = y.sigmoid()
                y_outputs.append(y)

        return y_outputs[0], y_outputs[1], y_outputs[2], y_outputs[3], y_outputs[4]