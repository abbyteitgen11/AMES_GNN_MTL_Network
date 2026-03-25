import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv
from torch_geometric.nn import GINEConv
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
            + Graph-Convolutional (n_gc_layers) + fully-connected
            linear layers for GNN, using the specified non-linear activation layers
            interspaced between them.

            """
        self.activation_layer = eval("nn." + act + "()")
        #self.activation_layer2 = nn.Tanh()

        # GNN
        if not use_molecular_descriptors:
            if n_node_neurons > n_node_features: # Expand node features if desired
                self.GNNlinear1 = nn.Linear(n_node_features, n_node_neurons)
            if n_edge_neurons > n_edge_features: # Expand edge features if desired
                self.GNNlinear2 = nn.Linear(n_edge_features, n_edge_neurons)

            if n_node_neurons > n_node_features:
                ni = n_node_neurons
            else:
                ni = n_node_features

            if n_edge_neurons > n_edge_features:
                ne = n_edge_neurons
            else:
                ne = n_edge_features

            #for i in range(n_gc_layers):
            #    setattr(self, f"conv_GNN{i + 1}", CGConv(ni, dim=ne))
            #    setattr(self, f"dropout_GNN{i + 1}", nn.Dropout(dropout_GNN))
            #    setattr(self, f"bn_GNN{i + 1}", BatchNorm(ni, momentum=momentum_batch_norm))

            for i in range(n_gc_layers):
                mlp = nn.Sequential(
                    nn.Linear(ni, ni),
                    nn.ReLU(),
                    nn.Linear(ni, ni)
                )
                setattr(self, f"conv_GNN{i + 1}", GINEConv(mlp, edge_dim=ne))
                setattr(self, f"dropout_GNN{i + 1}", nn.Dropout(dropout_GNN))
                setattr(self, f"bn_GNN{i + 1}", BatchNorm(ni, momentum=momentum_batch_norm))

        else:
            ni = n_inputs

        #Shared core
        if n_s_layers > 0:
            prev_dim = ni
            for i, (n_units, dropout) in enumerate(zip(n_shared, dropout_shared)):
                setattr(self, f"linear_shared{i + 1}", nn.Linear(prev_dim, n_units))
                setattr(self, f"bn_shared{i + 1}", nn.BatchNorm1d(n_units, momentum=momentum_batch_norm))
                setattr(self, f"dropout_shared{i + 1}", nn.Dropout(dropout))
                prev_dim = n_units

            output_n = prev_dim

        else:
            output_n = ni


        #Target specific core
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




    def forward(self, x, edge_index, edge_attr, batch, n_node_neurons, n_node_features, n_edge_neurons, n_edge_features, n_gc_layers, n_s_layers, n_ts_layers, use_molecular_descriptors):
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