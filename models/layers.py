import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torch_geometric.nn import GINConv
from models.model_utils import create_norm, create_activation
logger = logging.getLogger(__name__)


class NodeTransform(nn.Module):
    """update node feature with MLP, BN and ReLU"""
    def __init__(self, mlp, norm="batchnorm", activation="relu"):
        super(NodeTransform, self).__init__()
        self.mlp = mlp

        if norm not in ["layernorm", "batchnorm"]:
            self.norm = nn.Identity()
        else:
            norm_func = create_norm(norm)
            self.norm = norm_func(self.mlp.output_dim)

        self.act = create_activation(activation)

    def forward(self, h):
        h = self.mlp(h)
        h = self.norm(h)
        h = self.act(h)
        return h


class MLP(nn.Module):
    def __init__(self, layer_dim=[], norm="batchnorm", activation="relu"):
        super().__init__()
        assert len(layer_dim) >= 2, f"MLP layer_dim={layer_dim}, at least specify input & output dim!"
        self.num_layers = len(layer_dim) - 1
        self.output_dim = layer_dim[-1]
        if self.num_layers == 1:  # linear
            self.linear = nn.Linear(layer_dim[0], layer_dim[1])
        else:  # non-linear
            self.linear_layers = torch.nn.ModuleList()
            self.norms = torch.nn.ModuleList()
            self.activations = torch.nn.ModuleList()

            for l in range(self.num_layers):
                self.linear_layers.append(nn.Linear(layer_dim[l], layer_dim[l + 1]))
                if l != self.num_layers - 1:
                    self.norms.append(create_norm(norm)(layer_dim[l + 1]))
                    self.activations.append(create_activation(activation))

    def forward(self, x):
        if self.num_layers == 1:
            return self.linear(x)
        else:
            for i in range(self.num_layers - 1):
                x = self.linear_layers[i](x)
                x = self.norms[i](x)
                x = self.activations[i](x)
            return self.linear_layers[-1](x)


class BilinearDiscriminator(nn.Module):
    def __init__(self, in_dim):
        super(BilinearDiscriminator, self).__init__()
        self.bilinear = nn.Bilinear(in_dim, in_dim, 1, bias=True)
        torch.nn.init.xavier_normal_(self.bilinear.weight.data)
        self.bilinear.bias.data.fill_(0.0)

    def forward(self, x, x_contrast):
        logits = self.bilinear(x, x_contrast)  # no softmax here if using BCEWithLogitsLoss
        return logits


class BatchDiscriminator(nn.Module):
    def __init__(self, layer_dim=[], norm='batchnorm', activation="relu"):
        super().__init__()
        mlp = MLP(layer_dim=layer_dim, norm=norm, activation=activation)
        self.layers = NodeTransform(mlp, norm, activation)

    def forward(self, x):
        return self.layers(x)


class BatchEncoder(nn.Module):
    def __init__(self, layer_dim=[], norm='no norm', activation="relu"):
        super().__init__()
        mlp = MLP(layer_dim=layer_dim, norm=norm, activation=activation)
        self.layers = NodeTransform(mlp, norm, activation)

    def forward(self, x):
        return self.layers(x)


class GNNLayers(nn.Module):
    def __init__(self, layer_dim, dropout, norm='batchnorm', activation='elu', last_norm=False, res=False):
        super().__init__()
        self.layer_dim = layer_dim
        self.num_layers = len(layer_dim) - 1
        self.gnn_layers = torch.nn.ModuleList()
        self.dropout = dropout
        self.norm = norm
        self.activation = create_activation(activation)
        self.norm_layers = torch.nn.ModuleList()  # For batch normalization layers
        self.proj_layers = nn.ModuleList()
        self.res = res
        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                if last_norm:
                    self.norm_layers.append(create_norm(self.norm)(layer_dim[i + 1]))
                else:
                    self.norm_layers.append(nn.Identity())
            else:
                self.norm_layers.append(create_norm(self.norm)(layer_dim[i + 1]))

            if res and layer_dim[i] != layer_dim[i + 1]:
                self.proj_layers.append(nn.Linear(layer_dim[i], layer_dim[i + 1]))
            else:
                self.proj_layers.append(None)

    def forward(self, inputs, edge_index):
        h = inputs
        for l in range(self.num_layers):
            if self.res:
                h_new = self.gnn_layers[l](h, edge_index)
                if self.proj_layers[l] is not None:
                    h = self.proj_layers[l](h)  # Project to match dimensions
                h = h_new + h
            else:
                h = self.gnn_layers[l](h, edge_index)
            h = self.norm_layers[l](h)
            h = self.activation(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class GINLayers(GNNLayers):
    def __init__(self, layer_dim, dropout, norm, activation, last_norm, res=False):
        super().__init__(layer_dim=layer_dim, dropout=dropout, norm=norm, activation=activation, last_norm=last_norm, res=res)
        for i in range(self.num_layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(layer_dim[i], (layer_dim[i] + layer_dim[i + 1]) // 2),
                create_norm(norm)((layer_dim[i] + layer_dim[i + 1]) // 2),
                self.activation,
                torch.nn.Linear((layer_dim[i] + layer_dim[i + 1]) // 2, layer_dim[i + 1]),
                create_norm(norm)(layer_dim[i + 1]),
                self.activation,
            )
            self.gnn_layers.append(GINConv(mlp))
            