import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torch_geometric.nn.conv import GCNConv, GATConv, GINConv
from torch.nn import BatchNorm1d
from models.model_utils import create_norm, create_activation
from torch.autograd import Function
logger = logging.getLogger(__name__)


class GNNLayers(nn.Module):
    def __init__(self, layer_dim, dropout, norm='batchnorm', activation='relu', last_norm=False):
        super().__init__()
        self.layer_dim = layer_dim
        self.num_layers = len(layer_dim) - 1
        self.gnn_layers = torch.nn.ModuleList()
        self.dropout = dropout
        self.norm = norm
        self.activation = create_activation(activation)
        self.norm_layers = torch.nn.ModuleList()  # For batch normalization layers
        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                if last_norm:
                    self.norm_layers.append(create_norm(self.norm)(layer_dim[i + 1]))
                else:
                    self.norm_layers.append(nn.Identity())
            else:
                self.norm_layers.append(create_norm(self.norm)(layer_dim[i + 1]))

    def forward(self, inputs, edge_index):
        h = inputs
        for l in range(self.num_layers):
            h = self.gnn_layers[l](h, edge_index)
            h = self.norm_layers[l](h)
            h = self.activation(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class GCNLayers(GNNLayers):
    def __init__(self, layer_dim, dropout, norm, activation, last_norm):
        super().__init__(layer_dim=layer_dim, dropout=dropout, norm=norm, activation=activation, last_norm=last_norm)
        for i in range(self.num_layers):
            self.gnn_layers.append(GCNConv(layer_dim[i], layer_dim[i + 1]))


class GATLayers(GNNLayers):
    def __init__(self, layer_dim, dropout, norm, activation, last_norm, heads=2):
        super().__init__(layer_dim=layer_dim, dropout=dropout, norm=norm, activation=activation, last_norm=last_norm)
        for i in range(self.num_layers):
            self.gnn_layers.append(GATConv(layer_dim[i], layer_dim[i + 1] // heads, heads=heads))


class GINLayers(GNNLayers):
    def __init__(self, layer_dim, dropout, norm, activation, last_norm):
        super().__init__(layer_dim=layer_dim, dropout=dropout, norm=norm, activation=activation, last_norm=last_norm)
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



class NodeTransform(nn.Module):
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


class BatchTransform(nn.Module):
    def __init__(self, layer_dim=[], norm='no norm', activation="relu"):
        super().__init__()
        mlp = MLP(layer_dim=layer_dim, norm=norm, activation=activation)
        self.layers = NodeTransform(mlp, norm, activation)

    def forward(self, x):
        return self.layers(x)


class RevGradFunc(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output): 
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


class RevGrad(nn.Module):
    def __init__(self, alpha=1., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._alpha = tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return RevGradFunc.apply(input_, self._alpha)

