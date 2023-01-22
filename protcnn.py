from torch import nn
import torch as tr
import math

class ProtCNN(nn.Module):
    """Pytorch (adapted) implementation of ProtCNN from https://github.com/google-research/google-research/blob/master/using_dl_to_annotate_protein_universe/neural_network/protein_model.py"""
    
    def __init__(self, input_size, filters=1100, kernel_size=9, num_layers=5, first_dilated_layer=2, dilation_rate=3, 
    resnet_bottleneck_factor=.5, device="cpu", lr=1e-3):
        super().__init__()

        self.cnn = [nn.Conv1d(input_size, filters, kernel_size, padding="same")]

        for k in range(num_layers):
            self.cnn.append(ResidualLayer(k, first_dilated_layer, dilation_rate, resnet_bottleneck_factor, filters, kernel_size))

        self.cnn = nn.Sequential(*self.cnn)

        self.adaptivemax = nn.AdaptiveMaxPool1d(1)
    
        self.to(device)
        self.device = device

    def forward(self, x):
        y = self.cnn(x)
        y = self.adaptivemax(y)

        return y

class ResidualLayer(nn.Module):
    def __init__(self, layer_index, first_dilated_layer, dilation_rate, resnet_bottleneck_factor, filters, kernel_size):
        super().__init__()

        shifted_layer_index = layer_index - first_dilated_layer + 1
        dilation_rate = max(1, dilation_rate**shifted_layer_index)

        num_bottleneck_units = math.floor(
            resnet_bottleneck_factor * filters)

        self.layer = nn.Sequential(nn.BatchNorm1d(filters),
        nn.ReLU(),
        nn.Conv1d(filters, num_bottleneck_units, kernel_size, dilation=dilation_rate, padding="same"), 
        nn.BatchNorm1d(num_bottleneck_units),
        nn.ReLU(),
        nn.Conv1d(num_bottleneck_units, filters, kernel_size=1, padding="same"))

    def forward(self, x):
        return x + self.layer(x)
