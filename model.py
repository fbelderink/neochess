import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

NUM_FILTERS = 256

class Net(nn.Module):
    #input_shape is 4-dimensional
    def __init__(self, input_shape, device="cpu"):
        super(Net, self).__init__()
        self.input_shape = input_shape

        self.input = _ConvolutionBlock(input_shape)

        self.residualblocks = _ResidualBlocks(39, NUM_FILTERS, NUM_FILTERS, kernel_size=3, padding=1, device=device)

        self.value_head = _ValueHead(input_shape)
    
    def forward(self, X):
        
        X = self.input(X)
        X = self.residualblocks(X)
        X = self.value_head(X)
        return X
    
    def save(self, save_as):
        self.eval()
        example = torch.rand(self.input_shape)
        traced_script_module = torch.jit.trace(model, example)
        save_as = "models/%s" % save_as
        traced_script_module.save(save_as)

class _ConvolutionBlock(nn.Sequential):    
    def __init__(self, input_shape):
        super().__init__()
        self.add_module("conv", nn.Conv2d(input_shape[1], NUM_FILTERS, kernel_size=3, padding=1))
        self.add_module("norm", nn.BatchNorm2d(NUM_FILTERS))
        self.add_module("rect", nn.LeakyReLU())

class _ResidualBlocks(nn.Sequential):
    def __init__(self, num_blocks, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, device="cpu"):
        super().__init__()
        for i in range(num_blocks):
            block = _ResidualLayer(in_channels, out_channels, kernel_size, stride, padding, bias).to(device)
            self.add_module("resblock%d" % i, block)

class _ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False):
        super(_ResidualLayer, self).__init__()
        self.cl1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
                )
        
        self.cl2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                )
        
        self.shortcut = nn.Sequential()

        self.rectifier = nn.Sequential(nn.LeakyReLU())

    def forward(self, X):
        residual = X
        X = self.cl1(X)
        X = self.cl2(X)
        X = X + self.shortcut(residual)
        X = self.rectifier(X)
        return X

class _ValueHead(nn.Module):
    def __init__(self, input_shape):
        super(_ValueHead, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(NUM_FILTERS, 1, kernel_size=1),
                nn.BatchNorm2d(1), 
                nn.LeakyReLU(),
                )
        
        self.fc = nn.Linear(np.prod(input_shape[2:]), 256)
        self.rect = nn.LeakyReLU()
        self.fc_out = nn.Linear(256, 1)

    def forward(self, X):
        X = self.conv(X)
        X = X.view(X.shape[0], -1)
        X = self.fc(X)
        X = self.rect(X)
        X = self.fc_out(X)
        return torch.tanh(X)
