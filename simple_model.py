import torch
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    #3-dimensional input_shape
    def __init__(self, input_shape, device="cpu"):
        super(Net, self).__init__()
        
        self.input_shape = input_shape

        self.block1 = _ConvolutionBlock(input_shape, 128)
        self.block2 = _ConvolutionBlock([128], 256)
        self.block3 = _ConvolutionBlock([256], 128)
        
        self.block4 = nn.Sequential(
                nn.Conv2d(128, 1, kernel_size=1),
                nn.BatchNorm2d(1),
                nn.LeakyReLU(),
                )
        
        self.fc = nn.Linear(np.prod((1,) + tuple(input_shape[1:])), 1)

    def forward(self, X):
        X = self.block1(X)
        X = self.block2(X)
        X = self.block3(X)
        X = self.block4(X)
        X = X.view(X.shape[0], -1)
        X = self.fc(X)
        return torch.tanh(X)

    def save(self, save_as):
        self.eval()
        example = torch.rand(self.input_shape)
        traced_script_module = torch.jit.trace(model, example)
        save_as = "models/%" % save_as
        traced_script_module.save(save_as)
        

class _ConvolutionBlock(nn.Sequential):
    def __init__(self, input_shape, num_filters):
        super().__init__()
        self.add_module("conv1", nn.Conv2d(input_shape[0], num_filters, kernel_size=3, padding=1))
        self.add_module("norm1", nn.BatchNorm2d(num_filters))
        self.add_module("rect1", nn.LeakyReLU())
        self.add_module("conv2", nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1))
        self.add_module("norm2", nn.BatchNorm2d(num_filters))
        self.add_module("rect2", nn.LeakyReLU())
        self.add_module("conv3", nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1))
        self.add_module("norm3", nn.BatchNorm2d(num_filters))
        self.add_module("rect3", nn.LeakyReLU())

if __name__ == "__main__":
    model = Net([119, 8, 8])
    out = model(torch.ones((32,119,8,8)))
