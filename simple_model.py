import torch
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    #3-dimensional input_shape
    def __init__(self, input_shape, device="cpu"):
        super(Net, self).__init__()
        
        self.input_shape = input_shape

        self.block1 = _ConvolutionBlock(input_shape, [256, 256, 256])
        self.block2 = _ConvolutionBlock([256], [256, 256, 256])
        self.block3 = _ConvolutionBlock([256], [128, 128, 128])
        
        self.head = _PolicyHead(4672, input_shape, 128, [1,64])

    def forward(self, X):
        X = self.block1(X)
        X = self.block2(X)
        X = self.block3(X)
        X = self.head(X)
        return torch.tanh(X)

    def save(self, save_as):
        self.eval()
        example = torch.rand(self.input_shape)
        traced_script_module = torch.jit.trace(self, example)
        save_as = "models/%s" % save_as
        traced_script_module.save(save_as)
        

class _ConvolutionBlock(nn.Sequential):
    def __init__(self, input_shape, num_filters):
        super().__init__()
        
        assert isinstance(input_shape, list)
        assert isinstance(num_filters, list)
        assert len(num_filters) == 3

        self.add_module("conv1", nn.Conv2d(input_shape[0], num_filters[0], kernel_size=3, padding=1))
        self.add_module("norm1", nn.BatchNorm2d(num_filters[0]))
        self.add_module("rect1", nn.LeakyReLU())
        
        self.add_module("conv2", nn.Conv2d(num_filters[0], num_filters[1], kernel_size=3, padding=1))
        self.add_module("norm2", nn.BatchNorm2d(num_filters[1]))
        self.add_module("rect2", nn.LeakyReLU())
        
        self.add_module("conv3", nn.Conv2d(num_filters[1], num_filters[2], kernel_size=3, padding=1))
        self.add_module("norm3", nn.BatchNorm2d(num_filters[2]))
        self.add_module("rect3", nn.LeakyReLU())

class _ValueHead(nn.Module):
    def __init__(self, input_shape, input_filters, features):
        super().__init__()
        assert isinstance(input_shape, list)
        assert isinstance(features, list)
        self.block = nn.Sequential(
                nn.Conv2d(input_filters, features[0], kernel_size=1),
                nn.BatchNorm2d(features[0]),
                nn.LeakyReLU(),
                )
        
        self.seq = nn.Sequential()
        for i in range(len(features)):
            if i == 0:
                self.seq.add_module("fc%d" % i, nn.Linear(np.prod((features[0],) + tuple(input_shape[1:])),features[1]))
                self.seq.add_module("rect1", nn.LeakyReLU())
            elif i == len(features) - 1:
                self.seq.add_module("fc%d" %i, nn.Linear(features[i], 1))
            else:
                self.seq.add_module("fc%d" % i, nn.Linear(features[i], features[i + 1]))
                self.seq.add_module("rect%d" % i, nn.LeakyReLU())
    
    def forward(self, X):
        X = self.block(X)
        X = X.view(X.shape[0], -1)
        X = self.seq.forward(X)
        return X

class _PolicyHead(nn.Module):
    def __init__(self, num_actions, input_shape, num_input_filters, features):
        super().__init__()
        assert isinstance(num_actions, int)
        assert isinstance(input_shape, list)
        assert isinstance(num_input_filters, int)
        assert isinstance(features, list)
        self.block = nn.Sequential(
                nn.Conv2d(num_input_filters, features[0],kernel_size=1),
                nn.BatchNorm2d(features[0]),
                nn.LeakyReLU(),
                )

        self.seq = nn.Sequential()
        for i in range(len(features)):
            if i == 0:
                self.seq.add_module("fc%d" % i, nn.Linear(np.prod((features[0],) + tuple(input_shape[1:])),features[1]))
            elif i == len(features) - 1:
                self.seq.add_module("fc%d" %i, nn.Linear(features[i], num_actions))
            else: 
                self.seq.add_module("fc%d" % i, nn.Linear(features[i], features[i + 1]))

    def forward(self, X):
        X = self.block(X)
        X = X.view(X.shape[0], -1)
        X = self.seq.forward(X)
        return X

if __name__ == "__main__":
    net = Net([119, 8, 8])
    out = net(torch.ones((32, 119, 8, 8)))
