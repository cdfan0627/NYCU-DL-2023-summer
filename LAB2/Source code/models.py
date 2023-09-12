import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

class EEGNet(nn.Module):
    def __init__(self, activation="relu"):
        super(EEGNet, self).__init__()
        self.activation = activation

        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.get_activation(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.get_activation(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25)
        )

        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )

    def get_activation(self):
        if self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "leakyrelu":
            return nn.LeakyReLU()
        elif self.activation == "elu":
            return nn.ELU(alpha=1.0)

    def forward(self, x):
        x = self.firstConv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(x.size(0), -1) 
        x = self.classify(x)
        return x

class DeepConvNet(nn.Module):
    def __init__(self, activation="relu"):
        super(DeepConvNet, self).__init__()
        self.activation = activation

        self.cov1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), stride=(1, 1)),
            nn.Conv2d(25, 25, kernel_size=(2, 1), stride=(1, 1)),
            nn.BatchNorm2d(25),
            self.get_activation(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.75)
        )
        self.cov2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), stride=(1, 1)),
            nn.BatchNorm2d(50),
            self.get_activation(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.75)
        )
        self.cov3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), stride=(1, 1)),
            nn.BatchNorm2d(100),
            self.get_activation(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.75)
        )
        self.cov4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), stride=(1, 1)),
            nn.BatchNorm2d(200),
            self.get_activation(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.75),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(8600, 2, bias=True)
        )

    def get_activation(self):
        if self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "leakyrelu":
            return nn.LeakyReLU()
        elif self.activation == "elu":
            return nn.ELU(alpha=1.0)


    def forward(self, x):
        x = self.cov1(x)
        x = self.cov2(x)
        x = self.cov3(x)
        x = self.cov4(x)
        x = self.fc(x)
        return x

class ShallowcovNet(nn.Module):
    def __init__(self, activation="relu"):
        super(ShallowcovNet, self).__init__()
        self.activation = activation
        
        self.cov1 = nn.Sequential( 
            nn.Conv2d(1, 64, kernel_size=(1, 25)),
            self.get_activation(),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(p=0.5)
            )
        self.cov2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(2, 25), stride=(1, 1)),
            self.get_activation(),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(p=0.5)
            )
        self.fc = nn.Sequential(
            nn.Linear(21632, 2)
            )

    def get_activation(self):
        if self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "leakyrelu":
            return nn.LeakyReLU()
        elif self.activation == "elu":
            return nn.ELU(alpha=1.0)

    def forward(self, x):
        x = self.cov1(x)
        x = self.cov2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x