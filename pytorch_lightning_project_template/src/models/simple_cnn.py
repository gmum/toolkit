# -*- coding: utf-8 -*-
"""
SimpleCNN model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import gin


@gin.configurable
class SimpleCNN(nn.Module):
    def __init__(self, n_filters, n_dense):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, int(n_filters), kernel_size=5)
        self.conv2 = nn.Conv2d(int(n_filters), int(n_filters), kernel_size=5)
        self.fc1 = nn.Linear(int(n_filters) * 25, int(n_dense))  # Oh well..
        self.fc2 = nn.Linear(int(n_dense), 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
