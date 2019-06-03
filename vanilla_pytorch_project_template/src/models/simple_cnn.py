# -*- coding: utf-8 -*-
"""
SimpleCNN model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, config):
        super(SimpleCNN, self).__init__()
        self.config = config
        self.conv1 = nn.Conv2d(3, int(config['n_filters']), kernel_size=5)
        self.conv2 = nn.Conv2d(int(config['n_filters']), int(config['n_filters']), kernel_size=5)
        self.fc1 = nn.Linear(int(config['n_filters']) * 25, int(config['n_dense']))  # Oh well..
        self.fc2 = nn.Linear(int(config['n_dense']), 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def small_simple_cnn(**config):
    config_base = {
        "n_dense": 100,
        "n_filters": 100,
    }
    config_base.update(config)
    return SimpleCNN(config_base)


def medium_simple_cnn(**config):
    config_base = {
        "n_dense": 500,
        "n_filters": 100,
    }
    config_base.update(config)
    return SimpleCNN(config_base)
