#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple model definitions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, config):
        super(SimpleCNN, self).__init__()
        self.config = config
        self.conv1 = nn.Conv2d(3, config['n_filters'], kernel_size=5)
        self.conv2 = nn.Conv2d(config['n_filters'], config['n_filters'], kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(config['n_filters'] * 25, 50) # Oh well..
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x) # TODO: Shoulnt be linear