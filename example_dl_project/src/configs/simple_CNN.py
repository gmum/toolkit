# -*- coding: utf-8 -*-
"""
Configs used in the project
"""

from src.vegab import ConfigRegistry

simple_CNN_configs = ConfigRegistry()

simple_CNN_configs.set_root_config({
  "n_layers": 1,
  "n_epochs": 100,
  "dataset": "cifar10",
  "lr_schedule": [[10, 0.1], [20, 0.01]],
  "dim_dense": 100,
  "n_filters": 100
})