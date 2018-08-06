# -*- coding: utf-8 -*-
"""
Configs used in the project
"""

from src.vegab import ConfigRegistry

simple_CNN_configs = ConfigRegistry()

simple_CNN_configs.set_root_config({
  "batch_size": 128,
  "augmented": True,
  "lr": 0.1,
  "n_epochs": 100,
  "lr_schedule": [[2,0.01],[10000,0.001]],
  "n_dense": 100,
  "n_filters": 100,
  "reload": False,
  "seed": 777,
})

c = simple_CNN_configs['root']
c['dataset'] = 'cifar10'
simple_CNN_configs['cifar10'] = c


c = simple_CNN_configs['root']
c['dataset'] = 'cifar100'
simple_CNN_configs['cifar100'] = c