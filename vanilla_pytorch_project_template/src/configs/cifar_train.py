# -*- coding: utf-8 -*-
"""
Configs used in the project
"""

from src.vegab import ConfigRegistry

cifar_train_configs = ConfigRegistry()

cifar_train_configs.set_root_config({
  "model_kwargs": {},
  "batch_size": 128,
  "model": "small_simple_cnn",
  "augmented": True,
  "lr": 0.1,
  "n_epochs": 100,
  "reload": False,
  "seed": 777,

  # Callbacks
  "lr_schedule_kwargs": {"schedule": [[2, 0.01], [10000, 0.001]]},
  "meta_saver": True,
  "lr_schedule": True,
})

c = cifar_train_configs['root']
c['dataset'] = 'cifar10'
cifar_train_configs['cifar10'] = c

c = cifar_train_configs['cifar10']
c['model'] = 'lenet'
cifar_train_configs['cifar10_lenet'] = c