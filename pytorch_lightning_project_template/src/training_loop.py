# -*- coding: utf-8 -*-
"""
Training loop based on PytorchLightning
"""

import logging
import os
import tqdm
import pickle

from functools import partial
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
import gin

from src.utils import save_weights

logger = logging.getLogger(__name__)

