# -*- coding: utf-8 -*-
"""
Callback module (inspired by Keras).
"""

from .base import *

import logging

logger = logging.getLogger(__name__)

_ALIASES = {
    # Base
    "batch_lr_schedule": BatchLRSchedule,
    "neptune_monitor": NeptuneMonitor,
    "lr_schedule": LRSchedule,
    "meta_saver": MetaSaver,
    "save_weights": SaveWeights,
    "weight_norm": WeightNorm,
}

def get_callback(clb_name, verbose=1, **kwargs):
    if clb_name in _ALIASES:
        return _ALIASES[clb_name](**kwargs)
    else:
        if verbose:
            logger.warning("Couldn't find {} callback. Skipping.".format(clb_name))
        return None
