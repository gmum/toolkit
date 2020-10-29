# -*- coding: utf-8 -*-
"""
Callbacks available in the project
"""
import logging
from src.callbacks.base import LRSchedule

logger = logging.getLogger(__name__)

# Add your callbacks here
_ALIASES = {
    "lr_schedule": LRSchedule
}

def get_callback(clb_name, verbose=1, **kwargs):
    if clb_name in _ALIASES:
        return _ALIASES[clb_name](**kwargs)
    else:
        if verbose:
            logger.warning("Couldn't find {} callback. Skipping.".format(clb_name))
        return None
