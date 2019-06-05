"""
Relevant constants
"""
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Configure paths
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(__file__), "results"))

# Configure logger
from .utils import configure_logger
configure_logger('')