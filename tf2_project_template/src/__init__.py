"""
Relevant constants and configurations
"""
# Env check
import os
if not int(os.environ.get("ENVCALLED", 0)):
    raise RuntimeError("Please source env before working on the project")

# Tensorflow and Keras specific constants
import tensorflow
assert tensorflow.__version__[0] == '2'
DATA_FORMAT = os.environ.get("DATA_FORMAT", "channels_first")
tensorflow.keras.backend.set_image_data_format(DATA_FORMAT)
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Configure paths
PROJECT_NAME = os.environ['PNAME']
PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(__file__), "results"))

# Configure logger
import logging
from .utils import configure_logger
configure_logger('')
logger = logging.getLogger()

# Opinionated default plotting styles
import matplotlib.style
import matplotlib as mpl
DEFAULT_FIGSIZE = 8
DEFAULT_LINEWIDTH = 3
DEFAULT_FONTSIZE = 22
mpl.style.use('seaborn-colorblind')
mpl.rcParams['figure.facecolor'] = 'w'
mpl.rcParams.update({'font.size': 14, 'lines.linewidth': 2, 'figure.figsize': (DEFAULT_FIGSIZE, DEFAULT_FIGSIZE / 1.61)})
mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['errorbar.capsize'] = 2
mpl.rcParams['image.cmap'] = 'cividis'
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['lines.markersize'] = 6
mpl.rcParams['lines.marker'] = None
mpl.rcParams['axes.grid'] = True
COLORS = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
mpl.rcParams.update({'font.size': DEFAULT_FONTSIZE, 'lines.linewidth': DEFAULT_LINEWIDTH,
                     'legend.fontsize': DEFAULT_FONTSIZE, 'axes.labelsize': DEFAULT_FONTSIZE,
                     'xtick.labelsize': DEFAULT_FONTSIZE, 'ytick.labelsize': DEFAULT_FONTSIZE,
                     'figure.figsize': (7, 7.0 / 1.4)})

# Misc
logger.info("GPU Available to Tensorflow:")
logger.info(tensorflow.test.is_gpu_available())
logger.info("TF can use Eager")
logger.info(tensorflow.executing_eagerly())
