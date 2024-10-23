from os import environ
from pathlib import Path

from numpy import array

TOP_MASS = 173
W_MASS = 80.4
METRIC = array([1, -1, -1, -1])
# Mass ranges
INVMS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3]

# "DATADIR" is the env variable for the path to where you save your data
# then DATA_DIR is where the data for this project is saved.
DATA_DIR = Path(environ["DATADIR"]) / "vqa"
# Event data from e.g. madgraph
EVT_DIR = DATA_DIR / "evt_data"
# Where the data from the QAs are outputted to
OUTPUT_DIR = DATA_DIR / "data"
# Same as OUTPUT_DIR but for simulations with noise
NOISY_DIR = DATA_DIR / "noisy_data"
# Output for the analyses of data from running "postdata.py"
POSTDATA_DIR = DATA_DIR / "post_data"
# Data holding index files used to split up large runs, e.g. 2000 events split up
# into 80 25 event files. Generated by "choose_inds.py"
IND_DIR = DATA_DIR / "ind_data"
# Output for log files. Code ran using NOHUP cmd in "batch_run.py" where log file
# path is specified
LOG_DIR = DATA_DIR / "logs"
# Directory for plots as created by the Autosavinator 3000 in "analysis.ipynb"
PLOTS_DIR = DATA_DIR / "plots"

# nonFALQON-specific defaults
DEFAULT_OPTIMIZER = "adam"
DEFAULT_STEPS = 1000
DEFAULT_STEPSIZE = 0.01
# FALQON-specific defaults
DEFAULT_BETA0 = 0
DEFAULT_DT = 0.08

ALG_CHOICES = ["qaoa", "maqaoa", "xqaoa", "falqon", "nw_maqaoa", "hybrid"]
QUADCOEFF_CHOICES = ["H0", "H1", "QA"]
LAMBDA_CHOICES = ["QA", "avg", "Pijavg", "Pijmax"]
# All possible choices for Hamiltonian. H0 and H1 are defined in the paper
# QA specifies the choice for the lambda coefficient
QCL_CHOICES = [qc for qc in QUADCOEFF_CHOICES if qc != "QA"] + [
    f"QA_{lmbda}" for lmbda in LAMBDA_CHOICES
]
DATA_CHOICES = ["parton", "smeared", "delphes"]
EVENT_CHOICES = ["ttbar", "tW", "6jet"]
OPTIMIZERS = ["grad_descent", "adagrad", "adam"]

SYM_TRUE_BS_DICT = {"ttbar": "000111", "tW": "00111", "6jet": None}
MASS_NORM_DICT = {"ttbar": 2 * TOP_MASS, "tW": TOP_MASS + W_MASS}
NUM_FSP_DICT = {"ttbar": 6, "tW": 5, "6jet": 6}