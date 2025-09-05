from itertools import product
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
DATA_DIR = Path(environ["DATADIR"]) / "data"
# Event data from e.g. madgraph, data used in simulations are subsets of this
# data massaged into a common format and saved as an `.npz` under the "data"
# directory in the repo's root.
EVT_DIR = DATA_DIR / "evt_data"
# Where the data from the QAs are outputted to
OUTPUT_DIR = DATA_DIR / "debug"
# Same as OUTPUT_DIR but for simulations with finite shots
SHOT_DIR = DATA_DIR / "shot_data"
# Same as OUTPUT_DIR but for simulations with noise
NOISY_DIR = DATA_DIR / "noisy_data"
# Output for log files. Code ran using NOHUP cmd in "batch_run.py" where log file
# path is specified
LOG_DIR = DATA_DIR / "logs"
# Log directory for finite shot runs
SHOT_LOG_DIR = DATA_DIR / "shot_logs"
# Directory for plots as created by the Autosavinator 3000 in "analysis.ipynb"
PLOTS_DIR = DATA_DIR / "plots"

# nonFALQON-specific defaults
DEFAULT_OPTIMIZER = "adam"
DEFAULT_STEPS = 1000
DEFAULT_STEPSIZE = 0.01
# FALQON-specific defaults
DEFAULT_BETA0 = 0
DEFAULT_DT = 0.08
# VarQITE-specific defaults
DEFAULT_PRECISION = 1e-5
DEFAULT_DTAU = 0.5
# Device to use
DEFAULT_DEVICE = "default.qubit"

ALG_CHOICES = ["qaoa", "maqaoa", "xqaoa", "falqon", "varqite"]
HAMILTONIAN_CHOICES = ["H0", "H1", "H2"]
LAMBDA_OPERS = ["min", "max", "mean"]
LAMBDA_VALS = ["Jij", "Pij"]
NORM_CHOICES = ["none", "max", "min", "trace", "mean", "sum", "minmax", "std"]
DATA_CHOICES = ["parton", "smeared"]
EVENT_CHOICES = ["ttbar", "tW", "6jet"]
OPTIMIZERS = ["grad_descent", "adagrad", "adam"]

SYM_TRUE_BS_DICT = {
    "ttbar": "000111",
    "tW": "00111",
    "6jet": None,
}
MASS_NORM_DICT = {
    "ttbar": 2 * TOP_MASS,
    "tW": TOP_MASS + W_MASS,
    # There is no normalization for 6 jets, this is arbitrary. But I set this
    # just to have some comparison/concept of mass bins vis a vis `INVMS`
    "6jet": 2 * TOP_MASS,
    "4top": 4 * TOP_MASS,
}
NUM_FSP_DICT = {"ttbar": 6, "tW": 5, "6jet": 6}

# This is for the PSG events, adds a lot more since we have choice for top mass
# and for number of jets (per quark)
top_masses = [173, 346, 500, 1000]
n_jets = [3, 4, 5, 8, 10, 12, 15]
for mt, nj in product(top_masses, n_jets):
    # Labeled as e.g. mt500_j10 for the case where the top quarks have a mass of
    # 500 GeV and each produce 10 jets.
    label = f"mt{mt}j{nj}"

    EVENT_CHOICES.append(label)
    SYM_TRUE_BS_DICT |= {label: "0" * nj + "1" * nj}
    MASS_NORM_DICT |= {label: 2 * mt}
    NUM_FSP_DICT |= {label: 2 * nj}
