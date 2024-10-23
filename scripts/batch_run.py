"""
Script used to run multiple instances of efficiency.py at the same time by passing
arguments via CLI.
"""
from argparse import ArgumentParser, BooleanOptionalAction
from subprocess import Popen

from constants import (
    ALG_CHOICES,
    DATA_CHOICES,
    DEFAULT_BETA0,
    DEFAULT_DT,
    DEFAULT_OPTIMIZER,
    DEFAULT_STEPS,
    DEFAULT_STEPSIZE,
    EVENT_CHOICES,
    LOG_DIR,
    OPTIMIZERS,
    QUADCOEFF_CHOICES,
)

parser = ArgumentParser()
# Algorithm to use, e.g. QAOA, FALQON, etc
parser.add_argument(
    "--algorithm", "-a", required=True, type=str.lower, choices=ALG_CHOICES
)
# Type of event to run on, e.g. ttbar or tW
parser.add_argument("--event", "-e", type=str, choices=EVENT_CHOICES)
# Data to run on, e.g. parton or smeared
parser.add_argument(
    "--dtype", "-D", required=True, type=str.lower, choices=DATA_CHOICES
)
# Coefficient of the quadratic term, e.g. og for Jij or qa for Jij + 2λPij
parser.add_argument(
    "--quadcoeff", "-c", required=True, type=str, choices=QUADCOEFF_CHOICES
)
# The lower and upper limit of events to run, must match an index file
parser.add_argument("--indlims", "-i", required=True, type=int, nargs=2)
# Optional unique ID for index file (needed if files are otherwise ambiguous)
parser.add_argument("--uid", "-u", type=int)
# Depth of circuit
parser.add_argument("--depth", "-d", required=True, type=int)
# If set, will not run simulation or save data at the end
parser.add_argument("--dryrun", action=BooleanOptionalAction, default=False)
# Lower limits for invariant mass
parser.add_argument("--invmlow", "-L", type=float, nargs="+")
# Upper limits for invariant mass
parser.add_argument("--invmhi", "-H", type=float, nargs="+")
## ARGUMENTS FOR HYBRID ALGORITHMS
# Max number of steps for optimizer
parser.add_argument("--steps", "-S", default=DEFAULT_STEPS, type=int)
# Optimizer to use, e.g. adam
parser.add_argument(
    "--optimizer", "-o", type=str, default=DEFAULT_OPTIMIZER, choices=OPTIMIZERS
)
# Stepsize of optimizer
parser.add_argument("--stepsize", "-s", default=DEFAULT_STEPSIZE, type=float)
## ARGUMENTS FOR FALQON
# Time step
parser.add_argument("--dt", "-t", default=DEFAULT_DT, type=float)
# Initial parameter value
parser.add_argument("--initbeta", "-b", default=DEFAULT_BETA0, type=float)

args = parser.parse_args()

# Have same number of low and high limits
if len(args.invmlow) != len(args.invmhi):
    raise Exception("Need same number of arguments for invm!")

# Run a job for each invariant mass range
for ind in range(len(args.invmlow)):
    invmlow = args.invmlow[ind]
    invmhi = args.invmhi[ind]
    print(f"Running for range {invmlow}-{invmhi}")

    if invmlow >= invmhi:
        raise Exception(f"Can't have invmlow={invmlow} and invmhi={invmhi}")

    # Build command
    cmd = " ".join(
        [
            "python efficiency.py",
            f"--algorithm {args.algorithm}",
            f"--event {args.event}",
            f"--dtype {args.dtype}",
            f"--quadcoeff {args.quadcoeff}",
            f"--indlims {str(args.indlims[0])} {str(args.indlims[1])}",
            f"--invmlims {str(invmlow)} {str(invmhi)}",
            f"--uid {str(args.uid)}" if args.uid is not None else "",
            f"--depth {str(args.depth)}",
            "--dryrun" if args.dryrun else "",
            # nonFALQON-specific arguments
            f"--steps {str(args.steps)}" if args.algorithm != "falqon" else "",
            f"--optimizer {args.optimizer}" if args.algorithm != "falqon" else "",
            f"--stepsize {args.stepsize}" if args.algorithm != "falqon" else "",
            # FALQON-specific arguments
            f"--dt {str(args.dt)}" if args.algorithm == "falqon" else "",
            f"--initbeta {str(args.initbeta)}" if args.algorithm == "falqon" else "",
        ]
    ).split()

    # Log file name
    log_file = (
        f"eff_{args.dtype}_{args.event}_{args.algorithm}_p{args.depth}"
        + f"_{args.indlims[0]}to{args.indlims[1]}_{invmlow}to{invmhi}.log"
    )
    with open(LOG_DIR / log_file, "w") as log:
        run_cmd = Popen(cmd, stdout=log)
