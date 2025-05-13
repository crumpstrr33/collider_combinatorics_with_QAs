"""
This script allows for the submission of multiple "efficiency.py"'s. It will run
through all 2000 events (per invm bin) submitting `runs_per_invm_per_core` jobs
to each core. If `runs_per_invm_per_core=100`, then, with 2000 events per bin,
that is 20 cores used total, with 2,000*6/20 = 12,000 / 20 = 600 jobs per core.
"""

import os
from pathlib import Path
from subprocess import Popen
from typing import Optional

import numpy as np

from .constants import LOG_DIR
from .data import split_data
from .events import get_data

runs_per_invm_per_core = 100


def main(
    alg: str,
    etype: str,
    dtype: str,
    hamiltonian: str,
    depth: int,
    steps: int,
    norm: str,
    lambda_nume: Optional[tuple[str, str]] = None,
    lambda_denom: Optional[tuple[str, str]] = None,
) -> None:
    # e.g. 2000
    evts_per_invm = split_data(get_data(etype=etype, dtype=dtype)[0])[0].shape[1]
    ind_lims = np.arange(0, evts_per_invm + 1, runs_per_invm_per_core)
    ind_pairs = np.dstack((ind_lims[:-1], ind_lims[1:]))[0]

    # Create string of attributes for log file name
    ham_str = hamiltonian
    if hamiltonian == "H2":
        lambda_nume = lambda_nume
        lambda_denom = lambda_denom
        ham_str = f"-{''.join(lambda_nume)}-{''.join(lambda_denom)}"
    attrs = f"{dtype}_{etype}_{alg}_p{depth}_{ham_str}_{norm}"

    print("\nCOMMANDS TO BE RAN:")
    cmds = []
    for ind_lo, ind_hi in ind_pairs:
        cmd = " ".join(
            [
                "python",
                str(Path(__file__).parent / "efficiency.py"),
                f"--algorithm {alg}",
                f"--etype {etype}",
                f"--dtype {dtype}",
                f"--depth {depth}",
                f"--hamiltonian {hamiltonian}",
                f"--indlims {ind_lo} {ind_hi}",
                f"--steps {steps}",
                f"--lambdanume {' '.join(lambda_nume)}" if lambda_nume else "",
                f"--lambdadenom {' '.join(lambda_denom)}" if lambda_denom else "",
                f"--norm {norm}",
            ]
        ).split()

        print(" ".join(cmd))
        cmds.append(cmd)

    print(f"\nUsing {len(ind_lims) - 1} cores.")
    cont = input("Continue? [y/N]")

    if cont.lower() == "y":
        os.makedirs(LOG_DIR, exist_ok=True)
        for cmd, ind_pair in zip(cmds, ind_pairs):
            ind_lo, ind_hi = ind_pair
            # Log file name
            log_file = f"log_{attrs}_{ind_lo}-{ind_hi}.log"
            err_file = f"err_{attrs}_{ind_lo}-{ind_hi}.log"
            with (
                open(LOG_DIR / log_file, "w") as log,
                open(LOG_DIR / err_file, "w") as err,
            ):
                Popen(cmd, stdout=log, stderr=err)
    else:
        print("Aborting...")


if __name__ == "__main__":
    alg = "MAQAOA"
    etype = "ttbar"
    dtype = "parton"
    hamiltonian = "H2"
    depth = 5
    steps = 1000
    lambda_nume = ["min", "Jij"]
    lambda_denom = ["max", "Pij"]
    norm = "max"

    main(
        alg=alg,
        etype=etype,
        dtype=dtype,
        hamiltonian=hamiltonian,
        depth=depth,
        steps=steps,
        norm=norm,
        lambda_nume=lambda_nume,
        lambda_denom=lambda_denom,
    )
