"""
This script allows for the submission of multiple "efficiency.py"'s. It will run
through all 2000 events (per invm bin) submitting `runs_per_invm_per_core` jobs
to each core. If `runs_per_invm_per_core=100`, then, with 2000 events per bin,
that is 20 cores used total, with 2,000*6/20 = 12,000 / 20 = 600 jobs per core.
"""

import os
from contextlib import redirect_stderr, redirect_stdout
from functools import partial
from multiprocessing import Pool
from pprint import pprint
from typing import Optional

import numpy as np

from .constants import DEFAULT_DEVICE, LOG_DIR
from .data import split_data
from .efficiency import run_jobs
from .events import get_data

runs_per_invm_per_core = 100


def worker(
    ind_lo: int,
    ind_hi: int,
    etype: str,
    dtype: str,
    alg: str,
    depth: int,
    hamiltonian: str,
    norm_scheme: str,
    device: str,
    steps: int,
    lambda_nume: tuple[str, str],
    lambda_denom: tuple[str, str],
    shots: Optional[int],
    bitflip_prob: int,
    evts_per_invm: int,
) -> None:
    """
    An individual, single-core worker to run jobs. To be called by Pool. The
    arguments are exactly the same as `main`, except in a slightly different
    order to allow the use of the `partial_worker` function to work, so check
    `main` for explanation of parameters.
    """
    # Create file names for output and error files in log directory
    ham_str = hamiltonian
    if hamiltonian == "H2":
        lambda_nume = lambda_nume
        lambda_denom = lambda_denom
        ham_str += f"-{''.join(lambda_nume)}-{''.join(lambda_denom)}"
    attrs = f"{dtype}_{etype}_{alg}_p{depth}_{ham_str}_{norm_scheme}"
    out_path = LOG_DIR / f"job_{attrs}_{ind_lo:0>6}-{ind_hi:0>6}.out"
    err_path = LOG_DIR / f"job_{attrs}_{ind_lo:0>6}-{ind_hi:0>6}.err"
    os.makedirs(LOG_DIR, exist_ok=True)

    # Create contexts for stdout and stderr files
    with open(out_path, "w") as fout, open(err_path, "w") as ferr:
        # Redirect stdout and stdout to these files
        with redirect_stdout(fout), redirect_stderr(ferr):
            # Run job
            try:
                print(f" {'-' * 10} PARAMS {'-' * 10} ")
                for k, v in locals().items():
                    if k not in ["fout", "ferr", "ham_str"]:
                        print(f"{k} -- {v}")

                print(f"\n {'-' * 10} START OF JOBS {'-' * 10} ")
                run_jobs(
                    etype=etype,
                    dtype=dtype,
                    ind_lo=ind_lo,
                    ind_hi=ind_hi,
                    alg=alg,
                    depth=depth,
                    hamiltonian=hamiltonian,
                    norm_scheme=norm_scheme,
                    device=device,
                    steps=steps,
                    lambda_nume=lambda_nume,
                    lambda_denom=lambda_denom,
                    shots=shots,
                    bitflip_prob=bitflip_prob,
                    evts_per_invm=evts_per_invm,
                )
            # Or error to file
            except Exception:
                from traceback import print_exc

                # Writes error to file
                print_exc(file=ferr)
                raise Exception("Job failed.")


def main(
    alg: str,
    etype: str,
    dtype: str,
    hamiltonian: str,
    depth: int,
    norm_scheme: str,
    steps: Optional[int] = None,
    device: str = DEFAULT_DEVICE,
    lambda_nume: Optional[tuple[str, str]] = None,
    lambda_denom: Optional[tuple[str, str]] = None,
    shots: Optional[int] = None,
    bitflip_prob: int = 0,
    evts_per_invm: Optional[int] = None,
    dryrun: bool = True,
) -> None:
    """
    Main function to run jobs. Distributes jobs in Pool to run.

    Parameters:
    etype - The event type, currently can be "ttbar", "tW" or "6jet".
    dtype - The data type, currently can be "parton" or "smeared".
    ind_lo - The lower index of events to run the algorithm on, inclusive.
    ind_hi - The higher index of events to run the algorithm on, exclusive.
    alg - The algorithm used for the data. Currently can be "qaoa", "maqaoa",
        "xqaoa", or "falqon".
    depth - The depth of the circuit ran
    hamiltonian - Which Hamiltonian used. Can be "H0", "H1", or "H2". If "H2",
        must define `lambda_nume` and `lambda_denom`.
    norm_scheme - The normalization scheme used for the coefficient matrix. Can
        be "max", "mean", or "sum".
    device - The Pennylane device to use, e.g. "default.qubit".
    steps - The number of classical optimization steps for the VQA to take
    lambda_nume - The numerator of the lambda coefficient used in
        the H2 Hamiltonian.
    lambda_denom - The denominator of the lambda coefficient used
        in the H2 Hamiltonian.
    shots - The number of shots to do each circuit run. If None, use infinite
        shots, the ideal case.
    bitflip_prob - The probability of a bitflip error for a gate execution.
        The device must be set to "default.mixed" if this is nonzero.
    evts_per_invm (default None) - Total number of events to use per invariant
        mass bin. If None, uses all of them.
    dryrun (default True) - If True, will not actually run jobs.
    """
    # Find number of events each invariant mass bin will have (assuming equal
    # numbers per bin). Can be specified so as to not use all data
    evts_per_invm = (
        split_data(
            evts=get_data(etype=etype, dtype=dtype, print_num_evts=False)[0],
            etype=etype,
        )[0].shape[1]
        if evts_per_invm is None
        else evts_per_invm
    )
    # Print used parameters
    print("Parameters:")
    pprint(locals())
    ignored = ["ignored", "dryrun"]
    partial_worker = partial(
        worker, **{k: v for k, v in locals().items() if k not in ignored}
    )

    # Find the index limits of each job, e.g. [0, 100, 200, 300, ...]
    ind_lims = np.arange(0, evts_per_invm + 1, runs_per_invm_per_core)
    # Turn those limits into tuple for low and high limits for each job
    ind_pairs = np.dstack((ind_lims[:-1], ind_lims[1:]))[0]

    # Print index limits that are going to be used
    print("\nIndex pairs:")
    for ind_pair in ind_pairs:
        print(f"\t{ind_pair}")
    print(f"\nUsing {len(ind_lims) - 1} cores.")

    # Run jobs!
    if dryrun:
        print("This was a dryrun. Ending...")
    else:
        with Pool() as pool:
            pool.starmap(partial_worker, ind_pairs)


if __name__ == "__main__":
    # For
    # - VarQITE: set depth=1, steps=500 (can vary)
    # - FALQON: set depth=2500 (can vary) and comment out steps
    # - QAOA-like: set steps=1000 (can vary)
    # - H0: comment out "lambda_X" (not needed)
    evts_per_invm = None

    alg = "MAQAOA"
    etype = "ttbar"
    dtype = "parton"
    hamiltonian = "H2"
    depth = 5
    steps = 1000
    lambda_nume = ["min", "Jij"]
    lambda_denom = ["max", "Pij"]
    norm_scheme = "max"
    bitflip_prob = 0.0
    device = DEFAULT_DEVICE if bitflip_prob == 0 else "default.mixed"

    main(
        alg=alg,
        etype=etype,
        dtype=dtype,
        hamiltonian=hamiltonian,
        depth=depth,
        steps=steps,
        device=device,
        norm_scheme=norm_scheme,
        lambda_nume=lambda_nume,
        lambda_denom=lambda_denom,
        bitflip_prob=bitflip_prob,
        evts_per_invm=evts_per_invm,
        dryrun=False,
    )
