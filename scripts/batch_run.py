"""
Allows for the batch submission of multiple events over multiple thread via
Python's multithreading package. This used to be set up as follows: all events
would be split up into groups of N and then each group, via a single thread,
would be sent to the class (now called `JobsRunner`) to run over every event and
save the output. Now, it, by default, send one event to one thread. Instead of
saving each event individually, it returns the data in this file and saves them
all together. I shouldn't have ever done it the previous way and I think I will
eventually remove the previous usecase.
"""

import os
import re
from contextlib import redirect_stderr, redirect_stdout
from functools import partial
from multiprocessing import Pool, current_process
from pprint import pprint
from typing import Optional

import numpy as np
from my_favorite_things import save

# from numpy.typing import NDArray
from .constants import DEFAULT_DEVICE, INVMS, LOG_DIR, NOISY_DIR, OUTPUT_DIR
from .job_runner import run_jobs

unique_kwargs = {
    "qaoa": ["gammas", "betas"],
    "maqaoa": ["gammas", "betas"],
    "xqaoa": ["gammmas", "betas", "alphas"],
    "falqon": ["betas", "depth_probs"],
    "varqite": ["thetas"],
}


def worker(
    evt_ind: int,
    alg: str,
    etype: str,
    dtype: str,
    hamiltonian: str,
    depth: int,
    norm_scheme: str,
    steps: int,
    device: str,
    lambda_nume: tuple[str, str],
    lambda_denom: tuple[str, str],
    shots: Optional[int],
    bitflip_prob: int,
) -> None:
    """
    An individual, single-core worker to run jobs. To be called by Pool. The
    arguments are exactly the same as `main`, except in a slightly different
    order to allow the use of the `partial_worker` function to work, so check
    `main` for explanation of parameters.
    """
    worker_uid = f"{os.getpid()} -- {current_process().name}"
    # Create file names for output and error files in log directory
    ham_str = hamiltonian
    if hamiltonian == "H2":
        lambda_nume = lambda_nume
        lambda_denom = lambda_denom
        ham_str += f"-{''.join(lambda_nume)}-{''.join(lambda_denom)}"
    attrs = f"{dtype}_{etype}_{alg}_p{depth}_{ham_str}_{norm_scheme}"
    output_dir = LOG_DIR / "out" / attrs
    error_dir = LOG_DIR / "err" / attrs
    log_name = f"{re.findall(r'(\d+)', current_process().name)[0]:0>4}_{os.getpid()}"
    out_path = output_dir / f"{log_name}.out"
    err_path = error_dir / f"{log_name}.err"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)

    # Create contexts for stdout and stderr files
    with open(out_path, "a") as fout, open(err_path, "a") as ferr:
        # Redirect stdout and stdout to these files
        with redirect_stdout(fout), redirect_stderr(ferr):
            # Run job
            try:
                # Print out info for entire one only once, when file it empty
                if not os.stat(out_path).st_size:
                    print(f"{worker_uid}\n")
                    print(f" {'-' * 10} PARAMS {'-' * 10} ")
                    for k, v in locals().items():
                        if k not in ["fout", "ferr", "ham_str"]:
                            print(f"{k} -- {v}")

                    print(f"\n {'-' * 10} START OF JOBS {'-' * 10} ")
                data_dict = run_jobs(
                    evt_ind=evt_ind,
                    etype=etype,
                    dtype=dtype,
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
                )
                return_tuple = (
                    data_dict["invm_p4s"],
                    data_dict["invms"],
                    data_dict["coeffs"],
                    data_dict["norm_coeffs"],
                    data_dict["probs"],
                    data_dict["costs"],
                    data_dict["expvals"],
                    data_dict["evals"],
                    data_dict["ranks"],
                    data_dict["rank_probs"],
                    data_dict["min_bitstrings"],
                    data_dict["min_energies"],
                )
                match alg.lower():
                    case "qaoa" | "maqaoa":
                        return_tuple += (data_dict["gammas"], data_dict["betas"])
                    case "xqaoa":
                        return_tuple += (
                            data_dict["gammas"],
                            data_dict["betas"],
                            data_dict["alphas"],
                        )
                    case "falqon":
                        return_tuple += (data_dict["betas"], data_dict["depth_probs"])
                    case "varqite":
                        return_tuple += data_dict["thetas"]

                return return_tuple
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
    ind_lo: str,
    ind_hi: str,
    hamiltonian: str,
    depth: int,
    norm_scheme: str,
    workers: Optional[int] = None,
    steps: Optional[int] = None,
    device: str = DEFAULT_DEVICE,
    lambda_nume: Optional[tuple[str, str]] = None,
    lambda_denom: Optional[tuple[str, str]] = None,
    shots: Optional[int] = None,
    bitflip_prob: int = 0,
    dryrun: bool = True,
) -> None:
    """
    Main function to run jobs. Distributes jobs in Pool to run.

    Parameters:
    workers - The number of threads to create.
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
    dryrun (default True) - If True, will not actually run jobs.
    """
    # Print used parameters
    print("Parameters:")
    pprint(locals())
    ignored = ["ignored", "dryrun", "workers", "ind_lo", "ind_hi"]
    partial_worker = partial(
        worker, **{k: v for k, v in locals().items() if k not in ignored}
    )
    evt_inds = np.arange(ind_lo, ind_hi)

    if dryrun:
        print("This was a dryrun. Ending...")
        return

    # Run jobs!
    with Pool(workers) as pool:
        invm_data = pool.map(partial_worker, evt_inds)

    kwargs = unique_kwargs[alg.lower()]
    # Number of events
    N_evts = len(invm_data)
    # Number of different output data
    N_outs = len(invm_data[0])
    # We are saving to individual per-invm directory so loop over them
    for ind, invm in enumerate(INVMS[:-1]):
        # Make a more specific string if we need to specify the lambda coefficient
        ham_str = hamiltonian
        if hamiltonian == "H2":
            lambda_nume = lambda_nume
            lambda_denom = lambda_denom
            ham_str += f"-{''.join(lambda_nume)}-{''.join(lambda_denom)}"
        # Make run specific directory
        root_dir = (
            (NOISY_DIR if bitflip_prob != 0 else OUTPUT_DIR)
            / alg.lower()
            / f"{etype}_{dtype}_{depth}_{ham_str}_{norm_scheme}"
        )
        os.makedirs(root_dir, exist_ok=True)

        # Create save directory
        invm_dir = root_dir / f"{invm:.2f}"
        print(f"Created directory: {invm_dir}")
        os.makedirs(invm_dir, exist_ok=True)

        # Save as permil, not percent
        noise = f"{1000 * bitflip_prob:0>3.0f}_" if bitflip_prob != 0 else ""
        name = f"eff_{noise}{ind_lo:0>{5}}-{ind_hi:0>{5}}"

        # Swapping from (N_evts, N_outs) to (N_outs, N_evts)
        outs = []
        for M in range(N_outs):
            out_res = []
            for N in range(N_evts):
                out_res.append(invm_data[N][M][ind])

            outs.append(np.array(out_res))
        # Save everything
        save(
            name=name,
            savedir=invm_dir,
            stype="npz",
            absolute=True,
            invm_p4s=outs[0],
            invms=outs[1],
            coeffs=outs[2],
            norm_coeffs=outs[3],
            probs=outs[4],
            costs=outs[5],
            expvals=outs[6],
            evals=outs[7],
            ranks=outs[8],
            rank_probs=outs[9],
            min_bitstrings=outs[10],
            min_energies=outs[11],
            # e.g. gammas, betas, and so on
            **dict(zip(kwargs, outs[-len(kwargs) :])),
        )


if __name__ == "__main__":
    # For
    # - VarQITE: set depth=1, steps=500 (can vary)
    # - FALQON: set depth=2500 (can vary) and comment out steps
    # - QAOA-like: set steps=1000 (can vary)
    # - H0: comment out "lambda_X" (not needed)
    ind_lo = 1000
    ind_hi = 2000

    alg = "MAQAOA"
    etype = "ttbar"
    dtype = "parton"
    hamiltonian = "H2"
    depth = 3
    steps = 100
    lambda_nume = ["min", "Jij"]
    lambda_denom = ["max", "Pij"]
    norm_scheme = "max"
    bitflip_prob = 0.0
    device = DEFAULT_DEVICE if bitflip_prob == 0 else "default.mixed"

    main(
        workers=10,
        alg=alg,
        etype=etype,
        dtype=dtype,
        ind_lo=ind_lo,
        ind_hi=ind_hi,
        hamiltonian=hamiltonian,
        depth=depth,
        steps=steps,
        device=device,
        norm_scheme=norm_scheme,
        lambda_nume=lambda_nume,
        lambda_denom=lambda_denom,
        bitflip_prob=bitflip_prob,
        dryrun=False,
    )
