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
from argparse import ArgumentParser
from contextlib import redirect_stderr, redirect_stdout
from functools import partial
from multiprocessing import Pool, current_process, set_start_method
from pprint import pprint
from typing import Optional

import numpy as np
from my_favorite_things import save

# from numpy.typing import NDArray
from .constants import (
    ALG_CHOICES,
    DATA_CHOICES,
    DEFAULT_DEVICE,
    EVENT_CHOICES,
    HAMILTONIAN_CHOICES,
    INVMS,
    LAMBDA_OPERS,
    LAMBDA_VALS,
    LOG_DIR,
    NOISY_DIR,
    NORM_CHOICES,
    OUTPUT_DIR,
)
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
    bitflip_prob: float,
) -> None:
    """
    An individual, single-core worker to run jobs. To be called by Pool. The
    arguments are exactly the same as `main`, except in a slightly different
    order to allow the use of the `partial_worker` function to work, so check
    `main` for explanation of parameters.
    """
    worker_pid, worker_name = os.getpid(), current_process().name
    worker_title = f"{worker_pid} -- {worker_name}"
    # Create file names for output and error files in log directory
    ham_str = hamiltonian
    if hamiltonian == "H2":
        lambda_nume = lambda_nume
        lambda_denom = lambda_denom
        ham_str += f"-{''.join(lambda_nume)}-{''.join(lambda_denom)}"
    attrs = f"{dtype}_{etype}_{alg}_p{depth}_{ham_str}_{norm_scheme}"
    output_dir = LOG_DIR / "out" / attrs
    error_dir = LOG_DIR / "err" / attrs
    uid_num = re.findall(r"(\d+)", worker_name)[0]
    log_name = f"{uid_num:0>4}_{worker_pid}"
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
                    print(f"{worker_title}\n")
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
                        return_tuple += (data_dict["thetas"],)

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
    bitflip_prob: float = 0.0,
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
    if workers is None:
        workers = len(os.sched_get_affinity(0))
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
    set_start_method("spawn")
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
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--alg", "-a", type=lambda s: s.lower(), required=True, choices=ALG_CHOICES
    )
    parser.add_argument("--etype", "-e", type=str, required=True, choices=EVENT_CHOICES)
    parser.add_argument("--dtype", "-d", type=str, required=True, choices=DATA_CHOICES)
    parser.add_argument(
        "--hamiltonian", "-h", type=str, required=True, choices=HAMILTONIAN_CHOICES
    )
    parser.add_argument("--depth", "-p", type=int, required=True)
    parser.add_argument("--steps", "-s", type=int)
    parser.add_argument("--lambda-nume", "-N", type=str, nargs="?", const="min_Jij")
    parser.add_argument("--lambda-denom", "-D", type=str, nargs="?", const="max_Pij")
    parser.add_argument(
        "--norm_scheme", "-n", type=str, default="max", choices=NORM_CHOICES
    )
    parser.add_argument("--bitflip-prob", "-b", type=float, default=0.0)
    parser.add_argument("--device", type=str)
    parser.add_argument("--indlo", "-L", type=int, required=True)
    parser.add_argument("--indhi", "-H", type=int, required=True)
    parser.add_argument("--dryrun", action="store_true")
    args = parser.parse_args()

    alg = args.alg
    etype = args.etype
    dtype = args.dtype
    norm_scheme = args.norm_scheme
    bitflip_prob = args.bitflip_prob
    dryrun = args.dryrun

    # Steps should not be set for FALQON
    steps = args.steps
    if alg.lower() == "falqon" and steps is not None:
        parser.error(
            f"--steps cannot be set if algorithm is FALQON, but steps is {steps}."
        )
    if alg.lower() != "falqon" and steps is None:
        parser.error("--steps must be set if algorithm is not FALQON.")

    # Default device differs based on whether we have noise or not
    device = args.device
    if device is None:
        device = DEFAULT_DEVICE if bitflip_prob == 0 else "default.mixed"

    # Only constraint on depth is if we're doing VarQITE
    depth = args.depth
    if alg.lower() == "varqite" and depth != 1:
        parser.error(
            f"Algorithm set as VarQITE, so depth must be one. It is {depth} instead."
        )

    # Make sure index limits are ordered correctly
    ind_lo = args.indlo
    ind_hi = args.indhi
    if ind_lo >= ind_hi:
        parser.error(f"--indlo must be lesser than --indhi: {ind_lo} !< {ind_hi}")

    # The H0 Hamiltonian means lambda should not be set and if H2, then lambda
    # should be set
    hamiltonian = args.hamiltonian
    lambda_nume = args.lambda_nume
    lambda_denom = args.lambda_denom
    if hamiltonian == "H0" and (lambda_nume is not None or lambda_denom is not None):
        parser.error(
            "With Hamiltonian H0, the lambda arguments should not be set but:\n"
            f"\t{lambda_nume = }\n\t{lambda_denom = }"
        )
    elif hamiltonian == "H2" and (lambda_nume is None or lambda_denom is None):
        parser.error(
            "With Hamiltonian H2, the lambda arguments need to be set but:\n"
            f"\t{lambda_nume = }\n\t{lambda_denom = }"
        )

    # Make sure the values for lambda are valid if they are given
    if lambda_nume is not None and lambda_denom is not None:
        lambda_nume = lambda_nume.split("_")
        lambda_denom = lambda_denom.split("_")
        if lambda_nume[0] not in LAMBDA_OPERS:
            parser.error(
                f"The --lambda-nume operator, `{lambda_nume[0]}`, isn't valid. Must "
                f"be from the following list:\n{LAMBDA_OPERS}"
            )
        if lambda_denom[0] not in LAMBDA_OPERS:
            parser.error(
                f"The --lambda-denom operator, `{lambda_denom[0]}`, isn't valid. Must "
                f"be from the following list:\n{LAMBDA_OPERS}"
            )
        if lambda_nume[1] not in LAMBDA_VALS:
            parser.error(
                f"The --lambda-nume value, `{lambda_nume[1]}`, isn't valid. Must "
                f"be from the following list:\n{LAMBDA_VALS}"
            )
        if lambda_denom[1] not in LAMBDA_VALS:
            parser.error(
                f"The --lambda-denom value, `{lambda_denom[1]}`, isn't valid. Must "
                f"be from the following list:\n{LAMBDA_VALS}"
            )

    main(
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
        dryrun=dryrun,
    )
