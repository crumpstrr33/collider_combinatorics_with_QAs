"""
In this script, the method `efficiency` is given a list of event indices and the chosen
algorithm is ran for those indices and the passed parameters. Various data are saved in
a dictionary but of important is placement, i.e. whether the algorithm found the correct
answer as most likely (placement = 1), 2nd most likely (placement = 2) and so on.

Below "symmetric" or `sym` refers to the fact that "111000" and "000111" represent the
same answer and so the number counts for each are combined. Also "3+3" or `3p3` refers
to assuming that the answer must have three 1's and three 0's, i.e. three particle
assigned to each decaying top quark.
"""

import os
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime as dt
from pathlib import Path
from typing import Sequence

import numpy as np
from constants import (
    ALG_CHOICES,
    DATA_CHOICES,
    DEFAULT_BETA0,
    DEFAULT_DEVICE,
    DEFAULT_DT,
    DEFAULT_OPTIMIZER,
    DEFAULT_STEPS,
    DEFAULT_STEPSIZE,
    EVENT_CHOICES,
    HAMILTONIAN_CHOICES,
    INVMS,
    LAMBDA_OPERS,
    LAMBDA_VALS,
    NORM_CHOICES,
    NUM_FSP_DICT,
    OPTIMIZERS,
    OUTPUT_DIR,
    SYM_TRUE_BS_DICT,
)
from hamiltonians import get_coefficients, get_minimum_energies
from my_favorite_things import save
from numpy.typing import NDArray
from pennylane_algs import FALQON, MAQAOA, QAOA, XQAOA
from qc_utilities import get_data, swap
from scipy.special import comb

from data import split_data


class Efficiency:
    def __init__(
        self,
        etype: str,
        dtype: str,
        ind_lo: int,
        ind_hi: int,
        alg: str,
        depth: int,
        hamiltonian: str,
        norm_scheme: str,
        device: str,
        root_dir: Path,
        alg_kwargs: dict[str, ...],
        lambda_kwargs: dict[str, Sequence[str]],
    ):
        self.etype = etype
        self.dtype = dtype
        self.ind_lo = ind_lo
        self.ind_hi = ind_hi
        self.alg_str = alg.lower()
        self.depth = depth
        self.hamiltonian = hamiltonian
        self.norm_scheme = norm_scheme
        self.device = device
        self.root_dir = root_dir
        self.steps = alg_kwargs["steps"]
        self.alg_kwargs = alg_kwargs
        self.lambda_kwargs = lambda_kwargs

        # Info on the shape of the params for each alg, used for creating
        # the numpy arrays that saves them.
        self.num_fsp = NUM_FSP_DICT[self.etype]
        match self.alg_str.lower():
            case "qaoa":
                self.param_shapes = ((self.depth,), (self.depth,))
            case "maqaoa":
                self.param_shapes = (
                    (self.depth, int(comb(self.num_fsp, 2))),
                    (self.depth, self.num_fsp),
                )
            case "xqaoa":
                self.param_shapes = (
                    (self.depth, int(comb(self.num_fsp, 2))),
                    (self.depth, self.num_fsp),
                    (self.depth, self.num_fsp),
                )
            case "falqon":
                self.param_shapes = (self.depth,)

        # Total number of events
        self.N = self.ind_hi - self.ind_lo
        # Bit string that solves combinatorial problem
        self.soln_bitstring = SYM_TRUE_BS_DICT[self.etype]

    def find_rank_and_prob(self, make_symmetric: bool) -> tuple[int, float]:
        """
        For the current algorithm (assigned to self.alg), find the rank of `self.soln_bitstring`.

        Parameters:
        make_symmetric - If True, will combine the probabilities of symmetric
            bit strings, e.g. "010" and "101".
        """
        probs = self.alg.get_probs(as_dict=True)
        if make_symmetric:
            probs = {
                k: v + probs[swap(k)]
                for k, v in probs.items()
                if k.startswith(self.soln_bitstring[0])
            }

        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        return [
            (ind, float(bs_prob[1]))
            for ind, bs_prob in enumerate(sorted_probs)
            if bs_prob[0] == self.soln_bitstring
        ][0]

    def get_input_data(self) -> None:
        """
        Get all the relevant input data.
        """
        self.p4s, self.Jijs, self.Pijs, self.invms = get_data(
            etype=self.etype, dtype=self.dtype
        )
        print()
        self.coeffs = get_coefficients(
            hamiltonian=self.hamiltonian, evts=self.p4s, **self.lambda_kwargs
        )

        # Create arrays of shape: [num_invm_bins, num_evts, ...], splits it all
        # up to be per event per invariant mass bin
        split_evts, split_inds = split_data(evts=self.p4s)
        self.p4s = split_evts[:, self.ind_lo : self.ind_hi, ...]
        self.Jijs = self.Jijs[split_inds][:, self.ind_lo : self.ind_hi, ...]
        self.Pijs = self.Pijs[split_inds][:, self.ind_lo : self.ind_hi, ...]
        self.coeffs = self.coeffs[split_inds][:, self.ind_lo : self.ind_hi, ...]
        self.invms = self.invms[split_inds][:, self.ind_lo : self.ind_hi, ...]
        self.norm_coeffs = self.normalize_coeffs()

    def run_event(self, coeff: NDArray[NDArray[np.float64]]) -> None:
        """
        Creates and Runs the algorithm.
        """
        match self.alg_str:
            case "qaoa":
                Alg = QAOA
            case "maqaoa":
                Alg = MAQAOA
            case "xqaoa":
                Alg = XQAOA
            case "falqon":
                Alg = FALQON

        self.alg = Alg(
            coeff=coeff, depth=self.depth, device=self.device, **self.alg_kwargs
        )

        match self.alg_str:
            case "falqon":
                self.alg.run(print_it=False)
            case _:
                self.alg.optimize(print_it=False)

    def normalize_coeffs(self) -> NDArray[np.float64]:
        """
        Normalizes the coefficient matrix based on the normalization scheme.

        Parameters:
        coeff - The coefficient matrix.
        """
        # Do operation over the last two dimensions
        axes = (-2, -1)
        match self.norm_scheme:
            case "none":
                return self.coeffs
            case "max":
                oper = np.max
            case "mean":
                oper = np.mean
            case "sum":
                oper = np.sum

        return self.coeffs / oper(self.coeffs, axis=axes)[..., None, None]

    def get_output_data(self) -> None:
        """
        Main function. Runs the algorithm on each event and saves the output on
        a per invariant mass basis.
        """
        # Loop over invariant mass bins
        for invm_ind, (coeffs, minimums) in enumerate(
            zip(self.norm_coeffs, self.minima)
        ):
            invm = INVMS[invm_ind]
            N_evts, num_fsp, _ = coeffs.shape
            print(f"        ---- INVARIANT MASS = {invm:.2f} ----")

            # Create numpy arrays to save data in
            probs_arr = np.empty((N_evts, 2**num_fsp))
            sym_probs_arr = np.empty((N_evts, 2 ** (num_fsp - 1)))
            costs_arr = np.empty((N_evts, self.steps))
            params_arr = [
                np.empty((N_evts, *param_shape))
                for param_shape in self.param_shapes
            ]
            expval_arr = np.empty(N_evts)
            evals_arr = np.empty(N_evts)
            rank_arr = np.empty(N_evts)
            prob_arr = np.empty(N_evts)
            sym_rank_arr = np.empty(N_evts)
            sym_prob_arr = np.empty(N_evts)
            min_bitstring_arr = np.empty(N_evts)
            min_energy_arr = np.empty(N_evts)

            # Loop over individual events
            for ind, (coeff, minimum) in enumerate(zip(coeffs, minimums)):
                print(
                    f"Index: {ind + 1:>{len(str(self.N))}} / {self.N} "
                    + f"(p = {self.depth}) | norm inv mass = {invm:.2f} "
                    + f" | Event: {self.ind_lo + ind}",
                    end=" ",
                    flush=True,
                )
                # Run the algorithm
                t0 = dt.now()
                self.run_event(coeff=coeff)
                tot_time = (dt.now() - t0).total_seconds()

                # Gather the data
                probs = self.alg.get_probs(as_dict=True)
                sym_probs = {
                    k: v + probs[swap(k)]
                    for k, v in probs.items()
                    if k.startswith(self.soln_bitstring[0])
                }
                costs = self.alg.costs.numpy()
                params = [param.numpy() for param in self.alg.params]
                expval = costs[-1]
                evals = self.alg.evals
                rank, prob = self.find_rank_and_prob(make_symmetric=False)
                sym_rank, sym_prob = self.find_rank_and_prob(
                    make_symmetric=True
                )

                print(
                    f"| {self.alg_str.upper()} time: {tot_time:.2f} "
                    + f"seconds | Steps: {evals}",
                    flush=True,
                )

                # Store data in arrays
                probs_arr[ind] = np.array(list(probs.values()))
                sym_probs_arr[ind] = np.array(list(sym_probs.values()))
                costs_arr[ind] = np.pad(costs, (0, self.steps - len(costs)))
                for param_arr, one_params in zip(params_arr, params):
                    param_arr[ind] = one_params
                expval_arr[ind] = expval
                evals_arr[ind] = evals
                rank_arr[ind] = rank
                prob_arr[ind] = prob
                sym_rank_arr[ind] = sym_rank
                sym_prob_arr[ind] = sym_prob
                min_bitstring_arr[ind] = minimum[0]
                min_energy_arr[ind] = minimum[1]

            # Create save directory
            invm_dir = self.root_dir / f"{invm:.2f}"
            print(f"Created directory: {invm_dir}")
            os.makedirs(invm_dir, exist_ok=True)

            # Save params based on actual names which is algorithm-specific
            match self.alg_str.lower():
                case "qaoa" | "maqaoa":
                    params_dict = {
                        "gammas": params_arr[0],
                        "betas": params_arr[1],
                    }
                case "xqaoa":
                    params_dict = {
                        "gammas": params_arr[0],
                        "betas": params_arr[1],
                        "alphas": params_arr[2],
                    }
                case "falqon":
                    params_dict = {"betas": params_arr[0]}

            # Save all the info
            save(
                name=f"eff_{self.ind_lo}-{self.ind_hi}",
                savedir=invm_dir,
                stype="npz",
                absolute=True,
                invm_p4s=self.p4s[invm_ind],
                invms=self.invms[invm_ind],
                coeffs=self.coeffs[invm_ind],
                norm_coeffs=self.norm_coeffs[invm_ind],
                probs=probs_arr,
                sym_probs=sym_probs_arr,
                costs=costs_arr,
                expvals=expval_arr,
                evals=evals_arr,
                ranks=rank_arr,
                rank_probs=prob_arr,
                sym_ranks=sym_rank_arr,
                sym_rank_probs=sym_prob_arr,
                min_bitstrings=min_bitstring_arr,
                min_energies=min_energy_arr,
                **params_dict,
            )
            print()

    def brute_force(self) -> None:
        """
        Finds the minimum bitstring and corresponding energy by brute force for
        each event.
        """
        # Array split by invm with minimum bitstrings and energies
        self.minima = np.empty((*self.p4s.shape[:2], 2), dtype=object)
        for ind, invm_p4s in enumerate(self.p4s):
            minimums = get_minimum_energies(
                evts=invm_p4s,
                hamiltonian=self.hamiltonian,
                **self.lambda_kwargs,
            )
            self.minima[ind] = minimums

    def run(self) -> None:
        """
        Function to call. Runs each section.
        """
        self.get_input_data()
        self.brute_force()
        self.get_output_data()


if __name__ == "__main__":
    parser = ArgumentParser()
    # Algorithm to use, e.g. QAOA, FALQON, etc
    parser.add_argument(
        "--algorithm", "-a", required=True, type=str.lower, choices=ALG_CHOICES
    )
    # Type of event to run on, e.g. ttbar or tW
    parser.add_argument(
        "--etype", "-e", required=True, type=str, choices=EVENT_CHOICES
    )
    # Data to run on, e.g. parton or smeared
    parser.add_argument(
        "--dtype", "-D", required=True, type=str.lower, choices=DATA_CHOICES
    )
    # Coefficient of the quadratic term, e.g. og for Jij or qa for Jij + 2λPij
    parser.add_argument(
        "--hamiltonian",
        "-H",
        required=True,
        type=str,
        choices=HAMILTONIAN_CHOICES,
    )
    # How to normalize the coefficient matrix
    parser.add_argument(
        "--norm", required=True, type=str.lower, choices=NORM_CHOICES
    )
    # Specify lambda, e.g. --lambda-nume min Jij
    parser.add_argument("--lambdanume", required=False, type=str, nargs=2)
    parser.add_argument("--lambdadenom", required=False, type=str, nargs=2)
    # The device to use for pennylane
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    # Number of shots, defaults to None == infinite
    parser.add_argument("--shots", "-s", required=False)
    # The lower and upper limit of events to run, must match an index file
    parser.add_argument("--indlims", "-i", required=True, type=int, nargs=2)
    # Depth of circuit
    parser.add_argument("--depth", "-d", required=True, type=int)
    # If set, will not run simulation or save data at the end
    parser.add_argument("--dryrun", action=BooleanOptionalAction, default=False)
    ## ARGUMENTS FOR HYBRID ALGORITHMS
    # Max number of steps for optimizer
    parser.add_argument("--steps", "-S", default=DEFAULT_STEPS, type=int)
    # Optimizer to use, e.g. adam
    parser.add_argument(
        "--optimizer",
        "-o",
        type=str,
        default=DEFAULT_OPTIMIZER,
        choices=OPTIMIZERS,
    )
    # Stepsize of optimizer
    parser.add_argument("--stepsize", default=DEFAULT_STEPSIZE, type=float)
    ## ARGUMENTS FOR FALQON
    # Time step
    parser.add_argument("--dt", "-t", default=DEFAULT_DT, type=float)
    # Initial parameter value
    parser.add_argument("--initbeta", "-b", default=DEFAULT_BETA0, type=float)

    args = parser.parse_args()

    # Make sure order of lims is enforced
    if not (args.indlims[0] < args.indlims[1]):
        raise Exception(
            "The first value for `indlims` must be the smaller of the two."
            + f" It is {args.indlims}"
        )
    # If we need arguments for lambda, make sure they are of a correct form
    if args.hamiltonian == "H2":
        if (
            args.lambdanume[0] not in LAMBDA_OPERS
            or args.lambdanume[1] not in LAMBDA_VALS
            or args.lambdadenom[0] not in LAMBDA_OPERS
            or args.lambdadenom[1] not in LAMBDA_VALS
        ):
            raise Exception(
                "Lambda arguments are incorrect. Operators must be from: "
                f"{LAMBDA_OPERS} and values from {LAMBDA_VALS}."
            )

    # Temporary stops
    if args.shots is not None:
        raise Exception("Finite shot functionality has been removed! (for now)")
    if args.algorithm.lower() == "falqon":
        raise Exception("FALQON doesn't work! (yet)")

    # Make a more specific string if we need to specify the lambda coefficient
    ham_str = args.hamiltonian
    if args.hamiltonian == "H2":
        lambda_nume = args.lambdanume
        lambda_denom = args.lambdadenom
        ham_str = (
            f"{args.hamiltonian}-{''.join(lambda_nume)}-{''.join(lambda_denom)}"
        )
    # Make run specific directory, if it already exists we DO want an error
    root_dir = (
        OUTPUT_DIR
        / args.algorithm
        / f"{args.etype}_{args.dtype}_{args.depth}_{ham_str}_{args.norm}"
    )
    os.makedirs(root_dir, exist_ok=True)

    # Stuff that differs between QAOA and FALQON
    match args.algorithm.lower():
        case "falqon":
            alg_kwargs = {
                "shots": args.shots,
                "dt": args.dt,
                "initbeta": args.initbeta,
            }
        case _:
            alg_kwargs = {
                "shots": args.shots,
                "steps": args.steps,
            }
    # Run it!
    efficiency = Efficiency(
        etype=args.etype,
        dtype=args.dtype,
        ind_lo=args.indlims[0],
        ind_hi=args.indlims[1],
        alg=args.algorithm,
        depth=args.depth,
        hamiltonian=args.hamiltonian,
        norm_scheme=args.norm,
        device=args.device,
        root_dir=root_dir,
        alg_kwargs=alg_kwargs,
        lambda_kwargs={"nume": args.lambdanume, "denom": args.lambdadenom},
    )
    efficiency.run()
