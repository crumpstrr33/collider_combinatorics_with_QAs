import os
from datetime import datetime as dt
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from my_favorite_things import save
from numpy.typing import NDArray
from scipy.special import comb

from .constants import (
    DEFAULT_BETA0,
    DEFAULT_DT,
    DEFAULT_DTAU,
    DEFAULT_PRECISION,
    INVMS,
    LAMBDA_OPERS,
    LAMBDA_VALS,
    NUM_FSP_DICT,
    OUTPUT_DIR,
    SYM_TRUE_BS_DICT,
)
from .data import split_data
from .events import get_data
from .hamiltonians import get_coefficients, get_minimum_energies, swap
from .pennylane_algs import FALQON, MAQAOA, QAOA, XQAOA, VarQITE


class JobRunner:
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
        match self.alg_str:
            case "falqon":
                self.steps = self.depth
            case _:
                self.steps = alg_kwargs["steps"]
        self.alg_kwargs = alg_kwargs
        self.lambda_kwargs = lambda_kwargs

        # Info on the shape of the params for each alg, used for creating
        # the numpy arrays that saves them.
        self.num_fsp = NUM_FSP_DICT[self.etype]
        match self.alg_str:
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
                self.param_shapes = ((self.depth,),)
            case "varqite":
                # This assume p=1 for VarQITE always
                self.param_shapes = ((int(comb(self.num_fsp, 2)),),)
            case _:
                raise Exception(f"Unknown algorithm {self.alg_str}")

        # Total number of events
        self.N = self.ind_hi - self.ind_lo
        # Bit string that solves combinatorial problem
        self.soln_bitstring = SYM_TRUE_BS_DICT[self.etype]

    def find_rank_and_prob(self, make_symmetric: bool) -> tuple[int, float]:
        """
        For the current algorithm (assigned to self.alg), find the rank of
        `self.soln_bitstring`.

        Parameters:
        make_symmetric - If True, will combine the probabilities of symmetric
            bit strings, e.g. "010" and "101".
        """
        match self.alg_str:
            case "varqite":
                probs = self.alg.get_probs()
            case _:
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
        split_evts, split_inds = split_data(evts=self.p4s, etype=self.etype)
        # Total number of event, e.g. 200
        self.tot_evts = split_evts.shape[1]
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
            case "varqite":
                Alg = VarQITE

        self.alg = Alg(
            coeff=coeff, depth=self.depth, device=self.device, **self.alg_kwargs
        )

        match self.alg_str:
            case "falqon":
                self.alg.run(print_it=False)
            case "varqite":
                self.alg.optimize(print_progress=False)
            case _:
                self.alg.optimize(print_it=False)

    def normalize_coeffs(self) -> NDArray[np.float64]:
        """
        Normalizes the coefficient matrix based on the normalization scheme.

        Parameters:
        coeff - The coefficient matrix.
        """
        axes = (-2, -1)
        match self.norm_scheme:
            # No normalization
            case "none":
                norm = np.ones(self.coeffs.shape[:2])
                shift = np.zeros(self.coeffs.shape[:2])
            # Divide by the maximum value
            case "max":
                norm = np.max(self.coeffs, axis=axes)
                shift = np.zeros(self.coeffs.shape[:2])
            # Divide by the minimum value
            case "min":
                norm = np.min(self.coeffs, axis=axes)
                shift = np.zeros(self.coeffs.shape[:2])
            # Divide by the trace of the matrix
            case "trace":
                norm = np.trace(self.coeffs, axis1=-2, axis2=-1)
                shift = np.zeros(self.coeffs.shape[:2])
            # Divide by the mean of the matrix values
            case "mean":
                norm = np.mean(self.coeffs, axis=axes)
                shift = np.zeros(self.coeffs.shape[:2])
            # Divide by the sum of the matrix values
            case "sum":
                norm = np.sum(self.coeffs, axis=axes)
                shift = np.zeros(self.coeffs.shape[:2])
            # Shift by the min and divide by range, goes from 0 to 1
            case "minmax":
                norm = np.max(self.coeffs, axis=axes) - np.min(self.coeffs, axis=axes)
                shift = np.min(self.coeffs, axis=axes)
            # Shift by the mean and divide by the standard deviation
            case "std":
                norm = np.std(self.coeffs, axis=axes)
                shift = np.mean(self.coeffs, axis=axes)
            case _:
                raise Exception(f"Don't have norm: {self.norm_scheme}!")

        # Do operation over the last two dimensions and reshape to shape of
        # full coeffs array
        return (self.coeffs - shift[..., None, None]) / norm[..., None, None]

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
            expval_arr = np.empty(N_evts)
            evals_arr = np.empty(N_evts)
            rank_arr = np.empty(N_evts)
            prob_arr = np.empty(N_evts)
            sym_rank_arr = np.empty(N_evts)
            sym_prob_arr = np.empty(N_evts)
            min_bitstring_arr = np.empty(N_evts)
            min_energy_arr = np.empty(N_evts)
            params_arr = [
                np.empty((N_evts, *param_shape)) for param_shape in self.param_shapes
            ]
            if self.alg_str == "falqon":
                depth_probs_arr = np.empty((N_evts, self.depth, 2**num_fsp))

            # Loop over individual events
            for ind, (coeff, minimum) in enumerate(zip(coeffs, minimums)):
                print(
                    f"Index: {ind + 1:>{len(str(self.N))}} / {self.N} "
                    + f"(p = {self.depth}) | norm inv mass = {self.invms[invm_ind][ind]:.2f} "
                    + f" | Event: {self.ind_lo + ind}",
                    end=" ",
                    flush=True,
                )
                # Run the algorithm
                t0 = dt.now()
                self.run_event(coeff=coeff)
                tot_time = (dt.now() - t0).total_seconds()

                # Gather the data
                match self.alg_str:
                    case "varqite":
                        probs = self.alg.get_probs()
                        costs = self.alg.energies
                        evals = self.alg.total_steps
                    case _:
                        probs = self.alg.get_probs(as_dict=True)
                        costs = self.alg.costs.numpy()
                        evals = self.alg.evals
                if self.soln_bitstring is not None:
                    sym_probs = {
                        k: v + probs[swap(k)]
                        for k, v in probs.items()
                        if k.startswith(self.soln_bitstring[0])
                    }
                    rank, prob = self.find_rank_and_prob(make_symmetric=False)
                    sym_rank, sym_prob = self.find_rank_and_prob(make_symmetric=True)
                expval = costs[-1]
                match self.alg_str:
                    case "varqite":
                        params = [self.alg.current_thetas]
                    case "falqon":
                        params = self.alg.params
                    case _:
                        params = [param.numpy() for param in self.alg.params]

                print(
                    f"| {self.alg_str.upper()} time: {tot_time:.2f} "
                    + f"seconds | Steps: {evals}",
                    flush=True,
                )

                # Store data in arrays
                probs_arr[ind] = np.array(list(probs.values()))
                if self.soln_bitstring is not None:
                    sym_probs_arr[ind] = np.array(list(sym_probs.values()))
                    rank_arr[ind] = rank
                    prob_arr[ind] = prob
                    sym_rank_arr[ind] = sym_rank
                    sym_prob_arr[ind] = sym_prob
                costs_arr[ind] = np.pad(costs, (0, self.steps - len(costs)))
                for param_arr, one_params in zip(params_arr, params):
                    param_arr[ind] = one_params
                expval_arr[ind] = expval
                evals_arr[ind] = evals
                min_bitstring_arr[ind] = minimum[0]
                min_energy_arr[ind] = minimum[1]
                if self.alg_str == "falqon":
                    depth_probs_arr[ind] = self.alg.depth_probs

            # Create save directory
            invm_dir = self.root_dir / f"{invm:.2f}"
            print(f"Created directory: {invm_dir}")
            os.makedirs(invm_dir, exist_ok=True)

            # Save params based on actual names which is algorithm-specific
            match self.alg_str:
                case "qaoa" | "maqaoa":
                    extras_dict = {
                        "gammas": params_arr[0],
                        "betas": params_arr[1],
                    }
                case "xqaoa":
                    extras_dict = {
                        "gammas": params_arr[0],
                        "betas": params_arr[1],
                        "alphas": params_arr[2],
                    }
                case "falqon":
                    extras_dict = {
                        "betas": params_arr[0],
                        "depth_probs": depth_probs_arr,
                    }
                case "varqite":
                    extras_dict = {"thetas": params_arr[0]}
            # Things that only make sense if there is a correct bitstring
            if self.soln_bitstring is not None:
                extras_dict |= {
                    "sym_probs": sym_probs_arr,
                    "ranks": rank_arr,
                    "rank_probs": prob_arr,
                    "sym_ranks": sym_rank_arr,
                    "sym_rank_probs": sym_prob_arr,
                }

            # Save all the info
            pad = len(str(self.tot_evts))
            save(
                name=f"eff_{self.ind_lo:0>{pad}}-{self.ind_hi:0>{pad}}",
                savedir=invm_dir,
                stype="npz",
                absolute=True,
                invm_p4s=self.p4s[invm_ind],
                invms=self.invms[invm_ind],
                coeffs=self.coeffs[invm_ind],
                norm_coeffs=self.norm_coeffs[invm_ind],
                probs=probs_arr,
                costs=costs_arr,
                expvals=expval_arr,
                evals=evals_arr,
                min_bitstrings=min_bitstring_arr,
                min_energies=min_energy_arr,
                **extras_dict,
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


def run_jobs(
    etype: str,
    dtype: str,
    ind_lo: int,
    ind_hi: int,
    alg: str,
    depth: int,
    hamiltonian: str,
    norm_scheme: str,
    device: str,
    steps: int,
    evts_per_invm: int,
    lambda_nume: Optional[tuple[str, str]] = None,
    lambda_denom: Optional[tuple[str, str]] = None,
    shots: Optional[int] = None,
):
    """
    Wrapper function for the JobRunner class. Essentially will run an algorithm
    with all the given parameters for a specific number of jobs for every
    invariant mass bin. For example, if `ind_lo=10` and `ind_hi=25`. Then it
    will run 15 jobs for each of the bins starting with the 10th event that is
    between each bin, then the 11th, etc.

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
    evts_per_invm - Number of events per invariant mass bin. Added to name of
        data directory so that verifying the data can proceed correctly.
    lambda_nume (default None) - The numerator of the lambda coefficient used in
        the H2 Hamiltonian.
    lambda_denom (default None) - The denominator of the lambda coefficient used
        in the H2 Hamiltonian.
    shots (default None)- The number of shots to do each circuit run. If None,
        use infinite shots, the ideal case.
    """
    if steps is None and alg.lower() != "falqon":
        raise Exception(
            f"`steps` can only be `None` if algorithm is FALQON but it is {alg}."
        )
    # Make sure order of lims is enforced
    if not ind_lo < ind_hi:
        raise Exception(
            "The first value for `indlims` must be the smaller of the two."
            + f" It is [{ind_lo}, {ind_hi}]"
        )
    # If we need arguments for lambda, make sure they are of a correct form
    if hamiltonian == "H2":
        if (
            lambda_nume[0] not in LAMBDA_OPERS
            or lambda_nume[1] not in LAMBDA_VALS
            or lambda_denom[0] not in LAMBDA_OPERS
            or lambda_denom[1] not in LAMBDA_VALS
        ):
            raise Exception(
                "Lambda arguments are incorrect. Operators must be from: "
                f"{LAMBDA_OPERS} and values from {LAMBDA_VALS}."
            )

    # Temporary stops
    if shots is not None:
        raise Exception("Finite shot functionality has been removed! (for now)")

    # Make a more specific string if we need to specify the lambda coefficient
    ham_str = hamiltonian
    if hamiltonian == "H2":
        lambda_nume = lambda_nume
        lambda_denom = lambda_denom
        ham_str += f"-{''.join(lambda_nume)}-{''.join(lambda_denom)}"
    # Make run specific directory
    root_dir = (
        OUTPUT_DIR
        / alg.lower()
        / f"{etype}_{dtype}_{depth}_{ham_str}_{norm_scheme}_{evts_per_invm}"
    )
    os.makedirs(root_dir, exist_ok=True)

    # Stuff that differs between QAOA and FALQON
    match alg.lower():
        case "varqite":
            alg_kwargs = {
                "shots": shots,
                "steps": steps,
                "dtau": DEFAULT_DTAU,
                "prec": DEFAULT_PRECISION,
            }
        case "falqon":
            alg_kwargs = {"dt": DEFAULT_DT, "init_beta": DEFAULT_BETA0}
        case _:
            alg_kwargs = {"shots": shots, "steps": steps, "optimizer": "adam"}
    # Run it!
    job_runner = JobRunner(
        etype=etype,
        dtype=dtype,
        ind_lo=ind_lo,
        ind_hi=ind_hi,
        alg=alg,
        depth=depth,
        hamiltonian=hamiltonian,
        norm_scheme=norm_scheme,
        device=device,
        root_dir=root_dir,
        alg_kwargs=alg_kwargs,
        lambda_kwargs={"nume": lambda_nume, "denom": lambda_denom},
    )
    job_runner.run()
