"""
Holds the class(es) for running simulations based on the inputted parameters.
Originally, there was only `JobsRunner` which was called `JobRunner`. I changed
this to include the "s" because it is running multiple events (defined by
the `ind_lo` and `ind_hi` keywords) and that gives me `JobRunner` to refer to
the class that runs only a single event. I made this change so that there is
more flexibility for multithreading, e.g. saying "hey 128 cores, here's X events
and I want to to continually run through them 1 event at a time". So they are
essentially identical; the differences arise in the simplifying for `JobRunner`.
"""

from datetime import datetime as dt
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import comb

from .constants import (
    DEFAULT_BETA0,
    DEFAULT_DT,
    DEFAULT_DTAU,
    DEFAULT_OPTIMIZER,
    DEFAULT_PRECISION,
    INVMS,
    LAMBDA_OPERS,
    LAMBDA_VALS,
    NUM_FSP_DICT,
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
        evt_ind: int,
        alg: str,
        depth: int,
        hamiltonian: str,
        norm_scheme: str,
        device: str,
        alg_kwargs: dict[str, ...],
        lambda_kwargs: dict[str, Sequence[str]],
    ):
        """
        Class used to to run a single event based on the given parameters.
        Gathers the input data, and runs the simulation. The output data can be
        collected with other events to save them. The kwargs above are what
        define a specific "run".

        Parameters:
        etype - The event type, can be "ttbar", "tW", or "6jet"
        dtype - The data type, can be "parton", or "smeared"
        evt_ind - Which event to run from the data file based on `etype` and
            `dtype`.
        alg - The algorithm to simulate, can be "qaoa", "maqaoa", "xqaoa",
            "falqon", or "varqite".
        depth - How many layers in the circuit. This isn't implemented for
            VarQITE, so it will be one if `alg="varqite"`
        hamiltonian - The Hamiltonian to use as the cost function, can be "H0",
            "H1", or "H2" where H2 = H0 + λ/2 * H1.
        norm_scheme - How to normalize the coefficients of the Hamiltonian
            terms (which are given as a symmetric matrix), can be "none", "max"
            "min", "trace", "mean", "sum", "minmax", or "std" which are defined
            in the `normalize_coeffs` method below. This shouldn't affect the
            physics but can affect the convergence rates and such.
        device - The Pennylane device to use.
        alg_kwargs - A dictionary of kwargs to pass specifically to the
            algorithm.
        lambda_kwargs - A dictionary of the form
                {"nume": lambda_nume, "denom": lambda_denom}
            where `lambda_nume` and `lambda_denom` are the numerator and
            denominator of the λ coefficient used in the H2 Hamiltonian.
        """
        self.etype = etype
        self.dtype = dtype
        self.evt_ind = evt_ind
        self.alg_str = alg.lower()
        self.depth = depth
        self.hamiltonian = hamiltonian
        self.norm_scheme = norm_scheme
        self.device = device
        self.alg_kwargs = alg_kwargs
        self.lambda_kwargs = lambda_kwargs

        # Info on what "steps" means. For FALQON, it's just the total depth
        # (since it's iterative) but for the others, it's whatever is passed,
        # i.e. the number of circuit optimizations. Also, info on the shape of
        # the params for each alg, used for creating the numpy arrays that saves
        # them.
        self.num_fsp = NUM_FSP_DICT[self.etype]
        match self.alg_str:
            case "qaoa":
                self.steps = alg_kwargs["steps"]
                self.param_shapes = ((self.depth,), (self.depth,))
            case "maqaoa":
                self.steps = alg_kwargs["steps"]
                self.param_shapes = (
                    (self.depth, int(comb(self.num_fsp, 2))),
                    (self.depth, self.num_fsp),
                )
            case "xqaoa":
                self.steps = alg_kwargs["steps"]
                self.param_shapes = (
                    (self.depth, int(comb(self.num_fsp, 2))),
                    (self.depth, self.num_fsp),
                    (self.depth, self.num_fsp),
                )
            case "falqon":
                self.steps = self.depth
                self.param_shapes = ((self.depth,),)
            case "varqite":
                self.steps = alg_kwargs["steps"]
                # This assume p=1 for VarQITE always
                self.param_shapes = ((int(comb(self.num_fsp, 2)),),)
            case _:
                raise Exception(f"Unknown algorithm {self.alg_str}")

        # Bit string that solves combinatorial problem
        self.soln_bitstring = SYM_TRUE_BS_DICT[self.etype]

    def normalize_coeffs(self) -> NDArray[np.float64]:
        """
        Normalizes the coefficient matrix based on the normalization scheme.

        Parameters:
        coeff - The coefficient matrix.
        """
        # min, max, etc over the last two axes (e.g. over each Jij)
        axes = (-2, -1)
        match self.norm_scheme:
            # No normalization
            case "none":
                norm = np.ones(self.coeffs.shape[0])
                shift = np.zeros(self.coeffs.shape[0])
            # Divide by the maximum value
            case "max":
                norm = np.max(self.coeffs, axis=axes)
                shift = np.zeros(self.coeffs.shape[0])
            # Divide by the minimum value
            case "min":
                norm = np.min(self.coeffs, axis=axes)
                shift = np.zeros(self.coeffs.shape[0])
            # Divide by the trace of the matrix
            case "trace":
                norm = np.trace(self.coeffs, axis1=-2, axis2=-1)
                shift = np.zeros(self.coeffs.shape[0])
            # Divide by the mean of the matrix values
            case "mean":
                norm = np.mean(self.coeffs, axis=axes)
                shift = np.zeros(self.coeffs.shape[0])
            # Divide by the sum of the matrix values
            case "sum":
                norm = np.sum(self.coeffs, axis=axes)
                shift = np.zeros(self.coeffs.shape[0])
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

        # Do operation over the last two dimensions
        return (self.coeffs - shift[..., None, None]) / norm[..., None, None]

    def get_input_data(self) -> None:
        """
        Get all the relevant input data.
        """
        self.p4s, self.Jijs, self.Pijs, self.invms = get_data(
            etype=self.etype, dtype=self.dtype, print_num_evts=False
        )
        print()
        self.coeffs = get_coefficients(
            hamiltonian=self.hamiltonian, evts=self.p4s, **self.lambda_kwargs
        )

        # Create arrays of shape: [num_invm_bins, num_evts, ...], splits it all
        # up to be per event per invariant mass bin
        split_evts, split_inds = split_data(evts=self.p4s, etype=self.etype)
        # Total number of event, e.g. 2000
        self.tot_evts = split_evts.shape[1]
        self.p4s = split_evts[:, self.evt_ind]
        self.Jijs = self.Jijs[split_inds][:, self.evt_ind]
        self.Pijs = self.Pijs[split_inds][:, self.evt_ind]
        self.coeffs = self.coeffs[split_inds][:, self.evt_ind]
        self.invms = self.invms[split_inds][:, self.evt_ind]
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

    def get_output_data(self) -> None:
        """
        Main function. Runs the algorithm the event for each invariant mass.
        Stores the results in specific class variables:
            expvals -- The expectation value of the final circuit
            params -- A list of the params of the circuit
            costs -- The evolution of the expectation value per step
            evals -- Number of circuit updates (evaluations) completed
            probs -- The probability for each bitstring ordered by bitstring value
            sym_probs -- Same as `probs` but combining symmetric bitstrings,
                i.e. when swapping 0 <--> 1
            rank -- The placement/rank of the correct bitstring
                e.g. 0 == first place == most probable
            prob --  The probability of this correct bitstring
            sym_rank -- The same as `rank` assuming bit symmetry
            sym_prob -- The same of `prob` assuming bit symmetry
            min_bitstring -- The bitstring that minimizes the energy
            min_energy -- Said energy of this ground state
            depth_probs -- (only for FALQON) The probability for each depth
        Then `params` is further broken up for their specific parameter equation
        symbols: `gammas` and `betas` for QAOA and ma-QAOA, additionally `alpha`
        for XQAOA, `betas` for FALQON and `thetas` for VarQITE.
        """
        iterable = zip(INVMS, self.invms, self.norm_coeffs, self.minima)
        print(
            "---------------- "
            f"EVENT START: {self.evt_ind} [{self.etype}, {self.dtype}, "
            f"{self.hamiltonian}, {self.alg_str}, p={self.depth}]"
            " ----------------"
        )
        self.expvals = []
        self.params = []
        self.costs = []
        self.evals = []
        self.probs = []
        self.sym_probs = []
        self.rank = []
        self.prob = []
        self.sym_rank = []
        self.sym_prob = []
        self.min_bitstring = []
        self.min_energy = []
        if self.alg_str == "falqon":
            self.depth_probs = []
        for invm_bin, invm, coeff, minimum in iterable:
            print(
                f"Event index: {self.evt_ind} -- invm bin: {invm_bin:.2f} "
                + f"(p = {self.depth}) | "
                + f"evt inv mass = {invm:.2f}",
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

            self.expvals.append(expval)
            self.params.append(params)
            # Pad with nans so it's not ragged, can be made into numpy array
            self.costs.append(list(costs) + [np.nan] * (self.steps - len(costs)))
            self.evals.append(evals)
            self.probs.append(list(probs.values()))
            self.sym_probs.append(list(sym_probs.values()))
            self.rank.append(rank)
            self.prob.append(prob)
            self.sym_rank.append(sym_rank)
            self.sym_prob.append(sym_prob)
            self.min_bitstring.append(minimum[0])
            self.min_energy.append(minimum[1])
            if self.alg_str == "falqon":
                self.depth_probs.append(self.alg.depth_probs)

        # Save params based on actual names which is algorithm-specific
        match self.alg_str:
            case "qaoa" | "maqaoa":
                self.gammas = np.array([invm_evt[0] for invm_evt in self.params])
                self.betas = np.array([invm_evt[1] for invm_evt in self.params])
            case "xqaoa":
                self.gammas = np.array([invm_evt[0] for invm_evt in self.params])
                self.betas = np.array([invm_evt[1] for invm_evt in self.params])
                self.alphas = np.array([invm_evt[2] for invm_evt in self.params])
            case "falqon":
                self.betas = np.array([invm_evt[0] for invm_evt in self.params])
            case "varqite":
                self.thetas = np.array([invm_evt[0] for invm_evt in self.params])

        print(
            "---------------- "
            f"EVENT END: {self.evt_ind} [{self.etype}, {self.dtype}, "
            f"{self.hamiltonian}, {self.alg_str}, p={self.depth}]"
            " ----------------"
        )

        # I know this is not best practice, shhhhh
        self.expvals = np.array(self.expvals)
        self.costs = np.array(self.costs)
        self.evals = np.array(self.evals)
        self.probs = np.array(self.probs)
        self.sym_probs = np.array(self.sym_probs)
        self.rank = np.array(self.rank)
        self.prob = np.array(self.prob)
        self.sym_rank = np.array(self.sym_rank)
        self.sym_prob = np.array(self.sym_prob)
        self.min_bitstring = np.array(self.min_bitstring)
        self.min_energy = np.array(self.min_energy)
        if self.alg_str == "falqon":
            self.depth_probs = np.array(self.depth_probs)

    def brute_force(self):
        """
            Finds the minimum bitstring and corresponding energy by brute force for
            each invariant mass event.
        it's real tough spending two weeks"""
        self.minima = get_minimum_energies(
            evts=self.p4s, hamiltonian=self.hamiltonian, **self.lambda_kwargs
        )

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

    def run(self) -> None:
        """
        Function to call. Runs each section.
        """
        self.get_input_data()
        self.brute_force()
        self.get_output_data()


class JobsRunner:
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
        # root_dir: Path,
        alg_kwargs: dict[str, ...],
        lambda_kwargs: dict[str, Sequence[str]],
    ):
        """
        The same as `JobRunner` (note the lack of "s") except will run for a
        range of events and save them to a file in `root_dir`. This is more
        involved since it involves N events per invariant mass, not just one,
        so there's an extra dimension for the numpy calculations.

        Parameters:
        etype - The event type, can be "ttbar", "tW", or "6jet"
        dtype - The data type, can be "parton", or "smeared"
        ind_lo - The lower limit on the event index. If we have N events per
            invariant mass bin, this says to start running the simulation on
            the "ind_lo"th event for each bin.
        ind_hi - The upper limit on the event index. We stop on (one before)
            the "ind_hi"th event.
        alg - The algorithm to simulate, can be "qaoa", "maqaoa", "xqaoa",
            "falqon", or "varqite".
        depth - How many layers in the circuit. This isn't implemented for
            VarQITE, so it will be one if `alg="varqite"`
        hamiltonian - The Hamiltonian to use as the cost function, can be "H0",
            "H1", or "H2" where H2 = H0 + λ/2 * H1.
        norm_scheme - How to normalize the coefficients of the Hamiltonian
            terms (which are given as a symmetric matrix), can be "none", "max"
            "min", "trace", "mean", "sum", "minmax", or "std" which are defined
            in the `normalize_coeffs` method below. This shouldn't affect the
            physics but can affect the convergence rates and such.
        device - The Pennylane device to use.
        # root_dir - The root directory into which to save the output data.
        alg_kwargs - A dictionary of kwargs to pass specifically to the
            algorithm.
        lambda_kwargs - A dictionary of the form
                {"nume": lambda_nume, "denom": lambda_denom}
            where `lambda_nume` and `lambda_denom` are the numerator and
            denominator of the λ coefficient used in the H2 Hamiltonian.
        """
        self.etype = etype
        self.dtype = dtype
        self.ind_lo = ind_lo
        self.ind_hi = ind_hi
        self.alg_str = alg.lower()
        self.depth = depth
        self.hamiltonian = hamiltonian
        self.norm_scheme = norm_scheme
        self.device = device
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

    def _get_alg(self, coeff: NDArray[NDArray[np.float64]]) -> None:
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

        return Alg(coeff=coeff, depth=self.depth, device=self.device, **self.alg_kwargs)

    def run_event(self, coeff: NDArray[NDArray[np.float64]]) -> None:
        """
        Creates and Runs the algorithm.
        """
        self.alg = self._get_alg(coeff=coeff)

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
        N_invms, N_evts, num_fsp, _ = self.norm_coeffs.shape
        # Done just to get the shape of the parameters, doesn't depend on the
        # coefficients used.
        self.param_shapes = self._get_alg(coeff=self.norm_coeffs[0][0]).param_shapes

        # Create numpy arrays to save data in
        self.params = [np.empty((N_invms, *shape)) for shape in self.param_shapes]
        self.probs = np.empty((N_invms, N_evts, 2**num_fsp))
        self.sym_probs = np.empty((N_invms, N_evts, 2 ** (num_fsp - 1)))
        self.costs = np.empty((N_invms, N_evts, self.steps))
        self.expvals = np.empty((N_invms, N_evts))
        self.evals = np.empty((N_invms, N_evts))
        self.rank = np.empty((N_invms, N_evts))
        self.prob = np.empty((N_invms, N_evts))
        self.sym_rank = np.empty((N_invms, N_evts))
        self.sym_prob = np.empty((N_invms, N_evts))
        self.min_bitstring = np.empty((N_invms, N_evts))
        self.min_energy = np.empty((N_invms, N_evts))
        self.params = [
            np.empty((N_invms, N_evts, *param_shape)) for param_shape in self.param_shapes
        ]
        if self.alg_str == "falqon":
            self.depth_probs = np.empty((N_invms, N_evts, self.depth, 2**num_fsp))

        # Loop over invariant mass bins
        for invm_ind, (coeffs, minimums) in enumerate(zip(self.norm_coeffs, self.minima)):
            invm = INVMS[invm_ind]
            print(f"        ---- INVARIANT MASS = {invm:.2f} ----")

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
                self.probs[invm_ind][ind] = np.array(list(probs.values()))
                if self.soln_bitstring is not None:
                    self.sym_probs[invm_ind][ind] = np.array(list(sym_probs.values()))
                    self.rank[invm_ind][ind] = rank
                    self.prob[invm_ind][ind] = prob
                    self.sym_rank[invm_ind][ind] = sym_rank
                    self.sym_prob[invm_ind][ind] = sym_prob
                self.costs[invm_ind][ind] = np.pad(costs, (0, self.steps - len(costs)))
                self.expvals[invm_ind][ind] = expval
                self.evals[invm_ind][ind] = evals
                self.min_bitstring[invm_ind][ind] = minimum[0]
                self.min_energy[invm_ind][ind] = minimum[1]
                for param_ind in range(len(params)):
                    self.params[param_ind][invm_ind][ind] = params[param_ind]
                if self.alg_str == "falqon":
                    self.depth_probs[invm_ind][ind] = self.alg.depth_probs

        # Save params based on actual names which is algorithm-specific
        match self.alg_str:
            case "qaoa" | "maqaoa":
                self.gammas = self.params[0]
                self.betas = self.params[1]
            case "xqaoa":
                self.gammas = self.params[0]
                self.betas = self.params[1]
                self.alphas = self.params[2]
            case "falqon":
                self.betas = self.params[0]
            case "varqite":
                self.thetas = self.params[0]

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
    alg: str,
    depth: int,
    hamiltonian: str,
    norm_scheme: str,
    device: str,
    steps: Optional[int] = None,
    ind_lo: Optional[int] = None,
    ind_hi: Optional[int] = None,
    evt_ind: Optional[int] = None,
    lambda_nume: Optional[tuple[str, str]] = None,
    lambda_denom: Optional[tuple[str, str]] = None,
    shots: Optional[int] = None,
    bitflip_prob: int = 0,
):
    """
    Wrapper function for the JobRunner class. Essentially will run an algorithm
    with all the given parameters for a specific number of jobs for every
    invariant mass bin. For example, if `ind_lo=10` and `ind_hi=25`. Then it
    will run 15 jobs for each of the bins starting with the 10th event that is
    between each bin, then the 11th, etc. If instead `evt_ind=34`, then it will
    run 1 job for event 34 only. Returns a tuple of data in the following order:
        # Input data
        invm_p4s: The 4-momenta
        invms: The invariant masses
        coeffs: The coefficient matrix for the VQA
        norm_coeffs: The normalized coefficient matrix as by `norm_scheme`
        expvals: The final expectation value for the circuit
        costs: The cost (i.e. `expval`) for each step of the circuit, the last
            value of this array == `expvals`
        evals: Number of circuit evaluations, can reach a max of `steps`
        probs: The probabilities of each
        "sym_probs": runner.sym_probs,
        "ranks": runner.rank,
        "rank_probs": runner.prob,
        "sym_ranks": runner.sym_rank,
        "sym_rank_probs": runner.sym_prob,
        "min_bitstrings": runner.min_bitstring,
        "min_energies": runner.min_energy,

    Parameters:
    etype - The event type, currently can be "ttbar", "tW" or "6jet".
    dtype - The data type, currently can be "parton" or "smeared".
    alg - The algorithm used for the data. Currently can be "qaoa", "maqaoa",
        "xqaoa", or "falqon".
    depth - The depth of the circuit ran
    hamiltonian - Which Hamiltonian used. Can be "H0", "H1", or "H2". If "H2",
        must define `lambda_nume` and `lambda_denom`.
    norm_scheme - The normalization scheme used for the coefficient matrix. Can
        be "max", "mean", or "sum".
    device - The Pennylane device to use, e.g. "default.qubit".
    steps (default None) - The number of classical optimization steps for the
        VQA to take. Must be defined if algorithm is not FALQON.
    ind_lo (default None) - The lower index of events to run the algorithm on,
        inclusive. Either this and `ind_hi` are defined (for multiple events)
        or `evt_ind` is defined (for a single event).
    ind_hi (default None) - The higher index of events to run the algorithm on,
        exclusive.
    evt_ind (default None) - If defined, then will run for the single event
        defined by this index. Both `ind_lo` and `ind_hi` must not be defined to
        use this.
    lambda_nume (default None) - The numerator of the lambda coefficient used in
        the H2 Hamiltonian.
    lambda_denom (default None) - The denominator of the lambda coefficient used
        in the H2 Hamiltonian.
    shots (default None)- The number of shots to do each circuit run. If None,
        use infinite shots, the ideal case.
    bitflip_prob (default 0) - The probability of a bitflip error for a gate
        execution. The device must be set to "default.mixed" if this is nonzero.
    """
    if steps is None and alg.lower() != "falqon":
        raise Exception(
            f"`steps` can only be `None` if algorithm is FALQON but it is {alg}."
        )
    # Make sure are indices are defined (for either one event or multiple)
    if evt_ind is not None and (ind_lo is not None or ind_hi is not None):
        raise Exception(
            f"{evt_ind = } is set, thus ind_lo and ind_hi must be None."
            f"Instead they are {ind_lo = } and {ind_hi = }"
        )
    if evt_ind is None and (ind_lo is None or ind_hi is None):
        raise Exception(
            "evt_ind = None, thus ind_lo and ind_hi must be defined."
            f"Instead they are {ind_lo = } and {ind_hi = }"
        )

    # Make sure order of lims is enforced if applicable
    if evt_ind is None and (not ind_lo < ind_hi):
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
            alg_kwargs = {
                "shots": shots,
                "steps": steps,
                "optimizer": DEFAULT_OPTIMIZER,
                "bitflip_prob": bitflip_prob,
            }
    # Run it!
    if evt_ind is not None:
        runner = JobRunner(
            etype=etype,
            dtype=dtype,
            evt_ind=evt_ind,
            alg=alg,
            depth=depth,
            hamiltonian=hamiltonian,
            norm_scheme=norm_scheme,
            device=device,
            alg_kwargs=alg_kwargs,
            lambda_kwargs={"nume": lambda_nume, "denom": lambda_denom},
        )
    else:
        runner = JobsRunner(
            etype=etype,
            dtype=dtype,
            ind_lo=ind_lo,
            ind_hi=ind_hi,
            alg=alg,
            depth=depth,
            hamiltonian=hamiltonian,
            norm_scheme=norm_scheme,
            device=device,
            alg_kwargs=alg_kwargs,
            lambda_kwargs={"nume": lambda_nume, "denom": lambda_denom},
        )

    runner.run()
    # Collect data
    return_dict = {
        "invm_p4s": runner.p4s,
        "invms": runner.invms,
        "coeffs": runner.coeffs,
        "norm_coeffs": runner.norm_coeffs,
        "expvals": runner.expvals,
        "costs": runner.costs,
        "evals": runner.evals,
        "probs": runner.probs,
        "sym_probs": runner.sym_probs,
        "ranks": runner.rank,
        "rank_probs": runner.prob,
        "sym_ranks": runner.sym_rank,
        "sym_rank_probs": runner.sym_prob,
        "min_bitstrings": runner.min_bitstring,
        "min_energies": runner.min_energy,
    }

    # Algorithm-specific data
    match alg.lower():
        case "qaoa" | "maqaoa":
            return_dict |= {"gammas": runner.gammas, "betas": runner.betas}
        case "xqaoa":
            return_dict |= {
                "gammas": runner.gammas,
                "betas": runner.betas,
                "alphas": runner.alphas,
            }
        case "falqon":
            return_dict |= {"betas": runner.betas, "depth_probs": runner.depth_probs}
        case "varqite":
            return_dict |= {"thetas": runner.thetas}

    return return_dict
