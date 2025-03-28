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

import re
from argparse import ArgumentParser, BooleanOptionalAction
from collections import Counter
from datetime import datetime as dt
from typing import Sequence

import numpy as np
from constants import (
    ALG_CHOICES,
    DATA_CHOICES,
    DEFAULT_BETA0,
    DEFAULT_DT,
    DEFAULT_OPTIMIZER,
    DEFAULT_STEPS,
    DEFAULT_STEPSIZE,
    EVENT_CHOICES,
    EVT_DIR,
    IND_DIR,
    NOISY_DIR,
    OPTIMIZERS,
    OUTPUT_DIR,
    QUADCOEFF_CHOICES,
    SYM_TRUE_BS_DICT,
)
from my_favorite_things import save
from pennylane_algs import FALQON, HYBRID_MAQAOA, MAQAOA, NO_WEIGHT_MAQAOA, QAOA, XQAOA
from qc_utilities import (
    format_m4s,
    get_coeffs,
    get_lambdas,
    get_minimum_energies,
)


class Efficiency:
    def __init__(
        self,
        etype: str,
        dtype: str,
        invm_lo: float,
        invm_hi: float,
        chosen_inds: Sequence[int],
        alg: str,
        quadcoeff: str,
        alg_kwargs: dict[str, ...],
        alg_opt_kwargs: dict[str, ...],
    ):
        self.etype = etype
        self.dtype = dtype
        self.invm_lo = invm_lo
        self.invm_hi = invm_hi
        self.chosen_inds = chosen_inds
        self.alg_str = alg.lower()
        self.quadcoeff_str = quadcoeff
        self.alg_kwargs = alg_kwargs
        self.alg_opt_kwargs = alg_opt_kwargs

        # Get data for running alg
        self.get_input_data()

        # Total number of events
        self.N = len(chosen_inds)
        # Bit string that solves combinatorial problem
        self.soln_bs = SYM_TRUE_BS_DICT[self.etype]

    @staticmethod
    def find_placement(bit_str: str, probs: dict[str, float]) -> int:
        """
        Finds the placement of a bit string based on it's probability, i.e. it returns 1
        if the bit string is the most likely.

        Parameters:
        bit_str - Bit string whose placement to find
        probs - Dictionary of probabilities for all the probabilities

        returns (
            placement of bit string in terms of probability
        )
        """
        if bit_str is None:
            return None
        return [bsc[0] for bsc in Counter(probs).most_common()].index(bit_str) + 1

    @staticmethod
    def swap(bit_str: str) -> str:
        """
        Swaps the 0's and 1's in a bit string, i.e. if `bit_str=111000` then it returns
        `000111`.

        Parameters:
        bit_str - Bit string to swap
        """
        return bit_str.replace("0", "2").replace("1", "0").replace("2", "1")

    def sym_swap(self, bit_str: str) -> str:
        """
        Same as self.swap but does nothing if the bit string starts with a 0.

        Parameters:
        bit_str - Bit string to swap
        """
        if bit_str is None:
            return None
        return bit_str if bit_str.startswith("0") else self.swap(bit_str)

    def get_input_data(self):
        """
        Get invariants masses and Jijs for either dtype='smeared' or 'parton' or 'test'.

        Choose coefficient of the quadratic term (by default is just Jij)
        """
        # Get all 4-momenta
        fpath = EVT_DIR / f"{self.etype}_{self.dtype}.npy"
        self.m4s = np.load(fpath)

        # Pick the correct events
        self.m4s = self.m4s[self.chosen_inds]
        # Reshape data properly
        self.num_fsp, self.m4s, self.invms = format_m4s(self.m4s, return_extra=True)
        # Can be just Jij or Jij + 2λPij, etc
        if self.quadcoeff_str == "QA":
            self.lambdas = get_lambdas(m4s=self.m4s, ltype="QA")
        else:
            self.lambdas = np.ones(len(self.m4s))
        self.quadcoeffs = get_coeffs(
            m4s=self.m4s, htype=self.quadcoeff_str, lambdas=self.lambdas
        )

    def get_output_data(
        self, quadcoeff: Sequence[Sequence[float]]
    ) -> (dict[str, float], Sequence[float], Sequence[Sequence[float]], float, int):
        """
        Runs the algorithm.
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
            case "nw_maqaoa":
                Alg = NO_WEIGHT_MAQAOA
            case "hybrid":
                Alg = HYBRID_MAQAOA

        self.alg = Alg(Jij=quadcoeff, **self.alg_kwargs, **self.alg_opt_kwargs)

        match self.alg_str:
            case "falqon":
                self.alg.run(print_it=False)
            case _:
                self.alg.optimize(print_it=False)

        params = self.alg.params
        costs = self.alg.costs
        expval = self.alg.costs[-1]
        probs = self.alg.get_probs(as_dict=True)
        evals = self.alg.evals

        return probs, costs, params, expval, evals

    def run(self):
        """
        Runs algorithm for each event and saves info in an array of dictionaries.
        """
        # Bit strings that minimize the Hamiltonian found by brute force
        _, _, bf_bss, _ = get_minimum_energies(
            m4s=self.m4s, htype=self.quadcoeff_str, lambdas=self.lambdas
        )

        self.datas = []
        for ind, (quadcoeff, m4, invm) in enumerate(
            zip(self.quadcoeffs, self.m4s, self.invms)
        ):
            # Event number
            evt = chosen_inds[ind]

            print(
                f"Index: {ind + 1:>{len(str(self.N))}} / {self.N} (p = {depth}) | "
                + f"norm inv mass = {invm:.2f}"
                + f" | Event: {evt:>{len(str(max(chosen_inds)))}}",
                end=" ",
                flush=True,
            )

            # Run the algorithm
            t0 = dt.now()
            norm_quadcoeff = quadcoeff / quadcoeff.max()
            probs, costs, params, expval, evals = self.get_output_data(
                quadcoeff=norm_quadcoeff
            )
            t1 = dt.now()

            print(
                f"| {self.alg_str.upper()} time: {(t1 - t0).total_seconds():.2f} "
                + f"seconds | Steps: {evals}",
                flush=True,
            )

            # Probabilities summing bit string and it's symmetric bit string
            # Only keeping bit strings that start with 0
            sym_probs = {
                bs: p + probs[self.swap(bs)]
                for bs, p in probs.items()
                if bs.startswith("0")
            }

            # Given the energy for each bit string (in terms of the given Hamiltonian),
            # this is the placement (1st, 2nd, 3rd, etc.) of the bit string that
            # solves the combinatorial problem
            soln_placement = self.find_placement(bit_str=self.soln_bs, probs=probs)
            # And this is its placement in for the symmetric probabilities
            sym_soln_placement = self.find_placement(
                bit_str=self.sym_swap(self.soln_bs), probs=sym_probs
            )

            data = {
                # Normalized invariant mass
                "invm": invm,
                # Coefficient of quadratic term of Hamiltonian in terms of spin
                "quadcoeff": quadcoeff,
                # Coefficient of quadratic term of Hamiltonian normalized
                "norm_quadcoeff": norm_quadcoeff,
                # 4 momentum of event
                "m4": m4,
                # Index of event
                "evt": evt,
                # Dict of probabilities for bit strings
                "probs": probs,
                # Dict of probabilities with assuming symmetry beteween 0 <-> 1
                "sym_probs": sym_probs,
                # Cost function per step
                "costs": costs,
                # Optimized parameters
                "params": params,
                # Expectation value for final algorithm circuit
                "expval": expval,
                # Number of evaluations
                "evals": evals,
                # Bruteforce minimum bit string
                "bruteforce_bs": bf_bss[ind],
                # Symmetric bruteforce minimum bit string
                "sym_bruteforce_bs": self.sym_swap(bf_bss[ind]),
                # Bit string that solves combinatorial problem
                "solution_bs": self.soln_bs,
                # Symmetric bit string that solves combinatorial problem
                "sym_solution_bs": self.sym_swap(self.soln_bs),
                # Placement of correct bit string in terms of energy
                "solution_placement": soln_placement,
                # Symmetric placement of correct bit string in terms of energy
                "sym_solution_placement": sym_soln_placement,
            }
            if self.alg_str == "falqon":
                # Save "depth_probs", the probability dictionary per depth and the same
                # for it's symmetric cousin
                depth_probs, depth_probs2 = [], []
                depth_bss, depth_bss2 = [], []
                sym_depth_probs, sym_depth_probs2 = [], []
                sym_depth_bss, sym_depth_bss2 = [], []
                for depth_prob in self.alg.depth_probs:
                    highest_probs = Counter(depth_prob).most_common(2)

                    sym_depth_prob = {
                        bs: p + depth_prob[self.swap(bs)]
                        for bs, p in depth_prob.items()
                        if bs.startswith("0")
                    }
                    sym_highest_probs = Counter(sym_depth_prob).most_common(2)

                    depth_bss.append(str(highest_probs[0][0]))
                    depth_probs.append(float(highest_probs[0][1]))
                    depth_bss2.append(str(highest_probs[1][0]))
                    depth_probs2.append(float(highest_probs[1][1]))
                    sym_depth_bss.append(str(sym_highest_probs[0][0]))
                    sym_depth_probs.append(float(sym_highest_probs[0][1]))
                    sym_depth_bss2.append(str(sym_highest_probs[1][0]))
                    sym_depth_probs2.append(float(sym_highest_probs[1][1]))

                data |= {
                    "depth_bss": depth_bss,
                    "depth_probs": depth_probs,
                    "depth_bss2": depth_bss2,
                    "depth_probs2": depth_probs2,
                    "sym_depth_bss": sym_depth_bss,
                    "sym_depth_probs": sym_depth_probs,
                    "sym_depth_bss2": sym_depth_bss2,
                    "sym_depth_probs2": sym_depth_probs2,
                }
            self.datas.append(data)
        self.datas = np.array(self.datas)


if __name__ == "__main__":
    parser = ArgumentParser()
    # Algorithm to use, e.g. QAOA, FALQON, etc
    parser.add_argument(
        "--algorithm", "-a", required=True, type=str.lower, choices=ALG_CHOICES
    )
    # Type of event to run on, e.g. ttbar or tW
    parser.add_argument("--event", "-e", required=True, type=str, choices=EVENT_CHOICES)
    # Data to run on, e.g. parton or smeared
    parser.add_argument(
        "--dtype", "-D", required=True, type=str.lower, choices=DATA_CHOICES
    )
    # Coefficient of the quadratic term, e.g. og for Jij or qa for Jij + 2λPij
    parser.add_argument(
        "--quadcoeff", "-c", required=True, type=str, choices=QUADCOEFF_CHOICES
    )
    # Number of shots, defaults to None == infinite
    parser.add_argument("--shots", "-s", required=False)
    # The lower and upper limit of events to run, must match an index file
    parser.add_argument("--indlims", "-i", required=True, type=int, nargs=2)
    # The lower and upper limit of the allowed invariant mass
    parser.add_argument("--invmlims", "-m", required=True, type=float, nargs=2)
    # Only required if glob finds more than one index file
    parser.add_argument("--uid", "-u", type=int)
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
    if not (args.invmlims[0] < args.invmlims[1]):
        raise Exception(
            "The first value for `invmlims` must be the smaller of the two."
            + f" It is {args.invmlims}"
        )

    # Gather parameters
    etype = args.event
    dtype = args.dtype
    shots = args.shots
    invmlims = args.invmlims
    depth = args.depth
    indlims = args.indlims
    uid = args.uid
    alg = args.algorithm
    quadcoeff_type = args.quadcoeff

    # Get directory holding the relevant index file(s)
    uid_str = uid or "*"
    ind_dir = list(
        IND_DIR.glob(
            f"inds*_{etype}_{dtype}_{uid_str}x*_{invmlims[0]:.2f}to{invmlims[1]:.2f}"
        )
    )
    # Find number of digits (e.g. padding) in index limits
    num_digs = len(
        re.findall(r"_(\d+)to\d+\.npz$", list(ind_dir[0].glob("*"))[0].name)[0]
    )
    # Find file matching the arguments
    ind_file = list(
        ind_dir[0].glob(f"*{indlims[0]:0>{num_digs}}to*{indlims[1]:0>{num_digs}}*")
    )

    # Make sure we didn't get more than one (if so, might need to specify UID)
    if len(ind_dir) != 1:
        raise Exception(f"Found {len(ind_dir)} matching directories:\n{ind_dir}")
    if len(ind_file) != 1:
        raise Exception(f"Found {len(ind_file)} matching files:\n{ind_file}")

    # Get the indices to pass to function
    chosen_inds = np.load(f"{ind_file[0]}")["inds"]
    # Grab UID from file name
    uid = re.findall(r"_(\d{3})x\d{3}_", str(ind_file[0]))[0]
    # Formatted string for index range
    inds = f"{indlims[0]:0>{num_digs}}to{indlims[1]:0>{num_digs}}"
    # Formatted string for invariant mass range
    invm_range_str = f"{invmlims[0]:.2f}to{invmlims[1]:.2f}"
    # Create save directory and file names
    save_name = f"eff_{etype}_{dtype}_{alg}_{quadcoeff_type}_{uid}_{inds}_p{depth}"
    save_dir = f"eff_{etype}_{dtype}_{alg}_{quadcoeff_type}_p{depth}"
    if shots is not None:
        save_dir += f"_shots{shots}"

    # Only add initial beta and/or dt if they aren't the defaults
    if args.algorithm == "falqon":
        save_name += f"_b{args.initbeta}"
        save_name += f"_dt{args.dt}"
        save_dir += f"_b{args.initbeta}"
        save_dir += f"_dt{args.dt}"
    # Add on invariant mass limits to save file name
    save_name += f"_{invm_range_str}"

    # Make sure we aren't running something we already ran
    if (OUTPUT_DIR / save_dir / save_name).exists():
        raise Exception(f"{OUTPUT_DIR / save_dir / save_name} already exists...")

    # Don't run anything if this is a dryrun
    if args.dryrun:
        datas = {}
    else:
        # Prepare arguments depending on algorithm
        match alg.lower():
            case "falqon":
                alg_opt_kwargs = {"dt": args.dt, "init_beta": args.initbeta}
                alg_kwargs = {"depth": depth, "shots": shots}
            case _:
                alg_opt_kwargs = {
                    "optimizer": args.optimizer,
                    "opt_kwargs": {"stepsize": args.stepsize},
                }
                alg_kwargs = {"depth": depth, "steps": args.steps, "shots": shots}

        # i was too lazy to do this proper for noisy runs, so i did this :)
        if False:
            bitflip_prob = 0.001
            prob_str = "0.1"
            save_dir = (
                NOISY_DIR
                / f"eff_{prob_str}_{etype}_{dtype}_{alg}_{quadcoeff_type}_p{depth}"
            )
            alg_kwargs |= {"bitflip_prob": bitflip_prob, "device": "default.mixed"}
        else:
            bitflip_prob = 0

        # Run events
        efficiency = Efficiency(
            etype=etype,
            dtype=dtype,
            invm_lo=invmlims[0],
            invm_hi=invmlims[1],
            chosen_inds=chosen_inds,
            alg=alg,
            quadcoeff=quadcoeff_type,
            alg_kwargs=alg_kwargs,
            alg_opt_kwargs=alg_opt_kwargs,
        )
        efficiency.run()

    # Collect event-nonspecific data
    metadata = {
        "lims": invmlims,
        "N_range": indlims,
        "uid": uid,
        "alg_type": alg,
        "shots": shots,
        "depth": depth,
        "etype": etype,
        "dtype": dtype,
        "quadcoeff": quadcoeff_type,
        "bitflip_prob": bitflip_prob,
    }
    if alg == "falqon":
        metadata |= {"dt": args.dt, "init_beta": args.initbeta}
    else:
        metadata |= {
            "steps": args.steps,
            "stepsize": args.stepsize,
            "optimizer": args.optimizer,
        }

    # Save everything
    save(
        name=save_name,
        savedir=OUTPUT_DIR / save_dir,
        stype="pkl",
        absolute=True,
        dryrun=args.dryrun,
        data=efficiency.datas,
        metadata=metadata,
    )
