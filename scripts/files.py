import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .constants import (
    DATA_CHOICES,
    DEFAULT_BETA0,
    DEFAULT_DICT,
    DEFAULT_DT,
    DEFAULT_DTAU,
    EVENT_CHOICES,
    HAMILTONIAN_CHOICES,
    INVMS,
    LAMBDA_COMBOS,
    NORM_CHOICES,
    OUTPUT_DIR,
)
from .type_hints import DatumType

ParamType: TypeAlias = str | float | int


def _parse_extras(extras: str, kw: str, default: ParamType) -> str:
    """
    Parses the string `extras` for the keyword `kw` and returns the number that
    comes immediately after it. If not found, returns `default`.

    We are assuming that the default value is number-like.
    """
    # Round to 3 decimal points to be consistent
    return next(
        (f"{float(s.removeprefix(kw)):.3f}" for s in extras if s.startswith(kw)),
        f"{float(default):.3f}",
    )


def _check_for_error(name: str, val: ParamType, options: Sequence[ParamType]):
    """
    Error for when a parameter, `name` is not one of the required values found
    in `options` but is instead `val`.
    """
    if val not in options:
        raise ValueError(f"{name} is {val}. It must be one of the following: {options}")


def find_dir(
    alg: str,
    depth: int | str,
    hamiltonian: str,
    norm: str = "max",
    etype: str = "ttbar",
    dtype: str = "parton",
    lambda_nume: str | None = None,
    lambda_denom: str | None = None,
    **extras: str,
) -> Path:
    """
    For the given parameters, returns the directory path as a Path object.

    Parameters:
    alg - The algorithm used for the data. Currently can be "qaoa", "maqaoa",
        "xqaoa", "varqite", or "falqon".
    depth - The depth of the circuit ran
    hamiltonian - Which Hamiltonian used. Can be "H0", "H1", or "H2". If "H2",
        must define `lambda_nume` and `lambda_denom`.
    norm (default "max") - The normalization scheme used for the coefficient
        matrix. Can be "max", "mean", or "sum".
    etype (default "ttbar") - The event type, currently can be "ttbar",
        "tW" or "6jet".
    dtype (default "parton") - The data type, currently can be "parton"
        or "smeared".
    lambda_nume (default None) - The numerator of the lambda coefficient used in
        the H2 Hamiltonian.
    lambda_denom (default None) - The denominator of the lambda coefficient used
        in the H2 Hamiltonian.
    """
    lambda_nume = lambda_nume or "minJij"
    lambda_denom = lambda_denom or "maxPij"
    # Find the root directory for data
    ham_str = hamiltonian
    if hamiltonian == "H2":
        ham_str += f"-{lambda_nume}-{lambda_denom}"
    dir_name = f"{etype}_{dtype}_{depth}_{ham_str}_{norm}"
    # We add on the algorithm-specific parameters ONLY IF they aren't the defaul
    dir_name += "_".join(
        [""] + ["".join((k, v)) for k, v in extras.items() if DEFAULT_DICT[k] != float(v)]
    )
    return OUTPUT_DIR / alg / dir_name


def parse_data() -> NDArray[NDArray[str]]:
    """
    Searches the data directory defined by `OUTPUT_DIR` and finds the params of
    every directory.
    """
    alg_dirs = OUTPUT_DIR.iterdir()
    params = []
    # Loop through the directories of each algorithm
    for alg_dir in alg_dirs:
        alg = alg_dir.name

        param_dirs = alg_dir.iterdir()
        # Loop through each directory representing a choice of parameters
        for param_dir in param_dirs:
            split_name = param_dir.name.split("_")

            etype = split_name[0]
            dtype = split_name[1]
            depth = split_name[2]
            ham = split_name[3].split("-")
            norm = split_name[4]
            extras = split_name[5:]

            # Make sure all these values are valid
            _check_for_error("Event type", etype, EVENT_CHOICES)
            _check_for_error("Data type", dtype, DATA_CHOICES)
            _check_for_error("Hamiltonian", ham[0], HAMILTONIAN_CHOICES)
            if len(ham) > 1:
                _check_for_error("Lambda denominator", ham[1], LAMBDA_COMBOS)
                _check_for_error("Lambda numerator", ham[2], LAMBDA_COMBOS)
            _check_for_error("Normalization", norm, NORM_CHOICES)

            # Get algorithm-specific info if it's there
            if alg == "falqon":
                beta0 = _parse_extras(extras, "beta0", DEFAULT_BETA0)
                dt = _parse_extras(extras, "dt", DEFAULT_DT)
                extras = [beta0, dt]
            elif alg == "varqite":
                dtau = _parse_extras(extras, "dtau", DEFAULT_DTAU)
                extras = [dtau, ""]
            else:
                extras = ["", ""]

            lambda_type = ["", ""] if len(ham) == 1 else ham[1:]
            datum = [alg, depth, ham[0], norm, etype, dtype, *lambda_type, *extras]
            params.append(datum)

    return np.array(params, dtype=str)


def verify_data(
    dir_path: Path,
) -> bool:
    """
    Makes basic checks on the saved data. If this function returns True, then
    the data is available to be analyzed. It checks
        2) That none of the events are duplicated
        3) That there are no events missing
    It doesn't check if there are 12,000 / 6 = 2,000 total events found.

    Parameters:
    dir_path: Path to the directory with the data. That is, it will have
        directories labeled by the invariant mass bin lower values.
    """
    # Loop over the invariant mass subdirectories
    for invm in INVMS[:-1]:
        invm_dir = dir_path / f"{invm:.2f}"
        if not invm_dir.exists():
            print(f"{invm:.2f} -- Invariant mass file {invm_dir} does not exist.")
            return False

        ind_pairs = []
        # Iterate over each .npz file
        for fpath in invm_dir.iterdir():
            if fpath.name.startswith("."):
                print(
                    f"{fpath.name} exists in {fpath.parent}... This implies that files are currently being copied over."
                )
                return False
            # Get the event indices in this specific file
            low_ind, hi_ind = re.findall(r"^eff_(\d+)-(\d+)\.npz$", fpath.name)[0]
            low_ind, hi_ind = int(low_ind), int(hi_ind)

            # Save those indicies
            ind_pairs.append([low_ind, hi_ind])

        # Find maximum index
        try:
            max_ind = np.max(ind_pairs)
        except ValueError:
            print(f"{invm_dir} is empty.")
            return False

        # Make sure we cover every event and every event only once
        all_inds = np.arange(0, max_ind)
        for pair in ind_pairs:
            inds = np.arange(*pair)
            # Make sure we don't have duplicates of events
            if not np.all(np.isin(inds, all_inds)):
                print(f"Path: {invm_dir}\nFound duplicates at {pair}.")
                return False

            all_inds = all_inds[~np.isin(all_inds, inds)]

        if len(all_inds):
            print(f"Path: {invm_dir}\nMissing {len(all_inds)} events: {all_inds}.")
            return False

    # All checks passed
    return True


def load_data(
    dir_path: Path,
) -> DatumType:
    """
    Return a dictionary with the data for a given run of events. The returned
    data structure is a dictionary with keys representing the invariant mass
    bin (specifically the lower bound of the bin) as defined by the `INVMS`
    list in `constants.py`. The values are dictionaries representing the keys
    and values of the loaded .npz files. Since not all 2000 events are
    necessarily saved to the same file, this loads all the files and appends
    them together.

    Parameters:
    dir_path: Path to the directory with the data. That is, it will have
        directories labeled by the invariant mass bin lower values.
    """
    data = {}
    # Iterate over the invariant masses
    for invm in INVMS[:-1]:
        invm_dir = dir_path / f"{invm:.2f}"

        invm_data = defaultdict(list)
        # Iterate over the different files
        fpaths = list(invm_dir.iterdir())
        # Files may not be in order, so find next correct file by index
        lo_ind = 0
        while len(fpaths):
            for fpath in fpaths:
                hi_ind = re.findall(rf"^eff_0*{lo_ind}-(\d+)\.npz", fpath.name)

                if len(hi_ind) == 1:
                    fdata = np.load(fpath)
                    # Iterate over the keys in each file
                    for key in fdata.keys():
                        invm_data[key].append(fdata[key])

                    lo_ind = int(hi_ind[0])
                    fpaths.remove(fpath)
                    break

        # Flatten all values into single per-event array
        for key, val in invm_data.items():
            invm_data[key] = np.concatenate(val, axis=0)

        data[invm] = dict(invm_data)
    return data


def _extract_falqon_data(dfile: Path, depths: NDArray[int], invm) -> list[DatumType]:
    """
    Properly extract the FALQON data for specific depths.
    """
    file_data = np.load(dfile)
    data = []

    invm_p4s = file_data["invm_p4s"]
    invms = file_data["invms"]
    coeffs = file_data["coeffs"]
    norm_coeffs = file_data["norm_coeffs"]
    min_bitstrings = file_data["min_bitstrings"]
    min_energies = file_data["min_energies"]
    costs = file_data["costs"]
    betas = file_data["betas"]
    inds = file_data["depth_inds"]

    # Indices for the depths we want
    depth_inds = np.where(np.isin(inds, depths - 1))[0]
    if len(depth_inds) != len(depths):
        missing = depths[~np.isin(depths - 1, inds[np.isin(inds, depths - 1)])]
        raise ValueError(f"Missings depths: {missing}")
    # Get probabilities as specific depth
    depth_probs = np.swapaxes(file_data["depth_probs"][:, depth_inds, :], 0, 1)
    # Get expectation values at specific depth
    depth_expvals = np.swapaxes(file_data["costs"][:, depth_inds], 0, 1)
    for depth_ind, probs, expvals in zip(depth_inds, depth_probs, depth_expvals):
        datum = {}

        # Per event data, isn't dependent on depth
        datum["invm_p4s"] = invm_p4s
        datum["invms"] = invms
        datum["coeffs"] = coeffs
        datum["norm_coeffs"] = norm_coeffs
        datum["min_bitstrings"] = min_bitstrings
        datum["min_energies"] = min_energies

        # Depth-dependent data
        datum["costs"] = costs[:, :depth_ind]
        datum["expvals"] = expvals
        datum["probs"] = probs
        datum["betas"] = betas[:, :depth_ind]

        data.append(datum)
    return data


def load_falqon_depths(falqon_path: Path, depths: Sequence[int]) -> list[DatumType]:
    """
    FALQON data is stored for EVERY depth (since each depth is ran iteratively).
    Thus, this method will extract the data for only the depths specified.

    Parameters:
    falqon_path: Path to the directory with the data. That is, it will have
        directories labeled by the invariant mass bin lower values.
    depths: Depths to extract data for.
    """
    depths = np.array(depths).astype(int)
    num_p = len(depths)

    falqon_data = [{} for _ in range(num_p)]
    for invm in INVMS[:-1]:
        path = falqon_path / f"{invm:.2f}"

        invm_data = [defaultdict(list) for _ in range(num_p)]
        for dfile in sorted(path.iterdir()):
            # This is where the FALQON-specific parsing occurs
            partial_data = _extract_falqon_data(dfile, depths, invm)

            for invm_datum, partial_datum in zip(invm_data, partial_data):
                for key, val in partial_datum.items():
                    invm_datum[key].append(val)

        # Add together and concatenate individual files
        for falqon_datum, invm_datum in zip(falqon_data, invm_data):
            falqon_datum[invm] = {k: np.concatenate(v) for k, v in invm_datum.items()}
        del invm_data

    return falqon_data
