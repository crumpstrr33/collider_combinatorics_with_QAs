import re
from collections import defaultdict
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from .constants import INVMS, OUTPUT_DIR
from .data import split_data
from .events import get_data


def parse_data() -> NDArray[NDArray[str]]:
    """
    Searches the data directory defined by `OUTPUT_DIR` and finds the params of
    every directory.
    """
    # Regex strign for extracting info from directory name
    ch = "[a-zA-Z0-9]"
    re_dir = rf"^({ch}+)_({ch}+)_(\d+)_({ch}+)(?:-({ch}+)-({ch}+))?_({ch}+)_(\d+)$"

    alg_dirs = OUTPUT_DIR.iterdir()
    params = []
    for alg_dir in alg_dirs:
        alg = alg_dir.name

        param_dirs = alg_dir.iterdir()
        for param_dir in param_dirs:
            (
                etype,
                dtype,
                depth,
                hamiltonian,
                lambda_nume,
                lambda_denom,
                norm,
                num_evts,
            ) = re.findall(re_dir, param_dir.name)[0]

            if hamiltonian == "H2" and (lambda_nume == "" or lambda_denom == ""):
                raise Exception("Hamiltonian is H2, both lambda kwargs must be defined")

            params.append(
                [
                    alg,
                    depth,
                    hamiltonian,
                    norm,
                    etype,
                    dtype,
                    lambda_nume,
                    lambda_denom,
                    num_evts,
                ]
            )

    return np.array(params, dtype=str)


def verify_data(
    alg: str,
    depth: Union[int, str],
    hamiltonian: str,
    norm: str,
    etype: str = "ttbar",
    dtype: str = "parton",
    lambda_nume: Optional[str] = None,
    lambda_denom: Optional[str] = None,
    num_evts: Optional[Union[int, str]] = None,
) -> bool:
    """
    Makes basic checks on the saved data. If this function returns True, then
    the data is available to be analyzed. It checks
        1) That is does have the maximum number of events as per the data file
        2) That none of the events are duplicated
        3) That there are no events missing

    Parameters:
    alg - The algorithm used for the data. Currently can be "qaoa", "maqaoa",
        "xqaoa", or "falqon".
    depth - The depth of the circuit ran
    hamiltonian - Which Hamiltonian used. Can be "H0", "H1", or "H2". If "H2",
        must define `lambda_nume` and `lambda_denom`.
    norm - The normalization scheme used for the coefficient matrix. Can be
        "max", "mean", or "sum".
    etype (default "ttbar") - The event type, currently can be "ttbar",
        "tW" or "6jet".
    dtype (default "parton") - The data type, currently can be "parton"
        or "smeared".
    lambda_nume (default None) - The numerator of the lambda coefficient used in
        the H2 Hamiltonian.
    lambda_denom (default None) - The denominator of the lambda coefficient used
        in the H2 Hamiltonian.
    num_evts (default None) - Total number of events per invariant mass bin.
    """
    # Find the total number of events per invariant mass bin
    if num_evts is None:
        num_evts = split_data(
            evts=get_data(etype=etype, dtype=dtype, print_num_evts=False)[0]
        )[0].shape[1]
    num_evts = int(num_evts)
    depth = int(depth)

    # Find the root directory for data
    ham_str = hamiltonian
    if hamiltonian == "H2":
        ham_str += f"-{lambda_nume}-{lambda_denom}"
    root_dir = OUTPUT_DIR / alg / f"{etype}_{dtype}_{depth}_{ham_str}_{norm}_{num_evts}"

    # Loop over the invariant mass subdirectories
    for invm in INVMS[:-1]:
        invm_dir = root_dir / f"{invm:.2f}"
        if not invm_dir.exists():
            print(f"{invm:.2f} -- Invariant mass file {invm_dir} does not exist.")
            return False

        ind_pairs = []
        # Iterate over each .npz file
        for fpath in invm_dir.iterdir():
            # Get the event indices in this specific file
            low_ind, hi_ind = re.findall(r"^eff_(\d+)-(\d+)\.npz$", fpath.name)[0]
            low_ind, hi_ind = int(low_ind), int(hi_ind)

            # Save those indicies
            ind_pairs.append([low_ind, hi_ind])

        # Find maximum index
        max_ind = np.max(ind_pairs)
        # Check to make sure do we do indeed have maximum number of events
        if max_ind != num_evts:
            print(
                f"{invm:.2f} -- "
                "Do not have maximum number of events. Max number from event "
                + f"file is {num_evts} but found {max_ind} from data files."
            )
            return False
        # Make sure we cover every event and every event only once
        all_inds = np.arange(0, max_ind)
        for pair in ind_pairs:
            inds = np.arange(*pair)
            # Make sure we don't have duplicates of events
            if not np.all(np.in1d(inds, all_inds)):
                p_str = f"{etype} | {dtype} | {alg} | p={depth} | {norm} | {hamiltonian} "
                if lambda_nume is not None:
                    p_str += f" {lambda_nume} / {lambda_denom}"
                print(
                    f"{p_str}\n"
                    f"{invm:.2f} -- "
                    f"Found duplicate indices looking at range {pair}"
                )
                return False

            all_inds = all_inds[~np.isin(all_inds, inds)]

        if len(all_inds):
            print(f"{invm:.2f} -- Missing {len(all_inds)} events: {all_inds}")
            return False
    return True


def load_data(
    alg: str,
    depth: Union[int, str],
    hamiltonian: str,
    norm: str,
    etype: str = "ttbar",
    dtype: str = "parton",
    lambda_nume: Optional[str] = None,
    lambda_denom: Optional[str] = None,
    num_evts: Optional[Union[int, str]] = None,
) -> dict[float, dict[str, NDArray[np.float64]]]:
    """
    Return a dictionary with the data for a given run of events. The returned
    data structure is a dictionary with keys representing the invariant mass
    bin (specifically the lower bound of the bin) as defined by the `INVMS`
    list in `constants.py`. The values are dictionaries representing the keys
    and values of the loaded .npz files. Since not all 2000 events are
    necessarily saved to the same file, this loads all the files and appends
    them together.

    Parameters:
    alg - The algorithm used for the data. Currently can be "qaoa", "maqaoa",
        "xqaoa", or "falqon".
    depth - The depth of the circuit ran
    hamiltonian - Which Hamiltonian used. Can be "H0", "H1", or "H2". If "H2",
        must define `lambda_nume` and `lambda_denom`.
    norm - The normalization scheme used for the coefficient matrix. Can be
        "max", "mean", or "sum".
    etype (default "ttbar") - The event type, currently can be "ttbar",
        "tW" or "6jet".
    dtype (default "parton") - The data type, currently can be "parton"
        or "smeared".
    lambda_nume (default None) - The numerator of the lambda coefficient used in
        the H2 Hamiltonian.
    lambda_denom (default None) - The denominator of the lambda coefficient used
        in the H2 Hamiltonian.
    """
    # Find the root directory for data
    ham_str = hamiltonian
    if hamiltonian == "H2":
        ham_str += f"-{lambda_nume}-{lambda_denom}"
    root_dir = OUTPUT_DIR / alg / f"{etype}_{dtype}_{depth}_{ham_str}_{norm}_{num_evts}"

    data = {}
    # Iterate over the invariant masses
    for invm in INVMS[:-1]:
        invm_dir = root_dir / f"{invm:.2f}"

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
