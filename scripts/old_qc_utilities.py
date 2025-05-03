"""
File with utility functions used throughout
"""

import re
import warnings
from itertools import product
from math import sqrt
from pathlib import Path
from typing import Optional, Sequence, TypeVar

import numpy as np
from numba import njit
from numpy.typing import NDArray
from typing_extensions import Annotated

try:
    from constants import (
        IND_DIR,
        LAMBDA_CHOICES,
        MASS_NORM_DICT,
        METRIC,
        NOISY_DIR,
        OUTPUT_DIR,
    )
except ImportError:
    from scripts.constants import (
        IND_DIR,
        LAMBDA_CHOICES,
        MASS_NORM_DICT,
        METRIC,
        NOISY_DIR,
        OUTPUT_DIR,
    )

warnings.filterwarnings("ignore")

# Types
num_fsp = TypeVar("numFSP", bound=int)
p4s_type = Annotated[NDArray[np.float64], ("shape", (..., num_fsp, 4))]
p4_type = Annotated[NDArray[np.float64], ("shape", (num_fsp, 4))]
Jijs_type = Annotated[NDArray[np.float64], ("shape", (..., num_fsp, num_fsp))]
Pijs_type = Annotated[NDArray[np.float64], ("shape", (..., num_fsp, num_fsp))]
bs_type = Annotated[NDArray[bool], ("shape", (..., 2, num_fsp))]


def Pij_sum_iter(num_fsp: int) -> product:
    """
    Pij is summed over i and j from 1 to `num_fsp`. This returns an iterable that goes
    over the entire sum.
    """
    return product(range(num_fsp), repeat=2)


def Jij_sum_iter(num_fsp: int) -> product:
    """
    Pij is summed over i, j, k and ell from 1 to `num_fsp`. This returns an iterable
    that goes over the entire sum.
    """
    return product(range(num_fsp), repeat=4)


def bit_string_bool_combinations(num_fsp):
    """
    Iterable of a tuple of bools representing every possible bit string and their
    inverse.
    """
    all_bs_bools = np.array(
        [[0] + list(x) for x in product([0, 1], repeat=num_fsp - 1)]
    ).astype(bool)
    all_bs_bools = np.array([(x, np.invert(x)) for x in all_bs_bools])

    return all_bs_bools


def bit_string_str_combinations(num_fsp):
    """
    Returns single list (not tuple) of strings of all possible combinations of
    1's and 0's.
    """
    bs_combs = bit_string_bool_combinations(num_fsp)
    bs_combs = [
        "".join(bs)
        for bs in bs_combs.astype(int).astype(str).reshape(-1, num_fsp)
    ]
    return np.array(bs_combs)


def _mass_norm(p4: p4_type, etype: Optional[str] = None) -> float:
    """
    Gets the proper mass normalization. If `etype` is known, will just use that to
    pull from `MASS_NORM_DICT`. Otherwise, find number of FSP from `p4` (assuming `p4`)
    is in shape (num_fsp, 4).
    """
    if etype is None:
        # Assume 6 particles means ttbar and 5 mean tW. If ambiguous, must pass `etype`
        etype = {12: "4top", 6: "ttbar", 5: "tW"}[p4.shape[0]]
    return MASS_NORM_DICT[etype]


def format_p4s(
    p4s: p4s_type, etype: Optional[str] = None, return_extra: bool = False
):
    """
    Assuming the 4-momentum array is of shape:
        (num_evts, num_fsp, 4) or (num_evts, 4 * num_fsp + is_invm)
    where
        `num_evts` is the number of total events,
        `num_fsp` is the number of final state particles
        `is_invm` is 1 if the invariant mass is included, otherwise it's 0
        4 are the components of the 4-momentum.

    Return the number of final state particles, the 4-momentum in the shape:
        (num_evts, num_fsp, 4),
    and the invariant mass normalized by 2*mt
    """
    p4_shape = p4s.shape[1:]

    # Numpy shape: (num_evts, num_fsp, 4)
    if len(p4_shape) == 2:
        if p4_shape[1] != 4:
            raise Exception(f"Last dimension should be 4, not {p4_shape[1]}.")

        num_fsp = p4_shape[0]
        # Already in the correct form
        formatted_p4s = p4s
        mass_norm = _mass_norm(formatted_p4s[0], etype)

        invms = np.empty(len(p4s))
        # Iterate through for all the invariant masses
        for ind, p4 in enumerate(p4s):
            summed_p4 = np.sum(p4, axis=0)
            mass = sqrt(np.dot(summed_p4, METRIC * summed_p4))
            invms[ind] = mass / mass_norm
    # Numpy shape: (num_evts, 4 * num_fsp + is_invm)
    elif len(p4_shape) == 1:
        # There is one extra entry for invariant mass
        if (p4_shape[0] - 1) % 4 != 0:
            raise Exception(
                f"2nd dimension is {p4_shape[0]} which isn't one more than a number divisble by 4."
            )

        num_fsp = p4_shape[0] // 4
        # Remove last entry since it's the invariant mass
        formatted_p4s = p4s[:, :-1].reshape(-1, num_fsp, 4)
        mass_norm = _mass_norm(formatted_p4s[0], etype)

        invms = p4s[:, -1] / mass_norm

    if return_extra:
        return num_fsp, formatted_p4s, invms
    return formatted_p4s


def get_Jijs_Pijs(p4s: p4s_type) -> (Jijs_type, Pijs_type):
    """
    Wrapper to jitted function to find the Jij and Pij terms for events.

    Parameters:
    p4s - MxNx4 array of 4-momentum for each particle where M is the number of events,
        and N is the number of final state particles per event

    Returns (
        NxN Jij weight matrix,
        NxN Pij momentum matrix
    )
    """
    N = p4s.shape[0]
    num_fsp = p4s.shape[1]
    Jijs, Pijs = _get_Jijs_Pijs(
        p4s=p4s,
        N=N,
        num_fsp=num_fsp,
        Pij_iter=list(Pij_sum_iter(num_fsp=num_fsp)),
        Jij_iter=list(Jij_sum_iter(num_fsp=num_fsp)),
    )

    return Jijs, Pijs


@njit
def _get_Jijs_Pijs(
    p4s: p4s_type,
    N: int,
    num_fsp: int,
    Pij_iter: Sequence[tuple[int]],
    Jij_iter: Sequence[tuple[int]],
) -> (Jijs_type, Pijs_type):
    """
    Jitted function to calculate Jij and Pij for events
    """
    # Initialize
    Jijs = np.zeros((N, num_fsp, num_fsp))
    Pijs = np.zeros((N, num_fsp, num_fsp))

    for ind, p4 in enumerate(p4s):
        # Calculate Pij
        Pij = np.zeros((num_fsp, num_fsp))
        for i, j in Pij_iter:
            Pij[i, j] = np.dot(p4[i], METRIC * p4[j])

        # Calculate Jij
        Jij = np.zeros((num_fsp, num_fsp))
        for i, j, k, ell in Jij_iter:
            Jij[i, j] += Pij[i, k] * Pij[j, ell]

        # Write to array
        Jijs[ind] = Jij
        Pijs[ind] = Pij

    return Jijs, Pijs


def get_lambdas(
    ltype: str,
    p4s: Optional[p4s_type] = None,
    Jijs: Optional[Jijs_type] = None,
    Pijs: Optional[Pijs_type] = None,
) -> Sequence[float]:
    """
    Returns lambda values (the coefficient of the H1 term of the Hamiltonian) by either
    calculating the Jij/Pij values from the 4-momentum or using the Jij/Pij values
    passed explicitly. The Hamiltonian is:
                        H = H0 + λH1
    so this method calculates various choices for λ.
    """
    if p4s is None and (Jijs is None or Pijs is None):
        raise Exception(
            "Either p4s must be defined or both Jijs and Pijs must be defined."
        )

    if Jijs is None or Pijs is None:
        Jijs, Pijs = get_Jijs_Pijs(p4s)
    else:
        assert Jijs.shape == Pijs.shape

    match ltype:
        case "QA":
            min_Jijs = np.min(Jijs.reshape(len(Jijs), -1), axis=1)
            max_Pijs = np.max(Pijs.reshape(len(Pijs), -1), axis=1)
            lambdas = min_Jijs / max_Pijs
        case "avg":
            avg_Jijs = np.mean(Jijs.reshape(len(Jijs), -1), axis=1)
            avg_Pijs = np.mean(Pijs.reshape(len(Pijs), -1), axis=1)
            lambdas = avg_Jijs / avg_Pijs
        case "Pijavg":
            avg_Pijs = np.mean(Pijs.reshape(len(Pijs), -1), axis=1)
            lambdas = avg_Pijs
        case "Pijmax":
            max_Pijs = np.max(Pijs.reshape(len(Pijs), -1), axis=1)
            lambdas = max_Pijs
        case _:
            raise Exception(
                f"`ltype` must be in {LAMBDA_CHOICES} but is {ltype}"
            )

    return lambdas


def get_coeffs(
    htype: str,
    p4s: Optional[p4s_type] = None,
    Jijs: Optional[Jijs_type] = None,
    Pijs: Optional[Pijs_type] = None,
    lambdas: Optional[Sequence[float]] = None,
) -> Sequence[float]:
    """
    Finds the coefficient for the Hamiltonian when written in terms of spin values, i.e.
    the Hamiltonian is:
                        H = sum_{ij}C_{ij}s_is_j
    where C_{ij} is a matrix of values. Either calculates the Jijs and Pijs from the p4s
    or just used them Jijs and Pijs if passed explicitly. If `htype="QA"` then then
    lambdas must be passed
    """
    if p4s is None and (Jijs is None or Pijs is None):
        raise Exception(
            "Either p4s must be defined or both Jijs and Pijs must be defined."
        )

    if Jijs is None or Pijs is None:
        # Make sure it's an array of 4-momentum (if only 1 event is passed)
        if len(p4s.shape) == 2:
            p4s = p4s.reshape(1, -1, 4)
        Jijs, Pijs = get_Jijs_Pijs(p4s)
    else:
        assert Jijs.shape == Pijs.shape

    num_fsp = p4s.shape[1] if p4s is not None else Jijs.shape[1]
    N = p4s.shape[0] if p4s is not None else Jijs.shape[0]
    coeffs = np.zeros((N, num_fsp, num_fsp))
    match htype:
        case "H0":
            coeffs = Jijs
        case "H1":
            coeffs = Pijs
        case "QA":
            if lambdas is None:
                raise Exception(
                    f"`lambdas` must be defined if `htype={htype}` (QA)."
                )
            if len(lambdas) != len(Jijs) or len(lambdas) != len(Pijs):
                print(lambdas)
                raise Exception(
                    f"`lambdas`, `Pijs` and `Jijs` must all have the same length, but {len(lambdas)=}, {len(Jijs)=} and {len(Pijs)=}."
                )
            # Do this silly thing to create arrays of proper shape
            lmbdas_mat = np.full(Pijs.T.shape, lambdas).T
            # To do this arithmatic correctly
            coeffs = Jijs + 2 * lmbdas_mat * Pijs

    return coeffs


def get_minimum_energies(
    htype: str,
    p4s: p4s_type,
    lambdas: Optional[Sequence[float]] = None,
    ltype: Optional[str] = None,
) -> Sequence[float]:
    """
    For a given Hamiltonian and 4-momenta, this finds the bit strings and corresponding
    energies for the minimum and first excited eigenstate. If `htype="QA", then either
    `lambdas` must be given or `ltype` (the latter of which will then calculate the
    chose of lambda).
    """
    num_fsp = p4s.shape[1]
    # Every combination for the bit strings as an array of booleans
    bit_string_indices = bit_string_bool_combinations(num_fsp)

    if lambdas is None:
        if htype == "QA":
            if ltype is None:
                raise Exception(
                    "If `htype=='QA'` and `lambdas=None`, then `ltype` must be defined."
                )
            Jijs, Pijs = get_Jijs_Pijs(p4s)
            lambdas = get_lambdas(ltype=ltype, Jijs=Jijs, Pijs=Pijs)
        else:
            lambdas = np.zeros(len(p4s))

    (
        min_energies,
        min2nd_energies,
        min_bitstrings,
        min2nd_bitstrings,
    ) = _get_minimum_energies(
        htype=htype,
        p4s=p4s,
        lambdas=lambdas,
        bit_string_indices=bit_string_indices,
    )

    return min_energies, min2nd_energies, min_bitstrings, min2nd_bitstrings


@njit
def _get_minimum_energies(
    htype: str,
    p4s: p4s_type,
    lambdas: Sequence[float],
    bit_string_indices: bs_type,
):
    """
    Jitted function to find ground and first excited states for the Hamiltonian.
    """
    min_energies = np.empty(len(p4s), dtype=float)
    min2nd_energies = np.empty(len(p4s), dtype=float)
    min_bitstrings = np.empty(len(p4s), dtype="U6")
    min2nd_bitstrings = np.empty(len(p4s), dtype="U6")

    for ind, (p4, lmbda) in enumerate(zip(p4s, lambdas)):
        min_energy, min2nd_energy = 1e15, 1e15
        for bool1, bool2 in bit_string_indices:
            p1 = np.sum(p4[bool1], axis=0)
            p2 = np.sum(p4[bool2], axis=0)

            m1sq = np.dot(p1, METRIC * p1)
            m2sq = np.dot(p2, METRIC * p2)

            match htype:
                case "H0":
                    H = (m1sq - m2sq) ** 2
                case "H1":
                    H = m1sq + m2sq
                case "QA":
                    H0 = (m1sq - m2sq) ** 2
                    H1 = m1sq + m2sq
                    H = H0 + lmbda * H1

            # Check if we have a lower value
            if H < min_energy:
                min_energy = H
                min_bitstring = bool1.copy()
            elif H < min2nd_energy:
                min2nd_energy = H
                min2nd_bitstring = bool1.copy()

        # Turn from array of bools to string of 1's and 0's
        min_bitstring_str = "".join(["1" if b else "0" for b in min_bitstring])
        min2nd_bitstring_str = "".join(
            ["1" if b else "0" for b in min2nd_bitstring]
        )

        min_energies[ind] = min_energy
        min2nd_energies[ind] = min2nd_energy
        min_bitstrings[ind] = min_bitstring_str
        min2nd_bitstrings[ind] = min2nd_bitstring_str

    return min_energies, min2nd_energies, min_bitstrings, min2nd_bitstrings


@njit
def get_bitstring_energy(p4: p4_type, bs: str, htype: str, lmbda: float = 1.0):
    """
    Returns the energy for a given bitstring. If `htype="QA"`, then `lmbda` should be
    specified.
    """
    bool1, bool2 = [], []
    for ind, b in enumerate(bs):
        if b == "0":
            bool1.append(ind)
        else:
            bool2.append(ind)

    p1 = np.zeros(4)
    for b in bool1:
        p1 += p4[b]
    p2 = np.zeros(4)
    for b in bool2:
        p2 += p4[b]

    m1sq = np.dot(p1, METRIC * p1)
    m2sq = np.dot(p2, METRIC * p2)

    match htype:
        case "H0":
            H = (m1sq - m2sq) ** 2
        case "H1":
            H = m1sq + m2sq
        case "QA":
            H0 = (m1sq - m2sq) ** 2
            H1 = m1sq + m2sq
            H = H0 + lmbda * H1
    return H


# @njit
def get_all_bitstring_energies(
    p4s: p4s_type,
    bss: list[str],
    htype: str,
    lambdas: Optional[Sequence[float]] = None,
):
    """
    Calculates the energy for every bit string given in `bss` for the given Hamiltonian.
    """
    # Just use ones if lambdas aren't used
    if lambdas is None:
        lambdas = np.ones(len(p4s))
    bs_energies = np.zeros((len(p4s), len(bss)), dtype=float)
    for evt_ind, (p4, lmbda) in enumerate(zip(p4s, lambdas)):
        for bs_ind, bs in enumerate(bss):
            energy = get_bitstring_energy(
                p4=p4, bs=bs, htype=htype, lmbda=lmbda
            )
            bs_energies[evt_ind][bs_ind] = energy

    return bs_energies


@njit
def get_masses(p4s: p4s_type, bss: Sequence[str]):
    """
    Finds the masses given for a given list of 4-momenta for a given list of bitstrings.
    """
    N = len(p4s)
    Nq = len(bss[0])
    num_jets0 = np.zeros(len(bss[0]) + 1)
    num_jets1 = np.zeros(len(bss[0]) + 1)
    m1s = np.zeros(N, dtype=float)
    m2s = np.zeros(N, dtype=float)
    for ind in range(N):
        p4 = p4s[ind]
        bs = bss[ind]
        num_jet0 = str(bs).count("0")
        num_jet1 = str(bs).count("1")
        # Include possibility for 0 jets
        num_jets0[num_jet0] += 1
        num_jets1[num_jet1] += 1

        p1, p2 = np.zeros(4), np.zeros(4)
        for qind in range(Nq):
            if bs[qind] == "0":
                p1 += p4[qind]
            else:
                p2 += p4[qind]

        m1 = sqrt(np.dot(p1, METRIC * p1))
        m2 = sqrt(np.dot(p2, METRIC * p2))

        # np.nan value represents negative mass squared which only happens when mass
        # should be zero but is closer, e.g. -1.2e-7
        if np.isnan(m1):
            m1 = 0
        if np.isnan(m2):
            m2 = 0
        if np.isnan(m1) and np.isnan(m2):
            raise Exception()

        m1s[ind] = m1
        m2s[ind] = m2

    return m1s, m2s, num_jets0, num_jets1


# File types for data files
ftypes = ["pkl", "npy", "npz"]
# Regex formats for files and directories
# Optional non-capturing group for extra terms in FALQON directories/files
falqon = r"(?:_?b\d+\.\d+_dt\d+\.\d+_?)?"
# A invm float
re_float = r"\d\.\d{2}"
# Format for index directories
re_inds_dir = (
    rf"^inds(\d+)_(\w+)_(\w+)_(\d{{3}})x(\d{{3}})_({re_float})to({re_float})$"
)
# Common start of efficiency files/directories (with noncapturing of potential bitflip prob)
re_eff_start = r"eff(?:_0?\.?\d+)?_(\w+)_(\w+)_(\w+)_(\w+)_"
# Format for efficiency directories
re_eff_dir = rf"^{re_eff_start}p(\d+){falqon}$"
# Format ofr file directories
re_eff_file = rf"^{re_eff_start}(\d{{3}})_(\d+)to(\d+)_p(\d+)_{falqon}({re_float})to({re_float})\.pkl$"
re_str_dict = {
    "re_inds_dir": re_inds_dir,
    "re_eff_dir": re_eff_dir,
    "re_eff_file": re_eff_file,
}


def get_Nevts(names):
    """
    Finds total number of events from either: file name, dir name or list of files from
    a specific dir.
    """
    # If we are passed just a file or directory name
    if isinstance(names, str):
        etype, dtype, alg, quadcoeff, depth = get_info(
            names, ["etype", "dtype", "alg", "quadcoeff", "depth"]
        )
        names = get_files(
            ntype="file",
            ftype="eff",
            etype=etype,
            dtype=dtype,
            alg=alg,
            quadcoeff=quadcoeff,
            depth=depth,
        )
    N = 0
    for name in names:
        N = max(N, int(get_info(name, ["N_hi"])[0]))

    return N


def get_info(name, data_names):
    """
    Grabs data from a file or directory name.

    Parameters:
    name - Name of the file or directory
    data_names - Information to grab from `name`
    """
    # Make sure it's a string
    if isinstance(name, Path):
        name = name.name
    # Determine what data type it is, e.g. 'inds' or 'eff'
    dtype = "".join([c for c in name.split("_")[0] if not c.isdigit()])
    # Determine if it's a file or a directory
    ntype = "file" if name[-4:] in [f".{ftype}" for ftype in ftypes] else "dir"

    # Get the data
    data = re.findall(re_str_dict[f"re_{dtype}_{ntype}"], name)
    if len(data) != 1:
        print(name)
        print(re_str_dict[f"re_{dtype}_{ntype}"])
        raise Exception(f"Found {len(data)} results...")

    # Put it in a directory
    if ntype == "dir" and dtype == "inds":
        d = {
            "N_evts": int(data[0][0]),
            "etype": data[0][1],
            "dtype": data[0][2],
            "uid": data[0][3],
            "rngn": data[0][4],
            "invm_lo": float(data[0][5]),
            "invm_hi": float(data[0][6]),
        }
    elif ntype == "dir" and dtype == "eff":
        d = {
            "etype": data[0][0],
            "dtype": data[0][1],
            "alg": data[0][2],
            "quadcoeff": data[0][3],
            "depth": data[0][4],
        }
    elif ntype == "file" and dtype == "eff":
        d = {
            "etype": data[0][0],
            "dtype": data[0][1],
            "alg": data[0][2],
            "quadcoeff": data[0][3],
            "uid": data[0][4],
            "N_lo": data[0][5],
            "N_hi": data[0][6],
            "depth": data[0][7],
            "invm_lo": data[0][8],
            "invm_hi": data[0][9],
        }

    # Extract the ones we want
    return [d[dname] for dname in data_names]


def get_all_indices(dir_name):
    """
    Returns all indices given the directory name of said indices.
    """
    N_evts = get_info(dir_name, ["N_evts"])[0]
    ind_fname = list(
        dir_name.glob(f"*{'1':0>{len(str(N_evts))}}to{N_evts}.npz")
    )
    # Make sure we only got one
    if len(ind_fname) != 1:
        raise Exception(ind_fname)

    # Get the indices and then the events
    inds = np.load(ind_fname[0])["inds"]
    return inds


def get_all_depths(
    etype, dtype, alg, quadcoeff, noise=False, bitflip_prob="\d+"
):
    """
    Returns every depth inside a directory of efficiency files.
    """
    dirnames = get_files(
        ntype="dir",
        ftype="eff",
        noise=noise,
        bitflip_prob=bitflip_prob,
        etype=etype,
        dtype=dtype,
        alg=alg,
        quadcoeff=quadcoeff,
    )
    depths = []
    for dirname in dirnames:
        data = re.findall(re_eff_dir, dirname.name)
        if len(data) != 1:
            raise Exception()
        depth = int(data[0][4])
        depths.append(depth)

    return np.unique(sorted(depths))


def get_files(ntype, ftype, noise=False, **kwargs):
    """
    Opposite of `get_info`. Given keywords, returns all files or directories that match
    these keywords.

    Parameters:
    ntype - Name type, e.g. 'dir' or 'file'
    ftype - File type, e.g. 'eff' or 'inds
    noise - If True, searches the directory for noisy runs, otherwise searches the
        regular directory
    """
    N_evts = kwargs.get("N_evts", "\d+")
    etype = kwargs.get("etype", "\w+")
    dtype = kwargs.get("dtype", "\w+")
    alg = kwargs.get("alg", "\w+")
    quadcoeff = kwargs.get("quadcoeff", "\w+")
    uid = kwargs.get("uid", r"\d{3}")
    depth = kwargs.get("depth", "\d+")
    invm_lo = kwargs.get("invm_lo", r"\d\.\d{2}")
    if invm_lo != r"\d\.\d{2}":
        invm_lo = f"{invm_lo:.2f}"
    invm_hi = kwargs.get("invm_hi", r"\d\.\d{2}")
    if invm_hi != r"\d\.\d{2}":
        invm_hi = f"{invm_hi:.2f}"
    evtind_lo = kwargs.get("evtind_lo", "\d+")
    evtind_hi = kwargs.get("evtind_hi", "\d+")
    err_prob = (str(kwargs.get("bitflip_prob", "\d+")) + "_") if noise else ""

    data_dir = NOISY_DIR if noise else OUTPUT_DIR
    if ftype == "eff":
        if noise:
            re_dir_name = f"eff_{err_prob}{etype}_{dtype}_{alg}_{quadcoeff}_p{depth}{falqon}"
        else:
            re_dir_name = (
                f"eff_{etype}_{dtype}_{alg}_{quadcoeff}_p{depth}{falqon}"
            )

        if ntype == "dir":
            dirs = []
            for dir_name in data_dir.glob("*"):
                if re.match(re_dir_name, dir_name.name) is not None:
                    dirs.append(data_dir / dir_name.name)
            return dirs
        elif ntype == "file":
            re_file_name = f"eff_{etype}_{dtype}_{alg}_{quadcoeff}_{uid}_0*{evtind_lo}to0*{evtind_hi}_p{depth}_{falqon}{invm_lo}to{invm_hi}\.pkl"

            files = []
            for fdir_name in data_dir.glob(
                f"*_{err_prob}*/*" if noise else "*/*"
            ):
                if re.match(re_file_name, fdir_name.name) is not None:
                    files.append(data_dir / fdir_name)
            return files
    elif ftype == "inds":
        re_dir_name = (
            f"inds{N_evts}_{etype}_{dtype}_{uid}x\d{{3}}_{invm_lo}to{invm_hi}"
        )

        if ntype == "dir":
            dirs = []
            for dir_name in IND_DIR.glob("*"):
                if re.match(re_dir_name, dir_name.name) is not None:
                    dirs.append(IND_DIR / dir_name.name)
            return dirs
        elif ntype == "file":
            re_file_name = f"inds_{etype}_{dtype}_{uid}x\d{{3}}_{invm_lo}to{invm_hi}_0*{evtind_lo}to0*{evtind_hi}\.npz"

            files = []
            for fdir_name in IND_DIR.glob("*/*"):
                if re.match(re_file_name, fdir_name.name) is not None:
                    files.append(IND_DIR / fdir_name)
            return files
    raise Exception("Check your values for `ntype` and `dtype`!")
