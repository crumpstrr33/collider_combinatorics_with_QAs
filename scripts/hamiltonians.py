from typing import Optional, Sequence, Union

import numpy as np
from events import get_Jijs, get_Pijs
from numpy.typing import NDArray
from type_hints import evts_type


def swap(bs: str) -> str:
    """
    Swaps the 0's and 1's in a string

    Parameters:
    bs - Bitstring to invert.
    """
    return bs.replace("0", "9").replace("1", "0").replace("9", "1")


def get_lambdas(
    evts: evts_type,
    nume: Optional[Sequence[str]] = None,  # ["min", "Jij"],
    denom: Optional[Sequence[str]] = None,  # ["max", "Pij"],
) -> NDArray[np.float64]:
    """
    Finds the lambda coefficient for events. This is most commonly used in the
    Hamiltonian H = H0 + λH1, i.e. as the ratio between the contributions of the
    two terms. The lambdas are specified as a ratio between allowed terms as
    shown below. The default values give the the value used in the quantum
    annealing paper: arXiv:2111.07806.

    Parameters:
    evts - A (N, n_fsp, 4) numpy array where N is any number, and n_fsp is the
        number of final state particles for this specific event.
    nume (default None --> ["min", "Jij"]) - The numerator for lambda.
    denom (default None --> ["max", Pij"])- The denominator for lambda. For both `nume`
        and `denom`, the first element of the list is the operation to perform.
        It can be:
            "min" -- minimum, i.e. np.min
            "max" -- maximum, i.e. np.max
            "mean" -- mean average, i.e. np.mean
        And the second element is what to perform the operation on. It can be:
            "Jij" -- The matrix returned by `get_Jijs`
            "Pij" -- The matrix returned by `get_Pijs`

    """
    nume = ["min", "Jij"] if nume is None else nume
    denom = ["max", "Pij"] if denom is None else denom

    Jijs, Pijs = get_Jijs(evts=evts, return_Pijs=True)
    arg_dict = {"min": np.min, "max": np.max, "mean": np.mean}
    val_dict = {"Jij": Jijs, "Pij": Pijs}

    numerator = arg_dict[nume[0]](val_dict[nume[1]], axis=(1, 2))
    denominator = arg_dict[denom[0]](val_dict[denom[1]], axis=(1, 2))

    return numerator / denominator


def get_coefficients(
    hamiltonian: str,
    evts: evts_type,
    **lambda_kwargs: Sequence[str],
) -> NDArray[np.float64]:
    """
    Finds the coefficients of the quadratic spin terms for a specific
    Hamiltonian. For example, since H0 = J_ij * s_i * s_j, this will just return
    J_ij. Also available is:
            H1 = P_ij / 2 * si  * sj
            H2 = H0 + lambda * H1
    where lambda is given by `get_lambdas`.
    """
    match hamiltonian:
        case "H0":
            return get_Jijs(evts=evts)
        case "H1":
            return get_Pijs(evts=evts) / 2
        case "H2":
            lambdas = get_lambdas(evts=evts, **lambda_kwargs)
            Jijs, Pijs = get_Jijs(evts=evts, return_Pijs=True)
            return Jijs + lambdas[:, None, None] * Pijs / 2


def get_bitstrings(
    N: int, astype: str = "bits"
) -> NDArray[Union[str, NDArray[int]]]:
    """
    Gives all possible bitstrings of length `N`.

    Parameters:
    N - Length of bitstring.
    astype (default "int") - What type should be. Can be: "bits" or "spins". If
        "bits", each element of the array will be a string of "0"'s and "1"'s.
        If "spins", each element with be a list of the integers -1 or +1.
    """
    bitstrings = [format(bs, f"0{N}b") for bs in range(2**N)]
    match astype:
        case "bits":
            return np.array(bitstrings, dtype=f"<U{N}")
        case "spins":
            return np.array(
                [
                    [+1 if bit == "1" else -1 for bit in bitstring]
                    for bitstring in bitstrings
                ],
                dtype=int,
            )


def get_bitstring_energies(
    evts: evts_type, bs: str, hamiltonian: str, **lambda_kwargs: Sequence[str]
) -> NDArray[np.float64]:
    """
    Finds the energies for a given bitstring and Hamiltonian for the given
    events.

    Parameters:
    evts - A (N, n_fsp, 4) numpy array where N is any number, and n_fsp is the
        number of final state particles for this specific event.
    bs - The bitstring given as a string of 0's and 1's.
    hamiltonian - The Hamiltonian whose energies we wish to find. Can be:
        H0, H1 or H2.
    lambda_kwargs - The keyword argument that would be given to specify the
        lambda coefficient. Must be specified if `hamiltonian="H2"` and then
        must be given `nume` and `denom`.
    """
    coeffs = get_coefficients(
        hamiltonian=hamiltonian, evts=evts, **lambda_kwargs
    )
    bs_arr = [+1 if b == "1" else -1 for b in bs]

    return np.einsum("nij, i, j -> n", coeffs, bs_arr, bs_arr)


def get_all_bitstring_energies(
    evts: evts_type,
    hamiltonian: str,
    as_dict: bool = False,
    **lambda_kwargs: Sequence[str],
) -> Union[
    tuple[Sequence[str], NDArray[NDArray[np.float64]]],
    NDArray[dict[str, np.float64]],
]:
    """
    Finds the energies for all possible bitstrings and a Hamiltonian for the
    given events. With N events and S = 2**n bitstrings (where n is the number
    of final state particles), it returns an object specified by `as_dict`.

    Parameters:
    evts - A (N, n_fsp, 4) numpy array where N is any number, and n_fsp is the
        number of final state particles for this specific event.
    hamiltonian - The Hamiltonian whose energies we wish to find. Can be:
        H0, H1 or H2.
    as_dict (default False) - If False, returns an S-length numpy array of all
        possible bitstrings and an NxS numpy array of the corresponding energies
        for each event. If True, returns an N length numpy array where each
        entry is a S-length dict of entries {bitstring: energy}.
    lambda_kwargs - The keyword argument that would be given to specify the
        lambda coefficient. Must be specified if `hamiltonian="H2"` and then
        must be given `nume` and `denom`.
    """
    coeffs = get_coefficients(
        hamiltonian=hamiltonian, evts=evts, **lambda_kwargs
    )
    num_fsp = coeffs.shape[1]

    bitstrings = get_bitstrings(N=num_fsp)
    bs_arrs = get_bitstrings(N=num_fsp, astype="spins")
    energies = np.einsum("nij, si ,sj -> ns", coeffs, bs_arrs, bs_arrs)

    if as_dict:
        return np.array([dict(zip(bitstrings, energy)) for energy in energies])
    return bitstrings, energies


def get_minimum_energies(
    evts: evts_type, hamiltonian: str, **lambda_kwargs: Sequence[str]
) -> NDArray[object]:
    """
    Finds the energies and bitstring corresponding to the minimum energy for a
    given Hamiltonian for the given events. It returns a numpy array where each
    entry is [bitstring, energy].

    Parameters:
    evts - A (N, n_fsp, 4) numpy array where N is any number, and n_fsp is the
        number of final state particles for this specific event.
    hamiltonian - The Hamiltonian whose energies we wish to find. Can be:
        H0, H1 or H2.
    lambda_kwargs - The keyword argument that would be given to specify the
        lambda coefficient. Must be specified if `hamiltonian="H2"` and then
        must be given `nume` and `denom`.
    """
    bitstrings, energies = get_all_bitstring_energies(
        evts=evts, hamiltonian=hamiltonian, **lambda_kwargs
    )
    min_inds = energies.argmin(axis=1)
    num_evts = evts.shape[0]

    minima = np.empty((num_evts, 2), dtype=object)
    minima[:, 0] = bitstrings[min_inds]
    minima[:, 1] = energies[range(num_evts), min_inds]

    return minima
