from pathlib import Path
from random import choice
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from .constants import MASS_NORM_DICT, METRIC
from .type_hints import Jijs_type, Pijs_type, evt_type, evts_type


def get_Pijs(evts: evts_type) -> Pijs_type:
    """
    Returns an array of the matrix Pij = pi * pj, i.e. the dot product between
    two 4-momenta for the input of events

    Parameters:
    evts - A (N, n_fsp, 4) numpy array where N is any number, and n_fsp is the
        number of final state particles for this specific event.
    """
    evts_dagger = evts * METRIC
    return np.einsum("nij, nkj -> nik", evts_dagger, evts)


def get_Jijs(evts: evts_type, return_Pijs=False) -> Jijs_type:
    """
    Returns an array of the matrix Jij = P_ik P_jl (summing over k and l) for
    the input of events

    Parameters:
    evts - A (N, n_fsp, 4) numpy array where N is any number, and n_fsp is the
        number of final state particles for this specific event.
    return_Pijs (default False) - If True, returns the Pij matrices as the
        second result.
    """
    Pijs = get_Pijs(evts)
    row_sums = Pijs.sum(axis=2)
    Jijs = row_sums[:, :, None] * row_sums[:, None, :]

    if return_Pijs:
        return Jijs, Pijs
    return Jijs


def get_invms(evts: evts_type, etype: str = "ttbar") -> NDArray[np.float64]:
    """
    Finds the invariant mass of the events which is normalized by the mass sum
    of the final state particles.

    Parameters:
    evts - A (N, n_fsp, 4) numpy array where N is any number, and n_fsp is the
        number of final state particles for this specific event.
    etype (default "ttbar") - A string representing the event type and therefore
        the mass sum of the final state paticles. We have:
            "ttbar" -- a top and antitop,
            "tW" -- a top and a W boson,
            "4top" -- two top and antitop pairs
    """
    norm = MASS_NORM_DICT[etype]
    total = evts.sum(axis=-2)

    E = total[..., 0]
    px = total[..., 1]
    py = total[..., 2]
    pz = total[..., 3]

    invms = E**2 - (px**2 + py**2 + pz**2)
    return np.sqrt(invms) / norm


def get_data(
    etype: str = "ttbar",
    dtype: str = "parton",
    data_path: Optional[Union[str, Path]] = None,
) -> evts_type:
    """
    Grabs all the relevant data for a given event and data type. Returns events
    (i.e. the 4-momentum of the final state particles), the Jij and Pij
    matrices, and the normalized invariant masses.

    Parameters:
    etype (default "ttbar") - The event type, currently can be "ttbar",
        "tW" or "6jet".
    dtype (default "parton") - The data type, currently can be "parton"
        or "smeared".
    data_path (default None) - Argument that can be specified to give the
        explicit path to where the data file is. Replacing `etype` and `dtype`.
    """
    if data_path is None:
        data_dir = Path(__file__).parents[1] / "data"
        data_path = data_dir / f"{etype}_{dtype}_events.npz"
    else:
        data_path = Path(data_path)

    match data_path.suffix:
        case ".npz":
            evts = np.load(data_path)
            # assume only a single key in the npz file
            evts = evts[evts.files[0]]
        case ".npy":
            evts = np.load(data_path)
        case _:
            raise Exception(f"Unknown file extension: {data_path.suffix}")

    Jijs, Pijs = get_Jijs(evts=evts, return_Pijs=True)
    invms = get_invms(evts, etype=etype)

    print(f"Loaded {len(invms):,} events from: {data_path}")
    return evts, Jijs, Pijs, invms


def get_event(
    etype: str = "ttbar",
    dtype: str = "parton",
    low_invm: float = 1.0,
    hi_invm: float = 3.0,
    ind: Optional[int] = None,
    data_path: Optional[Union[str | Path]] = None,
) -> evt_type:
    """
    Returns a single event either randomly chosen between two invariant masses
    or a specific index N representing the Nth event between those masses.

    Parameters:
    etype (default "ttbar") - The event type, currently can be "ttbar",
        "tW" or "6jet".
    dtype (default "parton") - The data type, currently can be "parton"
        or "smeared".
    low_invm (default 1.0) - The minimum invariant mass to cut at.
    hi_invm (default 3.0) - The maximum invariant mass to cut at.
    ind (default None) - An index value that can be specified. After doing the
        filter with the invariant masses, it with give the `ind`th value of the
        result.
    data_path (default None) - Argument that can be specified to give the
        explicit path to where the data file is. Replacing `etype` and `dtype`.

    """
    evts, Jijs, Pijs, invms = get_data(
        etype=etype, dtype=dtype, data_path=data_path
    )
    invm_inds = np.where(np.logical_and(invms > low_invm, invms < hi_invm))[0]

    if ind is None:
        ind = choice(invm_inds)

    return evts[ind], Jijs[ind], Pijs[ind], invms[ind]
