import numpy as np
from constants import INVMS
from events import get_invms
from numpy.typing import NDArray
from type_hints import evts_type


def split_data(
    evts: evts_type,
) -> tuple[NDArray[evts_type], NDArray[NDArray[int]]]:
    """
    Splits up the events by their invariant masses into groups defined by
    the constant `INVMS`. Returns the split events and the indices for those events.
    Assumes each invariant mass bin contains the same number of events.


    Parameters:
    evts - A (N, n_fsp, 4) numpy array where N is any number, and n_fsp is the
        number of final state particles for this specific event.
    """
    invms = get_invms(evts=evts)
    split_evts, split_inds = [], []
    for low, hi in zip(INVMS[:-1], INVMS[1:]):
        mask = np.where(np.logical_and(invms > low, invms < hi))[0]
        split_evts.append(evts[mask])
        split_inds.append(mask)

    # Sanity check, each element should be same size
    if len(set([len(split) for split in split_evts])) != 1:
        raise Exception(
            "Length of each invariant mass split should be the same"
        )
    return np.array(split_evts), np.array(split_inds)
