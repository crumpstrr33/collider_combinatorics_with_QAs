from typing import Union

import numpy as np
from numpy.typing import NDArray

from .constants import INVMS, METRIC
from .events import get_invms
from .type_hints import evt_type, evts_type


def split_data(
    evts: evts_type,
) -> tuple[NDArray[evts_type], NDArray[NDArray[int]]]:
    """
    Splits up the events by their invariant masses into groups defined by
    the constant `INVMS`. Returns the split events and the indices for those
    events. Assumes each invariant mass bin contains the same number of events.

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
        raise Exception("Length of each invariant mass split should be the same")
    return np.array(split_evts), np.array(split_inds)


def get_bitstring_invms(
    evts: Union[evt_type, evts_type], bitstrings: Union[str, NDArray[str]]
) -> NDArray[NDArray[float]]:
    """
    Finds the invariant mass for the 4-momenta represented by the "1" in the
    bitstring and "0" in the bitstring separately.

    Parameters:
    evts - Either an array of events, i.e. their 4-momenta, or a single event.
    bitstrings - Either an array of bitstrings or a single bitstring. The length
        of the bitstring(s) should be the number of 4-momenta per event. If not
        using a single bitstring, the length of the array should equal the
        length of `evts`.
    """
    num_bits = evts.shape[-2]
    num_evts = evts.shape[0] if len(evts.shape) == 3 else 1
    # Turn single string in copies of itself for use below
    if isinstance(bitstrings, str):
        bitstrings = np.tile(bitstrings, num_evts).astype(f"S{num_bits}")

    # Turns bitstrings into a boolean mask, one N-length array per event
    mask = bitstrings.view(np.uint8).reshape(-1, num_bits) == ord("1")
    # Creates 4-length array for each event of summed 4-momenta for 0's and 1's
    summed_p4s_0 = np.sum(evts * np.invert(mask)[..., None], axis=1)
    summed_p4s_1 = np.sum(evts * mask[..., None], axis=1)

    # Find m^2 = p^2, round to avoid small negative masses due to floating point
    msq_0 = np.round(np.sum(summed_p4s_0 * (METRIC * summed_p4s_0), axis=1), 3)
    msq_1 = np.round(np.sum(summed_p4s_1 * (METRIC * summed_p4s_1), axis=1), 3)

    # Make sure we don't have any negative masses
    if ((msq_0 < 0.0) & np.isclose(msq_0, 0.0)).any():
        raise Exception(f"We have negative masses: {np.where(msq_0 < 0) = }")
    if ((msq_1 < 0.0) & np.isclose(msq_1, 0.0)).any():
        raise Exception(f"We have negative masses: {np.where(msq_1 < 0) = }")

    return np.stack((np.sqrt(msq_0), np.sqrt(msq_1))).T
