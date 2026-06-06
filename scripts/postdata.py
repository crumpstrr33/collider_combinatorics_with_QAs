from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .constants import INVMS, SYM_TRUE_BS_DICT
from .data import get_bitstring_invms
from .hamiltonians import swap
from .type_hints import DatumType


def find_efficiency(
    data: NDArray[DatumType],
    find_correct: bool = True,
    assume_symmetric: bool = True,
    etype: str | Sequence[str] = "ttbar",
) -> NDArray[NDArray[float]]:
    """
    From the given data arrays, finds the efficiency. That is, it finds the
    fraction of events for which the algorithm found the "correct" bitstring
    with the largest probability OR the "minimum" bitstring. Keep in mind, we
    are assuming that symmetric bitstring are equivalent, i.e. we are summing
    both cases together.

    Parameters:
    data - An array of data for runs as created by the `JobRunner` class and
        saved as an .npz file.
    find_correct (default True) - If True, the "correct" bitstring is the one
        that solves the problem, in our case it is "000111". If False, the
        "correct" bitstring is the one that minimizing the Hamiltonian.
    etype (default "ttbar") - The event type. If a string, it will apply it the
        same data type to every element of `data`. If a list, it will be
        element-specific.
    """
    # Number of data to go over
    data_len = len(data)
    n_bins = len(INVMS) - 1
    # Turn into a list of all the same etype
    if isinstance(etype, str):
        etype = [etype] * data_len

    effs = np.empty((data_len, n_bins), dtype=np.float64)
    for dind, (datum, etype) in enumerate(zip(data, etype)):
        # The correct bitstring
        cor_bs = SYM_TRUE_BS_DICT[etype]

        for iind, invm in enumerate(INVMS[:-1]):
            invm_datum = datum[invm]
            # Bitstrings minimizing the energy for each event
            min_bitstrings = invm_datum["min_bitstrings"]
            # All probabilities for each event
            evt_probs = invm_datum["probs"]

            # For each event, index of max probability, i.e. base 10 of bitstring
            max_probs_inds = np.argmax(evt_probs, axis=1)
            tot_evts = max_probs_inds.shape[0]
            if find_correct:
                bs_ind = int(cor_bs, 2)
                bs_swap_ind = int(swap(cor_bs), 2)
            else:
                bs_ind = np.array([int(bs, 2) for bs in min_bitstrings])
                bs_swap_ind = np.array([int(swap(bs), 2) for bs in min_bitstrings])

            # Make a mask of where our wanted bitstrings are. We are seeing if
            # the bitstrings with the max probability as stored in
            # `max_probs_inds` ARE these wanted bitstrings. Then we sum up all
            # instances where that is true.
            is_match = (max_probs_inds == bs_ind) | (max_probs_inds == bs_swap_ind)
            effs[dind, iind] = is_match.sum() / tot_evts

    return effs


def find_probabilities(
    data: NDArray[DatumType],
    find_correct: bool = True,
    etype: str | Sequence[str] = "ttbar",
) -> NDArray[NDArray[float]]:
    """
    Similar to `find_efficiency`, this will instead calculate, not how many
    events chose a specific bitstring as the most probable, but what the actual
    probabilities of these bitstrings are. This can be considered as a form of
    "confidence". For example, an algorithm could have an efficiency of 90%
    (that is, for 90% of events, the algorithm chose the correct bitstring
    with the most probability) but maybe it only chooses said bitstring with 25%
    probability. In this case, the quantum state is in a large superposition of
    states.

    Parameters:
    data - An array of data for runs as created by the `JobRunner` class and
        saved as an .npz file.
    find_correct (default True) - If True, the "correct" bitstring is the one
        that solves the problem, in our case it is "000111". If False, the
        "correct" bitstring is the one that minimizing the Hamiltonian.
    etype (default "ttbar") - The event type. If a string, it will apply it the
        same data type to every element of `data`. If a list, it will be
        element-specific.
    """
    # Number of data to go over
    data_len = len(data)
    n_bins = len(INVMS) - 1
    # Turn into a list of all the same etype
    etypes = etype
    if isinstance(etype, str):
        etypes = [etype] * data_len

    probs = np.empty((data_len, n_bins), dtype=np.float64)
    for dind, (datum, etype) in enumerate(zip(data, etypes)):
        # The correct bitstring
        cor_bs = SYM_TRUE_BS_DICT[etype]

        for iind, invm in enumerate(INVMS[:-1]):
            invm_datum = datum[invm]
            # Bitstrings minimizing the energy for each event
            min_bitstrings = invm_datum["min_bitstrings"]
            # All probabilities for each event
            evt_probs = invm_datum["probs"]

            if find_correct:
                # Probabilities for CORRECT bitstring
                bs_ind = int(cor_bs, 2)
                bs_swap_ind = int(swap(cor_bs), 2)

                bs_probs = evt_probs[:, bs_ind] + evt_probs[:, bs_swap_ind]
            else:
                # Probabilities for MINIMUM bitstrings
                bs_ind = np.array([int(bs, 2) for bs in min_bitstrings])
                bs_swap_ind = np.array([int(swap(bs), 2) for bs in min_bitstrings])

                bs_probs = np.take_along_axis(evt_probs, bs_ind[:, None]).squeeze()
                bs_probs += np.take_along_axis(evt_probs, bs_swap_ind[:, None]).squeeze()

            probs[dind, iind] = bs_probs.mean()
    return probs


def parse_with_metadata(
    infos: Sequence[Sequence[str]] | Sequence[str],
    metadata: NDArray[str],
    one_per: bool = True,
) -> NDArray[NDArray[str]] | NDArray[str]:
    """
    Given a list of keywords, it will find the data that corresponds to that
    data. For example, if `infos=[["qaoa", "max"], ["maqaoa" "5"]], it will find
    the indices for `data` and `metadata` which contains runs for either QAOA
    using the max normalization or MAQAOA with a depth of p=5. As of yet, this
    won't differentiate between the lambda numerator or denominator but I don't
    think I'll need that functionality due to dimensional analysis and hope.
    Also, I don't look at number of events, can get confused with depth.

    Parameters:
    infos - A sequence where each element contains info to search through the
        metadata for. Each element is a different search query. OR just a single
        search query, i.e. one element from the first sentence.
    metadata - An array of elements of the list output by the `parse_data`
        method. Contains the metadata info for a run.
    """
    is_single = isinstance(infos[0], str)
    if is_single:
        infos = [infos]

    indices = []
    # Iterate over each search query
    for info in infos:
        mask = np.ones(len(metadata), dtype=bool)
        # Iterate over each search term
        for ele in info:
            # Convert float to string
            if isinstance(ele, float):
                ele = f"{ele:.3f}"
            # If that term isn't in a metadatum, then it gets a False mask
            mask &= (metadata == ele).any(axis=1)

        # Just warn if the mask is completely False
        if one_per and mask.sum() > 1:
            raise Exception(
                f"`one_per=True` but we have {mask.sum()} results for: {info}."
                f" Found indices: {np.where(mask)[0]}"
            )
        if not mask.any():
            print(f"Warning: didn't find any results for: {info}")

        indices.append(np.where(mask)[0])

    if is_single:
        return indices[0]
    return np.concatenate(indices)


def int_to_bin(
    arr: NDArray[str | int], N: int, dtype: Literal["bits", "str"] = "bits"
) -> NDArray[str]:
    """
    Converts an array of ints (or strings of ints) into their binary values.
    Can be an array of either strings or bytes.

    Parameters:
    arr - The array of ints
    N - The number of bits for the bitstrings
    dtype (default "bits") - The type of numpy array, either bits or strings
    """
    dtype = {"bits": "S", "str": "U"}[dtype]
    return np.array([str(format(int(x), f"0{N}b")) for x in arr], dtype=f"{dtype}{N}")


def get_2dhist_invms(
    datum: DatumType, by_invm_bin: bool = False
) -> NDArray[tuple[float, float] | NDArray[tuple[float, float]]]:
    """
    Gets the masses of the final state particles based on what the algorithm
    chose at the most likely bitstring, i.e. if "110000" was chosen, then it
    will separately find the invariant mass of the first two 4-momenta and the
    last 4 4-momenta.

    Parameters:
    datum - The data for this run as created by the `JobRunner` class and saved
        as an .npz file.
    by_invm_bin (default True) - If True, will keep the array as (M, N, 2) where
        M is the number of invariant mass bins and N is the number of events per
        bin. Otherwise, return as (MN, 2).
    """
    invms = []
    num_bits = datum[2.50]["invm_p4s"].shape[1]
    for invm in INVMS[:-1]:
        # Most likely bitstrings per event
        bitstrings = int_to_bin(
            np.argmax(datum[invm]["probs"], axis=1), N=num_bits, dtype="str"
        )
        # Invariant masses for the 0's and the 1's separately
        invms.append(get_bitstring_invms(datum[invm]["invm_p4s"], bitstrings))

    invms = np.array(invms)

    if by_invm_bin:
        return invms
    return invms.reshape(-1, 2)


def get_chosen_bitstrings(datum: DatumType) -> NDArray[str]:
    """
    For the given run, returns the bitstring chosen with the highest probability
    by the algorithm for each event.

    Parameters:
    datum - The data for this run as created by the `JobRunner` class and saved
        as an .npz file.
    """
    N = datum[1.00]["invm_p4s"].shape[1]

    bitstrings = []
    for invm in INVMS[:-1]:
        # Convert from decimal to binary
        bitstrings.append(
            [format(val, f"0{N}b") for val in np.argmax(datum[invm]["probs"], axis=1)]
        )

    return np.array(bitstrings)
