from typing import Sequence, Union

import numpy as np
from numpy.typing import NDArray

from .constants import INVMS, NUM_FSP_DICT, SYM_TRUE_BS_DICT
from .data import get_bitstring_invms
from .type_hints import datum_type


def create_falqon_depth(
    datum: datum_type, metadatum: NDArray[str], depth: int
) -> tuple[datum_type, NDArray[str]]:
    """
    Since the FALQON data comes with the probabilities for every depth, as each
    circuit depth is created, this method returns in the proper format the
    `metadatum` and `datum` for a specific depth.

    Parameters:
    datum - The data for this run as created by the `JobRunner` class and saved
        as an .npz file.
    metadatum - An element of the list output by the `parse_data` method.
        Contains the metadata info for a run.
    depth - The depth to get the info for.
    """
    num_fsp = NUM_FSP_DICT[metadatum[4]]
    bitstrings = np.array([format(x, f"0{num_fsp}b") for x in range(2**num_fsp)])

    if metadatum[0] != "falqon":
        raise Exception(f"Not a FALQON instance:\n{metadatum}")
    if "depth_probs" not in datum[1.00]:
        raise Exception("No `depth_probs` key. Might be a copy already...")
    is_full_depth = depth == datum[2.50]["depth_probs"].shape[1]

    new_datum = {}
    for invm in INVMS[:-1]:
        if is_full_depth:
            invm_datum = {k: v for k, v in datum[invm].items() if k != "depth_probs"}
        else:
            invm_datum = {
                k: v
                for k, v in datum[invm].items()
                if k not in ["probs", "depth_probs", "ranks", "rank_probs"]
            }
            probs = datum[invm]["depth_probs"][:, depth, :]
            invm_datum["probs"] = probs
            invm_datum["rank_probs"] = probs[:, int("000111", 2)]

            # Get indices of sorted probabilities for each event
            sorted_inds = np.flip(np.argsort(probs, axis=1), axis=1)
            ranks = np.where(bitstrings[sorted_inds] == "000111")[1]
            # This is the assumption that the bitstring is symmetric, so rank = 1
            # is the same as rank == 0, so subtract 1 from all odd ranks
            ranks -= ranks % 2
            invm_datum["ranks"] = ranks

        new_datum[invm] = invm_datum

    new_metadatum = np.concatenate(([metadatum[0], depth], metadatum[2:]))
    return new_datum, new_metadatum


def find_efficiency(
    data: NDArray[datum_type],
    find_correct: bool = True,
    assume_symmetric: bool = True,
) -> NDArray[NDArray[float]]:
    """
    From the given data arrays, finds the efficiency. That is, it finds the
    fraction of events for which the algorithm found the "correct" bitstring
    with the largest probability.

    Parameters:
    data - An array of data for runs as created by the `JobRunner` class and
        saved as an .npz file.
    find_correct (default True) - If True, the "correct" bitstring is the one
        that solves the problem, in our case it is "000111". If False, the
        "correct" bitstring is the one that minimizing the Hamiltonian.
    assume_symmetric (default True) - If True, it will assume that symmetric
        bitstrings are equivalent. So a rank of 1 would be equivalent to a rank
        of 0 and so on.
    """
    effs = []
    # Iterate for each run
    for datum in data:
        datum_effs = []
        # Iterate for each invariant mass bin
        for invm in INVMS[:-1]:
            # Calculate the efficiency with possibility of symmetry
            if find_correct:
                # i.e solves the combinatorial problem, that is gives "000111"
                ranks = datum[invm]["ranks"]
                eff = np.sum(
                    (ranks - (ranks % (2 if assume_symmetric else 1))) == 0
                ) / len(ranks)
            else:
                # i.e. found the minimum energy bitstring
                # For some reason, the `min_bistrings` are saved as ints? I
                # don't know why and it's too late to change it, so we need to
                # turn them back into bitstrings by padding the ints with 0s
                min_bitstrings = [
                    f"{int(bs):0>6}" for bs in datum[invm]["min_bitstrings"]
                ]
                alg_bitstrings = get_bitstrings(datum=datum)[INVMS.index(invm)]
                eff = np.sum(min_bitstrings == alg_bitstrings) / len(min_bitstrings)

            datum_effs.append(eff)
        effs.append(datum_effs)

    return np.array(effs)


def parse_with_metadata(
    infos: Sequence[Sequence[str]], metadata: NDArray[str], one_per: bool = True
):
    """
    Given a list of keywords, it will find the data that corresponds to that
    data. For example, if `infos=[["qaoa", "max"], ["maqaoa" "5"]], it will find
    the indices for `data` and `metadata` which contains runs for either QAOA
    using the max normalization or MAQAOA with a depth of p=5. As of yet, this
    won't differentiate between the lambda numerator or denominator but I don't
    think I'll need that functionality due to dimensional analysis and hope.

    Parameters:
    infos - A sequence where each element contains info to search through the
        metadata for. Each element is a different search query.
    metadata - An array of elements of the list output by the `parse_data`
        method. Contains the metadata info for a run.
    """
    indices = []
    # Iterate over each search query
    for info in infos:
        mask = np.ones(len(metadata), dtype=bool)
        # Iterate over each search term
        for ele in info:
            # If that term isn't in a metadatum, then it gets a False mask
            mask &= (metadata == ele).any(axis=1)

        # Just warn if the mask is completely False
        if not mask.any():
            print(f"Warning: didn't find any results for: {info}")
        if one_per and mask.sum() > 1:
            raise Exception(
                f"`one_per=True` but we have {mask.sum()} results for: {info}."
            )

        indices.append(np.where(mask)[0])
    return np.concatenate(indices)


def int_to_bin(
    arr: NDArray[Union[str, int]], N: int, dtype: str = "bits"
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
    return np.array(
        [str(format(int(x), f"0{N}b")) for x in arr], dtype=f"{dtype}{N}"
    )


def get_2dhist_invms(
    datum: datum_type, metadatum: NDArray[str], by_invm_bin: bool = False
) -> NDArray[Union[tuple[float, float], NDArray[tuple[float, float]]]]:
    """
    Gets the masses of the final state particles based on what the algorithm
    chose at the most likely bitstring, i.e. if "110000" was chosen, then it
    will separately find the invariant mass of the first two 4-momenta and the
    last 4 4-momenta.

    Parameters:
    datum - The data for this run as created by the `JobRunner` class and saved
        as an .npz file.
    metadatum - An element of the list output by the `parse_data` method.
        Contains the metadata info for a run.
    by_invm_bin (default True) - If True, will keep the array as (M, N, 2) where
        M is the number of invariant mass bins and N is the number of events per
        bin. Otherwise, return as (MN, 2).
    """
    invms = []
    num_bits = datum[2.50]["invm_p4s"].shape[1]
    for invm in INVMS[:-1]:
        # Most likely bitstrings per event
        bitstrings = int_to_bin(np.argmax(datum[invm]["probs"], axis=1), N=num_bits)
        invms.append(
            get_bitstring_invms(evts=datum[invm]["invm_p4s"], bitstrings=bitstrings)
        )

    invms = np.array(invms)

    if by_invm_bin:
        return invms
    return invms.reshape(-1, 2)


def get_bitstrings(datum: datum_type) -> NDArray[str]:
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
            [
                format(val, f"0{N}b")
                for val in np.argmax(datum[invm]["probs"], axis=1)
            ]
        )

    return np.array(bitstrings)
