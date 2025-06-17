from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .constants import (
    INVMS,
    NUM_FSP_DICT,
)
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
    data: NDArray[datum_type], assume_symmetric: bool = True
) -> NDArray[NDArray[float]]:
    """
    From the given data arrays, finds the efficiency. That is, it finds the
    fraction of events for which the algorithm found the "correct" bitstring.

    Parameters:
    data - An array of data for runs as created by the `JobRunner` class and
        saved as an .npz file.
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
            ranks = datum[invm]["ranks"]

            # Calculate the efficiency with possibility of symmetry
            datum_effs.append(
                np.sum((ranks - (ranks % (2 if assume_symmetric else 1))) == 0)
                / len(ranks)
            )
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
