from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from my_favorite_things import save

from .constants import EVT_DIR, INVMS, NUM_FSP_DICT
from .events import get_invms
from .type_hints import evts_type


def draw_random_events(
    N: int,
    invm_bins: Optional[Sequence[tuple[float]]] = None,
    etype: str = "ttbar",
    dtype: str = "parton",
    prec_bins: int = 500,
) -> evts_type:
    """
    From the original `.npy` file of madevent data, randomly chooses events and
    saves them in a (N, N_fsp, 4) numpy array as a `.npz` file where N is the
    number of events and N_fsp is the number of final state particles. The
    invariant masses distribution of the events isn't uniform, so we use the
    inverse CDF to create a uniform distribution of invariant masses. We choose
    iteratively from the invariant mass bins and then join them all together
    into one big array.

    Parameters:
    N - Number of events per invariant mass bin.
    invm_bins (default None) - If None, the bins are from the `INVMS` constant,
        i.e. [1.00. 1.25], then [1.25, 1.50], then [1.50, 1.75], etc. Otherwise,
        the bins can be specified as a list of tuples.
    etype (default "ttbar") - The event type, currently can be "ttbar", "tW" or
        "6jet".
    dtype (default "parton") - The data type, currently can be "parton" or
        "smeared".
    prec_bins (default 500) - The number of bins used to split up the invariant
        masses to give a probability to each event of randomly choosing it. A
        higher probability means that the event's invariant mass is in a bin
        with a small number of events (such that those events are boosted to
        create the uniform distribution). Too small and the uniform distribution
        is less approximated and too large, there aren't enough events per bin.
    """
    evts = np.load(EVT_DIR / f"{etype}_{dtype}.npy")
    N_fsp = NUM_FSP_DICT[etype]
    shape = evts.shape
    if len(shape) == 2:
        # Not divisible by 4, assume last element is not part of 4-momentum
        if shape[1] // 4:
            # Remove last element and reshape as (N_fsp, 4)
            evts = evts[:, :-1].reshape(-1, N_fsp, 4)
        # Is divisible by 4, so just reshape
        else:
            evts = evts.reshape(-1, N_fsp, 4)
    # Get invariant masses
    invms = get_invms(evts=evts, etype=etype)

    invm_bins = np.vstack((INVMS[:-1], INVMS[1:])).T if invm_bins is None else invm_bins
    tot_evts = []
    for invm_lo, invm_hi in invm_bins:
        # Create mask based on upper and lower limits
        invm_mask = np.logical_and(invms > invm_lo, invms < invm_hi)
        # Filter events
        evts_filt = evts[invm_mask]
        # We can choose without replacement if there aren't enough events
        if len(evts_filt) < N:
            raise Exception(
                f"[{invm_lo:.2f} - {invm_hi:.2f}] -- there are {len(evts_filt)}"
                f" events, but we are asking for {N} events. Choose a number"
                "less than the total number of events."
            )
        # Filter invariant masses
        invms_filt = invms[invm_mask]
        counts, bins = np.histogram(invms_filt, bins=prec_bins)
        # Get indices for each invm of which bin it is in as per `bins`
        inds = np.clip(np.searchsorted(bins, invms_filt), 0, prec_bins - 1)
        # Probability for each invariant mass value with offset for Db0 errors
        probs = 1 / (counts[inds] + 0.00001)
        probs /= probs.sum()
        # res = np.random.choice(invms, N, p=probs)
        rand_choice = np.random.choice(evts_filt.shape[0], N, p=probs)
        # return invms[rand_choice], evts[rand_choice]
        tot_evts.append(evts_filt[rand_choice])

    return np.concatenate(tot_evts)


if __name__ == "__main__":
    # Call this file from root directory as: python -m scripts.choose_inds
    num_evts_per_bin = 2000
    etype = "ttbar"
    dtype = "parton"
    evts = draw_random_events(N=num_evts_per_bin, etype=etype, dtype=dtype)
    # Just for testing
    is_dryrun = False

    # Save events to data directory in root
    print(Path(__file__))
    save(
        name=f"{etype}_{dtype}_events",
        absolute=True,
        # Root directory
        savepath=Path(__file__).parents[1],
        savedir="data",
        stype="npz",
        overwrite=False,
        append=False,
        dryrun=is_dryrun,
        evts=evts,
    )
