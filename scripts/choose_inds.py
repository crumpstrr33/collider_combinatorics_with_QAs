# """
# Randomly chooses indices for events and saves those indices. Specify by invariant mass
# range (via `invms_lims`), which event data to use (via `dtype`). Saves with a unique
# 3-digit identifier and the 3-digit integer used for the numpy RNG. Also specify how to
# cut it up. For example, can decide to cut up 1000 events in 10 different files (for
# parallel use) so the first file has events 1-100, the second has events 101-200 and so
# on.
# """

# from typing import Optional, Sequence

# import numpy as np
# from constants import EVT_DIR, IND_DIR
# from my_favorite_things import save
# from qc_utilities import format_p4s


# class IndexMaker:
#     def __init__(
#         self,
#         invms_lims: Sequence[tuple[float, float]],
#         num_splits_choices: Sequence[int],
#         dtype: str,
#         etype: str,
#         N: int,
#         test: bool,
#         error: bool = False,
#         rngn: Optional[int] = None,
#     ):
#         """
#         Create files containing indices references events in a given event file
#         specified by `etype` and `dtype`, i.e. {etype}_{dtype}.npy

#         Parameters
#         invms_lims - List of tuples representing the limits on the invariant mass to
#             create an index file by, i.e. all indices in the file will point to
#             events that have an invariant mass between this range. It is a list,
#             so multiple index files can be created at once.
#         num_split_choices - Each element of the list represents how many files to
#             generate. If `N=1000` and `num_split_choices=[1, 5]`, it will create a
#             file with 1000 events and another 5 each with 200 events (per invariant
#             mass range)
#         dtype - What the data type is, e.g. parton or smeared
#         etype - The event type, e.g. ttbar or tW
#         N - Total number of events (indices)to save
#         test - If these files are to be test files, change the save name accordingly
#         error (default False) - If True, it will error out if there aren't enough
#             total events create the files.
#         rngn (default None) - The seed number for the Numpy RandomState object.
#         """
#         self.invms_lims = invms_lims
#         self.num_splits_choices = num_splits_choices
#         self.dtype = dtype
#         self.etype = etype
#         self.N = N
#         self.test = test
#         self.error = error

#         # Int representing the RNG used
#         self.rngn = rngn or np.random.randint(1, 1000)
#         self.rng = np.random.RandomState(self.rngn)
#         # Random unique identifier
#         self.uid = str(self.rng.randint(0, 1000)).zfill(3)

#     def run(self, print_it: bool = True) -> None:
#         """
#         Loops through the invariant mass ranges and runs the methods to create the index
#         files for each range.

#         Parameters:
#         print_it (default True) - If True will print each instance of a save.
#         """
#         for invm_lims in self.invms_lims:
#             if print_it:
#                 if invm_lims != invms_lims[0]:
#                     print()
#                 print(
#                     f"Running for mass range: {invm_lims[0]:.2f} -- {invm_lims[1]:.2f}"
#                 )
#             self.loop_and_save(invm_lo=invm_lims[0], invm_hi=invm_lims[1])

#     def loop_and_save(self, invm_lo: float, invm_hi: float) -> None:
#         """
#         Loop ran to generate files.

#         Parameters:
#         invm_lo - Lower limit on the invariant mass
#         invm_hi - Upper limit on the invariant mass
#         """
#         # Grab random indices within mass range
#         chosen_inds = self.choose_inds(
#             invm_lo=invm_lo,
#             invm_hi=invm_hi,
#         )
#         for num_splits in self.num_splits_choices:
#             # split up into `num_splits` equal-as-possible sized arrays to save separately
#             split_lim_inds = np.linspace(0, N, num_splits + 1, dtype=int)
#             splits = [
#                 (split_lim_inds[i], split_lim_inds[i + 1]) for i in range(num_splits)
#             ]
#             for split in splits:
#                 split_inds = chosen_inds[split[0] : split[1]]
#                 # ID to identify which set of indices this is and seed for numpy's RNG
#                 ids = f"{self.uid.zfill(3)}x{str(self.rngn).zfill(3)}"
#                 # The range of the total set of indices
#                 inds = f"{split[0] + 1:0>{len(str(self.N))}}to{split[1]:0>{len(str(self.N))}}"
#                 invm = f"{invm_lo:.2f}to{invm_hi:.2f}"

#                 dtype = "test" if self.test else self.dtype
#                 save(
#                     name=f"inds_{self.etype}_{dtype}_{ids}_{invm}_{inds}",
#                     savedir=IND_DIR / f"inds{self.N}_{self.etype}_{dtype}_{ids}_{invm}",
#                     stype="npz",
#                     absolute=True,
#                     parents=0,
#                     inds=split_inds,
#                 )

#     def choose_inds(self, invm_lo: float, invm_hi: float) -> Sequence[int]:
#         """
#         Returns an array of N random indices for events within the invariant mass range.

#         Parameters:
#         invm_lo - Lower limit on the invariant mass
#         invm_hi - Upper limit on the invariant mass

#         Returns (
#             Numpy array of `self.N` random ints
#         )
#         """
#         # Get all 4-momenta
#         fpath = EVT_DIR / f"{self.etype}_{self.dtype}.npy"
#         p4s = np.load(str(fpath))

#         # Reshape data properly
#         _, _, invms = format_p4s(p4s, return_extra=True)

#         # Filter on invm
#         evt_inds = np.where(np.logical_and(invms > invm_lo, invms < invm_hi))[0]

#         # Then randomly choose `N` of them
#         # If we have less than N to choose from, just use them all
#         if self.N > len(evt_inds):
#             if self.error:
#                 raise IndexError(
#                     f"**** ERROR: Asking for N={N}, "
#                     + f"but there are only {len(evt_inds)} events... ****",
#                 ) from None

#             print(
#                 f"**** WARNING: Asking for N={N}, "
#                 + f"but there are only {len(evt_inds)} events... ****",
#                 end=f"\n{' ' * 4}",
#             )
#             self.N = len(evt_inds)
#         return self.rng.choice(evt_inds, self.N, replace=False)


# if __name__ == "__main__":
#     # Total number of events
#     N = 2_000
#     is_test_file = False
#     # all the different number of splits you want, e.g. if [1, 5]
#     # make 1 file of all events and 5 files that split it evenly
#     # Numbers: [2000, 1000, 500, 200, 100]
#     # num_splits_choices = [1, 2, 3, 6, 12, 30, 60]  # For 6,000
#     num_splits_choices = [1, 2, 4, 10, 20]  # For 2,000
#     # Name of ttbar file to be used
#     dtype = "parton"
#     # Event type
#     etype = "ttbar"
#     # Mass ranges
#     invms_lims = [
#         (1.0, 1.25),
#         (1.25, 1.5),
#         (1.5, 1.75),
#         (1.75, 2.0),
#         (2, 2.5),
#         (2.5, 3.0),
#     ]

#     print(f"Total number of events: {N}")
#     print(f"Split choices: {', '.join([str(x) for x in num_splits_choices])}")
#     print(f"Event data set: {dtype}\n")

#     indexer = IndexMaker(
#         invms_lims=invms_lims,
#         num_splits_choices=num_splits_choices,
#         dtype=dtype,
#         etype=etype,
#         N=N,
#         test=is_test_file,
#         error=True,
#     )
#     indexer.run()

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
    etype = "tW"
    dtype = "parton"
    evts = draw_random_events(N=num_evts_per_bin, etype=etype, dtype=dtype)

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
        dryrun=False,
        evts=evts,
    )
