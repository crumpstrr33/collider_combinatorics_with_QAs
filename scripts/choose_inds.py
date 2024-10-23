"""
Randomly chooses indices for events and saves those indices. Specify by invariant mass
range (via `mtts_lims`), which event data to use (via `dtype`). Saves with a unique
3-digit identifier and the 3-digit integer used for the numpy RNG. Also specify how to
cut it up. For example, can decide to cut up 1000 events in 10 different files (for
parallel use) so the first file has events 1-100, the second has events 101-200 and so
on.
"""

from typing import Optional, Sequence

import numpy as np
from constants import EVT_DIR, IND_DIR
from my_favorite_things import save
from qc_utilities import format_m4s


class IndexMaker:
    def __init__(
        self,
        mtts_lims: Sequence[tuple[float, float]],
        num_splits_choices: Sequence[int],
        dtype: str,
        etype: str,
        N: int,
        test: bool,
        error: bool = False,
        rngn: Optional[int] = None,
    ):
        """
        Create files containing indices references events in a given event file
        specified by `etype` and `dtype`, i.e. {etype}_{dtype}.npy

        Parameters
        mtts_lims - List of tuples representing the limits on the invariant mass to
            create an index file by, i.e. all indices in the file will point to
            events that have an invariant mass between this range. It is a list,
            so multiple index files can be created at once.
        num_split_choices - Each element of the list represents how many files to
            generate. If `N=1000` and `num_split_choices=[1, 5]`, it will create a
            file with 1000 events and another 5 each with 200 events (per invariant
            mass range)
        dtype - What the data type is, e.g. parton or smeared
        etype - The event type, e.g. ttbar or tW
        N - Total number of events (indices)to save
        test - If these files are to be test files, change the save name accordingly
        error (default False) - If True, it will error out if there aren't enough
            total events create the files.
        rngn (default None) - The seed number for the Numpy RandomState object.
        """
        self.mtts_lims = mtts_lims
        self.num_splits_choices = num_splits_choices
        self.dtype = dtype
        self.etype = etype
        self.N = N
        self.test = test
        self.error = error

        # Int representing the RNG used
        self.rngn = rngn or np.random.randint(1, 1000)
        self.rng = np.random.RandomState(self.rngn)
        # Random unique identifier
        self.uid = str(self.rng.randint(0, 1000)).zfill(3)

    def run(self, print_it: bool = True) -> None:
        """
        Loops through the invariant mass ranges and runs the methods to create the index
        files for each range.

        Parameters:
        print_it (default True) - If True will print each instance of a save.
        """
        for mtt_lims in self.mtts_lims:
            if print_it:
                if mtt_lims != mtts_lims[0]:
                    print()
                print(f"Running for mass range: {mtt_lims[0]:.2f} -- {mtt_lims[1]:.2f}")
            self.loop_and_save(mtt_lo=mtt_lims[0], mtt_hi=mtt_lims[1])

    def loop_and_save(self, mtt_lo: float, mtt_hi: float) -> None:
        """
        Loop ran to generate files.

        Parameters:
        mtt_lo - Lower limit on the invariant mass
        mtt_hi - Upper limit on the invariant mass
        """
        # Grab random indices within mass range
        chosen_inds = self.choose_inds(
            mtt_lo=mtt_lo,
            mtt_hi=mtt_hi,
        )
        for num_splits in self.num_splits_choices:
            # split up into `num_splits` equal-as-possible sized arrays to save separately
            split_lim_inds = np.linspace(0, N, num_splits + 1, dtype=int)
            splits = [
                (split_lim_inds[i], split_lim_inds[i + 1]) for i in range(num_splits)
            ]
            for split in splits:
                split_inds = chosen_inds[split[0] : split[1]]
                # ID to identify which set of indices this is and seed for numpy's RNG
                ids = f"{self.uid.zfill(3)}x{str(self.rngn).zfill(3)}"
                # The range of the total set of indices
                inds = f"{split[0] + 1:0>{len(str(self.N))}}to{split[1]:0>{len(str(self.N))}}"
                mtt = f"{mtt_lo:.2f}to{mtt_hi:.2f}"

                dtype = "test" if self.test else self.dtype
                save(
                    name=f"inds_{self.etype}_{dtype}_{ids}_{mtt}_{inds}",
                    savedir=IND_DIR / f"inds{self.N}_{self.etype}_{dtype}_{ids}_{mtt}",
                    stype="npz",
                    absolute=True,
                    parents=0,
                    inds=split_inds,
                )

    def choose_inds(self, mtt_lo: float, mtt_hi: float) -> Sequence[int]:
        """
        Returns an array of N random indices for events within the invariant mass range.

        Parameters:
        mtt_lo - Lower limit on the invariant mass
        mtt_hi - Upper limit on the invariant mass

        Returns (
            Numpy array of `self.N` random ints
        )
        """
        # Get all 4-momenta
        fpath = EVT_DIR / f"{self.etype}_{self.dtype}.npy"
        m4s = np.load(str(fpath))

        # Reshape data properly
        _, _, mtts = format_m4s(m4s, return_extra=True)

        # Filter on mtt
        evt_inds = np.where(np.logical_and(mtts > mtt_lo, mtts < mtt_hi))[0]

        # Then randomly choose `N` of them
        # If we have less than N to choose from, just use them all
        if self.N > len(evt_inds):
            if self.error:
                raise IndexError(
                    f"**** ERROR: Asking for N={N}, "
                    + f"but there are only {len(evt_inds)} events... ****",
                ) from None

            print(
                f"**** WARNING: Asking for N={N}, "
                + f"but there are only {len(evt_inds)} events... ****",
                end=f"\n{' '*4}",
            )
            self.N = len(evt_inds)
        return self.rng.choice(evt_inds, self.N, replace=False)


if __name__ == "__main__":
    # Total number of events
    N = 2_000
    is_test_file = False
    # all the different number of splits you want, e.g. if [1, 5]
    # make 1 file of all events and 5 files that split it evenly
    # Numbers: [2000, 1000, 500, 200, 100]
    # num_splits_choices = [1, 2, 3, 6, 12, 30, 60]  # For 6,000
    num_splits_choices = [1, 2, 4, 10, 20]  # For 2,000
    # Name of ttbar file to be used
    dtype = "parton"
    # Event type
    etype = "ttbar"
    # Mass ranges
    mtts_lims = [
        (1.0, 1.25),
        (1.25, 1.5),
        (1.5, 1.75),
        (1.75, 2.0),
        (2, 2.5),
        (2.5, 3.0),
    ]

    print(f"Total number of events: {N}")
    print(f"Split choices: {', '.join([str(x) for x in num_splits_choices])}")
    print(f"Event data set: {dtype}\n")

    indexer = IndexMaker(
        mtts_lims=mtts_lims,
        num_splits_choices=num_splits_choices,
        dtype=dtype,
        etype=etype,
        N=N,
        test=is_test_file,
        error=True,
    )
    indexer.run()
