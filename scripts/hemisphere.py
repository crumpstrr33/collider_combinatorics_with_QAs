from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .type_hints import EvtsType, EvtType


def get_energy(evts: EvtType | EvtsType) -> NDArray[np.floating]:
    """
    For each momentum in each event, returns the 0th component. Can take in one
    event or an array of events.
    """
    return evts[..., 0]


def get_magnitude(evts: EvtType | EvtsType) -> NDArray[np.floating]:
    """
    For each momentum in each event, calculates the magnitude of the spatial
    components. Can take in one event or an array of events.
    """
    return np.sqrt(np.sum(evts[..., 1:] ** 2, axis=-1))


def get_mass(evts: EvtType | EvtsType) -> NDArray[np.floating]:
    """
    For each momentum in each event, find the mass from that momentum. Can take
    in one event or an array of events.
    """
    # Calculate mass squared
    msq = evts[..., 0] ** 2 - np.sum(evts[..., 1:] ** 2, axis=-1)
    msq = np.where(np.isclose(msq, 0, atol=1e-1), 0, msq)

    # Make sure we have no negative mass squared
    if np.any(msq < 0):
        neg_msqs = msq[np.where(msq < 0)]
        raise ValueError(f"Found a negative mass squared. Values: {neg_msqs}")

    return np.sqrt(msq)


def get_spatial_dot(
    evts1: EvtType | EvtsType, evts2: EvtType | EvtsType
) -> NDArray[np.floating]:
    """
    For each momentum for each event, calculates the dot product of the spatial
    components. Can take in one event or an array of events for either input.
    If one input is an array of events and the other is a single event, it will
    dot the single event with every event of the other input. Otherwise, it will
    go element-wise.
    """
    return np.sum(evts1[..., 1:] * evts2[..., 1:], axis=-1)


def get_cosine(
    evts1: EvtType | EvtsType, evts2: EvtType | EvtsType
) -> NDArray[np.floating]:
    """
    For each momentum for each event, calculates the cosine of the angle between
    the spatial vectors. Can take in one event or an array of events for either
    input. If one input is an array of events and the other is a single event,
    it will calculate the cosine of the single event with every event of the
    other input. Otherwise, it will go element-wise.
    """
    dot = get_spatial_dot(evts1, evts2)
    magnitude1 = get_magnitude(evts1)
    magnitude2 = get_magnitude(evts2)

    return dot / (magnitude1 * magnitude2 + 1e-12)


def get_lund(
    evts1: EvtType | EvtsType, evts2: EvtType | EvtsType
) -> NDArray[np.floating]:
    """
    For each momentum for each event, calculates the Lund distance for each
    4-momentum. Can take in one event or an array of events for either input. If
    one input is an array of events and the other is a single event, it will
    calculate the cosine of the single event with every event of the other
    input. Otherwise, it will go element-wise.
    """
    cosine = get_cosine(evts1, evts2)
    energy1 = get_energy(evts1)
    energy2 = get_energy(evts2)
    magnitude1 = get_magnitude(evts1)

    return (energy1 - magnitude1 * cosine) * energy1 / (energy1 + energy2) ** 2


def make_evttype(arr: NDArray[Any]) -> NDArray[Any]:
    """
    For an array of shape either (p,) or (N, p) will turn it into either
    (1, 1, p)  or (N, 1, p), respectively. The first case is a single 4-momentum
    becoming 1 event with 1 particle. The second case is N events with a single
    4-momenta each giving each event an explicit dimension for the 4-momenta.
    That is, an EvtType is (N, M, P)-shaped where:
        N is the number of events.
        M is the number of particles/4-momenta
        P == 4 is the 4-momenta components
    """
    if len(arr.shape) > 2:
        return arr
    return np.swapaxes(np.atleast_3d(arr), 1, 2)


class Hemisphere:
    def __init__(
        self, evts: EvtType | EvtsType, max_iter: int = 20, tol: float = 1e-6
    ) -> None:
        """ """
        # Carbon copy of the input
        self.evts = evts
        # Add a singlet dimension in the front to make it the correct shape
        self.n_evts = 1 if len(evts.shape) == 2 else len(evts)
        self._evts = evts[np.newaxis, ...] if self.n_evts == 1 else evts
        # Number of jets per event
        self.n_jets = evts.shape[-2]
        # Number of iterations of algorithm to do before giving up
        self.max_iter = max_iter
        # Tolerance (of Euclidean distance) for saying two axes are the same
        self.tol = tol

        # Mask to keep track of which events have converged
        self.converged = np.zeros(self.n_evts, dtype=bool)
        self.arrangement = np.zeros((self.n_evts, self.n_jets), dtype=int)

    def run(self) -> None:
        # Try different seeding methods for events that don't converge
        methods = ["mass", "pt"]
        for method in methods:
            self.init_assigns(method=method)
            self._iterate()
            if self.converged.all():
                break

        # Rewrite bit arrays as strings
        self.arrangement = np.array(
            ["".join(row.astype(str)) for row in self.arrangement]
        )

    def init_assigns(self, method: Literal["mass", "pt"]) -> None:
        """
        Does the initial assignment of particles. It seeds the axes based on the
        jet pair with the largest value according to `method`.

        Parameters:
        method: Method to use to seed the axes. Can be:
            "mass": pair with the highest invariant mass,
            "pt": pair with the highest total transverse momenta
        """
        if method == "mass":
            # Sums every pair of 4-momenta per event.
            # If `evts.shape == (..., n, 4)`, then
            #               `jet_pairs.shape == (..., n, n, 4)`
            # where the n by n matrix of 4-momenta is symmetric. The index
            # (i, j, k, l) represents, for the ith event, the sum of the lth
            # components of the jth and kth 4-momenta. If `evts` is a single
            # event, i.e. has shape (n, 4), then the above index i doesn't exist
            # but it still works (due to the ellipses).
            jet_pairs = (
                self._evts[..., :, np.newaxis, :] + self._evts[..., np.newaxis, :, :]
            )
            # Turns each 4-momenta into its mass: (..., n, n, 4) --> (..., n, n)
            seed_scores = get_mass(jet_pairs)
        elif method == "pt":
            # Find the transverse momenta for every jet
            pt = np.sqrt(self._evts[..., 1] ** 2 + self._evts[..., 2] ** 2)
            # Sum every pair of pt
            seed_scores = pt[..., :, np.newaxis] + pt[..., np.newaxis, :]

        # Get all possible pairs of indices for the masses
        jet_pair_inds = np.triu_indices(self.n_jets, k=1)
        # Index of max invariant mass per event. The shape evolution goes as
        #               (..., n, n) --> (..., nC2) --> (...,)
        max_ind = np.argmax(seed_scores[..., *jet_pair_inds], axis=-1)
        # Index for the jets that give the largest invariant mass/pt: (2, ...)
        best_inds = (jet_pair_inds[0][max_ind], jet_pair_inds[1][max_ind])

        # Create the two hemispheres as (N, 1, 4) where, if n_evts == 1, then
        # the first dimension (number of events) is 1 instead of not existing.
        # The second dimension would be the number of 4-momenta per event, i.e.
        # there's only one since it's the hemisphere.
        self.axis0 = make_evttype(self._evts[np.arange(self.n_evts), best_inds[0]])
        self.axis1 = make_evttype(self._evts[np.arange(self.n_evts), best_inds[1]])

    def _iterate(self) -> None:
        """
        Method to iterate over axes for `self.max_iter` number of times. Only
        events that haven't converged will be affected.
        """
        prev_axis0 = self.axis0.copy()
        prev_axis1 = self.axis1.copy()

        for ind in range(self.max_iter):
            # End loop if everything has converged
            if self.converged.all():
                break

            # Events we are still iterating
            active = ~self.converged
            active_evts = self._evts[active]

            l0 = get_lund(self.axis0, self._evts)
            l1 = get_lund(self.axis1, self._evts)
            # Boolean mask for jets going to first jet based on Lund distance
            jet0_mask = (l0 < l1)[active]

            # Add new dimension to account fo the components of the 4-momenta
            ext_mask = jet0_mask[..., np.newaxis]
            # Sum together all the jets according to the mask
            self.axis0[active] = (active_evts * ext_mask).sum(axis=1, keepdims=True)
            self.axis1[active] = (active_evts * ~ext_mask).sum(axis=1, keepdims=True)
            self.arrangement[active] = jet0_mask.astype(int)

            # How much have the axes changed since last iteration?
            delta0 = np.linalg.norm(self.axis0 - prev_axis0, axis=-1).squeeze()
            delta1 = np.linalg.norm(self.axis1 - prev_axis1, axis=-1).squeeze()

            # Consider an event converged if both axes don't change much at all
            # from a previous event
            new_converged = (delta0 < self.tol) & (delta1 < self.tol) & active
            self.converged |= new_converged

            prev_axis0 = self.axis0.copy()
            prev_axis1 = self.axis1.copy()


def run_hemisphere(evts):
    """
    Runs the hemisphere method on an array of events of shape (..., Njet, 4)
    and returns the resultant bitstrings in an array of shape (...,).
    """
    hemi = Hemisphere(evts.reshape(-1, *evts.shape[-2:]))
    hemi.run()
    return hemi.arrangement.reshape(*evts.shape[:-2])
