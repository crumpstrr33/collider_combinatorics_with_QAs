"""
Runs events for finite number of shots. Yeah, this is a lot like efficiency.py
but that thing is large and I didn't feel like properly incorporating the
functionality of finite shots into that, so I wrote this which is simpler
and thus much less error prone.
"""

from argparse import ArgumentParser

import numpy as np
from constants import EVT_DIR, IND_DIR, SHOT_DIR
from pennylane_algs import MAQAOA, QAOA
from qc_utilities import format_p4s, get_coeffs, get_Jijs_Pijs, get_lambdas

DATA_PATH = EVT_DIR / "ttbar_parton.npy"

# Can't use XQAOA rn because `correct_prob` assumes the bitstring symmetry
ALGS = {"QAOA": QAOA, "MAQAOA": MAQAOA}


def get_data(ham):
    """
    Get coefficients for the Hamiltonian, e.g. the Jij matrix for H0.
    """
    p4s = np.load(DATA_PATH)
    p4s = format_p4s(p4s=p4s)
    Jijs, Pijs = get_Jijs_Pijs(p4s=p4s)
    lambdas = get_lambdas(ltype="QA", p4s=p4s, Jijs=Jijs, Pijs=Pijs)
    coeffs = get_coeffs(htype=ham, p4s=p4s, Jijs=Jijs, Pijs=Pijs, lambdas=lambdas)

    return coeffs


def get_inds(indlo, indhi, invm, etype="ttbar", dtype="parton", Nevt=2000):
    """
    Get appropriate event indices
    """
    invm_dir = list(IND_DIR.glob(f"inds{Nevt}_{etype}_{dtype}_*_{invm:.2f}to*"))
    assert len(invm_dir) == 1
    invm_dir = invm_dir[0]

    ind_file = list(
        invm_dir.glob(
            f"inds_{etype}_{dtype}_*_{invm:.2f}to*_{indlo:0>4}to{indhi:0>4}.npz"
        )
    )
    assert len(ind_file) == 1
    ind_file = ind_file[0]

    return np.load(ind_file)["inds"]


def do_event(coeff_matrix, shots, depth, steps, ALG, indlo, indhi, ind):
    """
    Runs a single event and returns placement - e.g. is the correct bitstring
    the most likely, 2nd most, etc - the probability of the correct bitstring,
    the values of the cost function and the probabilities of all bitstrings
    """
    device = "default.qubit"
    shots = None if shots == 0 else shots
    num_evts = indhi - indlo + 1
    print_pref = f"[{shots}] {ind + 1} / {num_evts} ({indlo} -- {indhi}) "

    alg = ALG(
        coeff_matrix,
        depth=depth,
        steps=steps,
        shots=shots,
        optimizer="adam",
        device=device,
        prec=1e-99,
    )
    alg.optimize(print_it=True, print_pref=print_pref)
    probs = alg.get_probs(as_dict=True)

    sorted_probs = np.array(sorted(probs.items(), key=lambda x: x[1], reverse=True))
    sorted_bss = list(sorted_probs[:, 0])
    sorted_probs = list(sorted_probs[:, 1])

    placement = 1 + min(sorted_bss.index("000111"), sorted_bss.index("111000"))
    correct_prob = 2 * float(sorted_probs[placement])

    return placement, correct_prob, alg.costs, probs


def run_events(coeff_matrices, shots, depth, steps, ALG, invm, indlo, indhi):
    """
    Iteratates over the events, running `do_event` for each one.
    """
    placements, correct_probs, costss, all_probs = [], [], [], []
    for ind, coeff_matrix in enumerate(coeff_matrices):
        # run event for given coefficient matrix
        placement, correct_prob, costs, probs = do_event(
            coeff_matrix=coeff_matrix,
            shots=shots,
            depth=depth,
            steps=steps,
            ALG=ALG,
            indlo=indlo,
            indhi=indhi,
            ind=ind,
        )
        placements.append(placement)
        correct_probs.append(correct_prob)
        costss.append(costs)
        all_probs.append(list(probs.items()))

    # Save data
    save_dir = (
        SHOT_DIR
        / f"shots_p{depth}_steps{steps}_{ALG.__name__.lower()}"
        / f"invm{invm:.2f}"
    )
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    ind_str = f"{indlo:0>4}to{indhi:0>4}"
    np.savez(
        save_dir / f"shots{shots}_{ind_str}.npz",
        placements=placements,
        correct_probs=correct_probs,
        costss=costss,
        all_probs=all_probs,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hamiltonian", type=str, choices=["H0", "H1", "QA"])
    parser.add_argument("--algorithm", type=str, choices=["QAOA", "MAQAOA", "XQAOA"])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--indlo", type=int)
    parser.add_argument("--indhi", type=int)
    parser.add_argument("--invm", type=float, choices=[1, 1.25, 1.5, 1.75, 2, 2.5])

    args = parser.parse_args()
    ham = args.hamiltonian
    alg = args.algorithm
    steps = args.steps
    depth = args.depth
    indlo = args.indlo
    indhi = args.indhi
    invm = args.invm

    coeff_matrices = get_data(ham=ham)
    inds = get_inds(indlo=indlo, indhi=indhi, invm=invm)
    coeff_matrices = coeff_matrices[inds]

    shotss = [0, 10, 25, 50, 100, 250, 500]

    for shots in shotss:
        print(f"Shots: {shots}", flush=True)
        run_events(
            coeff_matrices=coeff_matrices,
            shots=shots,
            depth=depth,
            steps=steps,
            ALG=ALGS[alg.upper()],
            invm=invm,
            indlo=indlo,
            indhi=indhi,
        )
