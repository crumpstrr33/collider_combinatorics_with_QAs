"""
From the data files found in `OUTPUT_DIR`, this file calculates additional data to be
used in analysis. It checks the post data directory, `POSTDATA_DIR`, to see if the
data has already been analysed and only does the calculations that hasn't been done yet.

It calculates:
I) Per Event per Hamiltonian
    1) The Pij and Jij matrices
    2) Lambda values -
        a) QA     = min(Jij) / max(Pij)
        b) avg    = avg(Jij) / avg(Pij)
        c) Pijavg = avg(Pij)
        d) Pijmax = max(Pij)
    3) Quadratic coefficients -
        a) H0 = Jij
        b) H1 = Pij
        c) QA = Jij + 2 * lambda * Pij
    3) Minimum bit string (the bit string that minimizes the Hamiltonians)
    4) Energy for said minimum bit string
II) Per Hamiltonian
    1) Efficiency
III) Per event per data file
    1) Ranking by energy of every bit string
    2) Array of most probable and 2nd most probable bit strings
    3) Energy for every bit string (as dict)
    4) Energy for most probable and 2nd most probable bit string
IV) Per data file
    1) Efficiency

The Hamiltonians are:
    H0 = (P1^2 - P2^2)^2
    H1 = (P1^2 + P2^2)
    QA = H0 + lambda * H1
"""

import pickle
from collections import Counter
from itertools import product
from typing import Optional, Sequence

import numpy as np
from constants import (
    DATA_CHOICES,
    EVENT_CHOICES,
    EVT_DIR,
    IND_DIR,
    INVMS,
    LAMBDA_CHOICES,
    NUM_FSP_DICT,
    POSTDATA_DIR,
    QCL_CHOICES,
    QUADCOEFF_CHOICES,
    SYM_TRUE_BS_DICT,
)
from hemisphere import run_hemisphere
from my_favorite_things import save
from qc_utilities import (
    bit_string_str_combinations,
    format_m4s,
    get_all_bitstring_energies,
    get_all_depths,
    get_all_indices,
    get_bitstring_energy,
    get_coeffs,
    get_files,
    get_info,
    get_Jijs_Pijs,
    get_lambdas,
    get_masses,
    get_minimum_energies,
    get_Nevts,
    swap,
)

# Constants for pretty printing (aligning stuff lol)
estr_max = max([len(e) for e in EVENT_CHOICES])
dstr_max = max([len(d) for d in DATA_CHOICES])

# File structure in output directory
aind_dir = POSTDATA_DIR / "algorithms"


def print_already_exists(name, *paths, tab=0):
    """Ran when file `name` already exists and skips analyzing `paths`."""
    path_str = ""
    for path in paths:
        path_str += f"{path}\n\t"
    print(
        f"{tabs(tab)}{name.capitalize()} file for all events already exists:\n"
        f"{tabs(tab + 1)}{path_str}Skipping saving..."
    )


def tabs(n):
    """More print formatting..."""
    return n * "\t"


def get_qcl_info(qcl):
    """
    `qcl` describes the Hamiltonian. If it is QA == H0 + lmbda*H1, then it also
    describes the lambda type. So this just splits it up.
    """
    qcl_choice = qcl.split("_")
    quadcoeff_choice = qcl_choice[0]
    lambda_choice = qcl_choice[1] if len(qcl_choice) == 2 else None
    return quadcoeff_choice, lambda_choice


def calculate_data(
    etype: str,
    dtype: str,
    savedir: str,
    indices: Optional[Sequence[int]] = None,
    tab: int = 0,
    dryrun: bool = True,
):
    """
    Calculates various values. The `exists_XXXX` booleans tell us if the file
    for that specific data exists. The `do_XXXX` booleans tell us if we need to
    run that data again. E.g. the coefficients need the Jij and Pij matrices
    so they would be `do_JPijs == True` but still have `exists_JPijs == True`.
    Could be quicker by pulling the intermediate data from the existing files
    but this is complicated enough and I don't mind waiting an extra minute or
    two.

    Parameters:
    etype - Event type, as given in `EVENT_CHOICES`
    dtype - Data type, as given in `DATA_CHOICES`
    savedir - Directory to save data to
    indices (default None) - List of indices to index the events. If None is given,
        then use all of the events
    tab (default 0) - Initial offset of tabbing in print statements
    dryrun (default True) - If True, don't actually save data
    """
    # First check to make sure there's an event file to use
    try:
        m4s = format_m4s(np.load(EVT_DIR / f"{etype}_{dtype}.npy"))
        if indices is not None:
            m4s = m4s[indices]
    except FileNotFoundError:
        print(f"{tabs(tab)}There is no event file: '{etype}_{dtype}.npy'\n")
        return

    # Initialize booleans representing whether we need to run the methods to calculate
    # them or not, along with their full path names. Some data requires previous data
    # to calculate, so we go through checks to see what we need to run and what we need
    # to save. Only save data if the file for that data already exists.
    # Hemisphere method data
    exists_hemisphere = False
    hemisphere_path = savedir / f"hemisphere_{etype}_{dtype}.npz"
    # Masses from minimum bit strings
    exists_masses = False
    masses_path = savedir / f"masses_{etype}_{dtype}.npz"
    num_jets_path = savedir / f"num_jets_{etype}_{dtype}.npz"
    # Efficiencies
    exists_efficiencies = False
    effs_path = savedir / f"effs_{etype}_{dtype}.npz"
    effs2_path = savedir / f"effs2_{etype}_{dtype}.npz"
    # Energies
    exists_energies = False
    min_engs_path = savedir / f"min_engs_{etype}_{dtype}.npz"
    min2_engs_path = savedir / f"min2_engs_{etype}_{dtype}.npz"
    min_bss_path = savedir / f"min_bss_{etype}_{dtype}.npz"
    min2_bss_path = savedir / f"min2_bss_{etype}_{dtype}.npz"
    bs_engs_path = savedir / f"bs_engs_{etype}_{dtype}.npz"
    # Quadratic coefficients
    exists_quadcoeff = False
    quadcoeffs_path = savedir / f"quadcoeffs_{etype}_{dtype}.npz"
    # Lambdas
    exists_lambdas = False
    lambdas_path = savedir / f"lambdas_{etype}_{dtype}.npz"
    # Jijs and Pijs
    exists_JPijs = False
    JPijs_path = savedir / f"JijsPijs_{etype}_{dtype}.npz"

    # Check for hemisphere method file
    if hemisphere_path.is_file():
        print_already_exists("hemisphere method", hemisphere_path, tab=tab)
        exists_hemisphere = True
    # Check for masses file
    epaths = [masses_path, num_jets_path]
    if all([epath.is_file() for epath in epaths]):
        print_already_exists("masses", epaths, tab=tab)
        exists_masses = True
    # Check for efficiency files
    epaths = [effs_path, effs2_path]
    if all([epath.is_file() for epath in epaths]):
        print_already_exists("efficiencies", *epaths, tab=tab)
        exists_efficiencies = True
    # Check if any of the energy files exist (since they are all done simultaenously)
    epaths = [min_engs_path, min2_engs_path, min_bss_path, min2_bss_path, bs_engs_path]
    if all([epath.is_file() for epath in epaths]):
        print_already_exists("energies", *epaths, tab=tab)
        exists_energies = True
    # Next check if the quadratic coefficient file exists
    if quadcoeffs_path.is_file():
        print_already_exists("quadratic coefficient", quadcoeffs_path, tab=tab)
        exists_quadcoeff = True
    # Then the lambdas file
    if lambdas_path.is_file():
        print_already_exists("lambdas", lambdas_path, tab=tab)
        exists_lambdas = True
    # And then the Jij/Pij file
    if JPijs_path.is_file():
        print_already_exists("Jij/Pij", JPijs_path, tab=tab)
        exists_JPijs = True

    do_hemisphere = not exists_hemisphere
    do_masses = not exists_masses
    do_quadcoeffs = not exists_quadcoeff
    do_efficiencies = not exists_efficiencies
    do_energies = not exists_energies or (do_efficiencies or do_masses)
    do_lambdas = not exists_lambdas or (do_quadcoeffs or do_energies)
    do_JPijs = not exists_JPijs or (do_lambdas or do_quadcoeffs)

    correct_bs = SYM_TRUE_BS_DICT[etype]
    num_fsp = NUM_FSP_DICT[etype]
    bss = list(bit_string_str_combinations(num_fsp=num_fsp))

    # Jij/Pij are used in all other files, so check if those files need to be made too
    if do_JPijs:
        print(f"{tabs(tab)}Calculating Jijs and Pijs...")
        Jijs, Pijs = get_Jijs_Pijs(m4s)
        if not exists_JPijs:
            save(
                name=JPijs_path.with_suffix("").name,
                savedir=savedir,
                absolute=True,
                dryrun=dryrun,
                Jijs=Jijs,
                Pijs=Pijs,
            )
    else:
        print(f"{tabs(tab)}Skipping calculating Jijs and Pijs...")

    # Lambdas are used by quadratic coefficients and energies files
    if do_lambdas:
        print(f"{tabs(tab)}Calculating lambdas...")
        lambdas_dict = {}
        for lambda_choice in LAMBDA_CHOICES:
            lambdas = get_lambdas(ltype=lambda_choice, Jijs=Jijs, Pijs=Pijs)
            lambdas_dict[lambda_choice] = lambdas

        if not exists_lambdas:
            save(
                name=lambdas_path.with_suffix("").name,
                savedir=savedir,
                absolute=True,
                dryrun=dryrun,
                **lambdas_dict,
            )
    else:
        print(f"{tabs(tab)}Skipping calculating lambdas...")

    # Nothing uses the quadratic coefficients
    if do_quadcoeffs:
        print(f"{tabs(tab)}Calculating quadratic coefficients...")
        quadcoeffs_dict = {}

        # Combine all quadcoeffs with lambdas to list all possible Hamiltonians
        for qcl in QCL_CHOICES:
            quadcoeff_choice, lambda_choice = get_qcl_info(qcl)
            lambdas = lambdas_dict.get(lambda_choice)
            quadcoeffs = get_coeffs(
                htype=quadcoeff_choice, Jijs=Jijs, Pijs=Pijs, lambdas=lambdas
            )
            # H0, H1, QA_QA, QA_avg, QA_Pijavg, QA_Pijmax
            quadcoeffs_dict[qcl] = quadcoeffs

        if not exists_quadcoeff:
            save(
                name=quadcoeffs_path.with_suffix("").name,
                savedir=savedir,
                absolute=True,
                dryrun=dryrun,
                **quadcoeffs_dict,
            )
    else:
        print(f"{tabs(tab)}Skipping calculating quadratic coefficients...")

    # Energies are used by efficiencies
    if do_energies:
        print(f"{tabs(tab)}Calculating energies...")
        min_engs_dict, min2_engs_dict = {}, {}
        min_bss_dict, min2_bss_dict = {}, {}
        bs_eng_dict = {}
        for qcl in QCL_CHOICES:
            quadcoeff_choice, lambda_choice = get_qcl_info(qcl)
            lambdas = lambdas_dict.get(lambda_choice)
            # Doesn't look at symmetric strings, e.g. top two won't be "000111" and
            # "111000"
            min_eng, min2_eng, min_bs, min2_bs = get_minimum_energies(
                htype=quadcoeff_choice, m4s=m4s, lambdas=lambdas
            )
            bs_eng = get_all_bitstring_energies(
                m4s=m4s, bss=bss, htype=quadcoeff_choice, lambdas=lambdas
            )

            min_engs_dict[qcl] = min_eng
            min2_engs_dict[qcl] = min2_eng
            min_bss_dict[qcl] = min_bs
            min2_bss_dict[qcl] = min2_bs
            bs_eng_dict[qcl] = bs_eng
            bs_eng_dict["bss"] = bss

        # Save them individually
        if not exists_energies:
            save(
                name=min_engs_path.with_suffix("").name,
                savedir=savedir,
                absolute=True,
                dryrun=dryrun,
                **min_engs_dict,
            )
            save(
                name=min2_engs_path.with_suffix("").name,
                savedir=savedir,
                absolute=True,
                dryrun=dryrun,
                **min2_engs_dict,
            )
            save(
                name=min_bss_path.with_suffix("").name,
                savedir=savedir,
                absolute=True,
                dryrun=dryrun,
                **min_bss_dict,
            )
            save(
                name=min2_bss_path.with_suffix("").name,
                savedir=savedir,
                absolute=True,
                dryrun=dryrun,
                **min2_bss_dict,
            )
            save(
                name=bs_engs_path.with_suffix("").name,
                savedir=savedir,
                absolute=True,
                dryrun=dryrun,
                **bs_eng_dict,
            )
    else:
        print(f"{tabs(tab)}Skipping calculating energies...")

    # Effeciencies are the final boss
    if do_efficiencies and etype != "6jet":
        print(f"{tabs(tab)}Calculating efficiencies...")
        effs_dict = {}
        for qcl, min_bs in min_bss_dict.items():
            count = np.sum(min_bs == correct_bs)
            tot = len(min_bs)
            effs_dict[qcl] = (count / tot, count, tot)
        effs2_dict = {}
        for qcl, min2_bs in min2_bss_dict.items():
            count = np.sum(min2_bs == correct_bs)
            tot = len(min2_bs)
            effs2_dict[qcl] = (count / tot, count, tot)

        if not exists_efficiencies:
            save(
                name=effs_path.with_suffix("").name,
                savedir=savedir,
                absolute=True,
                dryrun=dryrun,
                **effs_dict,
            )
            save(
                name=effs2_path.with_suffix("").name,
                savedir=savedir,
                absolute=True,
                dryrun=dryrun,
                **effs2_dict,
            )
    else:
        print(f"{tabs(tab)}Skipping calculating efficiencies...")

    # Post-credit cutscene
    if do_masses:
        print(f"{tabs(tab)}Calculating masses...")
        masses_dict = {}
        num_jets_dict = {}
        for qcl, min_bss in min_bss_dict.items():
            m1s, m2s, num_jets0, num_jets1 = get_masses(m4s=m4s, bss=min_bss)
            masses_dict[f"min_{qcl}"] = [m1s, m2s]
            num_jets_dict[f"{qcl}0"] = num_jets0
            num_jets_dict[f"{qcl}1"] = num_jets1
            num_jets_dict[f"{qcl}both"] = num_jets0 + num_jets1

        if etype != "6jet":
            correct_m1s, correct_m2s, _, _ = get_masses(
                m4s=m4s, bss=np.full(len(m4s), correct_bs)
            )
            correct_masses = [correct_m1s, correct_m2s]
        else:
            correct_masses = [None, None]

        if not exists_masses:
            save(
                name=masses_path.with_suffix("").name,
                savedir=savedir,
                absolute=True,
                dryrun=dryrun,
                correct=correct_masses,
                **masses_dict,
            )
            save(
                name=num_jets_path.with_suffix("").name,
                savedir=savedir,
                absolute=True,
                dryrun=dryrun,
                **num_jets_dict,
            )
    else:
        print(f"{tabs(tab)}Skipping calculating masses...")

    # Stuff for the hemisphere method
    if do_hemisphere:
        print(f"{tabs(tab)}Calculating hemisphere method results...")
        # 4-momentum should have each event array flattened
        bss = run_hemisphere(m4s.reshape(len(m4s), -1))
        # Turn from array of floats of 0's and 1's to string of 0's and 1's
        bss = np.array(["".join([str(int(bit)) for bit in bs]) for bs in bss])
        swapped_bss = np.array([swap(bs) for bs in bss])

        m1s, m2s, num_jets0, num_jets1 = get_masses(m4s=m4s, bss=bss)
        num_jetsboth = num_jets0 + num_jets1

        is_correct_bss = np.logical_or(bss == correct_bs, swapped_bss == correct_bs)
        eff = np.sum(is_correct_bss) / len(is_correct_bss)

        if not exists_hemisphere:
            save(
                name=hemisphere_path.with_suffix("").name,
                savedir=savedir,
                absolute=True,
                dryrun=dryrun,
                bss=bss,
                m1s=m1s,
                m2s=m2s,
                num_jets0=num_jets0,
                num_jets1=num_jets1,
                num_jetsboth=num_jetsboth,
                eff=eff,
            )
    else:
        print(f"{tabs(tab)}Skipping calculating hemisphere method results...")
    print()


def calculate_hamiltonian_data(dryrun: bool = True):
    """
    Calculate data for all of each event file, i.e. brute force
    """
    # Directory for all events
    savedir = POSTDATA_DIR / "hamiltonian" / "all_events"
    for dtype, etype in product(DATA_CHOICES, EVENT_CHOICES):
        print(f"{etype.upper()} -- {dtype.upper()}")
        calculate_data(etype=etype, dtype=dtype, savedir=savedir, dryrun=dryrun, tab=1)


def calculate_indexed_data(dryrun: bool = True):
    """
    Calculate data for the indices saved in the index files.
    """
    for ind_dir in IND_DIR.glob("*"):
        print(f"Index directory: {ind_dir.name}")
        etype, dtype, invm, N_evts = get_info(
            ind_dir, ["etype", "dtype", "invm_lo", "N_evts"]
        )
        inds = get_all_indices(ind_dir)
        savedir = POSTDATA_DIR / "hamiltonian" / f"indexed_{invm:.2f}"
        calculate_data(
            etype=etype,
            dtype=dtype,
            savedir=savedir,
            indices=inds,
            dryrun=dryrun,
            tab=1,
        )


def calculate_alg_data(
    alg: str, savedir: str, noise: bool = False, dryrun: bool = True, **noise_kwargs
):
    """
    Calculate data from the output of the various simulations.
    """
    # Loop through every combination of parameters
    for dtype, etype, quadcoeff in product(
        DATA_CHOICES, EVENT_CHOICES, QUADCOEFF_CHOICES
    ):
        # There could be bitflip noise...
        bitflip_prob = noise_kwargs.get("bitflip_prob", "")
        noise_prefix = f"noisy{bitflip_prob}_" if noise else ""
        savename = f"{noise_prefix}{alg}_{etype}_{dtype}_{quadcoeff}"
        correct_bs = SYM_TRUE_BS_DICT[etype]
        num_fsp = NUM_FSP_DICT[etype]
        # Find the values of p we must iterate through
        depths = get_all_depths(
            noise=noise,
            bitflip_prob=bitflip_prob,
            etype=etype,
            dtype=dtype,
            alg=alg,
            quadcoeff=quadcoeff,
        )
        if not depths.size:
            print(
                f"Skipping {etype} events, {dtype} data, {quadcoeff} coefficient..."
                f" Nothing found."
            )
            continue
        if (savedir / savename).with_suffix(".npz").is_file():
            print_already_exists(
                f"{alg.upper()} data", f"min_{etype}_{dtype}_{alg}_{quadcoeff}.npz"
            )
            continue

        effs_dict = {}
        masses_dict = {}
        energies_dict = {}
        skip_depth = False
        # Cycle through each value of p
        for depth in depths:
            print(
                f"[{alg.upper()}] Analyzing {etype:>{estr_max}} events, "
                f"{dtype:>{dstr_max}} data for depth {depth:>{len(str(max(depths)))}}, "
                f"coefficient {quadcoeff}..."
            )
            # Saving lotsa stuff
            correct_effs, correct2_effs = [], []
            min_effs, min2_effs = [], []
            tot_correct_effs, tot_correct2_effs = [], []
            tot_min_effs, tot_min2_effs = [], []
            m1ss, m2ss = [], []
            num_jetss0, num_jetss1, num_jetssboth = [], [], []
            correct_deltaEs, min_deltaEs = [], []
            tot_N_evts = 0
            # Cycle through each invariant mass range
            for invm in INVMS[:-1]:
                # Data from the brute forcing to compare to
                min_bss = np.load(
                    POSTDATA_DIR
                    / "hamiltonian"
                    / f"indexed_{invm:.2f}"
                    / f"min_bss_{etype}_{dtype}.npz"
                )[quadcoeff + ("_QA" if quadcoeff == "QA" else "")]
                min_engs = np.load(
                    POSTDATA_DIR
                    / "hamiltonian"
                    / f"indexed_{invm:.2f}"
                    / f"min_engs_{etype}_{dtype}.npz"
                )[quadcoeff + ("_QA" if quadcoeff == "QA" else "")]

                # Get specific files for simulation
                files = get_files(
                    ntype="file",
                    ftype="eff",
                    noise=noise,
                    bitflip_prob=bitflip_prob,
                    etype=etype,
                    dtype=dtype,
                    alg=alg,
                    quadcoeff=quadcoeff,
                    depth=depth,
                    invm_lo=invm,
                )
                if not files:
                    # lol
                    raise Exception("AHHHH", etype, dtype, alg, quadcoeff, depth, invm)

                N_evts = get_Nevts(files)
                m4s = np.empty((N_evts, num_fsp, 4))
                bss, bss2 = np.empty(N_evts, dtype="U6"), np.empty(N_evts, dtype="U6")
                foffset = 0
                # Cycle through the available files that fit the match
                for fname in sorted(files):
                    with open(fname, "rb") as f:
                        data = pickle.load(f)["data"]

                    # Find most common (highest prob) bit strings (top 2)
                    for ind, datum in enumerate(data):
                        top_two = Counter(datum["sym_probs"]).most_common(2)
                        bss[ind + foffset] = top_two[0][0]
                        bss2[ind + foffset] = top_two[1][0]

                        m4s[ind + foffset] = datum["m4"]
                    foffset += len(data)

                if quadcoeff == "QA":
                    lambdas = get_lambdas(ltype="QA", m4s=m4s)
                else:
                    lambdas = np.ones(len(m4s))

                engs = np.empty(len(m4s))
                correct_engs = np.empty(len(m4s))
                # Find energies of these bitstrings
                for ind, (m4, bs, bs2) in enumerate(zip(m4s, bss, bss2)):
                    engs[ind] = get_bitstring_energy(
                        m4=m4, bs=bs, htype=quadcoeff, lmbda=lambdas[ind]
                    )
                    # And energy of the "correct" bit string, e.g. 000111
                    if correct_bs is not None:
                        correct_engs[ind] = get_bitstring_energy(
                            m4=m4, bs=correct_bs, htype=quadcoeff, lmbda=lambdas[ind]
                        )

                try:
                    min_deltaEs.append(np.abs(engs - min_engs))
                    correct_deltaEs.append(np.abs(engs - correct_engs))
                except ValueError:
                    print(
                        f"Data is not finished. Have {N_evts} events but expect "
                        f"{len(min_engs)} events. Skipping..."
                    )
                    skip_depth = True
                    break

                # Just collect a whole bunch of stuff
                N_evts = len(bss)
                tot_N_evts += N_evts
                if correct_bs is not None:
                    is_correct_bss = bss == correct_bs
                    is_correct2_bss = bss2 == correct_bs
                    correct_effs.append(np.sum(is_correct_bss) / N_evts)
                    correct2_effs.append(np.sum(is_correct2_bss) / N_evts)
                    tot_correct_effs.append(np.sum(is_correct_bss))
                    tot_correct2_effs.append(np.sum(is_correct2_bss))

                is_min_bss = bss == min_bss
                is_min2_bss = bss2 == min_bss
                min_effs.append(np.sum(is_min_bss) / N_evts)
                min2_effs.append(np.sum(is_min2_bss) / N_evts)
                tot_min_effs.append(np.sum(is_min_bss))
                tot_min2_effs.append(np.sum(is_min2_bss))

                m1s, m2s, num_jets0, num_jets1 = get_masses(m4s=m4s, bss=bss)
                m1ss.append(m1s)
                m2ss.append(m2s)
                num_jetss0.append(num_jets0)
                num_jetss1.append(num_jets1)
                num_jetssboth.append(num_jets0 + num_jets1)

            # Skip if job isn't done running yet (i.e. the data ain't all there)
            if skip_depth:
                skip_depth = False
                continue

            # Turn lists into numpy array
            m1ss = np.array(m1ss)
            m2ss = np.array(m2ss)
            num_jetss0 = np.array(num_jetss0)
            num_jetss1 = np.array(num_jetss1)
            num_jetssboth = np.array(num_jetssboth)
            min_deltaEs = np.array(min_deltaEs)
            correct_deltaEs = np.array(correct_deltaEs)

            # Put it all together into dictionaries
            effs_dict[f"min_effs_p{depth}"] = min_effs
            effs_dict[f"min2_effs_p{depth}"] = min2_effs
            effs_dict[f"tot_min_effs_p{depth}"] = sum(tot_min_effs) / tot_N_evts
            effs_dict[f"tot_min2_effs_p{depth}"] = sum(tot_min2_effs) / tot_N_evts
            if correct_bs is not None:
                effs_dict[f"correct_effs_p{depth}"] = correct_effs
                effs_dict[f"correct2_effs_p{depth}"] = correct2_effs
                effs_dict[f"tot_correct_effs_p{depth}"] = (
                    sum(tot_correct_effs) / tot_N_evts
                )
                effs_dict[f"tot_correct2_effs_p{depth}"] = (
                    sum(tot_correct2_effs) / tot_N_evts
                )

            masses_dict[f"m1s_p{depth}"] = m1ss
            masses_dict[f"m2s_p{depth}"] = m2ss
            masses_dict[f"num_jets0_p{depth}"] = num_jetss0
            masses_dict[f"num_jets1_p{depth}"] = num_jetss1
            masses_dict[f"num_jetsboth_p{depth}"] = num_jetssboth
            masses_dict[f"tot_m1s_p{depth}"] = m1ss.flatten()
            masses_dict[f"tot_m2s_p{depth}"] = m2ss.flatten()
            masses_dict[f"tot_num_jets0_p{depth}"] = np.sum(num_jetss0, axis=0)
            masses_dict[f"tot_num_jets1_p{depth}"] = np.sum(num_jetss1, axis=0)
            masses_dict[f"tot_num_jetsboth_p{depth}"] = np.sum(num_jetssboth, axis=0)

            energies_dict[f"min_deltaE_p{depth}"] = min_deltaEs
            energies_dict[f"tot_min_deltaE_p{depth}"] = min_deltaEs.flatten()
            if correct_bs is not None:
                energies_dict[f"correct_deltaE_p{depth}"] = correct_deltaEs
                energies_dict[f"tot_correct_deltaE_p{depth}"] = (
                    correct_deltaEs.flatten()
                )

        # Save it all
        save(
            name=savename,
            savedir=savedir,
            absolute=True,
            dryrun=dryrun,
            **effs_dict,
            **masses_dict,
            **energies_dict,
        )
        print()


def calculate_falqon_data(savedir: str, dryrun: bool = True):
    """
    Similar to calculate_alg_data except more care is needed to be taken since we have
    data from every depth. Could I have abstracted this with the above function?
    Probably, but I'm not made of time and this works fine :)
    """
    for dtype, etype, quadcoeff in product(
        DATA_CHOICES, EVENT_CHOICES, QUADCOEFF_CHOICES
    ):
        # No FALQON + QCD bg stuff
        if etype == "6jet":
            continue

        savename = f"falqon_{etype}_{dtype}_{quadcoeff}"
        correct_bs = SYM_TRUE_BS_DICT[etype]
        depths = get_all_depths(
            etype=etype, dtype=dtype, alg="falqon", quadcoeff=quadcoeff
        )
        if len(depths) == 0:
            print(
                f"Didn't find any files for: {etype} events, {dtype} data and "
                f"Hamiltonian {quadcoeff}."
            )
            continue
        if (savedir / savename).with_suffix(".npz").is_file():
            print_already_exists(
                "FALQON data", f"min_{etype}_{dtype}_falqon_{quadcoeff}.npz"
            )
            continue
        if len(depths) != 1:
            print(f"Found {len(depths)} different depths. Should only be one.")
        N_evts = get_Nevts(
            get_files(
                ntype="file",
                ftype="eff",
                etype=etype,
                dtype=dtype,
                alg="falqon",
                quadcoeff=quadcoeff,
            )
        )

        depth = int(depths[0])
        tot_N_evts = 0

        effs_dict = {}
        masses_dict = {}
        energies_dict = {}

        min_deltaEs = []
        correct_deltaEs = []
        min_effs = []
        correct_effs = []
        tot_min_effs = np.zeros((len(INVMS) - 1, depth))
        tot_correct_effs = np.zeros((len(INVMS) - 1, depth))
        m1ss = np.empty((depth, len(INVMS) - 1, N_evts))
        m2ss = np.empty((depth, len(INVMS) - 1, N_evts))
        num_jetss0 = np.empty((depth, len(INVMS) - 1, len(correct_bs) + 1))
        num_jetss1 = np.empty((depth, len(INVMS) - 1, len(correct_bs) + 1))
        num_jetssboth = np.empty((depth, len(INVMS) - 1, len(correct_bs) + 1))
        for invm_ind, invm in enumerate(INVMS[:-1]):
            print(
                f"[FALQON] Analyzing {etype:>{estr_max}} events, "
                f"{dtype:>{dstr_max}} data for invm {invm:>.2f}, "
                f"coefficient {quadcoeff}..."
            )
            min_bss = np.load(
                POSTDATA_DIR
                / "hamiltonian"
                / f"indexed_{invm:.2f}"
                / f"min_bss_{etype}_{dtype}.npz"
            )[quadcoeff + ("_QA" if quadcoeff == "QA" else "")]
            min_engs = np.load(
                POSTDATA_DIR
                / "hamiltonian"
                / f"indexed_{invm:.2f}"
                / f"min_engs_{etype}_{dtype}.npz"
            )[quadcoeff + ("_QA" if quadcoeff == "QA" else "")]
            min_engs = np.full((depth, *min_engs.shape), min_engs)
            files = get_files(
                ntype="file",
                ftype="eff",
                etype=etype,
                dtype=dtype,
                alg="falqon",
                quadcoeff=quadcoeff,
                invm_lo=invm,
            )
            if not files:
                raise Exception("AHHHH", etype, dtype, quadcoeff, invm)

            m4s = []
            bss = []
            for fname in files:
                with open(fname, "rb") as f:
                    data = pickle.load(f)["data"]

                for ind, datum in enumerate(data):
                    m4s.append(datum["m4"])
                    bss.append(datum["sym_depth_bss"])
            m4s = np.array(m4s)
            # Turn array from (N_evts, p) to (p, N_evts) so to iterate through by depth
            bss = np.array(bss).T

            lambdas = (
                get_lambdas("QA", m4s=m4s) if quadcoeff == "QA" else np.ones(N_evts)
            )

            engs = []
            correct_engs = []
            # DIFFERENCE HERE! We cycle through the depth and check the state of the
            # algorithm at that point.
            for p in range(depth):
                # Hence the "depth_" prefix used
                depth_engs = []
                depth_correct_engs = []
                for m4, bs, lmbda in zip(m4s, bss[p], lambdas):
                    depth_engs.append(
                        get_bitstring_energy(
                            m4=m4, bs=bs, htype=quadcoeff, lmbda=lambdas[ind]
                        )
                    )
                    depth_correct_engs.append(
                        get_bitstring_energy(
                            m4=m4, bs=correct_bs, htype=quadcoeff, lmbda=lmbda
                        )
                    )
                # Save the per-depth data here
                engs.append(depth_engs)
                correct_engs.append(depth_correct_engs)
            engs = np.array(engs)
            correct_engs = np.array(correct_engs)

            min_deltaEs.append(np.abs(engs - min_engs))
            correct_deltaEs.append(np.abs(engs - correct_engs))

            tot_N_evts += N_evts
            is_correct_bss = np.sum(bss == correct_bs, axis=1)
            is_min_bss = np.sum(bss == min_bss, axis=1)

            correct_effs.append(is_correct_bss / N_evts)
            min_effs.append(is_min_bss / N_evts)

            tot_correct_effs[invm_ind] += is_correct_bss
            tot_min_effs[invm_ind] += is_min_bss

            for ind, depth_bss in enumerate(bss):
                m1s, m2s, num_jets0, num_jets1 = get_masses(m4s=m4s, bss=depth_bss)
                m1ss[ind][invm_ind] = m1s
                m2ss[ind][invm_ind] = m2s
                num_jetss0[ind][invm_ind] = num_jets0
                num_jetss1[ind][invm_ind] = num_jets1
                num_jetssboth[ind][invm_ind] = num_jets0 + num_jets1

        min_deltaEs = np.array(min_deltaEs)
        correct_deltaEs = np.array(correct_deltaEs)
        min_effs = np.array(min_effs)
        correct_effs = np.array(correct_effs)

        effs_dict["min_effs"] = min_effs.T
        effs_dict["correct_effs"] = correct_effs.T
        effs_dict["tot_min_effs"] = np.sum(tot_min_effs, axis=0) / tot_N_evts
        effs_dict["tot_correct_effs"] = np.sum(tot_correct_effs, axis=0) / tot_N_evts

        masses_dict["m1s"] = m1ss
        masses_dict["m2s"] = m2ss
        masses_dict["num_jets0"] = num_jetss0
        masses_dict["num_jets1"] = num_jetss1
        masses_dict["num_jetsboth"] = num_jetssboth
        masses_dict["tot_m1s"] = m1ss.reshape(depth, -1)
        masses_dict["tot_m2s"] = m2ss.reshape(depth, -1)
        masses_dict["tot_num_jets0"] = np.sum(num_jetss0, axis=1)
        masses_dict["tot_num_jets1"] = np.sum(num_jetss1, axis=1)
        masses_dict["tot_num_jetsboth"] = np.sum(num_jetssboth, axis=1)

        min_deltaEs = np.transpose(min_deltaEs, axes=[1, 0, 2])
        correct_deltaEs = np.transpose(correct_deltaEs, axes=[1, 0, 2])
        energies_dict["min_deltaE"] = min_deltaEs
        energies_dict["tot_min_deltaE"] = min_deltaEs.reshape(depth, -1)
        energies_dict["correct_deltaE"] = correct_deltaEs
        energies_dict["tot_correct_deltaE"] = correct_deltaEs.reshape(depth, -1)

        save(
            name=savename,
            savedir=savedir,
            absolute=True,
            dryrun=dryrun,
            **effs_dict,
            **masses_dict,
            **energies_dict,
        )


if __name__ == "__main__":
    calculate_hamiltonian_data(False)
    # calculate_indexed_data(False)
    # calculate_alg_data("qaoa", aind_dir, dryrun=False)
    # calculate_alg_data("maqaoa", aind_dir, dryrun=False)
    # calculate_alg_data("xqaoa", aind_dir, dryrun=False)
    # calculate_alg_data("maqaoa", aind_dir, noise=True, bitflip_prob=0.1, dryrun=False)
    # calculate_alg_data("maqaoa", aind_dir, noise=True, bitflip_prob=1, dryrun=False)
    # calculate_alg_data("maqaoa", aind_dir, noise=True, bitflip_prob=10, dryrun=False)

    # calculate_falqon_data(aind_dir, False)

    # calculate_hamiltonian_data(False)
    # calculate_indexed_data(False)
