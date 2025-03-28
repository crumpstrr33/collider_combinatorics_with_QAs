from itertools import combinations

import numpy as np
from skhep.math.vectors import LorentzVector


def init_assign(jetlist):
    remaining_jets = np.arange(len(jetlist))
    assignment = np.zeros(len(jetlist))
    # seeding2
    mass = 0
    max_index = [0, 0]
    for i in combinations(range(len(jetlist)), 2):
        mass_new = (jetlist[i[0]] + jetlist[i[1]]).m
        if mass_new > mass:
            mass = mass_new
            max_index = i
    j1 = jetlist[max_index[0]]
    assignment[max_index[0]] = 1
    j2 = jetlist[max_index[1]]
    remaining_jets = np.delete(remaining_jets, max_index)

    for i in remaining_jets:
        jet = jetlist[i]
        if lund_distance(j1, jet) < lund_distance(j2, jet):
            assignment[i] = 1
    for i in remaining_jets:
        jet = jetlist[i]
        if assignment[i]:
            j1 += jet
        else:
            j2 += jet

    return assignment, [j1, j2]


def lund_distance(jet1, jet2):
    cos = jet1.vector.cosdelta(jet2.vector)
    return (jet1.e - jet1.p * cos) * jet1.e / (jet1.e + jet2.e) ** 2


def assign_group(jetlist, axes):
    index = np.arange(len(jetlist))
    assignment = np.zeros(len(jetlist))
    j1, j2 = axes
    for i in index:
        jet = jetlist[i]
        if lund_distance(j1, jet) < lund_distance(j2, jet):
            assignment[i] = 1
    for i in index:
        jet = jetlist[i]
        if assignment[i]:
            j1 = j1 + jet
        else:
            j2 = j2 + jet

    return assignment, [j1, j2]


def hemisphere(event):
    jetlist = []
    for i in range(event.shape[0] // 4):
        jetlist.append(
            LorentzVector(
                event[1 + i * 4], event[2 + i * 4], event[3 + i * 4], event[0 + i * 4]
            )
        )

    assigned, axes = init_assign(jetlist)
    k = False
    while not k:
        assigned_new, axes = assign_group(jetlist, axes)
        if (assigned_new == assigned).all():
            k = True
        else:
            assigned = assigned_new
    return assigned


def run_hemisphere(events):
    pairing = []
    for event in events:
        pairing.append(hemisphere(event))
    return np.array(pairing)
