from datetime import datetime as dt
from itertools import combinations, product
from math import asin, sqrt
from multiprocessing import Pool
from random import choice

import numpy
import pennylane as qml
from pennylane import numpy as np
from scipy.optimize import minimize
from scipy.special import comb

from .qc_utilities import swap


class VQA:
    """
    Base class for the VQAs. For an algorithm like QAOA, it has two lists of parameters:
    gammas and betas (and e.g. XQAOA has three with the additional alphas). `params` is
    a list of each list of parameters and `self.param_shapes` is a list of the numpy
    shape for each list of parameters.
    """

    OPTIMIZERS = {
        "grad_descent": qml.GradientDescentOptimizer,
        "adagrad": qml.AdagradOptimizer,
        "adam": qml.AdamOptimizer,
    }

    def __init__(
        self,
        Jij,
        depth,
        steps=None,
        shots=None,
        prec=1e-4,
        optimizer="adam",
        opt_kwargs={},
        device="default.qubit",
        bitflip_prob=0,
    ):
        """
        In the __init__ for inheriting classes:
            1) Define the cost and mixer Hamiltonians using self.edges/self.weights for
            the cost Hamiltonian and self.vertices for the mixer Hamiltonian which are
            used in your definition for `layer`.
            2) Define the array shapes for the lists of parameters as a list defined as
            `self.param_shapes`.

        Parameters:
        Jij - Ising NxN matrix created from 4-momenta defined in 2111.07806.
        depth - How many layers in the circuit, written as `p`.
        steps - How many steps to run the optimizer.
        shots - Number of shots for the circuit, If None, use exact statevector, e.g.
            an infinite number of shots.
        prec - How precise the optimizer should be. That is the optimizer won't stop
            until the difference between the latest two evaluations are less that `prec`
            or `steps` number of evaluations have been completed. That is, which ever
            happens first.
        optimizer - Which optimizer to use: "grad_descent", "adagrad" or "adam" for the
            gradient descent, adagrad or adam optimizer.
        opt_kwargs - A dictionary of keyword arguments to pass to the optimizer object.
        device - The pennylane device to run on.
        """
        self._Jij = Jij
        self.max_Jij = np.max(Jij)
        self.Jij = self._Jij / self.max_Jij
        self.depth = depth
        self.steps = steps
        self.shots = shots
        self.prec = prec
        self.N = len(Jij)
        self.bitflip_prob = bitflip_prob

        self.device = qml.device(device, wires=self.N, shots=self.shots)
        self.str_opt = optimizer
        self.optimizer = self.OPTIMIZERS[optimizer](**opt_kwargs)

        self.bit_strs = numpy.array(
            ["".join(bs) for bs in product(["0", "1"], repeat=self.N)]
        )
        self.vertices = numpy.arange(self.N)
        self.edges = numpy.array(list(combinations(range(self.N), r=2)))
        self.weights = numpy.array([self.Jij[edge[0], edge[1]] for edge in self.edges])

        # Expectation value operator: problem Hamiltonian
        self.expval_op = qml.Hamiltonian(
            self.weights,
            [qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1]) for edge in self.edges],
        )

    def layer(self, *params):
        """
        Each layer of the VQA. Each arg from `params` is a list of parameters, e.g. QAOA
        has two: gammas and betas but XQAOA has three: gammas, betas and alphas.
        """
        pass

    def circuit(self, *params):
        """
        Variational circuit for given parameters.
        """
        # Initialize with Hadamards
        for ind in range(self.N):
            qml.Hadamard(wires=ind)
        qml.Barrier()

        # Add layers
        qml.layer(self.layer, self.depth, *params)

    def _cost_circuit(self, *params):
        """
        Define `self.expval_op` in inheriting class's __init__. It is the total cost
        function, usually problem/cost Hamiltonian.
        """
        self.circuit(*params)
        return qml.expval(self.expval_op)

    def _probs_circuit(self, *params):
        """
        Circuit for getting probabilities for each eigenstate.
        """
        self.circuit(*params)
        return qml.probs()

    def get_probs(self, as_dict=False):
        """
        Gets the probabilities for each eigenstate.
        """
        probs_qnode = qml.QNode(self._probs_circuit, self.device)
        probs = probs_qnode(*self.params)

        if as_dict:
            return dict(zip(self.bit_strs, probs))
        return probs

    def optimize(
        self,
        init_params=None,
        print_it=False,
        print_pref="",
        print_newlines=False,
        init_val=0.5,
    ):
        """
        Runs the optimization for given initial parameters. The shapes of those
        parameters should defined in a list as `self.param_shapes` in the __init__
        """
        self.params = init_params
        if self.params is None:
            self.params = [init_val * np.ones(shape) for shape in self.param_shapes]

        # Make sure the shapes are correct
        for ind, shape in enumerate(self.param_shapes):
            if self.params[ind].shape != shape:
                raise TypeError(
                    f"Index {ind} of keyword `init_params` should be of shape "
                    + f"{shape}, not {self.params[ind]}"
                )

        # Create QNode (a la the decorator way)
        if self.bitflip_prob != 0:
            noisy_device = qml.transforms.insert(
                self.device, op=qml.BitFlip, op_args=self.bitflip_prob
            )
            self.cost_qnode = qml.QNode(func=self._cost_circuit, device=noisy_device)
        else:
            self.cost_qnode = qml.QNode(func=self._cost_circuit, device=self.device)

        # Run through the steps of the optimizing
        self.costs = np.empty(self.steps)
        self.evals = self.steps
        start = dt.now()
        for ind in range(self.steps):
            # Save the value of the cost per step too
            self.params, self.costs[ind] = self.optimizer.step_and_cost(
                self.cost_qnode, *self.params
            )
            if ind:
                self.current_prec = abs(1 - self.costs[ind - 1] / self.costs[ind])
                if print_it:
                    print(
                        f"{print_pref}"
                        f"[{self.__class__.__name__}] {self.str_opt} > "
                        f"Calculating step {ind + 1}/{self.steps}..."
                        f" Time: {(dt.now() - start).total_seconds():.3f}"
                        f" | Current precision: {self.current_prec:.3e}",
                        end="\n" if print_newlines else "\r",
                        flush=True,
                    )
                if self.current_prec < self.prec:
                    # Trim off extra part of numpy array
                    self.costs = self.costs[: ind + 1]
                    self.evals = ind + 1
                    break
            else:
                if print_it:
                    print(end="\r", flush=True)
        self.params = [param for param in self.params]
        if print_it:
            print("\nDone!")


class QAOA(VQA):
    def __init__(
        self,
        Jij,
        depth,
        steps,
        shots=None,
        prec=1e-6,
        optimizer="grad_descent",
        opt_kwargs={},
        device="default.qubit",
        bitflip_prob=0,
    ):
        super().__init__(
            Jij=Jij,
            depth=depth,
            steps=steps,
            shots=shots,
            prec=prec,
            optimizer=optimizer,
            opt_kwargs=opt_kwargs,
            device=device,
            bitflip_prob=bitflip_prob,
        )

        self.param_shapes = ((self.depth,), (self.depth,))

    def layer(self, gamma, beta):
        # Create Pauli rotation gates
        for edge, weight in zip(self.edges, self.weights):
            qml.PauliRot(gamma * weight, "ZZ", wires=edge)
        qml.Barrier()
        for ind in range(self.N):
            qml.PauliRot(beta, "X", wires=ind)
        qml.Barrier()


class DQAOA:
    def __init__(
        self,
        Jij,
        depth,
        steps,
        num_iterations,
        num_constants,
        alg="qaoa",
        prec=1e-6,
        optimizer="grad_descent",
        opt_kwargs={},
        device="default.qubit",
        num_core_err=20,
    ):
        self.Jij = Jij
        self.depth = depth
        self.steps = steps
        self.alg = alg
        self.prec = prec
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs
        self.device = device

        # How many times to run sub-QAOAs
        self.num_iterations = num_iterations
        # How many bits to ignore in sub-QAOAs
        self.num_constants = num_constants
        # Length of global bit string
        self.size = self.Jij.shape[0]
        # Length of sub-bit string/size (number of qubits) of sub-QAOA
        self.subsize = self.size - self.num_constants
        # How many cores to use
        self.num_cores = self.num_constants + 1
        # Just in case, we don't wanna slam the computer with too many cores
        if self.num_cores > num_core_err:
            raise Exception(
                f"Number of cores to be used is {self.num_cores} which is greater than"
                f"the limit {num_core_err}. If this is ok, set `num_core_err` to a"
                "greater number than `num_constants` + 1."
            )

        # How many sub-qaoa iterations we've done
        self.subiter_count = 0
        # Global bitstring
        self.global_bs = list(numpy.random.choice([0, 1], self.size))
        self.init_global_bs = self.global_bs
        # self.global_bs = [0] * self.size
        self.global_energy = self.calculate_energy(self.global_bs)
        # Histories
        self.energy_history = [self.global_energy]
        self.bs_history = [self.global_bs]
        self.iter_history = [0]
        # self.total_energy_history = [self.global_energy]
        # self.total_bs_history = [self.global_bs]

    def create_sub_alg(self, Jij):
        match self.alg:
            case "qaoa":
                Alg = QAOA
            case "maqaoa":
                Alg = MAQAOA
            case "xqaoa":
                Alg = XQAOA

        alg = Alg(
            Jij=Jij,
            depth=self.depth,
            steps=self.steps,
            prec=self.prec,
            optimizer=self.optimizer,
            opt_kwargs=self.opt_kwargs,
            device=self.device,
        )
        return alg

    def get_most_likely_bitstrings(self, alg):
        alg.optimize()
        probs_dict = alg.get_probs(as_dict=True)
        # Gets the highest probability bit string
        top_bs = sorted(probs_dict.items(), reverse=True, key=lambda val: val[1])[0][0]

        return top_bs, swap(top_bs)

    def compare_energies(self, bit_inds, new_bits, iter_ind):
        new_energies, new_bitstrings, new_iters = [], [], []
        for bit_ind, bit in zip(bit_inds, new_bits):
            new_bs = (
                self.global_bs[:bit_ind] + [int(bit)] + self.global_bs[bit_ind + 1 :]
            )
            new_energy = self.calculate_energy(new_bs)

            if new_energy < self.global_energy:
                self.global_energy = new_energy
                old_bs_str = "".join([str(b) for b in self.global_bs])
                new_bs_str = "".join([str(b) for b in new_bs])
                # self.global_bs = new_bs
                print(f"    Global: {old_bs_str} --> {new_bs_str}")
                new_energies.append(self.global_energy)
                new_bitstrings.append(new_bs)
                new_iters.append(iter_ind)
                # self.energy_history.append(self.global_energy)
                # self.bs_history.append(self.global_bs)
                # self.iter_history.append(iter_ind)

        return new_energies, new_bitstrings, new_iters

    def calculate_energy(self, bitstring):
        s = [2 * bit - 1 for bit in bitstring]
        return s @ self.Jij @ s

    def do_iteration(self, iter_bits, iter_ind):
        # Create smaller weight matrix without the constant bits
        reduced_Jij = self.Jij[iter_bits][:, iter_bits]
        # Create algorithm class with said weightmatrix
        alg = self.create_sub_alg(Jij=reduced_Jij)
        # Run algorithm and find most likely bitstrings (there's 2)
        bitstrings = self.get_most_likely_bitstrings(alg=alg)
        # Randomly choose one of them (for now)
        bitstring = choice(bitstrings)

        if self.print_it:
            sub_bs_str = ["x"] * self.size
            for bit_ind, bit in zip(iter_bits, bitstring):
                sub_bs_str[bit_ind] = bit
            sorted_inds = numpy.argsort(iter_bits)
            print(
                f"  Bitstring to check: {''.join(sub_bs_str)} | "
                f"{''.join(numpy.array(list(bitstring))[sorted_inds])} -- "
                f"{', '.join([str(x) for x in iter_bits[sorted_inds]])}"
            )

        # Replace each bit with the sub-QAOA solution and check energy
        new_energies, new_bitstrings, new_iters = self.compare_energies(
            bit_inds=iter_bits, new_bits=bitstring, iter_ind=iter_ind
        )

        return new_energies, new_bitstrings, new_iters

    def create_sub_bits(self):
        sub_bits = numpy.empty(
            (self.num_iterations, self.num_cores, self.subsize), dtype=int
        )
        # init_bits = sliding_window_view(numpy.arange(self.size), window_shape=self.subsize)
        for iter_ind in range(self.num_iterations):
            for core_ind in range(self.num_cores):
                bit_choice = numpy.random.choice(
                    range(self.size), self.subsize, replace=False
                )
                sub_bits[iter_ind][core_ind] = bit_choice

        return sub_bits

    def run_pool(self, print_it=False):
        self.sub_bits = self.create_sub_bits()
        self.print_it = print_it

        for iter_ind, iter_bits in enumerate(self.sub_bits):
            if self.print_it:
                if iter_ind:
                    print()
                print(
                    f"  -----   {iter_ind + 1}/{self.num_iterations}   -----  ",
                    flush=True,
                )
            with Pool(self.num_cores) as pool:
                new_data = pool.starmap(
                    self.do_iteration, zip(iter_bits, [iter_ind] * self.num_cores)
                )
                new_energies = [n for new_datum in new_data for n in new_datum[0]]
                new_bitstrings = [n for new_datum in new_data for n in new_datum[1]]
                new_iters = [n for new_datum in new_data for n in new_datum[2]]

                if new_energies:
                    min_ind = np.argmin(new_energies)

                    if new_energies[min_ind] < self.global_energy:
                        old_bitstring = self.global_bs
                        old_energy = self.global_energy
                        self.global_bs = new_bitstrings[min_ind]
                        self.global_energy = new_energies[min_ind]
                        self.energy_history.append(self.global_energy)
                        self.bs_history.append(self.global_bs)
                        self.iter_history.append(new_iters)

                        if self.print_it:
                            old_bs_str = "".join([str(x) for x in old_bitstring])
                            g_bs_str = "".join([str(x) for x in self.global_bs])
                            print(f"New bitstring: {old_bs_str} --> {g_bs_str}")
                            print(
                                f"New energy: {old_energy:.4f} --> {self.global_energy}"
                            )


class MAQAOA(VQA):
    """
    Instead of a free parameter per layer for the cost Hamiltonian and the mixer
    Hamiltonian, each gate has its own free parameter. So for a depth of `p` and N
    qubits, there are (NChoose2 + N)p free parameters (instead of just 2p).
    See: https://www.nature.com/articles/s41598-022-10555-8
    """

    def __init__(
        self,
        Jij,
        depth,
        steps,
        shots=None,
        prec=1e-6,
        optimizer="grad_descent",
        opt_kwargs={},
        device="default.qubit",
        bitflip_prob=0,
    ):
        super().__init__(
            Jij=Jij,
            depth=depth,
            steps=steps,
            shots=shots,
            prec=prec,
            optimizer=optimizer,
            opt_kwargs=opt_kwargs,
            device=device,
            bitflip_prob=bitflip_prob,
        )
        # comb(N, 2) == N choose 2 edges between N vertices in maximally connected graph
        self.param_shapes = ((self.depth, int(comb(self.N, 2))), (self.depth, self.N))

    def layer(self, gammas, betas):
        # Each gate gets its own beta/gamma parameter
        for edge, weight, gamma in zip(self.edges, self.weights, gammas):
            qml.PauliRot(gamma * weight, "ZZ", wires=edge)
        qml.Barrier()
        for ind, beta in zip(range(self.N), betas):
            qml.PauliRot(beta, "X", wires=ind)
        qml.Barrier()


class NO_WEIGHT_MAQAOA(MAQAOA):
    """
    Exactly the same as MAQAOA except the weight, i.e. the Jij matrix, does not multiple
    the gamma parameters. This ~shouldn't~ make a difference since the gamma paramters
    are independent and gate-specific and thus the value of Jij should be able to be
    absorbed into the R_ZZ^ij gate parameter.
    """

    def layer(self, gammas, betas):
        # Each gate gets its own beta/gamma parameter
        for edge, gamma in zip(self.edges, gammas):
            qml.PauliRot(gamma, "ZZ", wires=edge)
        qml.Barrier()
        for ind, beta in zip(range(self.N), betas):
            qml.PauliRot(beta, "X", wires=ind)
        qml.Barrier()


class HYBRID_MAQAOA(VQA):
    """
    A ma-QAOA-like mixer layer and a QAOA-like problem layer.
    """

    def __init__(
        self,
        Jij,
        depth,
        steps,
        shots=None,
        prec=1e-6,
        optimizer="grad_descent",
        opt_kwargs={},
        device="default.qubit",
        bitflip_prob=0,
    ):
        super().__init__(
            Jij=Jij,
            depth=depth,
            steps=steps,
            shots=shots,
            prec=prec,
            optimizer=optimizer,
            opt_kwargs=opt_kwargs,
            device=device,
            bitflip_prob=bitflip_prob,
        )
        self.param_shapes = ((self.depth,), (self.depth, self.N))

    def layer(self, gamma, betas):
        # Each gate gets its own beta/gamma parameter
        for edge, weight in zip(self.edges, self.weights):
            qml.PauliRot(gamma * weight, "ZZ", wires=edge)
        qml.Barrier()
        for ind, beta in zip(range(self.N), betas):
            qml.PauliRot(beta, "X", wires=ind)
        qml.Barrier()


class XQAOA(VQA):
    """
    eXpressive QAOA adds to MA-QAOA by adding an addtional mixer layer of Pauli Y
    gates each with their own free parameter (like with MA-QAOA).
    See: https://arxiv.org/pdf/2302.04479.pdf
    """

    def __init__(
        self,
        Jij,
        depth,
        steps,
        shots=None,
        prec=1e-6,
        optimizer="grad_descent",
        opt_kwargs={},
        device="default.qubit",
        bitflip_prob=0,
    ):
        super().__init__(
            Jij=Jij,
            depth=depth,
            steps=steps,
            shots=shots,
            prec=prec,
            optimizer=optimizer,
            opt_kwargs=opt_kwargs,
            device=device,
            bitflip_prob=bitflip_prob,
        )
        # comb(N, 2) == N choose 2 edges between N vertices in maximally connected graph
        self.param_shapes = (
            (self.depth, int(comb(self.N, 2))),
            (self.depth, self.N),
            (self.depth, self.N),
        )

    def layer(self, gammas, betas, alphas):
        # Each gate gets its own beta/gamma parameter
        for edge, weight, gamma in zip(self.edges, self.weights, gammas):
            qml.PauliRot(gamma * weight, "ZZ", wires=edge)
        qml.Barrier()
        for ind, beta in zip(range(self.N), betas):
            qml.PauliRot(beta, "X", wires=ind)
        for ind, alpha in zip(range(self.N), alphas):
            qml.PauliRot(alpha, "Y", wires=ind)
        qml.Barrier()


class FALQON:
    """
    Feedback-based ALgorithm for Quantum OptimizatioN (FALQON) builds its circuit
    iteratively by calculating the per-layer parameter via the expectation value
    of i[Hm, Hc] thus removing the need for optimization/backpropagation.
    See: https://arxiv.org/pdf/2103.08619.pdf.

    Since the parameters are fixed once found, this class calculates the state for
    layer n, then initializes the the next layer with that state so the (n+1)th layer
    only needs to run a circuit of 1 layer deep.
    """

    def __init__(
        self, Jij, depth, dt=0.08, init_beta=0, shots=None, device="default.qubit"
    ):
        self._Jij = Jij
        self.max_Jij = np.max(Jij)
        # Keep the quadratic coefficient normalized
        self.Jij = self._Jij / self.max_Jij
        self.depth = depth
        self.shots = shots
        # Number of final state particles in problem
        self.N = len(Jij)

        self.device = qml.device(device, wires=self.N)

        # Stuff for weighting gates
        self.bit_strs = numpy.array(
            ["".join(bs) for bs in product(["0", "1"], repeat=self.N)]
        )
        self.vertices = numpy.arange(self.N)
        self.edges = numpy.array(list(combinations(range(self.N), r=2)))
        self.weights = numpy.array([self.Jij[edge[0], edge[1]] for edge in self.edges])

        # FALQON-specific parameters
        self.dt = dt
        self.init_beta = init_beta

        # List of parameters
        self.betas = []
        # List of probabilities per depth
        self.depth_probs = []
        # Temporarily stores the newest parameter
        self._new_beta = init_beta
        # Intial state to equal superposition (Hadamards on all qubits)
        self._cur_state = sqrt(1 / 2**self.N) * np.ones(2**self.N)

        # Commutator i[Hc, Hm] whose expval gives parameter of next layer
        comm_gates, comm_weights = [], []
        for i in range(self.N):
            for j, k in combinations(range(self.N), r=2):
                # [Xi, wjk*ZjZk] = -2iwjk(δijYiZk + δikZjYi)
                # delta_ij
                if i == j:
                    comm_weights.append(2 * self.Jij[j, k])
                    comm_gates.append(qml.PauliY(i) @ qml.PauliZ(k))
                # delta_jk
                elif i == k:
                    comm_weights.append(2 * self.Jij[j, k])
                    comm_gates.append(qml.PauliZ(j) @ qml.PauliY(i))
        self.commutator = qml.Hamiltonian(comm_weights, comm_gates)

        # Expectation value: problem Hamiltonian
        self.expval_op = qml.Hamiltonian(
            self.dt * self.weights,
            [qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1]) for edge in self.edges],
        )

    def _state_circuit(self):
        """
        Finds the output state of the current circuit. Saved in `self._cur_state` and
        used to initalize the next layer.
        """
        self.circuit(self._new_beta)
        return qml.state()

    def _probs_circuit(self):
        """
        Circuit to return the probabilities for each eigenstate.
        """
        self.circuit()
        return qml.probs()

    def _param_circuit(self):
        """
        Circuit to both minimize the cost function (i.e. the expectation value of
        `self.expval_op` with respect to the circuit) and the next parameter value.
        """
        self.circuit(self._new_beta)
        return qml.expval(self.expval_op), qml.expval(self.commutator)

    def circuit(self, beta=None):
        """
        Actual circuit to run.
        """
        # For an n layer circuit, initialize the first n-1 layers
        qml.QubitStateVector(self._cur_state, wires=range(self.N))

        # If you just want to run the circuit as is (without adding another layer)
        if beta is not None:
            # Then add the nth layer
            for edge, weight in zip(self.edges, self.weights):
                qml.PauliRot(self.dt * weight, "ZZ", wires=edge)
            for ind in range(self.N):
                qml.PauliRot(self.dt * beta, "X", wires=ind)

    def get_probs(self, as_dict=False):
        """
        Gets the probabilities for each eigenstate.
        """
        probs_qnode = qml.QNode(self._probs_circuit, self.device)
        probs = probs_qnode()
        if as_dict:
            return dict(zip(self.bit_strs, probs))
        return probs

    def run(self, print_it=False):
        """`
        Run the FALQON algorithm
        """
        # For other algorithms, they stop when cost function stops decreasing at a rate
        # below some threshold. We don't have that here, so there will a number of
        # evaluations equal to the depth of the circuit
        self.evals = self.depth
        self.costs = numpy.empty(self.depth)

        for ind in range(self.depth):
            if print_it:
                print(
                    f"[FALQON] > Calculating depth {ind + 1}/{self.depth}...",
                    end="\r",
                )

            # Save value of parameter
            self.betas.append(self._new_beta)
            # Get the current cost and new parameter values
            param_qnode = qml.QNode(self._param_circuit, self.device)
            cost, beta = param_qnode()
            # Get the current state of circuit
            state_qnode = qml.QNode(self._state_circuit, self.device)

            self.costs[ind] = cost
            # Save the probabilities for each depth
            self.depth_probs.append(self.get_probs(True))
            # Save state and parameter for next layer
            self._cur_state = state_qnode()
            self._new_beta = -beta

        self.betas = numpy.array(self.betas)
        # To put in same format as with the parameters for other algorithms
        self.params = [self.betas]

        if print_it:
            print("\nDone!")


class WSQAOA(VQA):
    """
    Warm start QAOA. There are more complicated variations of this algorithm but this
    one initializes the circuit with a continuous estimation of the solution via
    classical means. This then changes the form of the mixer Hamiltonian/layer.

    See https://arxiv.org/pdf/2009.10095.pdf.
    """

    def __init__(
        self,
        Jij,
        depth,
        steps,
        eps=0.5,
        prec=1e-6,
        optimizer="grad_descent",
        opt_kwargs={},
        device="default.qubit",
        bitflip_prob=0,
        method="COBYLA",
    ):
        super().__init__(
            Jij=Jij,
            depth=depth,
            steps=steps,
            prec=prec,
            optimizer=optimizer,
            opt_kwargs=opt_kwargs,
            device=device,
            bitflip_prob=bitflip_prob,
        )

        self.param_shapes = ((self.depth,), (self.depth,))

        self.eps = eps
        self.initial_guess = self.find_estimated_continuous_solns(method=method)
        # self.initial_guess = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
        self.get_angles()

    def find_estimated_continuous_solns(self, method):
        def func(x):
            energy = 0
            for i, j in product(range(self.N), repeat=2):
                energy += self.Jij[i, j] * (x[i] * x[j] - x[i])
            return energy

        self.x0 = np.random.uniform(0, 1, self.N)
        res = minimize(
            fun=func,
            x0=self.x0,
            bounds=[[0, 1]] * self.N,
            method="COBYLA",
        )
        return res.x

    def get_angles(self):
        self.thetas = []
        for ind, init_param in enumerate(self.initial_guess):
            if init_param <= self.eps:
                self.thetas.append(2 * asin(sqrt(self.eps)))
            elif init_param > 1 - self.eps:
                self.thetas.append(2 * asin(sqrt(1 - self.eps)))
            else:
                self.thetas.append(2 * asin(sqrt(init_param)))

    def circuit(self, *params):
        """
        Rewrite for different state initialization.
        """
        # Initialize with Hadamards
        for ind, theta in enumerate(self.thetas):
            qml.RY(theta, wires=ind)
        qml.Barrier()

        # Add layers
        qml.layer(self.layer, self.depth, *params)

    def layer(self, gamma, beta):
        # Create Pauli rotation gates
        for edge, weight in zip(self.edges, self.weights):
            qml.PauliRot(gamma * weight, "ZZ", wires=edge)
        qml.Barrier()
        for ind, theta in enumerate(self.thetas):
            qml.RY(-theta, wires=ind)
            qml.RZ(2 * beta, wires=ind)
            qml.RY(theta, wires=ind)
        qml.Barrier()
