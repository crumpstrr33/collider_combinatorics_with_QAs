from datetime import datetime as dt
from itertools import combinations, product
from math import sqrt

import numpy as np
import pennylane as qml
from pennylane import numpy as qmlnp
from scipy.special import comb


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
        coeff,
        depth,
        steps=None,
        shots=None,
        prec=1e-8,
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
        coeff - Matrix of coefficients for s_i * s_j spin coupling.
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
        self._coeff = coeff
        self.max_coeff = np.max(coeff)
        self.coeff = self._coeff / self.max_coeff
        self.depth = depth
        self.steps = steps
        self.shots = shots
        self.prec = prec
        self.N = len(coeff)
        self.bitflip_prob = bitflip_prob

        self.device = qml.device(device, wires=self.N, shots=self.shots)
        self.str_opt = optimizer
        self.optimizer = self.OPTIMIZERS[optimizer](**opt_kwargs)

        self.bit_strs = np.array(
            ["".join(bs) for bs in product(["0", "1"], repeat=self.N)]
        )
        self.vertices = np.arange(self.N)
        self.edges = np.array(list(combinations(range(self.N), r=2)))
        self.weights = np.array(
            [self.coeff[edge[0], edge[1]] for edge in self.edges]
        )

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
        if not isinstance(probs, np.ndarray):
            probs = probs.numpy()

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
            self.params = [
                init_val * qmlnp.ones(shape) for shape in self.param_shapes
            ]

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
            self.cost_qnode = qml.QNode(
                func=self._cost_circuit, device=noisy_device
            )
        else:
            self.cost_qnode = qml.QNode(
                func=self._cost_circuit, device=self.device
            )

        # Run through the steps of the optimizing
        self.costs = qmlnp.empty(self.steps)
        self.evals = self.steps
        start = dt.now()
        for ind in range(self.steps):
            # Save the value of the cost per step too
            self.params, self.costs[ind] = self.optimizer.step_and_cost(
                self.cost_qnode, *self.params
            )
            if ind:
                self.current_prec = abs(
                    1 - self.costs[ind - 1] / self.costs[ind]
                )
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
        coeff,
        depth,
        steps,
        shots=None,
        prec=1e-8,
        optimizer="adam",
        opt_kwargs={},
        device="default.qubit",
        bitflip_prob=0,
    ):
        super().__init__(
            coeff=coeff,
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


class MAQAOA(VQA):
    """
    Instead of a free parameter per layer for the cost Hamiltonian and the mixer
    Hamiltonian, each gate has its own free parameter. So for a depth of `p` and N
    qubits, there are (NChoose2 + N)p free parameters (instead of just 2p).
    See: https://www.nature.com/articles/s41598-022-10555-8
    """

    def __init__(
        self,
        coeff,
        depth,
        steps,
        shots=None,
        prec=1e-8,
        optimizer="adam",
        opt_kwargs={},
        device="default.qubit",
        bitflip_prob=0,
    ):
        super().__init__(
            coeff=coeff,
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
        )

    def layer(self, gammas, betas):
        # Each gate gets its own beta/gamma parameter
        for edge, weight, gamma in zip(self.edges, self.weights, gammas):
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
        coeff,
        depth,
        steps,
        shots=None,
        prec=1e-8,
        optimizer="adam",
        opt_kwargs={},
        device="default.qubit",
        bitflip_prob=0,
    ):
        super().__init__(
            coeff=coeff,
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
        self,
        coeff,
        depth,
        dt=0.08,
        init_beta=0,
        shots=None,
        device="default.qubit",
    ):
        self._coeff = coeff
        self.max_coeff = qmlnp.max(coeff)
        # Keep the quadratic coefficient normalized
        self.coeff = self._coeff / self.max_coeff
        self.depth = depth
        self.shots = shots
        # Number of final state particles in problem
        self.N = len(coeff)

        self.device = qml.device(device, wires=self.N)

        # Stuff for weighting gates
        self.bit_strs = np.array(
            ["".join(bs) for bs in product(["0", "1"], repeat=self.N)]
        )
        self.vertices = np.arange(self.N)
        self.edges = np.array(list(combinations(range(self.N), r=2)))
        self.weights = np.array(
            [self.coeff[edge[0], edge[1]] for edge in self.edges]
        )

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
        self._cur_state = sqrt(1 / 2**self.N) * qmlnp.ones(2**self.N)

        # Commutator i[Hc, Hm] whose expval gives parameter of next layer
        comm_gates, comm_weights = [], []
        for i in range(self.N):
            for j, k in combinations(range(self.N), r=2):
                # [Xi, wjk*ZjZk] = -2iwjk(δijYiZk + δikZjYi)
                # delta_ij
                if i == j:
                    comm_weights.append(2 * self.coeff[j, k])
                    comm_gates.append(qml.PauliY(i) @ qml.PauliZ(k))
                # delta_jk
                elif i == k:
                    comm_weights.append(2 * self.coeff[j, k])
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
        self.costs = qmlnp.empty(self.depth)

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

        self.betas = np.array(self.betas)
        # To put in same format as with the parameters for other algorithms
        self.params = [self.betas]

        if print_it:
            print("\nDone!")
