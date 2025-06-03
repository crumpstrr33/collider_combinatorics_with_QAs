from datetime import datetime as dt
from itertools import combinations, product
from math import comb, sqrt
from typing import Optional, Sequence

import numpy as np
import pennylane as qml
from numpy.typing import NDArray
from pennylane import numpy as qmlnp


class VQA:
    """
    Base class for the VQAs. For an algorithm like QAOA, it has two lists of
    parameters: gammas and betas (and e.g. XQAOA has three with the additional
    alphas). `params` is a list of each list of parameters and
    `self.param_shapes` is a list of the numpy shape for each list of parameters.
    """

    OPTIMIZERS = {
        "grad_descent": qml.GradientDescentOptimizer,
        "adagrad": qml.AdagradOptimizer,
        "adam": qml.AdamOptimizer,
    }

    def __init__(
        self,
        coeff: NDArray[NDArray[float]],
        depth: int,
        steps: Optional[int] = None,
        shots: Optional[int] = None,
        prec: float = 1e-8,
        optimizer: str = "adam",
        opt_kwargs: dict[str, ...] = {},
        device: str = "default.qubit",
        bitflip_prob: float = 0,
    ):
        """
        In the __init__ for inheriting classes:
            1) Define the cost and mixer Hamiltonians using self.edges/
            self.weights for the cost Hamiltonian and self.vertices for the
            mixer Hamiltonian which are used in your definition for `layer`.
            2) Define the array shapes for the lists of parameters as a list
            defined as `self.param_shapes`.

        Parameters:
        coeff - Matrix of coefficients for s_i * s_j spin coupling.
        depth - How many layers in the circuit, written as `p`.
        steps - How many steps to run the optimizer.
        shots - Number of shots for the circuit, If None, use exact statevector,
            e.g. an infinite number of shots.
        prec - How precise the optimizer should be. That is the optimizer won't
            stop until the difference between the latest two evaluations are
            less that `prec` or `steps` number of evaluations have been
            completed. That is, which ever happens first.
        optimizer - Which optimizer to use: "grad_descent", "adagrad" or "adam"
            for the gradient descent, adagrad or adam optimizer.
        opt_kwargs - A dictionary of keyword arguments to pass to the optimizer
            object.
        device - The pennylane device to run on.
        """
        self.coeff = coeff
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

    def layer(self, *params: NDArray[float]) -> None:
        """
        Each layer of the VQA. Each arg from `params` is a list of parameters,
        e.g. QAOA has two: gammas and betas but XQAOA has three: gammas, betas
        and alphas.
        """
        pass

    def circuit(self, *params: NDArray[float]) -> None:
        """
        Variational circuit for given parameters.
        """
        # Initialize with Hadamards
        for ind in range(self.N):
            qml.Hadamard(wires=ind)
        qml.Barrier()

        # Add layers
        qml.layer(self.layer, self.depth, *params)

    def _cost_circuit(self, *params: NDArray[float]):
        """
        Define `self.expval_op` in inheriting class's __init__. It is the total
        cost function, usually problem/cost Hamiltonian.
        """
        self.circuit(*params)
        return qml.expval(self.expval_op)

    def _probs_circuit(self, *params: NDArray[float]):
        """
        Circuit for getting probabilities for each eigenstate.
        """
        self.circuit(*params)
        return qml.probs()

    def get_probs(self, as_dict: bool = False) -> NDArray[int] | dict[str, float]:
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
        init_params: Optional[Sequence[NDArray[float]]] = None,
        print_it: bool = False,
        print_pref: str = "",
        print_newlines: bool = False,
        init_val: float = 0.5,
    ) -> None:
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
            self.cost_qnode = qml.QNode(func=self._cost_circuit, device=noisy_device)
        else:
            self.cost_qnode = qml.QNode(func=self._cost_circuit, device=self.device)

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
        coeff: NDArray[NDArray[float]],
        depth: int,
        steps: int,
        shots: Optional[int] = None,
        prec: float = 1e-8,
        optimizer: str = "adam",
        opt_kwargs: dict[str, ...] = {},
        device: str = "default.qubit",
        bitflip_prob: float = 0,
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

    def layer(self, gamma: NDArray[float], beta: NDArray[float]) -> None:
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
    Hamiltonian, each gate has its own free parameter. So for a depth of `p` and
    N qubits, there are (NChoose2 + N)p free parameters (instead of just 2p).
    See: https://www.nature.com/articles/s41598-022-10555-8
    """

    def __init__(
        self,
        coeff: NDArray[NDArray[float]],
        depth: int,
        steps: int,
        shots: Optional[int] = None,
        prec: float = 1e-8,
        optimizer: str = "adam",
        opt_kwargs: dict[str, ...] = {},
        device: str = "default.qubit",
        bitflip_prob: float = 0,
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

    def layer(self, gammas: NDArray[float], betas: NDArray[float]) -> None:
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
        coeff: NDArray[NDArray[float]],
        depth: int,
        steps: int,
        shots: Optional[int] = None,
        prec: float = 1e-8,
        optimizer: str = "adam",
        opt_kwargs: dict[str, ...] = {},
        device: str = "default.qubit",
        bitflip_prob: float = 0,
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

    def layer(
        self, gammas: NDArray[float], betas: NDArray[float], alphas: NDArray[float]
    ) -> None:
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

    Since the parameters are fixed once found, this class calculates the state
    for layer n, then initializes the the next layer with that state so the
    (n+1)th layer only needs to run a circuit of 1 layer deep.
    """

    def __init__(
        self,
        coeff: NDArray[NDArray[float]],
        depth: int,
        dt: float = 0.08,
        init_beta: float = 0,
        shots: Optional[int] = None,
        device: str = "default.qubit",
    ):
        self.coeff = coeff
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
        Finds the output state of the current circuit. Saved in `self._cur_state`
        and used to initalize the next layer.
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
        Circuit to both minimize the cost function (i.e. the expectation value
        of `self.expval_op` with respect to the circuit) and the next parameter
        value.
        """
        self.circuit(self._new_beta)
        return qml.expval(self.expval_op), qml.expval(self.commutator)

    def circuit(self, beta: Optional[float] = None):
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

    def get_probs(self, as_dict: bool = False) -> NDArray[float] | dict[str, float]:
        """
        Gets the probabilities for each eigenstate.
        """
        probs_qnode = qml.QNode(self._probs_circuit, self.device)
        probs = probs_qnode()
        if as_dict:
            return dict(zip(self.bit_strs, probs))
        return probs

    def run(self, print_it: bool = False) -> None:
        """`
        Run the FALQON algorithm
        """
        # For other algorithms, they stop when cost function stops decreasing at
        # a rate below some threshold. We don't have that here, so there will a
        # number of evaluations equal to the depth of the circuit
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


class VarQITE:
    def __init__(
        self,
        coeff: NDArray[NDArray[float]],
        depth: int = 1,
        steps: int = 10,
        shots: Optional[int] = None,
        prec: float = 1e-5,
        dtau: float = 0.5,
        device: str = "default.qubit",
    ):
        """
        Algorithm for Variational Quantum Imaginary Time Evolution. This
        algorithm is described in arxiv.org/pdf/2404.16135.
        """
        self.coeff = coeff
        self.depth = depth  # doesn't do anything atm
        self.steps = steps
        self.dtau = dtau
        self.nq = len(coeff)
        self.prec = prec
        self.device = qml.device(device, wires=self.nq)

        # Ordered array of all possible eigenstates
        self.bitstrings = np.array(
            [format(bs, f"0{self.nq}b") for bs in range(2**self.nq)]
        )
        # Verteices of our graph
        self.vertices = np.arange(self.nq)
        # The edges represented by tuples of vertices
        self.edges = np.stack(np.triu_indices(self.nq, k=1), axis=1)
        # The weights for each edge
        self.weights = np.array(
            [self.coeff[edge[0], edge[1]] for edge in self.edges]
        )
        # # The operators of the Hamiltonian (without their weights)
        self.ops = [qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1]) for edge in self.edges]

        # Expectation value operator: problem Hamiltonian
        self.hamiltonian = qml.Hamiltonian(self.weights, self.ops)
        self.qubit_order = self._order_qubits()
        # Ordered tuples for ZY gates
        self.qubit_pairs = [(q, r) for q in range(self.nq) for r in range(q)]
        # Creates circuits to run
        self._create_qnodes()

        # Per step info to record
        self.energies = []
        self.op_energies = []
        self.all_thetas = []
        self.Gmats = []
        self.Dvecs = []
        self.theta_dots = []
        self.energy_diffs = []
        # To assign to self.old_energy for the first step
        self.current_energy = None

    def _order_qubits(self) -> NDArray[int]:
        """
        Returns the order of the qubits based on Eq. S7 of the paper. It's the
        sum of the absolute value of the weights of a specific vertex with all
        others in the graph.
        """
        return np.argsort(np.sum(np.abs(self.coeff), axis=1))[::-1]

    def _circuit(self, thetas: NDArray[float]):
        """
        Builds ansatz.
        """
        for ind in range(self.nq):
            qml.Hadamard(ind)

        for ind, (q, r) in enumerate(self.qubit_pairs):
            qubits = [self.qubit_order[r], self.qubit_order[q]]
            qml.PauliRot(-2 * thetas[ind], "ZY", wires=qubits)

    def circuit_op_expvals(self, thetas: NDArray[float]):
        """
        Circuit for expectation value of each term of Hamiltonian.
        """
        self._circuit(thetas=thetas)
        return qmlnp.array([qml.expval(op) for op in self.ops])

    def circuit_ham_expval(self, thetas: NDArray[float]):
        """
        Circuit for expectation value for full Hamiltonian.
        """
        self._circuit(thetas=thetas)
        return qml.expval(self.hamiltonian)

    def circuit_probs(self, thetas: NDArray[float]):
        """
        Circuit for probability of each eigenstate.
        """
        self._circuit(thetas=thetas)
        return qml.probs()

    def _create_qnodes(self) -> None:
        """
        Create QNodes for each circuit.
        """
        self.get_op_expvals = qml.QNode(
            func=self.circuit_op_expvals, device=self.device
        )
        self.get_ham_expval = qml.QNode(
            func=self.circuit_ham_expval, device=self.device
        )
        self._get_probs = qml.QNode(func=self.circuit_probs, device=self.device)

    def get_probs(self, thetas: Optional[NDArray[float]] = None) -> dict[str, float]:
        """
        Gets the probabilities for each eigenstate as a dictionary
        """
        thetas = self.current_thetas if thetas is None else thetas
        probs = self._get_probs(thetas=thetas)
        return dict(zip(self.bitstrings, probs))

    def find_gradient(self, thetas: NDArray[float]) -> NDArray[float]:
        """
        Finds theta dot via inverting finding G and D and inverting D (Eq. 6).
        Returns the gradient for each parameter.
        """
        Dvec = np.empty(len(self.ops))
        for i, Hi in enumerate(self.ops):
            # Sum of ith column of covariance matrix, i.e. <H_i H_j>
            cov = 0
            for j, Hj in enumerate(self.ops):
                if i == j:
                    cov += self.weights[j]
                    continue

                @qml.qnode(device=self.device)
                def get_cov_expval(thetas):
                    self._circuit(thetas=thetas)
                    return qml.expval(Hi @ Hj)

                cov += np.real_if_close(
                    self.weights[j] * get_cov_expval(thetas), tol=1e-10
                )
                if cov.imag != 0:
                    raise Exception(f"Covariance is complex: {cov}! Fix!!")

            # D[i] = -∑_j c_j⟨H_iH_j⟩ + ⟨H⟩⟨H_i⟩
            Dvec[i] = -cov + self.current_energy * self.current_op_energies[i]

        # G[i][j] = ∂(⟨H_i⟩)/∂(params[j]).
        Gmat = qml.jacobian(self.get_op_expvals)(thetas) / 2
        # Calculate theta dot via SVD
        max_sv = np.linalg.svd(Gmat, compute_uv=False).max()
        theta_dot = np.linalg.lstsq(Gmat, Dvec, rcond=max_sv / 100)[0]

        self.Gmats.append(Gmat)
        self.Dvecs.append(Dvec)
        self.theta_dots.append(theta_dot)
        return theta_dot

    def step(self, thetas: NDArray[float]) -> NDArray[float]:
        """
        Does single optimization step of algorithm. Returns new values for
        parameters.
        """
        # expectation value of full Hamiltonian
        self.previous_energy = self.current_energy
        self.current_energy = self.get_ham_expval(thetas=thetas)
        self.energies.append(self.current_energy)
        # expectation values of each term of the Hamiltonian
        self.current_op_energies = self.get_op_expvals(thetas=thetas)
        self.op_energies.append(self.current_op_energies)

        self.all_thetas.append(thetas)
        # Modulo the angle FOR NOW, sometimes value is really large...
        theta_dots = self.find_gradient(thetas=thetas)

        return (thetas + self.dtau * theta_dots) % (2 * np.pi)

    def optimize(
        self,
        steps_till_newline: int = 10,
        print_progress: bool = True,
    ) -> None:
        """
        Runs full optimization.
        """
        start_time = dt.now()
        self.current_thetas = qmlnp.ones(comb(self.nq, 2)) * np.pi
        for self.step_ind in range(self.steps):
            step_start = dt.now()
            self.current_thetas = self.step(self.current_thetas)
            step_end = dt.now()

            self.energy_diff = (
                None
                if self.previous_energy is None
                else abs(self.current_energy - self.previous_energy)
            )
            if self.energy_diff is not None:
                self.energy_diffs.append(self.energy_diff)

            if print_progress:
                end = "\r" if (self.step_ind + 1) % steps_till_newline else "\n"
                step_str = (
                    f"Step time: {(step_end - step_start).total_seconds():.3f}"
                )
                total_str = (
                    f"Total time: {(step_end - start_time).total_seconds():.3f}"
                )
                diff_str = (
                    f"ΔE = {self.energy_diff / abs(self.current_energy):.3e}"
                    if self.energy_diff is not None
                    else ""
                )
                print(
                    f"Step: {self.step_ind + 1:>{len(str(self.steps))}} / {self.steps}"
                    f" | {step_str} | {total_str} | {diff_str}",
                    end=end,
                )

            if self.prec is not None and self.energy_diff is not None:
                if self.energy_diff / abs(self.current_energy) < self.prec:
                    break

        self.total_steps = self.step_ind + 1
        self.energies = np.array(self.energies)
        self.op_energies = np.array(self.op_energies)
        self.all_thetas = np.array(self.all_thetas)
        self.Gmats = np.array(self.Gmats)
        self.Dvecs = np.array(self.Dvecs)
        self.theta_dots = np.array(self.theta_dots)
        self.energy_diffs = np.array(self.energy_diffs)
