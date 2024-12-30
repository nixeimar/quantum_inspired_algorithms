import numpy as np
import pennylane as qml
from scipy.linalg import eigh

import pennylane as qml
from pennylane import numpy as np
from scipy.linalg import eigh


def hybrid_hhl_solution(matrix, vector, shots=1024):
    """
    Hybrid quantum-classical implementation of HHL for solving Ax = b.

    Args:
        matrix (np.ndarray): Hermitian matrix A (NxN).
        vector (np.ndarray): Vector b (size N).
        shots (int): Number of quantum measurement shots.

    Returns:
        np.ndarray: Approximate solution vector x.
    """
    # Validate inputs
    N = matrix.shape[0]
    assert matrix.shape == (N, N), "Matrix must be NxN."
    print(np.log2(N))
    assert np.log2(N).is_integer(), "Matrix dimension must be a power of 2."
    if not np.allclose(matrix, matrix.conj().T):
        print("Warning: Matrix is not Hermitian. Enforcing Hermitian symmetry.")
        matrix = (matrix + matrix.conj().T) / 2
#        raise ValueError("Matrix must be Hermitian.")

    n_qubits = int(np.log2(N))
    dev = qml.device("default.qubit", wires=n_qubits + 1, shots=shots)

    # Pre-compute eigenvalues and eigenvectors (classical step)
    eigenvalues, eigenvectors = eigh(matrix)

    @qml.qnode(dev)
    def quantum_circuit(invert=False):
        """Quantum subroutine for state preparation and eigenvalue inversion."""
        # Prepare the normalized |b> state
        qml.FockStateVector(vector / np.linalg.norm(vector), wires=range(n_qubits))

        # Apply eigenvalue simulation (classical eigenvalues are used here as proxy)
        for i, eigenval in enumerate(eigenvalues):
            angle = 2 * np.pi * eigenval
            qml.RZ(angle, wires=i)

        # Perform eigenvalue inversion if required
        if invert:
            for i, eigenval in enumerate(eigenvalues):
                if eigenval != 0:
                    qml.CRY(-2 * np.arcsin(1 / eigenval), wires=[i, n_qubits])

        # Measure expectation values for classical reconstruction
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    # Perform the quantum state preparation and measurement
    prepared_state = quantum_circuit(invert=False)

    # Invert eigenvalues (quantum eigenvalue inversion step)
    inverted_state = quantum_circuit(invert=True)

    # Classical reconstruction using eigenvectors
    state_vector = np.array(prepared_state)
    solution_vector = np.array(inverted_state)
    approximate_solution = eigenvectors @ solution_vector

    return approximate_solution


def quantum_hhl_large_system(matrix, vector, shots=1024):
    """
    Quantum-inspired approximation of the HHL algorithm for solving a system.

    Args:
        matrix (np.ndarray): Matrix A (NxN).
        vector (np.ndarray): Input vector b (size N).
        shots (int): Number of quantum measurement shots.

    Returns:
        np.ndarray: Approximate solution vector x of the system Ax = b.
    """
    # Check matrix properties
    N = matrix.shape[0]
    assert matrix.shape == (N, N), "Matrix must be NxN."

    # Enforce Hermitian property if it is not exact due to numerical issues
    if not np.allclose(matrix, matrix.conj().T):
#        print("Warning: Matrix is not Hermitian. Enforcing Hermitian symmetry.")
        matrix = (matrix + matrix.conj().T) / 2

    n_qubits = int(np.log2(matrix.shape[0]))
    dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

    @qml.qnode(dev)
    def quantum_solution_circuit():
        # Embed the Hermitian matrix as an observable
        Hermitian = qml.Hermitian(matrix, wires=range(n_qubits))

        # Initialize the state |b>
        qml.StateVector(vector / np.linalg.norm(vector), wires=range(n_qubits))

        # Measure in the basis of the Hermitian matrix
        return qml.expval(Hermitian)

    # Approximation of the inverse by eigenvalues
    eigenvalues, eigenvectors = eigh(matrix)
#    print( "kappa: ", np.min( eigenvalues)/np.max(eigenvalues) )
#    print( "min: ", np.min(eigenvalues)/np.linalg.norm(eigenvalues) )
#    print( "max: ", np.max(eigenvalues)/np.linalg.norm(eigenvalues)  )

    eigenvalues_inverse = 1 / eigenvalues

    # Calculate the state vector in the eigenbasis
    state_vector = eigenvectors.T @ (vector / np.linalg.norm(vector))

    # Perform elementwise multiplication of eigenvalues_inverse and state_vector
    inverted_solution = eigenvalues_inverse * state_vector

    # Reconstruct the solution vector by applying the eigenvectors
    approximate_solution = eigenvectors @ inverted_solution

    return approximate_solution
