from qiskit import QuantumCircuit
from qiskit.quantum_info import random_statevector, Pauli, Operator
from .unitary import U3_pauli
import numpy as np

def get_tensor_state(n_qubits, rotations):
    """
    Generates a tensor product state (separable state) where every qubit 
    is subjected to the same single-qubit rotation.
    
    State: |psi> = U|0> (tensor) U|0> ... (tensor) U|0>
    """
    qc = QuantumCircuit(n_qubits)
    # Create the single-qubit unitary operator from rotation parameters
    u = U3_pauli(rotations[0], rotations[1], rotations[2])
    unitary_gate = Operator(u)
    
    # Apply the same unitary to every qubit independently
    for i in range(n_qubits):
        qc.append(unitary_gate, [i])
    
    qc.barrier()
    return qc

def get_ghz_state(n_qubits):
    """
    Generates a Greenberger-Horne-Zeilinger (GHZ) state.
    
    State: (|00...0> + |11...1>) / sqrt(2)
    This is a maximally entangled state.
    """
    if n_qubits < 2:
        raise ValueError("GHZ state requires at least 2 qubits.")
        
    qc = QuantumCircuit(n_qubits)
    
    # 1. Put the last qubit into superposition |+>
    qc.h(n_qubits - 1)
    
    # 2. Entangle the qubits using a cascade of CNOTs (ladder structure)
    # Propagates the superposition from the last qubit down to the first.
    for qubit in range(n_qubits - 1, 0, -1):
        qc.cx(qubit, qubit - 1)
        
    qc.barrier()    
    return qc

def get_random_state(n_qubits):
    """
    Generates a random quantum state vector.
    This typically samples from the Haar measure, producing a highly entangled 
    state with high probability for larger n.
    """
    qc = QuantumCircuit(n_qubits)
    # Initialize the circuit with a random statevector of size 2^n
    qc.initialize(random_statevector(2 ** n_qubits), range(n_qubits))
    qc.barrier()
    return qc

def get_zero_state(n_qubits):
    """
    Generates the ground state |00...0>.
    """
    qc = QuantumCircuit(n_qubits)
    # Implicitly initialized to |0>, so no operations are needed.
    # The explicit initialization below is commented out for efficiency.
    # qc.initialize([1] + [0] * (2 ** n_qubits - 1), range(n_qubits))
    # qc.barrier()
    return qc

def get_one_state(n_qubits):
    """
    Generates the excited state |11...1>.
    """
    qc = QuantumCircuit(n_qubits)
    # Apply Pauli-X (NOT) gate to every qubit to flip |0> to |1>
    for i in range(n_qubits):
        qc.x(i)
    qc.barrier()
    return qc

def get_random_weighted_pauli(n, w):
    """
    Generates a random Pauli operator with a specific weight 'w'.
    
    The 'weight' is defined as the number of non-identity terms in the tensor product.
    Example (n=3, w=2): "XIZ" (Weight 2 because X and Z are non-identity).
    """
    if w > n:
        raise ValueError("Weight w cannot be greater than the number of qubits n.")
        
    # 1. Select 'w' unique indices to place non-identity gates
    indices = np.random.permutation(n)
    non_identity_indices = indices[:w]
    
    # 2. Build the Pauli string
    pauli_list = ['I'] * n
    for index in non_identity_indices:
        # Randomly choose between X, Y, Z for the active sites
        pauli_list[index] = np.random.choice(['X', 'Y', 'Z'])
        
    # Join list into string (e.g., "IXZY...") and convert to matrix
    pauli_str = ''.join(pauli_list)
    pauli_obj = Pauli(pauli_str)
    pauli_matrix = pauli_obj.to_matrix()
    
    return pauli_matrix, pauli_str

def get_pauli(pauli_str):
    """
    Helper function to convert a Pauli string (e.g., "IXZ") into its matrix representation.
    """
    return Pauli(pauli_str).to_matrix()