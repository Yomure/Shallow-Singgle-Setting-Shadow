# povms.py
import numpy as np
from typing import List, Union, Optional
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
# If IBM hardware is not used, qiskit_ibm_runtime can be removed to reduce dependencies.
# from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 

def expand_circuit_with_ancilla(
    original_state: Union[QuantumCircuit, np.ndarray, Statevector], 
    two_qubit_subcircuit: QuantumCircuit
) -> QuantumCircuit:
    """
    Constructs a POVM measurement circuit via Naimark dilation.
    Expands the system with ancilla qubits to perform projective measurements in a larger Hilbert space.
    """
    
    # 1. Determine number of qubits and construct initial circuit
    if isinstance(original_state, QuantumCircuit):
        original_circuit = original_state
        num_qubits = original_circuit.num_qubits
    else:
        # Handle vector input (convert numpy array or list to state initialization)
        vec = np.array(original_state)
        num_qubits = int(np.log2(len(vec)))
        original_circuit = QuantumCircuit(num_qubits)
        original_circuit.initialize(vec, range(num_qubits))

    # 2. Construct expanded circuit (System + Ancilla)
    # Naimark dilation typically requires doubling the number of qubits (System + Ancilla)
    expanded_circuit = QuantumCircuit(num_qubits * 2)
    expanded_circuit.compose(original_circuit, qubits=range(num_qubits), inplace=True)
    
    # 3. Apply POVM unitary (System-Ancilla interaction)
    for qubit in range(num_qubits):
        # Apply the interaction unitary to the pair: [system_qubit, corresponding_ancilla_qubit]
        # The ancilla index is offset by 'num_qubits'
        expanded_circuit.compose(
            two_qubit_subcircuit, 
            qubits=[qubit, num_qubits + qubit], 
            inplace=True
        )
        
    expanded_circuit.measure_all()
    return expanded_circuit

def get_result_bitstrings(
    circuit: QuantumCircuit, 
    shots: int = 1024, 
    backend_name: str = 'qasm_simulator',
    seed: Optional[int] = None
) -> List[str]:
    """
    Get list of bitstrings from measurement results.
    Note: Supports local simulators. Cloud execution logic should be handled separately.
    """
    simulator = AerSimulator(method='automatic')
    
    # Use run() to retrieve raw memory (list of bitstrings)
    # memory=True is crucial to get individual shot results rather than just counts
    result = simulator.run(circuit, shots=shots, memory=True, seed_simulator=seed).result()
    bitstrings = result.get_memory()
    
    return bitstrings