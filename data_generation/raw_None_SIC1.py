import sys
import os
sys.path.append(os.getcwd())
from packages.unitary import *
from packages.povms import *
from packages.utils import *
from packages.states import *
from packages.linalg import *
from qiskit.quantum_info import (
    random_statevector, 
    DensityMatrix, 
    Pauli, 
    partial_trace, 
    Operator,Statevector
)
from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
)
from qiskit import QuantumCircuit
import numpy as np
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from packages.decomposition import kak_decomposition
from scipy.optimize import brute, brentq

sic_povms_states = np.load(r'./data/sic_phi_vectors/PRA_86_062107.npy')
shots = 2000000
num_qubits = 6
cloud = False
post_progress_data = True
optimization_level = 2
noise_model = None
ghz_state = get_ghz_state(num_qubits)
init_state = ghz_state
obs = DensityMatrix.from_instruction(init_state).data

def expand_circuit_with_ancilla(qc, measure_list, two_qubit_subcircuit):
    num_qubits = qc.num_qubits
    m_qubits = len(measure_list)
    expanded_circuit = QuantumCircuit(num_qubits + m_qubits, m_qubits * 2)
    expanded_circuit.compose(qc, qubits=range(num_qubits), inplace=True)
    for i, qubit in enumerate(measure_list):
        ancilla_index = num_qubits + i 
        expanded_circuit.compose(two_qubit_subcircuit, qubits=[qubit, ancilla_index], inplace=True)
    expanded_circuit.barrier()
    for i, qubit in enumerate(measure_list):
        expanded_circuit.measure(qubit, i)
    for i, qubit in enumerate(measure_list):
        ancilla_index = num_qubits + i 
        expanded_circuit.measure(ancilla_index, m_qubits + i)
    return expanded_circuit

def get_measurement_circuit(qc,  sic_phi, method = 0, measure_list= None):
    Ub = sic_phi_2_unitary(sic_phi)
    if method == 0:
        ucirc = unitary2circuit(Ub)
        cnotf = False
    if method == 1:
        def z_function(gamma2):
            cu = controlled_U3_pauli(0, gamma2, 0)
            _, _, z = kak_decomposition(Ub @ cu)
            return np.abs(z)  # 我们想要最小化绝对值
        def z_function2(gamma2):
            cu = controlled_U3_pauli(0, gamma2, 0)
            _, _, z = kak_decomposition(Ub @ cu)
            return z 
        ranges = (0, 1 * np.pi)
        gamma2_zero_a = brute(z_function, [ranges], Ns=1000, full_output=False, finish=None)
        gamma2_zero = brentq(z_function2, gamma2_zero_a - np.pi/100, gamma2_zero_a + np.pi/100, xtol=1e-30)
        cu = controlled_U3_pauli(0, gamma2_zero, 0)
        U = Ub @ cu
        ucirc = unitary2circuit(U)
        cnotf = False
    if method == 2:
        u,phases,cnotf = get_maps_basic2input(Ub)
        ucirc = QuantumCircuit(2)
        unitary_gate = UnitaryGate(u)
        ucirc.append(unitary_gate,[0])
        ucirc.rx(np.arccos(1/np.sqrt(3)), 1)
        ucirc.rz(3 * np.pi / 4, 1)
        ucirc.cx(1, 0)
        ucirc.h(1)       
    if measure_list is None:
        measure_list = range(qc.num_qubits)
    expanded_circ = expand_circuit_with_ancilla(qc,measure_list,ucirc)
    return expanded_circ, cnotf

init_qc = init_state
qc, cnotf = get_measurement_circuit(init_qc,sic_povms_states,optimization_level)

sim_thermal = AerSimulator(noise_model=noise_model)
passmanager = generate_preset_pass_manager(optimization_level=0, backend=sim_thermal)
circ_tthermal = passmanager.run(qc)

for i in range(20):
    path = f'./data/raw_bitstrings/noise_{noise_model}_SIC_1'
    os.makedirs(path, exist_ok=True)
    results = sim_thermal.run(circ_tthermal, shots=shots,memory = True).result()
    res = results.get_memory()
    np.save(f'{path}/opt_{optimization_level}_shots_{shots}_cnotf_{cnotf}_{i}.npy', res)