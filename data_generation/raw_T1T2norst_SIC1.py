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
optimization_level = [0,1,2]
ghz_state = get_ghz_state(num_qubits)
init_state = ghz_state
obs = DensityMatrix.from_instruction(init_state).data

# Error model
noise_qubits = 2*num_qubits
T1s = np.random.normal(50e3, 10e3, noise_qubits) # Sampled from normal distribution mean 50 microsec
T2s = np.random.normal(70e3, 10e3, noise_qubits)  # Sampled from normal distribution mean 50 microsec
# Truncate random T2s <= T1s
T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(noise_qubits)])
# Instruction times (in nanoseconds)
time_u1 = 0   # virtual gate
time_u2 = 50  # (single X90 pulse)
time_u3 = 100 # (two X90 pulses)
time_cx = 300
time_reset = 1000  # 1 microsecond
time_measure = 1000 # 1 microsecond
errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                for t1, t2 in zip(T1s, T2s)]
errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                  for t1, t2 in zip(T1s, T2s)]
errors_u1  = [thermal_relaxation_error(t1, t2, time_u1)
              for t1, t2 in zip(T1s, T2s)]
errors_u2  = [thermal_relaxation_error(t1, t2, time_u2)
              for t1, t2 in zip(T1s, T2s)]
errors_u3  = [thermal_relaxation_error(t1, t2, time_u3)
              for t1, t2 in zip(T1s, T2s)]
errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
             thermal_relaxation_error(t1b, t2b, time_cx))
              for t1a, t2a in zip(T1s, T2s)]
               for t1b, t2b in zip(T1s, T2s)]
noise_thermal = NoiseModel()
for j in range(noise_qubits):
    # if j == num_qubits - 1:  # 检查是否是 num_qubits - 1
    #     noise_thermal.add_quantum_error(errors_u3[j], "u3", [j])
    #     continue
    noise_thermal.add_quantum_error(errors_u1[j], "u1", [j])
    noise_thermal.add_quantum_error(errors_u2[j], "u2", [j])
    noise_thermal.add_quantum_error(errors_u3[j], "u3", [j])
    # noise_thermal.add_quantum_error(errors_reset[j], "reset", [j])
    # noise_thermal.add_quantum_error(errors_measure[j], "measure", [j])
for j in range(num_qubits,noise_qubits):
    noise_thermal.add_quantum_error(errors_cx[j][j-num_qubits], "cx", [j, j-num_qubits])
    noise_thermal.add_quantum_error(errors_cx[j-num_qubits][j], "cx", [j-num_qubits, j])

noise_model = noise_thermal


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


for opt in optimization_level:
    init_qc = init_state
    qc, cnotf = get_measurement_circuit(init_qc,sic_povms_states,opt)

    sim_thermal = AerSimulator(noise_model=noise_model)
    passmanager = generate_preset_pass_manager(optimization_level=0, backend=sim_thermal)
    circ_tthermal = passmanager.run(qc)

    for i in range(20):
        path = f'./data/raw_bitstrings/noise_T1T2norst_SIC_1'
        os.makedirs(path, exist_ok=True)
        results = sim_thermal.run(circ_tthermal, shots=shots,memory = True).result()
        res = results.get_memory()
        np.save(f'{path}/opt_{opt}_shots_{shots}_cnotf_{cnotf}_{i}.npy', res)