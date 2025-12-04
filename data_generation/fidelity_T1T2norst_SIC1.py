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
from tqdm import tqdm
import glob
import re

sic_povms_states = np.load(r'./data/sic_phi_vectors/PRA_86_062107.npy')
post_progress_data = True
num_qubits = 6
ghz_state = get_ghz_state(num_qubits)
init_state = ghz_state
obs = DensityMatrix.from_instruction(init_state).data
noises = ['T1T2norst']
sics = ['SIC_1']
opts = ['opt_0','opt_1','opt_2']
shots = 2000000
num_datasets = 20

def shadow_estimation(res, sic_phi, cnotf, progress_data, func, *state):
    Sp_inv, vec_povms_elements_dict = compute_sp_inv_and_dict(sic_phi, cnotf)
    base_results_cache = {sub_result: devectorize_matrix(Sp_inv @ vec_povms_elements_dict[sub_result]) for sub_result in vec_povms_elements_dict}
    processed_results_cache = {}
    accumulated_result = 0
    
    def process_result(result):
        nonlocal processed_results_cache, base_results_cache, func, state
        if result not in processed_results_cache:
            split_results = snapshot_data_split(result)
            sigmas = [base_results_cache[sub_result] for sub_result in split_results]
            rho = compute_kronecker_product(sigmas)
            processed_results_cache[result] = func(rho, *state)
        return processed_results_cache[result]
    
    if not progress_data:
        for result in res:
            accumulated_result += process_result(result)
        return accumulated_result / len(res)
    else:
        ans = []
        for i, result in enumerate(res):
            # accumulated_result += process_result(result)
            # ans.append(accumulated_result / (i + 1))
            ans.append(process_result(result))
        return ans
    
sets = [(noise,sic,opt) for noise in noises for sic in sics for opt in opts]

for set_name in tqdm(sets, desc="Processing sets"):
    folder_path = f'./data/raw_bitstrings/noise_{set_name[0]}_{set_name[1]}'
    for data_index in tqdm(range(num_datasets)):
        pattern = f"{folder_path}/{set_name[2]}_shots_{shots}_cnotf_*_{data_index}.npy"
        file_paths = glob.glob(pattern)
        for file_path in file_paths:
            match = re.search(r'cnotf_(.*?)_', file_path)
            cnotf = match.group(1)
            data = np.load(file_path)
            f_hats = shadow_estimation(data, sic_povms_states, cnotf, post_progress_data, expect , obs)
            save_path = f'./data/fidelity/noise_{set_name[0]}_{set_name[1]}'
            os.makedirs(save_path, exist_ok=True)
            np.save(f'{save_path}/{set_name[2]}_shots_{shots}_{data_index}.npy', f_hats)
            
