from qiskit.compiler import transpile
from qiskit_aer import Aer
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator
from .exceptions import UnitaryError
from .decomposition import gram_schmidt
import numpy as np
from scipy.linalg import expm

# Note: is_unitary, is_vec_orthogonal, get_column_matrix are assumed to be 
# simple helper functions. If they are not imported from elsewhere, 
# standard numpy checks can be used.
def is_unitary(m, tol=1e-5):
    """Check if matrix m is unitary: U^dagger @ U = I."""
    return np.allclose(np.eye(len(m)), m.conj().T @ m, atol=tol)

def is_vec_orthogonal(v1, v2, tol=1e-5):
    """Check if two vectors are orthogonal."""
    return np.abs(np.vdot(v1, v2)) < tol

def unitary2circuit(
    unitary: np.ndarray, 
    basis_gates: list = ['ry', 'rz', 'cx'], 
    backend = Aer.get_backend('qasm_simulator')
) -> QuantumCircuit:
    """
    Decomposes a unitary matrix into a Qiskit QuantumCircuit.
    
    Args:
        unitary (np.ndarray): The target unitary matrix.
        basis_gates (list): List of allowed gates for transpilation.
        backend: The backend used for transpilation target.
        
    Returns:
        QuantumCircuit: The transpiled circuit implementing the unitary.
    """
    if not is_unitary(unitary):
        raise UnitaryError("Input matrix is not unitary.")
    if len(unitary.shape) != 2:
        raise UnitaryError("Input matrix is not 2-dimensional.")
        
    unitary_gate = UnitaryGate(unitary)
    circuit = QuantumCircuit(unitary_gate.num_qubits)
    circuit.append(unitary_gate, range(unitary_gate.num_qubits))
    
    transpiled_circuit = transpile(
        circuits=circuit, 
        basis_gates=basis_gates,
        backend=backend
    )
    return transpiled_circuit

def random_2rows_fixed_unitary(row1: np.ndarray, row2: np.ndarray) -> np.ndarray:
    """
    Generates a random 4x4 unitary matrix where the first two rows are fixed.
    
    This is often used to construct a unitary that maps specific states |00> and |01>
    to target states defined by row1 and row2.
    """
    if not is_vec_orthogonal(row1, row2, tol=1e-5):
        raise ValueError("Input vectors are not orthogonal.")
        
    # Normalize inputs
    row1 = row1 / np.linalg.norm(row1)
    row2 = row2 / np.linalg.norm(row2)
    
    # Generate random complex matrix
    Q = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    
    # Fix the first two rows
    Q[0] = row1
    Q[1] = row2
    
    # Apply Gram-Schmidt to orthogonalize the remaining rows against the fixed ones
    Q = gram_schmidt(Q)
    
    if not is_unitary(Q):
        raise ValueError("Gram-Schmidt process failed.")
        
    # Return Conjugate Transpose because Gram-Schmidt usually returns orthogonal rows,
    # and we want to ensure the final structure fits the unitary definition used.
    # (Note: Depending on the specific Gram-Schmidt implementation, this ensures 
    # row orthonormality).
    return Q.conj().T

def sic_phi_2_unitary(phis):
    """
    Constructs a unitary matrix from a set of SIC (Symmetric Informationally Complete) 
    fiducial vectors.
    """
    row1 = [x[0] for x in phis]
    row2 = [x[1] for x in phis]
    Ub = random_2rows_fixed_unitary(row1, row2)
    return Ub
 
# --- Basic Rotation Gates ---

def U3(theta, phi, lambda_):
    """Standard U3 parametrization for single qubit unitary."""
    return np.array([
        [np.cos(theta/2), -np.exp(1j*lambda_) * np.sin(theta/2)],
        [np.exp(1j*phi) * np.sin(theta/2), np.exp(1j*(phi + lambda_)) * np.cos(theta/2)]
    ])

def rz(theta):
    """Rotation around Z axis."""
    return np.array([
        [np.exp(-1j*theta/2), 0],
        [0, np.exp(1j*theta/2)]
    ])
    
def ry(theta):
    """Rotation around Y axis."""
    return np.array([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2), np.cos(theta/2)]
    ])
    
def rx(theta):  
    """Rotation around X axis."""
    return np.array([
        [np.cos(theta/2), -1j*np.sin(theta/2)],
        [-1j*np.sin(theta/2), np.cos(theta/2)]
    ])

def rz_1(theta):
    """Alternative Z rotation (global phase shifted)."""
    return np.array([
        [1, 0],
        [0, np.exp(1j*theta)]
    ])

# --- Composite and Controlled Gates ---

def U3_xzx(theta, phi, lambda_):
    """Euler angle decomposition using X-Z-X sequence."""
    rx1 = rx(theta)
    rz1 = rz(phi)
    rx2 = rx(lambda_)
    return rx2 @ rz1 @ rx1

def controlled_U3_xzx(gamma1, gamma2, gamma3, phase=0.0):
    """Controlled version of the XZX decomposed unitary."""
    u_matrix = np.exp(1j*phase) * U3_xzx(gamma1, gamma2, gamma3)
    top_left = np.eye(2)
    bottom_right = u_matrix
    return np.block([[top_left, np.zeros((2, 2))], [np.zeros((2, 2)), bottom_right]])

def random_U3():
    """Generate a random single-qubit unitary uniformly sampled."""
    theta = np.random.uniform(0, 2*np.pi)
    phi = np.random.uniform(0, 2*np.pi)
    lambda_ = np.random.uniform(0, 2*np.pi)
    return U3(theta, phi, lambda_)

def controlled_U3(theta, phi, lambda_):
    """Standard Controlled-U3 gate."""
    u_matrix = U3(theta, phi, lambda_)
    top_left = np.eye(2)
    bottom_right = u_matrix
    return np.block([[top_left, np.zeros((2, 2))], [np.zeros((2, 2)), bottom_right]])

# --- Pauli Matrices and Interactions ---

simga_i = np.array([[1,0],[0,1]])
sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0,-1j],[1j,0]])
sigma_z = np.array([[1,0],[0,-1]])

sigma_xx = np.kron(sigma_x, sigma_x)
sigma_yy = np.kron(sigma_y, sigma_y)
sigma_zz = np.kron(sigma_z, sigma_z)


def U3_pauli(gamma1, gamma2, gamma3):
    """Unitary generated by Hamiltonian H = g1*X + g2*Y + g3*Z."""
    return expm(1j * (gamma1*sigma_x + gamma2*sigma_y + gamma3*sigma_z))
    
def controlled_U3_pauli(gamma1, gamma2, gamma3, phase=0.0):
    """Controlled version of the Pauli-generated unitary."""
    u_matrix = np.exp(1j*phase) * U3_pauli(gamma1, gamma2, gamma3)
    top_left = np.eye(2)
    bottom_right = u_matrix
    return np.block([[top_left, np.zeros((2, 2))], [np.zeros((2, 2)), bottom_right]])


# --- Standard Gates Definitions ---

Cnot1 = np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]]) # Control on q0
Cnot2 = np.array([[1,0,0,0], [0,0,0,1], [0,0,1,0], [0,1,0,0]]) # Control on q1

DCnot12 = Cnot2 @ Cnot1
DCnot21 = Cnot1 @ Cnot2

Swap = np.array([[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]])
X = np.array([[0,1],[1,0]])
IX = np.kron(np.eye(2), X)
XI = np.kron(X, np.eye(2))
XX = np.kron(X, X)
II = np.eye(4)

def Ztheta(phase):
    """Phase gate: diag(1, e^{i*phase})."""
    return np.array([[1,0], [0, np.exp(1j * phase)]])

def ZthetaI(phase):
    """Phase gate on qubit 1, Identity on qubit 2."""
    I = np.eye(2)
    return np.kron(Ztheta(phase), I)

def IZtheta(phase):
    """Identity on qubit 1, Phase gate on qubit 2."""
    I = np.eye(2)
    return np.kron(I, Ztheta(phase))

def ZZtheta(phase1, phase2):
    """Two-qubit phase gate."""
    return ZthetaI(phase1) @ IZtheta(phase2)

def CZtheta(phase):
    """Controlled-Phase gate with arbitrary phase."""
    I = np.eye(4, dtype=complex)
    I[-1,-1] = np.exp(1j * phase)
    return I

# --- SIC POVM Algorithm Implementation ---

def get_sic_parameters(target_u_sic):
    """
    Calculates the circuit parameters for a SIC (Symmetric Informationally Complete) POVM 
    based on Algorithm 1 and Algorithm 2 from the paper:
    Phys. Rev. Applied 23, 014021 (2025).
    
    Args:
        target_u_sic (np.ndarray): The target SIC POVM unitary matrix U_{SIC}^{(1)} 
                                   (referred to as Ub1 in original context).
        
    Returns:
        u_s (np.ndarray): The single-qubit rotation gate U_S.
        betas (tuple): A tuple of phase parameters (beta1, beta2, beta3).
        c_flag (bool): Relabeling parameter 'c' (True implies c=0/CNOT required, 
                       False implies c=1/Identity).
    """
    
    # --- Helper Functions ---
    def get_phase_angle(complex_val):
        """Extracts the argument (phase) of a complex number."""
        return np.angle(complex_val)

    def get_orthogonal_vec(v):
        """Returns a vector orthogonal to the 2D input vector v."""
        return np.array([-v[1].conjugate(), v[0].conjugate()])

    def construct_unitary_from_vec_mapping(src_vec, target_vec):
        """
        Constructs a unitary matrix that maps src_vec to target_vec.
        Corresponds to constructing U^{(i)} such that <phi_1| U = <0|.
        """
        src_vec = src_vec / np.linalg.norm(src_vec)
        target_vec = target_vec / np.linalg.norm(target_vec)
        
        # Construct U = |target><src| + |target_orth><src_orth|
        return (np.outer(src_vec.conjugate(), target_vec) + 
                np.outer(get_orthogonal_vec(src_vec).conjugate(), get_orthogonal_vec(target_vec)))

    # --- 1. Prepare Reference Unitary U_{SIC}^{(2)} (Bell Measurement Circuit) ---
    # Corresponds to Appendix D, Eq. (D1) and Sec III.D of the paper.
    # Bell basis measurement circuit: H on target, CNOT (control=target, target=ancilla).
    qc = QuantumCircuit(2)
    qc.rx(np.arccos(1/np.sqrt(3)), 1) # Set 2 fiducial state preparation part 1
    qc.rz(3 * np.pi / 4, 1)           # Set 2 fiducial state preparation part 2
    qc.cx(1, 0)                       # CNOT
    qc.h(1)                           # Hadamard
    ref_u_sic = Operator(qc).data     # U_{SIC}^{(2)}

    # --- 2. Extract Upper and Lower Blocks of Matrix V (V_up, V_down) ---
    # Corresponds to Eq. (E2): V_up contains the first two rows, V_down the last two.
    def get_v_matrix_elements(matrix):
        # Extract first two columns (corresponds to V^dagger transpose in paper, or columns of U_P)
        return matrix[:, :2]

    v_matrix_target = get_v_matrix_elements(target_u_sic) # V^{(1)}
    v_matrix_ref = get_v_matrix_elements(ref_u_sic)       # V^{(2)}

    # Split into Up (first 2 rows) and Down (last 2 rows)
    # See Eq. (E2)
    v_up_target = v_matrix_target[[0, 1], :]  
    v_up_ref = v_matrix_ref[[0, 1], :]        
    
    v_down_target = v_matrix_target[[2, 3], :] 
    v_down_ref = v_matrix_ref[[2, 3], :]       

    # --- 3. Compute U_S (Single Qubit Unitary) ---
    # Corresponds to Algorithm 1, Steps 3-5.
    
    # Extract the first row vector (corresponding to <phi_1|)
    phi1_target = v_up_target[0, :] # <phi_1^{(1)}| 
    phi1_ref = v_up_ref[0, :]       # <phi_1^{(2)}| 
    
    # Construct rotation matrix to map phi1 to |0> (or <0|)
    # U^{(1)}: <phi_1^{(1)}| U^{(1)} = <0|
    u_rot_target = construct_unitary_from_vec_mapping(phi1_target, np.array([1, 0])) # U_2r
    # U^{(2)}: <phi_1^{(2)}| U^{(2)} = <0|
    u_rot_ref = construct_unitary_from_vec_mapping(phi1_ref, np.array([1, 0]))       # U_1r

    # Rotate the second row vector phi2
    # <phi_2'| = <phi_2| U
    phi2_prime_target = v_up_target[1, :] @ u_rot_target 
    phi2_prime_ref = v_up_ref[1, :] @ u_rot_ref          

    # Extract phases (Algorithm 1 Step 4)
    # theta ~ alpha_{i1} (coefficient of <0|) or alpha_{i2} (coefficient of <1|)
    alpha_target = get_phase_angle(phi2_prime_target[0])[1] if hasattr(phi2_prime_target[0], '__len__') else get_phase_angle(phi2_prime_target[0]) # theta21
    alpha_ref = get_phase_angle(phi2_prime_ref[0])       # theta11

    # Compute beta2 (Logic near Eq. (E5): beta2 = alpha11 - alpha21)
    beta2 = alpha_target - alpha_ref 

    # Construct intermediate correction matrix A_theta 
    # (Corresponds to part of U_r functionality or R_z(beta2))
    a_theta = np.eye(2, dtype=complex)
    a_theta[1, 1] = np.exp(1j * beta2) # Phase correction term

    # Calculate U_S
    # From Eq. (E3): R_z(beta2) V_{up}^{(2)} U_S = V_{up}^{(1)}
    # => U_S = (R_z(beta2) V_{up}^{(2)})^{-1} V_{up}^{(1)}
    v_up_ref_corrected = a_theta @ v_up_ref 
    u_s = np.linalg.inv(v_up_ref_corrected) @ v_up_target

    # --- 4. Compute Relabeling Parameter c and Remaining Phases ---
    # Corresponds to Algorithm 1 Steps 6-9 and Algorithm 2 Step 3.
    
    # Compute U_{pr} (Permutation & Phase Matrix)
    # Eq. (E9): U_{pr} = V_{down}^{(1)} * (V_{down}^{(2)} * U_S)^{-1}
    u_pr = v_down_target @ np.linalg.inv(v_down_ref @ u_s)

    # Determine c (Relabeling flag)
    # Algorithm 1 Step 7: If U_{pr} is diagonal, c=1 (Identity); otherwise c=0 (CNOT).
    # If U_pr[0,0] is close to 0, it indicates an anti-diagonal matrix (X/CNOT operation) -> c=0.
    # Note: The implementation returns 'True' for 'CNOT needed' (anti-diagonal), which corresponds to c=0 in the paper.
    cnot_flag = np.allclose(u_pr[0, 0], 0)

    # Compute beta1, beta3
    # Algorithm 2 Step 3:
    # beta1 = arg(sum(first column of U_pr))
    # beta3 = arg(sum(second column of U_pr)) - beta1 - beta2
    beta1 = get_phase_angle(np.sum(u_pr[:, 0]))
    beta_sum_col2 = get_phase_angle(np.sum(u_pr[:, 1]))
    beta3 = beta_sum_col2 - beta1 - beta2

    return u_s, (beta1, beta2, beta3), cnot_flag