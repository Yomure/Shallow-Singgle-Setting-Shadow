from qiskit.quantum_info import Operator
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from ipywidgets import interactive

# --- Basic Quantum/Linear Algebra Utilities ---

def is_unitary(matrix):
    """Checks if the given matrix is Unitary (U^dagger U = I)."""
    return Operator(matrix).is_unitary()

def is_vec_orthogonal(vector1, vector2, tol=1e-10):
    """Checks if two vectors are orthogonal (inner product is close to 0)."""
    return abs(np.vdot(vector1, vector2)) < tol

def is_vec_normalized(vector, tol=1e-10):
    """Checks if the vector has a Euclidean norm of 1."""
    return abs(np.linalg.norm(vector) - 1) < tol

def vec2density(vector):
    """
    Converts a state vector |psi> to a density matrix rho = |psi><psi|.
    Uses outer product.
    """
    return np.outer(vector, vector.conj())

def vec2povm(vector):
    """
    Converts a vector to a POVM (Positive Operator-Valued Measure) element.
    Normalizes the vector and scales by 1/dimension.
    Formula: E = (1/d) * |psi><psi|
    """
    vector = vector / np.linalg.norm(vector)
    return np.outer(vector, vector.conj()) / (vector.size)

def vectorize_matrix(matrix):
    """
    Vectorizes a matrix by stacking columns (Column-major / Fortran order).
    This corresponds to the standard |A>> notation in quantum information.
    """
    return matrix.flatten(order='F')

def devectorize_matrix(vector):
    """
    Reshapes a vectorized operator back into a square matrix.
    Assumes the original matrix was square.
    """
    dim = int(np.sqrt(vector.size))
    return vector.reshape(dim, dim, order='F')

# --- Visualization: Weyl Chamber (2-Qubit Gate Geometry) ---

def plot_weyl_chamber(points, x_rotation=30, y_rotation=30, framework=1):
    """
    Plots points inside the Weyl Chamber (geometric representation of 2-qubit gates).
    
    Args:
        points: List of (x, y, z) coordinates (canonical coordinates).
        framework: Determines the shape of the Weyl Chamber boundaries (0 or 1).
                   Framework 1 typically corresponds to the standard tetrahedron
                   bounded by Identity, ISWAP, SWAP, etc.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate colors based on the index of points (gradient)
    num_points = len(points)
    color_range = np.linspace(0, 1, num_points)
    colors = cm.viridis(color_range)
    
    xs, ys, zs = zip(*points)
    for x, y, z, color in zip(xs, ys, zs, colors):
        ax.scatter(x, y, z, c=[color], marker='.')

    # Define vertices for the Weyl Chamber tetrahedron
    if framework == 0:
        vertices = np.array([
            [0, 0, 0],                  # Identity / Origin
            [np.pi/2, 0, 0],            # X tensor X direction
            [np.pi/4, np.pi/4, 0],      # Y tensor Y direction
            [np.pi/4, np.pi/4, np.pi/4] # Z tensor Z direction
        ])   
    elif framework == 1:
        # Standard Weyl Chamber vertices
        vertices = np.array([
            [0, 0, 0],                    # Origin
            [np.pi/4, 0, 0],              # Point A (e.g., CNOT equivalent class)
            [np.pi/4, np.pi/4, -np.pi/4], # Point L
            [np.pi/4, np.pi/4, np.pi/4]   # Point M (SWAP)
        ]) 
    
    # Define edges connecting the vertices
    boundary_lines = [
        [vertices[0], vertices[1]],
        [vertices[0], vertices[2]],
        [vertices[0], vertices[3]],
        [vertices[1], vertices[2]],
        [vertices[1], vertices[3]],
        [vertices[2], vertices[3]]
    ]
    
    # Plot the wireframe of the chamber
    for line in boundary_lines:
        ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], [line[0][2], line[1][2]], 'r-')
        
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(x_rotation, y_rotation)
    plt.show()
    
def interactive_plot(points, default_x=30, default_y=30, default_height='800px', default_width='800px', framework=1):
    """Creates an interactive widget to rotate the Weyl Chamber plot."""
    interactive_plot = interactive(
        lambda x_rotation=default_x, y_rotation=default_y: plot_weyl_chamber(points, x_rotation, y_rotation, framework),
        x_rotation=(0, 360, 1), 
        y_rotation=(0, 360, 1)
    )
    output = interactive_plot.children[-1]
    output.layout.height = default_height
    output.layout.width = default_width 
    return interactive_plot

# --- SIC-POVM (Symmetric Informationally Complete POVM) Analysis ---

def get_pi_vec(matrix):
    """
    Extracts column vectors from the first two columns of a matrix 
    and computes their tensor products. 
    (Context dependent: Likely used for generating SIC vectors from a unitary).
    """
    first_two_columns = matrix[:, :2]
    result_list = []
    for row in first_two_columns:
        # Create a column vector from the row elements
        column_vector = np.array([row]).T.conj()
        # Compute tensor product: v \otimes v
        tensor_product = np.tensordot(column_vector.conj(), column_vector, axes=0)
        result_list.append(tensor_product.reshape(-1))
    return result_list

def is_SIC_U(matrix):
    """
    Verifies if a matrix generates a SIC-POVM (Symmetric Informationally Complete).
    
    Conditions checked:
    1. Normalization: Inner product <v|v> = 1/d^2 (based on specific scaling in get_pi_vec)
    2. Equiangularity: Pairwise overlap |<v_i|v_j>|^2 is constant.
    """
    d = np.sqrt(matrix.shape[0]) # Dimension
    vector_list = get_pi_vec(matrix)
    
    # Check 1: Self-overlap (Normalization scaling)
    for vector in vector_list:
        if not np.isclose(np.vdot(vector, vector), 1 / d**2):
            return False
            
    # Check 2: Pairwise overlap (SIC condition)
    # The expected constant is 1 / (d^2 * (d + 1))
    for i in range(len(vector_list)):
        for j in range(i + 1, len(vector_list)):
            if not np.isclose(np.vdot(vector_list[i], vector_list[j]), 1 / (d**2 * (d + 1))):
                return False
    return True

# --- Linear Algebra Helpers ---

def get_column_matrix(matrix, column: list):
    """Extracts specific columns from a matrix to form a new sub-matrix."""
    return np.array([matrix[:, i] for i in column]).T

def have_solution(m1, m2):
    """
    Determines the number of solutions for a linear system using the Rouché–Capelli theorem.
    Compares Rank(A) vs Rank(A|b).
    
    Returns:
        1: Unique solution
        2: Infinite solutions
        0: No solution
    """
    rank1 = np.linalg.matrix_rank(m1)
    n = m1.shape[1]
    imp = np.hstack((m1, m2))
    rank2 = np.linalg.matrix_rank(imp)
    
    if rank1 == rank2 == n:
        return 1
    if rank1 == rank2 < n:
        return 2
    if rank1 < rank2:
        return 0
        
def snapshot_data_split(data):
    """
    Splits and reorders data strings.
    Example: Takes a list, splits it in half, and concatenates the i-th element 
    of the first half with the i-th element of the second half.
    Likely used for processing readout bitstrings.
    """
    data_s = []
    half = len(data) // 2
    for i in range(half):
        data_s.append(f'{data[i]}{data[half+i]}')
    return data_s   

def compute_kronecker_product(matrices):
    """Computes the cumulative Kronecker (Tensor) product for a list of matrices."""
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result

# --- Tomography / Frame Operator Construction ---

def compute_frame_operator_inv_and_dict(sic_povms_states: list, apply_key_swap: bool = False):
    """
    Computes the inverse of the Frame Operator and the dictionary of POVM elements.
    This is essential for Linear Inversion Tomography.
    
    Args:
        sic_povms_states: List of fiducial vectors defining the SIC-POVM.
        apply_key_swap: If True, swaps keys '10' and '11' (correction for specific CNOT readout logic).
        
    Returns:
        Sp_inv: The inverse of the Frame Operator (Superoperator).
        vec_povms_dict: Dictionary mapping bitstrings to vectorized POVM elements.
    """
    # 1. Generate POVM elements: E_i = (1/d) * |psi><psi|
    povms_elements = [vec2povm(vec) for vec in sic_povms_states]
    
    # 2. Vectorize the POVM matrices into supervectors: |E_i>>
    vec_povms_elements = [vectorize_matrix(element) for element in povms_elements]
    
    # 3. Compute the Frame Operator S (Superoperator)
    # S = Sum |E_i>> <<E_i|
    povms_pis = [np.outer(element, element.conj()) for element in vec_povms_elements]
    Sp = np.sum(povms_pis, axis=0)
    
    # 4. Invert the Frame Operator
    # We use pseudo-inverse (pinv) as a fallback if the matrix is near-singular
    try:
        Sp_inv = np.linalg.inv(Sp)
    except np.linalg.LinAlgError:
        Sp_inv = np.linalg.pinv(Sp)

    # 5. Build the dictionary mapping binary strings to POVM elements
    # Determine number of bits needed for binary representation
    num_bits = (len(vec_povms_elements) - 1).bit_length()
    fmt_str = f'0{num_bits}b'
    
    vec_povms_dict = {
        format(i, fmt_str): vec
        for i, vec in enumerate(vec_povms_elements)
    }
    
    # 6. Apply key swapping (Relabeling)
    # This is often needed in hardware experiments where readout lines map differently
    if apply_key_swap:
        if '10' in vec_povms_dict and '11' in vec_povms_dict:
            vec_povms_dict['10'], vec_povms_dict['11'] = \
                vec_povms_dict['11'], vec_povms_dict['10']
            
    return Sp_inv, vec_povms_dict