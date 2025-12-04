# decomposition.py
import numpy as np
import cirq
from scipy.linalg import qr

def gram_schmidt(A: np.ndarray) -> np.ndarray:
    """
    Implements numerically stable Gram-Schmidt orthogonalization using QR decomposition.
    Note: QR decomposition typically operates on column vectors.
    """
    # Assume the rows of A are the vectors to be orthogonalized.
    # We perform QR decomposition on A.T (transpose) because scipy.linalg.qr
    # operates on columns, but our input vectors are rows.
    # Decomposition: A.T = Q * R
    q, r = qr(A.T) 
    
    # Return the transpose of Q to get the orthogonalized vectors back as rows.
    # Property: Q.T @ Q = I (Identity matrix)
    return q.T 

def kak_decomposition(matrix: np.ndarray):
    """
    Compute the KAK decomposition coefficients (x, y, z) for a 2-qubit unitary.
    """
    # Ensure the input is a 4x4 Numpy array with complex data type
    matrix = np.array(matrix, dtype=complex)
    
    # Validate dimensions: KAK decomposition applies specifically to 2-qubit unitaries (4x4)
    if matrix.shape != (4, 4):
        raise ValueError(f"KAK decomposition requires 4x4 matrix, got {matrix.shape}")
        
    # Perform the decomposition using Cirq's built-in utility
    kak = cirq.kak_decomposition(matrix)
    
    # Extract and return the canonical interaction coefficients (x, y, z)
    return kak.interaction_coefficients