from scipy.linalg import sqrtm
import numpy as np

def trace_distance(rho, sigma):
    """
    Computes the trace distance between two density matrices.
    
    Formula: D(rho, sigma) = 0.5 * Tr|rho - sigma|
    where |A| is the unique positive square root of A^dagger * A.
    """
    # np.linalg.norm with ord=1 computes the nuclear norm (sum of singular values),
    # which is equivalent to the trace norm for the difference of density matrices.
    return 0.5 * np.linalg.norm(rho - sigma, ord=1)

def fidelity(rho, sigma):
    """
    Computes the squared fidelity between two density matrices.
    
    Formula: F(rho, sigma) = (Tr(sqrt(sqrt(rho) @ sigma @ sqrt(rho))))^2
    """
    # Calculate the matrix square root of rho
    srho = sqrtm(rho)
    
    # Compute the inner term: sqrt(rho) * sigma * sqrt(rho)
    # Then take the square root of that result
    intermediate = sqrtm(srho @ sigma @ srho)
    
    # Calculate the trace of the result
    fid = np.trace(intermediate)
    
    # Return the squared magnitude (handling potential small imaginary parts from numerical errors)
    return np.abs(fid)**2

def reconstruction(rho):
    """
    Placeholder for state reconstruction. 
    In a full implementation, this might perform Maximum Likelihood Estimation (MLE)
    or other methods to reconstruct a physical density matrix from raw data.
    """
    return rho  

def expect(rho, obs):
    """
    Calculates the expectation value of an observable given a density matrix.
    
    Formula: <O> = Tr(rho * O)
    """
    return np.trace(rho @ obs)