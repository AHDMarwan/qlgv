import numpy as np

def partial_trace(statevec, keep, n_wires):
    """Computes the reduced density matrix rho_A by tracing out the complement of 'keep'."""
    if not isinstance(statevec, np.ndarray) or statevec.ndim != 1 or statevec.size != 2**n_wires:
        raise ValueError("State vector must be a 1D array of size 2^n_wires.")
        
    state_tensor = statevec.reshape([2]*n_wires)
    keep = sorted(list(keep))
    trace_out = [i for i in range(n_wires) if i not in keep]
    
    perm = keep + trace_out
    permuted = np.transpose(state_tensor, perm)
    
    k = len(keep)
    dim_keep = 2**k
    dim_trace = 2**(n_wires - k)
    
    mat = permuted.reshape(dim_keep, dim_trace)
    rho = np.tensordot(mat, mat.conj(), axes=([1],[1]))
    return rho

def entropy(statevec, keep, n_wires, eps=1e-12):
    """Computes the von Neumann entanglement entropy S(rho_A) for a partition 'keep'."""
    rho = partial_trace(statevec, keep, n_wires)
    
    vals = np.linalg.eigvalsh(rho)
    vals = np.clip(np.real(vals), eps, None)
    
    return float(-np.sum(vals * np.log2(vals)))