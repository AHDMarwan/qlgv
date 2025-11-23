import pennylane as qml

def z_obs(n_wires, target=0):
    """Returns the PauliZ observable on the target wire."""
    if target >= n_wires:
        raise ValueError("Target wire index must be less than the number of wires.")
    return qml.Hamiltonian([1.0], [qml.PauliZ(target)])

def hea(params, wires):
    """Hardware Efficient Ansatz (HEA) with alternating CZ entangling layers.
    Assumes params shape is (L, W, G). Only the first parameter (index 0) is used for RY rotations."""
    
    # Ensure params is 3D for consistent indexing
    if params.dim() == 2:
        params = params.unsqueeze(2)
        
    L, W, G = params.shape
    if W != len(wires):
        raise ValueError("Parameter width must match the number of wires.")

    # Use only the first parameter group for RY
    p_ry = params[:, :, 0] 

    for l in range(L):
        # Apply single-qubit rotations
        for w in range(W):
            qml.RY(p_ry[l, w].item(), wires=wires[w])
        
        # Apply entangling layer
        if l < L - 1:
            if l % 2 == 0:
                for i in range(0, W-1, 2):
                    qml.CZ(wires=[wires[i], wires[i+1]])
            else:
                for i in range(1, W-1, 2):
                    qml.CZ(wires=[wires[i], wires[i+1]])

def basic_ry_cnot(params, wires):
    """Basic Ansatz: RY rotations followed by linear CNOT chain entanglers.
    Assumes params shape is (L, W, G). Only the first parameter (index 0) is used for RY rotations."""
    
    # Ensure params is 3D for consistent indexing
    if params.dim() == 2:
        params = params.unsqueeze(2)
        
    L, W, G = params.shape
    if W != len(wires):
        raise ValueError("Parameter width must match the number of wires.")

    # Use only the first parameter group for RY
    p_ry = params[:, :, 0]

    for l in range(L):
        # Apply single-qubit rotations
        for w in range(W):
            qml.RY(p_ry[l, w].item(), wires=wires[w])
        
        # Apply linear CNOT entanglement
        if l < L - 1:
            for i in range(W - 1):
                qml.CNOT(wires=[wires[i], wires[i+1]])