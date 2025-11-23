import pennylane as qml
import numpy as np

def angle_product(x, wires, scale=1.0):
    """Angle Encoding: map input features to RX rotation angles."""
    feats = np.array(x).flatten()
    n_wires = len(wires)
    
    if feats.size < n_wires:
        feats = np.resize(feats, n_wires)
    elif feats.size > n_wires:
        feats = feats[:n_wires]
        
    for i, w in enumerate(wires):
        qml.RX(float(feats[i]*scale), wires=w)

def amplitude(x, wires):
    """Amplitude Encoding: encode normalized feature vector into state amplitudes."""
    feats = np.array(x).flatten().astype(float)
    n = len(wires)
    required_size = 2**n
    
    if feats.size != required_size:
        raise ValueError(f"Amplitude encoding requires input size={required_size} for {n} qubits, got {feats.size}")
    
    norm = np.linalg.norm(feats)
    if norm < 1e-12:
        feats = np.zeros_like(feats)
        feats[0] = 1.0
    else:
        feats /= norm
        
    qml.AmplitudeEmbedding(feats, wires=wires, normalize=False)