import pennylane as qml

def make_node(n_wires, ansatz_fn, obs=None, encoder_fn=None, device_name='default.qubit', shots=None):
    """Creates a PennyLane QNode returning either state vector or expectation value, using 'torch' interface."""
    dev = qml.device(device_name, wires=n_wires, shots=shots)

    if obs is None:
        @qml.qnode(dev, interface='torch')
        def node(params, x=None):
            # Pass 3D tensor to ansatz (or 2D if the original was 2D)
            if encoder_fn is not None and x is not None:
                encoder_fn(x, list(range(n_wires)))
            ansatz_fn(params, list(range(n_wires)))
            return qml.state()
    else:
        @qml.qnode(dev, interface='torch')
        def node(params, x=None):
            # Pass 3D tensor to ansatz (or 2D if the original was 2D)
            if encoder_fn is not None and x is not None:
                encoder_fn(x, list(range(n_wires)))
            ansatz_fn(params, list(range(n_wires)))
            return qml.expval(obs)

    return node