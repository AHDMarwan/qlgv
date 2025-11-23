# QLGV: Quantum Local Gradient Variance Toolkit

QLGV is a Python package designed for researchers and practitioners to analyze the **trainability** of **Variational Quantum Circuits (VQC)**, specifically by measuring the **Local Gradient Variance (LGV)**. A vanishing LGV is a strong indicator of the **Barren Plateau** phenomenon .

The package is built for maximum flexibility, supporting both simple 2D parameter tensors ($L \times W$) and complex 3D tensors ($L \times W \times G$) to accommodate various circuit architectures.

-----

## Installation and Setup

Execute the following commands to install the package :

```bash
# Clone the repository
!git clone https://github.com/AHDMarwan/qlgv.git
```

### Standard Import

Once the generation cell has run and the package is built locally:

```python
import qlgv
import torch
import pandas as pd
```
-----

## Key Features

| Feature | Description | Relevant Module |
| :--- | :--- | :--- |
| **Flexible Parameters** | Supports parameter tensors of shape $(L, W, G)$, where $G$ is the number of rotation gates per qubit per layer (defaults to 1). | `lgv.py`, `samplers.py` |
| **LGV Calculation** | Computes the **variance** of the gradient components over a distribution of random parameters. | `qlgv.compute_lgv` |
| **Parameter-Shift** | Implements the exact parameter-shift rule ($\pm \pi/2$ shifts) for gradient calculation . | `qlgv.lgv.gradient` |
| **Pre-built Ansatzes** | Includes standard circuits like the **Hardware Efficient Ansatz (HEA)** and a basic RY-CNOT chain. | `qlgv.ansatzes` |
| **Benchmarking** | High-level function to systematically test LGV across different depths, samplers, and encoders. | `qlgv.benchmark` |
| **Entanglement** | Tools to compute the **Von Neumann Entropy** and **Partial Trace**, often used in analyzing circuit expressivity. | `qlgv.entropy` |

-----

## Usage Example: Running a QLGV Benchmark

This example demonstrates how to configure and run a comprehensive test of LGV vs. circuit depth.

```python
import qlgv
import torch
import pandas as pd

# --- Configuration ---
N_QUBITS = 4
L_LAYERS = [1, 3, 5] 
N_FEATURES = 4
G_PARAMS = 1 # Using the default 2D parameter structure

# 1. Data and Setup
qlgv.set_seed(42)
data = qlgv.make_synthetic_data(n_samples=50, n_features=N_FEATURES)

# 2. Define Components (Samplers must accept L, W, G)
ansatz_config = {"HEA": qlgv.hea}
encoder_config = {"AngleProduct": qlgv.angle_product}

sampler_config = {
    # The lambda function must accept L, W, AND G
    "UniformPi": lambda L, W, G: qlgv.uniform_sampler(L, W, G=G, scale=torch.pi), 
    "Normal0.1": lambda L, W, G: qlgv.normal_sampler(L, W, G=G, std=0.1) 
}

# 3. Run Benchmark
df_results = qlgv.benchmark(
    ansatzes=ansatz_config,
    qubits=[N_QUBITS],
    layers=L_LAYERS,
    samplers=sampler_config,
    encoders=encoder_config,
    data=data,
    G=G_PARAMS,      # Pass G=1 to the benchmark
    M=10,            # Number of parameter sets for variance calculation
    shots=None       # Exact simulation
)

# 4. Analyze Results
print(df_results[['Layers', 'Sampler', 'LGV_Mean', 'LGV_Max']])
```

-----

## Advanced Usage: Custom 3D Ansatz (G \> 1)

This shows how to define and test a circuit that requires **multiple parameters per qubit per layer** (e.g., $G=3$ for $\text{RX}-\text{RY}-\text{RZ}$ blocks).

### Step 1: Define a Custom Ansatz

The custom ansatz must be written to explicitly access the $G$ indices of the 3D `params` tensor.

```python
import pennylane as qml

# This function expects a 3D parameter tensor (L, W, G=3)
def rx_ry_rz_ansatz(params, wires):
    L, W, G = params.shape
    if G != 3: raise ValueError("RXYZ ansatz requires G=3.")
    
    for l in range(L):
        for w in range(W):
            qlgv.qml.RX(params[l, w, 0].item(), wires=wires[w]) # G=0
            qlgv.qml.RY(params[l, w, 1].item(), wires=wires[w]) # G=1
            qlgv.qml.RZ(params[l, w, 2].item(), wires=wires[w]) # G=2
        
        # Linear CNOT entanglement layer
        if l < L - 1:
            for i in range(W - 1):
                qml.CNOT(wires=[wires[i], wires[i+1]])
```

### Step 2: Run Benchmark with G=3

```python
# Run benchmark with G=3
results_3d = qlgv.benchmark(
    ansatzes={"RXYZ": rx_ry_rz_ansatz},
    qubits=[4],
    layers=[2],
    samplers=sampler_config,
    encoders=encoder_config,
    data=data,
    G=3 # <-- CRITICAL: Specifies G=3 for both sampler and LGV calculation
)

print("\n--- RXYZ (G=3) Results ---")
print(results_3d[['Layers', 'G_Params', 'LGV_Mean']])
```

-----

## Development and Contribution

Feel free to clone the repository, test the package, and propose enhancements\!

**Repository Link:** [https://github.com/AHDMarwan/qlgv](https://github.com/AHDMarwan/qlgv)
