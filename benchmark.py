import pandas as pd
from typing import List, Dict, Callable
from .ansatzes import z_obs
from .qnodes import make_node
from .lgv import compute_lgv, compute_shot_noise
from .encoders import angle_product, amplitude 
from .samplers import uniform_sampler, normal_sampler, tiny_noise 

def benchmark(ansatzes: Dict[str,Callable], qubits: List[int], layers: List[int],
              samplers: Dict[str,Callable], encoders: Dict[str,Callable], data: List,
              device_name: str = 'default.qubit', shots: int = None, 
              G: int = 1, M: int=5, repeats: int=2):
    """
    Runs a systematic benchmark of LGV and shot noise across various configurations.
    G parameter is passed to the samplers.
    """
    results=[]
    for ans_name, ans in ansatzes.items():
        for n in qubits:
            for enc_name, enc in encoders.items():
                for L in layers:
                    if enc_name == 'amplitude' and data and len(data[0]) != 2**n:
                        print(f"[SKIP] Amplitude encoding on {n} qubits requires input size {2**n}. Skipping.")
                        continue

                    # Create the QNode for the specific setup
                    node = make_node(n, ans, obs=z_obs(n), encoder_fn=enc, 
                                     device_name=device_name, shots=shots)
                    
                    for sname, sampler_factory in samplers.items():
                        # Pass the G parameter to the sampler factory
                        sampler = sampler_factory(L, n, G=G)
                        
                        p = sampler() 
                        
                        # --- Compute LGV (Parameter Variance) ---
                        lgv_val = compute_lgv(node, sampler, n_samples=M, data=data)
                        
                        # --- Compute Shot Noise (Measurement Variance) ---
                        noise_val = compute_shot_noise(node, p, repeats=repeats, data=data)
                        
                        results.append({"Ansatz":ans_name, "Qubits":n, "Layers":L,
                                        "Sampler":sname, "Encoder":enc_name, "G_Params":G, "Shots":shots,
                                        "LGV_Mean":float(lgv_val.mean()),
                                        "LGV_Max":float(lgv_val.max()),
                                        "Noise_Mean":float(noise_val.mean()),
                                        "Noise_Max":float(noise_val.max())})
    return pd.DataFrame(results)