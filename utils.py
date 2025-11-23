import numpy as np
import torch

def set_seed(seed=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

def make_synthetic_data(n_samples, n_features, seed=123):
    rng = np.random.RandomState(seed)
    return [rng.normal(scale=0.6, size=(n_features,)) for _ in range(n_samples)]