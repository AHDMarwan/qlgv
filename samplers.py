import torch
import math

def uniform_sampler(L, W, G=1, scale=math.pi):
    """Returns a function to sample parameters uniformly in [-scale, scale]. G is the number of parameters per qubit per layer."""
    return lambda: (torch.rand((L, W, G))*2 - 1)*scale

def normal_sampler(L, W, G=1, std=0.1):
    """Returns a function to sample parameters from a Gaussian distribution N(0, std). G is the number of parameters per qubit per layer."""
    return lambda: torch.randn((L, W, G))*std

def tiny_noise(L, W, G=1, std=1e-3):
    """Returns a function to sample parameters from a Gaussian distribution N(0, std) for perturbation. G is the number of parameters per qubit per layer."""
    return lambda: torch.randn((L, W, G))*std