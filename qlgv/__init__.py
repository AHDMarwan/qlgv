"""QLGV: Quantum Local Gradient Variance Toolkit"""

from .utils import set_seed, make_synthetic_data
from .ansatzes import hea, basic_ry_cnot, z_obs
from .encoders import angle_product, amplitude
from .samplers import uniform_sampler, normal_sampler, tiny_noise
from .qnodes import make_node
from .lgv import compute_lgv, compute_shot_noise
from .entropy import entropy, partial_trace
from .benchmark import benchmark

__version__ = "0.1.0"
__author__ = "AHDMarwan"

__all__ = [
    "set_seed",
    "make_synthetic_data",
    "hea",
    "basic_ry_cnot",
    "z_obs",
    "angle_product",
    "amplitude",
    "uniform_sampler",
    "normal_sampler",
    "tiny_noise",
    "make_node",
    "compute_lgv",
    "compute_shot_noise",
    "entropy",
    "partial_trace",
    "benchmark",
]
