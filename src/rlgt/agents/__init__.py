"""
The `rlgt.agents` package defines the `GraphAgent` abstract base class, which formalizes a
reinforcement learning agent for graph theory applications. It also provides three concrete agent
classes that inherit from `GraphAgent` and implement the Deep-Cross Entropy, REINFORCE and Proximal
Policy Optimization (PPO) methods, all based on `PyTorch`. In addition, the package includes
several auxiliary classes for defining random action mechanisms.

This package can only be used if the optional ``agents`` extra dependencies are installed.
"""

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    print("To use the 'rlgt.agents' package, please install the 'agents' extra dependencies.")
    exit(1)

from .deep_cross_entropy_agent import DeepCrossEntropyAgent
from .graph_agent import GraphAgent
from .ppo_agent import PPOAgent
from .random_action_mechanisms import (
    ConstantRandomActionMechanism,
    ExponentialRandomActionMechanism,
    NoRandomActionMechanism,
    RandomActionMechanism,
)
from .reinforce_agent import ReinforceAgent


__all__ = [
    "DeepCrossEntropyAgent",
    "GraphAgent",
    "PPOAgent",
    "ConstantRandomActionMechanism",
    "ExponentialRandomActionMechanism",
    "NoRandomActionMechanism",
    "RandomActionMechanism",
    "ReinforceAgent",
]
