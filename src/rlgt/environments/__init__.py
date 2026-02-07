"""
The `rlgt.environments` package defines the `GraphEnvironment` abstract base class, which
formalizes a reinforcement learning environment for graph theory applications. It also provides
seven concrete environment classes that inherit from `GraphEnvironment` and implement specific
graph-building dynamics. In addition, the package defines the `EpisodeStatus` enumeration, which
describes the execution status of a batch of episodes, and includes several utility functions for
constructing graph generator functions used to initialize batches of underlying graphs.
"""

from .global_environments import GlobalFlipEnvironment, GlobalSetEnvironment
from .graph_environment import (
    EpisodeStatus,
    GraphEnvironment,
    GraphInvariant,
    GraphInvariantDiff,
)
from .graph_generators import (
    GraphGenerator,
    create_choose_two_graph_generator,
    create_edge_perturbation_graph_generator,
    create_fixed_graph_generator,
    create_random_graph_generator,
)
from .linear_environments import (
    LinearBuildEnvironment,
    LinearFlipEnvironment,
    LinearSetEnvironment,
)
from .local_environments import LocalFlipEnvironment, LocalSetEnvironment


__all__ = [
    "GlobalFlipEnvironment",
    "GlobalSetEnvironment",
    "EpisodeStatus",
    "GraphEnvironment",
    "GraphInvariant",
    "GraphInvariantDiff",
    "GraphGenerator",
    "create_choose_two_graph_generator",
    "create_edge_perturbation_graph_generator",
    "create_fixed_graph_generator",
    "create_random_graph_generator",
    "LinearBuildEnvironment",
    "LinearFlipEnvironment",
    "LinearSetEnvironment",
    "LocalFlipEnvironment",
    "LocalSetEnvironment",
]
