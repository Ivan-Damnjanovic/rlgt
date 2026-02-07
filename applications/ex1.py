import networkx as nx
import numpy as np
import torch.nn as nn
import torch.optim as optim

from rl_graph_theory.graphs.graph import Graph
from rl_graph_theory.agents.deep_cross_entropy_agent import DeepCrossEntropyAgent
from rl_graph_theory.environments.linear_environments import LinearBuildEnvironment, LinearFlipEnvironment, LinearSetEnvironment
from rl_graph_theory.environments.global_environments import GlobalSetEnvironment, GlobalFlipEnvironment
from rl_graph_theory.environments.local_environments import LocalFlipEnvironment, LocalSetEnvironment
from rl_graph_theory.environments.graph_environment import RewardType
from rl_graph_theory.graphs.graph_formats import FlattenedOrdering
from rl_graph_theory.agents.random_action_mechanisms import create_multiplication_factor_random_action_mechanism


def graph_invariant(graph_batch: Graph):
    adj = graph_batch.adjacency_matrix_colors.astype(np.float32)

    edges = np.sum(adj, axis=(1, 2)) // 2
    cycles = np.trace(adj @ adj @ adj, axis1=1, axis2=2) // 6

    return edges - cycles


policy_network = nn.Sequential(
    nn.Linear(306, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(128, 2),
)


dcem = DeepCrossEntropyAgent(
    environment=LinearBuildEnvironment(
        reward_type=RewardType.SPARSE,
        reward_function=graph_invariant,
        graph_order=18,
        flattened_ordering=FlattenedOrdering.CLOCKWISE,
    ),
    policy_network=policy_network,
    optimizer=optim.Adam(policy_network.parameters(), lr=0.001),
    loss_function=nn.CrossEntropyLoss(),
    candidates_count=300,
    elite_count=30,
    survivors_count=20,
    random_action_mechanism=create_multiplication_factor_random_action_mechanism(
        initial_random_action_probability=0.001,
        waiting_period=10,
        multiplication_factor=1.1,
        maximum_random_action_probability=0.05,
    ),
)

dcem.reset()

while True:
    dcem.step()

    if dcem.step_count % 1 == 0:
        print(f"Generations: {dcem.step_count}. Best score: {dcem.best_score}.")

        # print(dcem.best_graph.adjacency_matrix_colors)

        if dcem.best_score > 70:
            print(dcem.best_graph.adjacency_matrix_colors)
            break
