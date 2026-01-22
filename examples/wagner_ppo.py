import networkx as nx
import numpy as np
import torch.nn as nn
import torch.optim as optim

from rl_graph_theory.graphs.graph import Graph
from rl_graph_theory.agents.ppo_agent import PPOAgent
from rl_graph_theory.environments.linear_environments import LinearBuildEnvironment
from rl_graph_theory.environments.graph_environment import RewardType
from rl_graph_theory.graphs.graph_formats import FlattenedOrdering


def graph_invariant(graph_batch: Graph):
    adj_matrices = graph_batch.adjacency_matrix_colors
    batch_size = adj_matrices.shape[0]
    order = adj_matrices.shape[1]
    result = np.zeros(batch_size, dtype=np.float32)

    for index in range(batch_size):
        nx_graph = nx.from_numpy_array(adj_matrices[index])
        
        if not nx.is_connected(nx_graph):
            return -1000

        evals = np.linalg.eigvalsh(adj_matrices[index].astype(np.float64))
        evalsRealAbs = np.zeros_like(evals)
        for i in range(len(evals)):
            evalsRealAbs[i] = abs(evals[i])
        lambda1 = max(evalsRealAbs)
        
        maxMatch = nx.max_weight_matching(nx_graph)
        mu = len(maxMatch)
            
        result[index] = np.sqrt(order - 1) + 1 - lambda1 - mu

    return result


def main(graph_order: int):
    # Policy network (Actor)
    policy_network = nn.Sequential(
        nn.Linear(graph_order * (graph_order - 1), 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 2),
    )

    # Value network (Critic)
    value_network = nn.Sequential(
        nn.Linear(graph_order * (graph_order - 1), 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
    )

    ppo = PPOAgent(
        environment=LinearBuildEnvironment(
            reward_type=RewardType.SPARSE,
            reward_function=graph_invariant,
            graph_order=graph_order,
            flattened_ordering=FlattenedOrdering.ROW_MAJOR,
        ),
        policy_network=policy_network,
        value_network=value_network,
        optimizer=optim.Adam(
            list(policy_network.parameters()) + list(value_network.parameters()), 
            lr=0.0003
        ),
        batch_size=20,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4,
        entropy_coef=0.01,
        value_coef=0.5,
    )

    ppo.reset()

    while True:
        ppo.step()
        print(f"Generations: {ppo.step_count}. Best score: {ppo.best_score:.3f}.")

        if ppo.best_score > 0.0001:
            print("Success!")
            print(ppo.best_graph.adjacency_matrix_colors)
            break


if __name__ == "__main__":
    main(graph_order=16)
