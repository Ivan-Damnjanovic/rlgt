import networkx as nx
import numpy as np
import torch.nn as nn
import torch.optim as optim

from rl_graph_theory.graphs.graph import Graph
from rl_graph_theory.agents.deep_cross_entropy_method import DeepCrossEntropyMethod
from rl_graph_theory.environments.linear_environments import LinearBuildEnvironment, LinearFlipEnvironment, LinearSetEnvironment
from rl_graph_theory.environments.global_environments import GlobalSetEnvironment, GlobalFlipEnvironment
from rl_graph_theory.environments.local_environments import LocalFlipEnvironment, LocalSetEnvironment
from rl_graph_theory.environments.graph_environment import RewardType
from rl_graph_theory.graphs.graph_formats import FlattenedOrdering
from rl_graph_theory.agents.random_action_mechanisms import create_multiplication_factor_random_action_mechanism


def graph_invariant(graph_batch: Graph):
    result = np.zeros((graph_batch.batch_size,), dtype=np.float32)

    for index in range(graph_batch.batch_size):
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(list(range(graph_batch.graph_order)))

        count = 0
        for i in range(graph_batch.graph_order):
            for j in range(i + 1, graph_batch.graph_order):
                if graph_batch.flattened_row_major_colors[index, count] == 1:
                    nx_graph.add_edge(i,j)
                count += 1
        
        if not nx.is_connected(nx_graph):
            return -1000

        evals = np.linalg.eigvalsh(nx.adjacency_matrix(nx_graph).todense())
        evalsRealAbs = np.zeros_like(evals)
        for i in range(len(evals)):
            evalsRealAbs[i] = abs(evals[i])
        lambda1 = max(evalsRealAbs)
        
        maxMatch = nx.max_weight_matching(nx_graph)
        mu = len(maxMatch)
            
        result[index] = np.sqrt(graph_batch.graph_order - 1) + 1 - lambda1 - mu

    return result


# policy_network = nn.Sequential(
#     nn.Linear(342, 128),
#     nn.ReLU(),
#     nn.Linear(128, 64),
#     nn.ReLU(),
#     nn.Linear(64, 4),
#     nn.ReLU(),
#     nn.Linear(4, 2),
#     nn.Sigmoid(),
# )


# policy_network = nn.Sequential(
#     nn.Linear(342, 72),
#     nn.ReLU(),
#     nn.Dropout(0.2),
#     nn.Linear(72, 12),
#     nn.ReLU(),
#     nn.Dropout(0.2),
#     nn.Linear(12, 2),
# )


def main(graph_order: int):
    policy_network = nn.Sequential(
        nn.Linear(graph_order * (graph_order - 1), 256),
        nn.ReLU(),
        # nn.Dropout(0.1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # nn.Dropout(0.1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # nn.Dropout(0.1),
        nn.Linear(256, 2),
    )

    dcem = DeepCrossEntropyMethod(
        environment=LinearSetEnvironment(
            reward_type=RewardType.SPARSE,
            reward_function=graph_invariant,
            graph_order=16,
            flattened_ordering=FlattenedOrdering.ROW_MAJOR,
        ),
        policy_network=policy_network,
        optimizer=optim.Adam(policy_network.parameters(), lr=0.003),
        loss_function=nn.CrossEntropyLoss(),
        new_candidates_count=1000,
        elite_count=70,
        survivors_count=30,
        random_action_mechanism=create_multiplication_factor_random_action_mechanism(
            initial_random_action_probability=0.001,
            waiting_period=10,
            multiplication_factor=1.1,
            maximum_random_action_probability=0.100,
        ),
    )

    dcem.reset()

    while True:
        dcem.step()
        print(f"Generations: {dcem.step_count}. Best score: {dcem.best_score:.3f}.")

        if dcem.best_score > 0.0001:
            print("Success!")
            print(dcem.best_graph.adjacency_matrix_colors)
            break


if __name__ == "__main__":
    main(graph_order=16)