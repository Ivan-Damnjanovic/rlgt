import networkx as nx
import numpy as np
import torch.nn as nn
import torch.optim as optim

from rl_graph_theory.graphs.graph_batch import GraphBatch
from rl_graph_theory.agents.cross_entropy_method import DeepCrossEntropyMethod
from rl_graph_theory.environments.linear_environments import LinearBuildEnvironment
from rl_graph_theory.environments.graph_environment import RewardType
from rl_graph_theory.graphs.graph_format import FlattenedOrdering


def graph_invariant(graph_batch: GraphBatch):
    result = np.zeros((graph_batch.batch_size,), dtype=np.float32)

    for index in range(graph_batch.batch_size):
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(list(range(graph_batch.order)))

        count = 0
        for i in range(graph_batch.order):
            for j in range(i + 1, graph_batch.order):
                if graph_batch.flattened_row_major[index, count] == 1:
                    nx_graph.add_edge(i,j)
                count += 1
        
        if not nx.is_connected(nx_graph):
            return -100000

        evals = np.linalg.eigvalsh(nx.adjacency_matrix(nx_graph).todense())
        evalsRealAbs = np.zeros_like(evals)
        for i in range(len(evals)):
            evalsRealAbs[i] = abs(evals[i])
        lambda1 = max(evalsRealAbs)
        
        maxMatch = nx.max_weight_matching(nx_graph)
        mu = len(maxMatch)
            
        result[index] = np.sqrt(graph_batch.order - 1) + 1 - lambda1 - mu

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


policy_network = nn.Sequential(
    nn.Linear(342, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 2),
)


dcem = DeepCrossEntropyMethod(
    environment=LinearBuildEnvironment(
        reward_type=RewardType.SPARSE,
        reward_function=graph_invariant,
        graph_order=19,
        flattened_ordering=FlattenedOrdering.ROW_MAJOR,
    ),
    policy_network=policy_network,
    optimizer=optim.Adam(policy_network.parameters(), lr=0.001),
    loss_function=nn.CrossEntropyLoss(),
    new_candidates_count=200,
    elite_count=20,
    survivors_count=5,
)

dcem.reset()
generation_number = 0

while True:
    dcem.step()
    generation_number += 1

    if generation_number % 100 == 0:
        best_score = dcem.best_score
        print(f"Generations: {generation_number}. Best score: {best_score}.")

        if best_score > 0:
            print(dcem.best_graph)
            break
