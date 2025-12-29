import numpy as np
from rl_graph_theory.environments.graph_generators import create_fixed_graph_generator, create_choose_two_graph_generator, create_edge_perturbation_graph_generator, create_random_graph_generator
from rl_graph_theory.graphs.graph import Graph
from rl_graph_theory.graphs.graph_format import FlattenedOrdering

# gb = GraphBatch()

# gb = GraphBatch.from_bitmask(np.array(
#     [
#         [
#             [6, 1, 1, 0],
#             [8, 4, 2, 1],
#             [0, 8, 0, 2],
#         ]
#     ]
# ))

# print(gb.bitmask_format_batch)
# print(gb.adjacency_matrix)



# generator = create_fixed_graph_generator(
#     Graph.from_flattened(np.array([1, 0, 1, 1, 1, 0, 0, 1, 0, 0], dtype=np.uint8))
# )

# print(generator(7).bitmask_out)



# generator = create_choose_two_graph_generator(
#     first_graph=Graph.from_flattened(np.array([1, 0, 1, 1, 1, 0, 0, 1, 0, 0], dtype=np.uint8)),
#     second_graph=Graph.from_flattened(np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 1], dtype=np.uint8)),
#     second_graph_probability=0.1,
# )

# print(generator(100).bitmask_out)



generator = create_edge_perturbation_graph_generator(
    initial_graph=Graph.from_flattened(np.array([1, 0, 1, 1, 1, 0, 0, 1, 0, 0], dtype=np.uint8), edge_colors=3),
    edge_perturbation_probability=0.30,
    color_selection_probabilities=np.array([0.3, 0.5, 0.2]),
)
result = generator(10000).flattened_row_major

# print(result)
# print((result != np.array([1, 0, 1, 1, 1, 0, 0, 1, 0, 0], dtype=np.uint8)).sum() / 100000)
print(result.dtype)
print((result == 0).sum() / 100000)
print((result == 1).sum() / 100000)
print((result == 2).sum() / 100000)



# generator = create_random_graph_generator(
#     graph_order=6,
#     color_selection_probabilities=np.array([0.2, 0.7, 0.1]),
#     flattened_ordering=FlattenedOrdering.CLOCKWISE,
#     edge_colors=3,
# )
# result = generator(10000).flattened_row_major
# print(result)
# print(result.dtype)
# print((result == 0).sum() / 150000)
# print((result == 1).sum() / 150000)
# print((result == 2).sum() / 150000)