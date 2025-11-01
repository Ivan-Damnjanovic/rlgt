import numpy as np
from rl_graph_theory.graphs.graph import GraphBatch


# gb = GraphBatch()

gb = GraphBatch.from_bitmask_format_batch(np.array(
    [
        [
            [6, 1, 1, 0],
            [8, 4, 2, 1],
            [0, 8, 0, 2],
        ]
    ]
))

print(gb.bitmask_format_batch)
print(gb.adjacency_matrix)