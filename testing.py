import numpy as np
from graphs.graph import GraphBatch


gb = GraphBatch.from_bitmask_format(np.array(
    [
        [
            [6, 1, 9, 4],
            [8, 4, 2, 1],
            [0, 8, 0, 2],
        ]
    ]
))

print(gb.bitmask_format)
print(gb.adjacency_matrix)