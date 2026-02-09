"""
The `rlgt.graphs` package provides the `Graph` class, which models a *k*-edge-colored looped
complete graph as well as a batch of *k*-edge-colored looped complete graphs of the same order. The
package also provides the `GraphFormat` enumeration, which defines the formats used to represent a
*k*-edge-colored looped complete graph, along with several auxiliary functions and supporting
classes for constructing *k*-edge-colored looped complete graphs with specific structures.
"""

from .graph import Graph
from .graph_formats import (
    BitmaskType,
    ColorRepresentation,
    FlattenedOrdering,
    GraphFormat,
)
from .special_graphs import (
    AlmostCompleteGraph,
    BookGraph,
    CompleteBipartiteGraph,
    CompleteGraph,
    CompleteKPartiteGraph,
    CycleGraph,
    EmptyGraph,
    FriendshipGraph,
    MonochromaticGraph,
    PathGraph,
    StarGraph,
    WheelGraph,
)
from .utils import (
    binary_slices_to_color_numbers,
    color_numbers_to_binary_slices,
    compute_edge_indices,
    flatten_from_adjacency_matrix,
    flattened_length_to_graph_order,
    graph_order_to_flattened_length,
    unflatten_to_adjacency_matrix,
)


__all__ = [
    "Graph",
    "BitmaskType",
    "ColorRepresentation",
    "FlattenedOrdering",
    "GraphFormat",
    "AlmostCompleteGraph",
    "BookGraph",
    "CompleteBipartiteGraph",
    "CompleteGraph",
    "CompleteKPartiteGraph",
    "CycleGraph",
    "EmptyGraph",
    "FriendshipGraph",
    "MonochromaticGraph",
    "PathGraph",
    "StarGraph",
    "WheelGraph",
    "binary_slices_to_color_numbers",
    "color_numbers_to_binary_slices",
    "compute_edge_indices",
    "flatten_from_adjacency_matrix",
    "flattened_length_to_graph_order",
    "graph_order_to_flattened_length",
    "unflatten_to_adjacency_matrix",
]
