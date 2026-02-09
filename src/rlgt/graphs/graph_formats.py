"""
This ``Python`` module contains several enumerations related to *k*-edge-colored looped complete
graphs:

* `GraphFormat` defines the formats that can be used to represent a *k*-edge-colored looped
  complete graph.
* `BitmaskType` defines the types of bitmasks that can be used in the two bitmask formats.
* `ColorRepresentation` defines the styles for representing edge colors in the two adjacency matrix
  formats and the four flattened formats.
* `FlattenedOrdering` defines the order of edges (or arcs) in the four flattened formats.
"""

from enum import Enum


class GraphFormat(Enum):
    r"""
    This enumeration encapsulates the concept of a format used to represent a *k*-edge-colored
    looped complete graph. Each graph is represented as a quintuple  ``(edge_colors, is_directed,
    allow_loops, graph_format, format_representation)``, where:

    * ``edge_colors`` is the number of proper edge colors, i.e., *k*, which is at least 2.
    * ``is_directed`` is a boolean that indicates whether the graph is a  *k*-edge-colored looped
      complete directed graph or a *k*-edge-colored looped complete undirected graph.
    * ``allow_loops`` is a boolean that indicates whether the graph is allowed to have loops. If
      loops are not allowed, then all loops are removed from the graph and they simply do not
      exist.
    * ``graph_format`` is an item of this enumeration that specifies the format used to represent
      the graph structure.
    * ``format_representation`` is a `numpy.ndarray` that encodes the structure of the graph in the
      chosen format.

    The enumeration provides support for the following eight graph formats:

    1. the bitmask format for the out-neighborhoods;
    2. the bitmask format for the in-neighborhoods;
    3. the adjacency matrix format with color numbers;
    4. the adjacency matrix format with binary slices;
    5. the flattened row-major format with color numbers;
    6. the flattened row-major format with binary slices;
    7. the flattened clockwise format with color numbers; and
    8. the flattened clockwise format with binary slices.

    All formats with color numbers require *k* ≤ 256. The two bitmask formats require that the
    graph order is at most 64.

    :cvar BITMASK_OUT: The bitmask format for the out-neighborhoods. When this format is used, the
        graph is represented by a `numpy.ndarray` matrix ``a`` of type `numpy.uint64` with *k* rows
        and *n* columns, where *n* is the graph order. Each entry ``a[i, j]`` is a nonnegative
        integer whose *h*-th bit indicates whether the edge (or arc) from vertex *j* to vertex *h*
        has color *i*. If loops are not allowed, the *j*-th bit of ``a[i, j]`` is zero. For fully
        colored graphs, the first row (corresponding to color 0) can be omitted; this variant is
        called the reduced bitmask format for the out-neighborhoods.
    :cvar BITMASK_IN: The bitmask format for the in-neighborhoods. When this format is used, the
        graph is represented by a `numpy.ndarray` matrix ``a`` of type `numpy.uint64` with *k* rows
        and *n* columns, where *n* is the graph order. Each entry ``a[i, j]`` is a nonnegative
        integer whose *h*-th bit indicates whether the edge (or arc) from vertex *h* to vertex *j*
        has color *i*. If loops are not allowed, the *j*-th bit of ``a[i, j]`` is zero. For fully
        colored graphs, the first row (corresponding to color 0) can be omitted; this variant is
        called the reduced bitmask format for the in-neighborhoods.
    :cvar ADJACENCY_MATRIX_COLORS: The adjacency matrix format with color numbers. When this format
        is used, the graph is represented by a `numpy.ndarray` square matrix ``a`` of type
        `numpy.uint8` with *n* rows and *n* columns, where *n* is the graph order. Each entry
        ``a[i, j]`` stores the color of the edge (or arc) from vertex *i* to vertex *j*. Uncolored
        edges (or arcs) are represented by the value *k*. If loops are not allowed, all diagonal
        entries are zero.
    :cvar ADJACENCY_MATRIX_BINARY: The adjacency matrix format with binary slices. When this format
        is used, the graph is represented by a binary `numpy.ndarray` tensor ``a`` of type
        `numpy.uint8` with shape ``(k, n, n)``, where *n* is the graph order. Each entry
        ``a[i, j, h]`` indicates whether the edge (or arc) from vertex *j* to vertex *h* has color
        *i*. If loops are not allowed, all diagonal entries ``a[i, j, j]`` are zero. For fully
        colored graphs, the first slice (corresponding to color 0) can be omitted; this variant is
        called the reduced adjacency matrix format with binary slices.
    :cvar FLATTENED_ROW_MAJOR_COLORS: The flattened row-major format with color numbers. When this
        format is used, the adjacency matrix entries are arranged in row-major order. For directed
        graphs without loops, the diagonal entries are skipped. For undirected graphs, only the
        upper-triangular entries are used, with or without the diagonal depending on whether loops
        are allowed. The arranged entries form a `numpy.ndarray` of type `numpy.uint8`; its length
        is called the flattened length.
    :cvar FLATTENED_ROW_MAJOR_BINARY: The flattened row-major format with binary slices. When this
        format is used, the graph is represented by a binary `numpy.ndarray` matrix of type
        `numpy.uint8`. Each row indicates which entries of the flattened row-major format with
        color numbers are equal to 0, 1, …, *k* - 1, respectively. For fully colored graphs, the
        first row (corresponding to color 0) can be omitted; this variant is called the reduced
        flattened row-major format with binary slices.
    :cvar FLATTENED_CLOCKWISE_COLORS: The flattened clockwise format with color numbers. When this
        format is used, the adjacency matrix entries are arranged in clockwise layer order: (0, 0),
        (0, 1), (1, 1), (1, 0), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), …. For directed graphs
        without loops, diagonal entries are skipped. For undirected graphs, only the
        upper-triangular entries are arranged, with or without the diagonal depending on whether
        loops are allowed. The arranged entries form a `numpy.ndarray` of type `numpy.uint8`; its
        length is called the flattened length.
    :cvar FLATTENED_CLOCKWISE_BINARY: The flattened clockwise format with binary slices. When this
        format is used, the graph is represented by a binary `numpy.ndarray` matrix of type
        `numpy.uint8`. Each row indicates which entries of the flattened clockwise format with
        color numbers are equal to 0, 1, …, *k* - 1, respectively. For fully colored graphs, the
        first row (corresponding to color 0) can be omitted; this variant is called the reduced
        flattened clockwise format with binary slices.

    :note: If the graph is undirected, the bitmask formats for the out- and in-neighborhoods are
        identical. Also, for undirected graphs, the flattened clockwise format can be interpreted
        as the flattened column-major format of the upper-triangular part of the adjacency matrix,
        with or without the diagonal depending on whether loops are allowed.
    """

    BITMASK_OUT = 0
    BITMASK_IN = 1
    ADJACENCY_MATRIX_COLORS = 2
    ADJACENCY_MATRIX_BINARY = 3
    FLATTENED_ROW_MAJOR_COLORS = 4
    FLATTENED_ROW_MAJOR_BINARY = 5
    FLATTENED_CLOCKWISE_COLORS = 6
    FLATTENED_CLOCKWISE_BINARY = 7


class BitmaskType(Enum):
    """
    This enumeration encapsulates the concept of a bitmask type in the context of the two bitmask
    formats from the `GraphFormat` enumeration.

    :cvar OUT_NEIGHBORS: This item corresponds to the bitmask format for the out-neighborhoods.
    :cvar IN_NEIGHBORS: This item corresponds to the bitmask format for the in-neighborhoods.
    """

    OUT_NEIGHBORS = 0
    IN_NEIGHBORS = 1


class ColorRepresentation(Enum):
    """
    This enumeration encapsulates the concept of an edge color representation style in the context
    of the two adjacency matrix formats and the four flattened formats from the `GraphFormat`
    enumeration.

    :cvar COLOR_NUMBERS: This item corresponds to the graph formats where each entry contains the
        color of an edge (resp. arc), namely the adjacency matrix format with color numbers, the
        flattened row-major format with color numbers, and the flattened clockwise format with
        color numbers.
    :cvar BINARY_SLICES: This item corresponds to the graph formats where binary slices indicate
        whether each edge (resp. arc) has a specific color, namely the adjacency matrix format with
        binary slices, the flattened row-major format with binary slices, and the flattened
        clockwise format with binary slices.
    """

    COLOR_NUMBERS = 0
    BINARY_SLICES = 1


class FlattenedOrdering(Enum):
    """
    This enumeration encapsulates the concept of an edge (resp. arc) ordering in the context of the
    four flattened formats from the `GraphFormat` enumeration.

    :cvar ROW_MAJOR: This item corresponds to the row-major edge (resp. arc) ordering.
    :cvar CLOCKWISE: This item corresponds to the clockwise edge (resp. arc) ordering.
    """

    ROW_MAJOR = 0
    CLOCKWISE = 1
