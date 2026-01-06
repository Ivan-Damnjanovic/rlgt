"""
This ``Python`` module contains the `GraphFormat` enumeration, which encapsulates the concept of a
format used to represent a $k$-edge-colored looped complete graph. The module additionally contains
the `BitmaskType` enumeration, which encapsulates the concept of a bitmask type in the context of
the two bitmask formats, the `ColorRepresentation` enumeration, which encapsulates the concept of
an edge color representation style in the context of the two adjacency matrix formats and the four
flattened formats, and the `FlattenedOrdering` enumeration, which encapsulates the concept of an
edge (resp. arc) ordering in the context of the four flattened formats.
"""

from enum import Enum


class GraphFormat(Enum):
    r"""
    This enumeration encapsulates the concept of a format used to represent a $k$-edge-colored
    looped complete graph. Each graph is essentially represented as a quintuple ``(edge_colors,
    is_directed, allow_loops, graph_format, format_representation)``, where:

    * ``edge_colors`` is the number of proper edge colors, i.e., $k$, which is at least 2;
    * ``is_directed`` is a boolean that indicates whether the considered graph is a
      $k$-edge-colored looped complete directed graph or a $k$-edge-colored looped complete
      undirected graph;
    * ``allow_loops`` is a boolean that indicates whether the considered graph is allowed to have
      loops (if loops are not allowed, then all the loops are removed from the considered looped
      complete graph and they simply do not exist);
    * ``graph_format`` is an item of this enumeration that describes the graph format that should
      be used to represent the structure of the considered graph; and
    * ``format_representation`` is a `numpy.ndarray` that represents the structure of the
      considered graph in the chosen graph format.

    The enumeration provides support for the following 8 graph formats:

    1. the bitmask format for the out-neighborhoods;
    2. the bitmask format for the in-neighborhoods;
    3. the adjacency matrix format with color numbers;
    4. the adjacency matrix format with binary slices;
    5. the flattened row-major format with color numbers;
    6. the flattened row-major format with binary slices;
    7. the flattened clockwise format with color numbers; and
    8. the flattened clockwise format with binary slices.

    All of the formats with color numbers require that the value $k$ be at most 255. Additionally,
    the two bitmask formats can only be used if the graph order is at most 64.

    :cvar BITMASK_OUT: The bitmask format for the out-neighborhoods. Here, the graph structure is
        represented through a `numpy.ndarray` matrix ``a`` of type `numpy.uint64` with $k$ rows and
        $n$ columns, where $n$ is the graph order. The value ``a[i, j]`` represents a nonnegative
        integer whose $h$-th bit indicates whether the edge (resp. arc) from the vertex $j$ to the
        vertex $h$ is of the color $i$. If loops are not allowed, then the $j$-bit of ``a[i, j]``
        is just zero. Additionally, if the graph is fully colored, then the starting row
        (corresponding to the color 0) of the said `numpy.ndarray` matrix can be optionally
        omitted, and we refer to such a format as the reduced bitmask format for the
        out-neighborhoods.
    :cvar BITMASK_IN: The bitmask format for the in-neighborhoods. Here, the graph structure is
        represented through a `numpy.ndarray` matrix ``a`` of type `numpy.uint64` with $k$ rows and
        $n$ columns, where $n$ is the graph order. The value ``a[i, j]`` represents a nonnegative
        integer whose $h$-th bit indicates whether the edge (resp. arc) from the vertex $h$ to the
        vertex $j$ is of the color $i$. If loops are not allowed, then the $j$-bit of ``a[i, j]``
        is just zero. Additionally, if the graph is fully colored, then the starting row
        (corresponding to the color 0) of the said `numpy.ndarray` matrix can be optionally
        omitted, and we refer to such a format as the reduced bitmask format for the
        in-neighborhoods.
    :cvar ADJACENCY_MATRIX_COLORS: The adjacency matrix format with color numbers. Here, the graph
        structure is represented through the adjacency matrix, i.e., a `numpy.ndarray` square
        matrix ``a`` of type `numpy.uint8` with $n$ rows and columns, where $n$ is the graph order.
        The value ``a[i, j]`` represents the color of the edge (resp. arc) from the vertex $i$ to
        the vertex $j$, with an uncolored edge (resp. arc) being represented by the value $k$. If
        loops are not allowed, then the diagonal entries of the adjacency matrix are all equal to
        zero.
    :cvar ADJACENCY_MATRIX_BINARY: The adjacency matrix format with binary slices. Here, the graph
        structure is represented through a binary `numpy.ndarray` tensor ``a`` of type
        `numpy.uint8` and shape ``(k, n, n)``, where $n$ is the graph order. The value
        ``a[i, j, h]`` represents whether the edge (resp. arc) from the vertex $j$ to the vertex
        $h$ is of the color $i$. If loops are not allowed, then all of the diagonal entries
        ``a[i, j, j]`` are equal to zero. Additionally, if the graph is fully colored, then the
        starting slice of the said `numpy.ndarray` tensor can be optionally omitted, so that a
        tensor of shape ``(k - 1, n, n)`` is reached. We refer to such a format as the reduced
        adjacency matrix format with binary slices.
    :cvar FLATTENED_ROW_MAJOR_COLORS: The flattened row-major format with color numbers. Here, the
        graph structure is represented through the adjacency matrix entries arranged in the
        row-major order. If the considered graph is directed and loops are not allowed, then the
        diagonal entries should be skipped when the entries are being arranged. If the considered
        graph is undirected, then only the entries from the upper triangular part of the adjacency
        matrix should be arranged in the row-major order (with or without the diagonal, depending
        on whether loops are allowed). The arranged entries form a `numpy.ndarray` list of type
        `numpy.uint8` of the required length, which is called the flattened length.
    :cvar FLATTENED_ROW_MAJOR_BINARY: The flattened row-major format with binary slices. Here, the
        graph structure is represented through the binary `numpy.ndarray` matrix of type
        `numpy.uint8` whose rows indicate which of the ``a`` entries are equal to $0, 1, 2,
        \\ldots, k - 1$, respectively, where ``a`` is the `numpy.ndarray` list from the flattened
        row-major format with color numbers. Additionally, if the graph is fully colored, then the
        starting row (corresponding to the color 0) of the said `numpy.ndarray` matrix can be
        optionally omitted, and we refer to such a format as the reduced flattened row-major format
        with binary slices.
    :cvar FLATTENED_CLOCKWISE_COLORS: The flattened clockwise format with color numbers. Here, the
        graph structure is represented through the adjacency matrix entries arranged in the
        clockwise layer order, i.e., the order $(0, 0), (0, 1), (1, 1), (1, 0), (0, 2), (1, 2),
        (2, 2), (2, 1), (2, 0), (0, 3), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1), (3, 0), \\ldots$.
        If the considered graph is directed and loops are not allowed, then the diagonal entries
        should be skipped when the entries are being arranged. If the considered graph is
        undirected, then only the entries from the upper triangular part of the adjacency matrix
        should be arranged in the clockwise order (with or without the diagonal, depending on
        whether loops are allowed). The arranged entries form a `numpy.ndarray` list of type
        `numpy.uint8` of the required length, which is called the flattened length.
    :cvar FLATTENED_CLOCKWISE_BINARY: The flattened clockwise format with binary slices. Here, the
        graph structure is represented through the binary `numpy.ndarray` matrix of type
        `numpy.uint8` whose rows indicate which of the ``a`` entries are equal to $0, 1, 2,
        \\ldots, k - 1$, respectively, where ``a`` is the `numpy.ndarray` list from the flattened
        clockwise format with color numbers. Additionally, if the graph is fully colored, then the
        starting row (corresponding to the color 0) of the said `numpy.ndarray` matrix can be
        optionally omitted, and we refer to such a format as the reduced flattened clockwise format
        with binary slices.

    :note: If the graph is undirected, then the bitmask format for the out-neighborhoods and the
        bitmask format for the in-neighborhoods are the same. Also, if the graph is undirected,
        then the flattened clockwise format can be regarded as the flattened column-major format
        over the upper triangular part of the adjacency matrix (with or without the diagonal,
        depending on whether loops are allowed).
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

    :cvar COLOR_NUMBERS: This items corresponds to the graph formats where each entry contains the
        color of some edge (resp. arc), i.e., the adjacency matrix format with color numbers, the
        flattened row-major format with color numbers and the flattened clockwise format with color
        numbers.
    :cvar BINARY_SLICES: This item corresponds to the graph formats where binary slices are used
        to indicate whether each of the edges (resp. arcs) is colored with a given color, i.e., the
        adjacency matrix format with binary slices, the flattened row-major format with binary
        slices and the flattened clockwise format with binary slices.
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
