"""
This ``Python`` module contains the classes `Graph` and `GraphBatch`, which encapsulate the
concept of an edge-colored complete graph and a batch of edge-colored complete graphs of the same
order, respectively.
"""

from __future__ import annotations

from enum import Enum
from math import isqrt
from typing import Optional

import numpy as np


class GraphFormat(Enum):
    r"""
    This enumeration encapsulates the concept of a format used to represent an edge-colored
    complete graph. Each of the edges can be colored in one of the colors from a given set $\\{ 0,
    1, 2, \\ldots, c - 1 \\}$, where $c$ is the number of available edge colors. Each edge can also
    be uncolored. The enumeration provides support for the following four graph formats:

    :cvar BITMASK: The bitmask format. Here, the graph is represented through the number of
        available edge colors $c$ and a `np.ndarray` integer matrix ``a`` with $c$ rows and $n$
        columns, where $n$ is the graph order. The value ``a[i, j]`` represents a nonnegative
        integer whose $k$-th bit indicates whether the edge between the vertices $j$ and $k$ is of
        the color $i$. Additionally, if there are no uncolored edges, then the starting row of the
        said `np.ndarray` matrix can be omitted, and we refer to this format as the reduced bitmask
        format.
    :cvar ADJACENCY_MATRIX: The adjacency matrix format. Here, the graph is represented through the
        number of available edge colors $c$ and its adjacency matrix, i.e., a `np.ndarray` integer
        square matrix ``a`` with $n$ rows and columns, where $n$ is the graph order. The value
        ``a[i, j]`` represents the color of the edge between the vertices $i$ and $j$, with an
        uncolored edge being represented by the value $c$. The diagonal values of the adjacency
        matrix are all equal to zero.
    :cvar FLATTENED_COLUMN_FIRST: The flattened column-first format. Here, the graph is represented
        through the number of available edge colors and its adjacency matrix entries that are
        strictly above the diagonal. The entries are arranged column-first, i.e., as $(0, 1),
        (0, 2), (1, 2), (0, 3), (1, 3), (2, 3), \\ldots$, and they form a `np.ndarray` list of
        length ``n(n - 1) // 2``, where $n$ is the graph order.
    :cvar FLATTENED_ROW_FIRST: The flattened row-first format. Here, the graph is represented
        through the number of available edge colors and its adjacency matrix entries that are
        strictly above the diagonal. The entries are arranged row-first, i.e., as $(0, 1), (0, 2),
        (0, 3), \\ldots, (0, n - 1), (1, 2), (1, 3), \\ldots$, and they form a `np.ndarray` list of
        length ``n(n - 1) // 2``, where $n$ is the graph order.
    """

    BITMASK_IN_NEIGHBORS = 0
    BITMASK_OUT_NEIGHBORS = 1
    ADJACENCY_MATRIX = 2
    FLATTENED_COLUMN_FIRST = 3
    FLATTENED_ROW_FIRST = 4


class EdgeOrdering(Enum):
    """
    This enumeration encapsulates the concept of an edge ordering in the context of the two
    flattened graph formats, `GraphFormat.FLATTENED_COLUMN_FIRST` and
    `GraphFormat.FLATTENED_ROW_FIRST`, from the `GraphFormat` enumeration.

    :cvar COLUMN_FIRST: This item corresponds to the column-first edge ordering.
    :cvar ROW_FIRST: This item corresponds to the row-first edge ordering.
    """

    COLUMN_FIRST = 0
    ROW_FIRST = 1


class BitmaskType(Enum):

    IN_NEIGHBORS = 0
    OUT_NEIGHBORS = 1


class Graph:
    r"""
    This class encapsulates the concept of an edge-colored complete graph. The graph is of order
    $n$, and each of its edges can be colored in one of the colors from the set $\\{ 0, 1, 2,
    \\ldots c - 1 \\}$, where $c$ is the number of available edge colors. The positive integer $c$
    must be at least two. An edge can also be uncolored, which is represented by the de facto
    auxiliary edge color $c$, as explained in the `GraphFormat` enumeration.

    The graph can be represented in any of the four graph formats from the `GraphFormat`
    enumeration. It is initialized in exactly one of these four formats so that the provided
    `np.ndarray` object representing the graph is internally stored. Afterwards, the graph can be
    accessed in any of the four formats, with all the required graph format conversions being
    performed automatically. With each performed conversion, the obtained `np.ndarray` object
    corresponding to the output graph format is internally stored. The class provides support for
    the reduced bitmask format, and while performing a conversion to the bitmask format
    (`GraphFormat.BITMASK`), the reduced bitmask format is always used if possible, i.e., if there
    are no uncolored edges.

    :ivar __edge_colors: The number of available edge colors, i.e., $c$.
    :ivar __order: The graph order, i.e., $n$.
    :ivar __bitmask: The `np.ndarray` from the bitmask format (`GraphFormat.BITMASK`)
        representation of the given graph, if it was used to initialize the graph or computed
        afterwards, and otherwise, `None`.
    :ivar __adjacency_matrix: The `np.ndarray` from the adjacency matrix format
        (`GraphFormat.ADJACENCY_MATRIX`) representation of the given graph, if it was used to
        initialize the graph or computed afterwards, and otherwise, `None`.
    :ivar __flattened_column_first: The `np.ndarray` from the flattened column-first format
        (`GraphFormat.FLATTENED_COLUMN_FIRST`) representation of the given graph, if it was used to
        initialize the graph or computed afterwards, and otherwise, `None`.
    :ivar __flattened_row_first: The `np.ndarray` from the flattened row-first format
        (`GraphFormat.FLATTENED_ROW_FIRST`) representation of the given graph, if it was used to
        initialize the graph or computed afterwards, and otherwise, `None`.

    :note: Although available, the constructor of this class is not intended to be directly used.
        Instead, the instances should be initialized by using the three class methods
        `from_bitmask`, `from_adjacency_matrix` and `from_flattened`.
    """

    def __init__(
        self,
        edge_colors: int,
        graph_format: GraphFormat,
        bitmask_in_neighbors: Optional[np.ndarray] = None,
        bitmask_out_neighbors: Optional[np.ndarray] = None,
        adjacency_matrix: Optional[np.ndarray] = None,
        flattened_row_major: Optional[np.ndarray] = None,
        flattened_column_major: Optional[np.ndarray] = None,
        flattened_clockwise_layers: Optional[np.ndarray] = None,
        flattened_counterclockwise_layers: Optional[np.ndarray] = None,
        allow_loops: bool = False,
        is_directed: bool = False,
    ):
        """
        This constructor initializes an instance using the chosen graph format. In other words, one
        of the four graph formats is selected, and then the corresponding `np.ndarray` is used
        together with the provided number of available edge colors to initialize a graph with
        respect to this format.

        :param graph_format: The selected graph format that is used to initialize the graph, given
            as an item of the `GraphFormat` enumeration.
        :param edge_colors: A positive integer (not below two) that represents the number of
            available edge colors.
        :param bitmask: Either `None`, or the `np.ndarray` that describes the given graph in the
            bitmask format (`GraphFormat.BITMASK`). If the argument ``graph_format`` equals
            `GraphFormat.BITMASK`, then this argument cannot be `None`, and otherwise, it is
            ignored.
        :param adjacency_matrix: Either `None`, or the `np.ndarray` that describes the given graph
            in the adjacency matrix format (`GraphFormat.ADJACENCY_MATRIX`). If the argument
            ``graph_format`` equals `GraphFormat.ADJACENCY_MATRIX`, then this argument cannot be
            `None`, and otherwise, it is ignored.
        :param flattened_column_first: Either `None`, or the `np.ndarray` that describes the given
            graph in the flattened column-first format (`GraphFormat.FLATTENED_COLUMN_FIRST`). If
            the argument ``graph_format`` equals `GraphFormat.FLATTENED_COLUMN_FIRST`, then this
            argument cannot be `None`, and otherwise, it is ignored.
        :param flattened_row_first: Either `None`, the `np.ndarray` that describes the given graph
            in the flattened row-first format (`GraphFormat.FLATTENED_ROW_FIRST`). If the argument
            ``graph_format`` equals `GraphFormat.FLATTENED_ROW_FIRST`, then this argument cannot be
            `None`, and otherwise, it is ignored.

        :note: This constructor is not intended to be directly used. The class methods
            `from_bitmask`, `from_adjacency_matrix` and `from_flattened` should be used to
            initialize the instances of the `Graph` class.
        """

        self.__edge_colors: int = edge_colors
        self.__allow_loops: bool = allow_loops
        self.__is_directed: bool = is_directed

        self.__bitmask_in_neighbors: Optional[np.ndarray] = None
        self.__bitmask_out_neighbors: Optional[np.ndarray] = None
        self.__adjacency_matrix: Optional[np.ndarray] = None
        self.__flattened_row_major: Optional[np.ndarray] = None
        self.__flattened_column_major: Optional[np.ndarray] = None
        self.__flattened_clockwise_layers: Optional[np.ndarray] = None
        self.__flattened_counterclockwise_layers: Optional[np.ndarray] = None

        if graph_format == GraphFormat.BITMASK_IN_NEIGHBORS:
            self.__order: int = bitmask_in_neighbors.shape[1]
            self.__bitmask_in_neighbors: Optional[np.ndarray] = bitmask_in_neighbors

        elif graph_format == GraphFormat.BITMASK_OUT_NEIGHBORS:
            self.__order: int = bitmask_out_neighbors.shape[1]
            self.__bitmask_out_neighbors: Optional[np.ndarray] = bitmask_out_neighbors

        elif graph_format == GraphFormat.ADJACENCY_MATRIX:
            self.__order: int = adjacency_matrix.shape[0]
            self.__adjacency_matrix: Optional[np.ndarray] = adjacency_matrix

        elif graph_format == GraphFormat.FLATTENED_COLUMN_FIRST:
            if self.__is_directed:
                if self.__allow_loops:
                    # Given $n^2$, find $n$.
                    self.__order: int = isqrt(flattened_column_major.shape[0])
                else:
                    # Given $n^2 - n$, find $n$.
                    self.__order: int = (isqrt(4 * flattened_column_major.shape[0] + 1) + 1) // 2
            else:
                if self.__allow_loops:
                    # Given \binom{n + 1}{2}$, find $n$.
                    self.__order: int = (isqrt(8 * flattened_column_major.shape[0] + 1) - 1) // 2
                else:
                    # Given $\binom{n}{2}$, find $n$.
                    self.__order: int = (isqrt(8 * flattened_column_major.shape[0] + 1) + 1) // 2

            self.__bitmask_in_neighbors: Optional[np.ndarray] = None
            self.__bitmask_out_neighbors: Optional[np.ndarray] = None
            self.__adjacency_matrix: Optional[np.ndarray] = None
            self.__flattened_column_major: Optional[np.ndarray] = flattened_column_major
            self.__flattened_row_major: Optional[np.ndarray] = None

        else:
            if self.__is_directed:
                if self.__allow_loops:
                    # Given $n^2$, find $n$.
                    self.__order: int = isqrt(flattened_row_major.shape[0])
                else:
                    # Given $n^2 - n$, find $n$.
                    self.__order: int = (isqrt(4 * flattened_row_major.shape[0] + 1) + 1) // 2
            else:
                if self.__allow_loops:
                    # Given \binom{n + 1}{2}$, find $n$.
                    self.__order: int = (isqrt(8 * flattened_row_major.shape[0] + 1) - 1) // 2
                else:
                    # Given $\binom{n}{2}$, find $n$.
                    self.__order: int = (isqrt(8 * flattened_row_major.shape[0] + 1) + 1) // 2

            self.__bitmask_in_neighbors: Optional[np.ndarray] = None
            self.__bitmask_out_neighbors: Optional[np.ndarray] = None
            self.__adjacency_matrix: Optional[np.ndarray] = None
            self.__flattened_column_major: Optional[np.ndarray] = None
            self.__flattened_row_major: Optional[np.ndarray] = flattened_row_major

    @classmethod
    def from_bitmask(cls, edge_colors: int, bitmask: np.ndarray) -> Graph:
        """
        This class method initializes a `Graph` object by using the bitmask or reduced bitmask
        format (`GraphFormat.BITMASK`).

        :param edge_colors: A positive integer (not below two) that represents the number of
            available edge colors.
        :param bitmask: The `np.ndarray` that describes the bitmask or reduced bitmask format
            (`GraphFormat.BITMASK`) of the graph that should be initialized.

        :return: The initialized `Graph` object.
        """

        return cls(graph_format=GraphFormat.BITMASK, edge_colors=edge_colors, bitmask=bitmask)

    @classmethod
    def from_adjacency_matrix(cls, edge_colors: int, adjacency_matrix: np.ndarray) -> Graph:
        """
        This class method initializes a `Graph` object by using the adjacency matrix format
        (`GraphFormat.ADJACENCY_MATRIX`).

        :param edge_colors: A positive integer (not below two) that represents the number of
            available edge colors.
        :param adjacency_matrix: The `np.ndarray` that describes the adjacency matrix format
            (`GraphFormat.ADJACENCY_MATRIX`) of the graph that should be initialized.

        :return: The initialized `Graph` object.
        """

        return cls(
            graph_format=GraphFormat.ADJACENCY_MATRIX,
            edge_colors=edge_colors,
            adjacency_matrix=adjacency_matrix,
        )

    @classmethod
    def from_flattened(
        cls,
        edge_colors: int,
        edge_ordering: EdgeOrdering,
        flattened: np.ndarray,
    ) -> Graph:
        """
        This class method initializes a `Graph` object by using either the flattened column-first
        format (`GraphFormat.FLATTENED_COLUMN_FIRST`) or the flattened row-first format
        (`GraphFormat.FLATTENED_ROW_FIRST`).

        :param edge_colors: A positive integer (not below two) that represents the number of
            available edge colors.
        :param edge_ordering: An item of the `EdgeOrdering` enumeration that determines whether the
            flattened column-first format should be used or the flattened row-first format.
        :param flattened: The `np.ndarray` that describes the flattened column-first format
            (`GraphFormat.FLATTENED_COLUMN_FIRST`) or the flattened row-first format
            (`GraphFormat.FLATTENED_ROW_FIRST`) of the graph that should be initialized.

        :return: The initialized `Graph` object.
        """

        if edge_ordering == EdgeOrdering.COLUMN_FIRST:
            return cls(
                graph_format=GraphFormat.FLATTENED_COLUMN_FIRST,
                edge_colors=edge_colors,
                flattened_column_first=flattened,
            )
        else:
            return cls(
                graph_format=GraphFormat.FLATTENED_ROW_FIRST,
                edge_colors=edge_colors,
                flattened_row_first=flattened,
            )

    @property
    def edge_colors(self) -> int:
        """
        This property returns the number of available edge colors for the given graph, as an `int`.
        """

        return self.__edge_colors

    @property
    def order(self) -> int:
        """
        This property returns the given graph order, as an `int`.
        """

        return self.__order

    @property
    def bitmask(self) -> np.ndarray:
        """
        This property returns the `np.ndarray` that represents the given graph in the bitmask
        format (`GraphFormat.BITMASK`).
        """

        # If the output `np.ndarray` is already known, then just return it.
        if self.__bitmask is not None:
            return self.__bitmask

        # Otherwise, compute the output `np.ndarray` by using the adjacency matrix format
        # representation. If the adjacency matrix format representation is also unknown, then it
        # will first get computed by using one of the remaining two format representations, which
        # is surely known.
        color_indices = np.arange(self.__edge_colors, dtype=int)
        masks = 1 << np.arange(self.__order, dtype=int)

        # If there are uncolored edges, then the reduced bitmask format cannot be used.
        if np.max(self.adjacency_matrix) == self.__edge_colors:
            temp = (self.adjacency_matrix == color_indices[:, None, None]).astype(int)
            np.fill_diagonal(temp[0], 0)
            result = temp @ masks
        # Otherwise, we use the reduced bitmask format.
        else:
            temp = (self.adjacency_matrix == color_indices[1:, None, None]).astype(int)
            result = temp @ masks

        # Update the bitmask format representation to make it available for further use, so that
        # the same conversion does not have to be performed twice.
        self.__bitmask = result

        return self.__bitmask

    @property
    def adjacency_matrix(self) -> np.ndarray:
        """
        This property returns the `np.ndarray` that represents the given graph in the adjacency
        matrix format (`GraphFormat.ADJACENCY_MATRIX`).
        """

        # If the output `np.ndarray` is already known, then just return it.
        if self.__adjacency_matrix is not None:
            return self.__adjacency_matrix

        # If the flattened column-first format representation is known, use it to obtain the
        # adjacency matrix format representation.
        if self.__flattened_column_major is not None:
            tril_rows, tril_columns = np.tril_indices(self.__order, k=-1)

            result = np.zeros((self.__order, self.__order), dtype=int)
            result[tril_rows, tril_columns] = self.__flattened_column_major
            result[tril_columns, tril_rows] = self.__flattened_column_major

            # Update the adjacency matrix format representation to make it available for further
            # use, so that the same conversion does not have to be performed twice.
            self.__adjacency_matrix = result

            return self.__adjacency_matrix

        # If the flattened row-first format representation is known, use it to obtain the adjacency
        # matrix format representation.
        if self.__flattened_row_major is not None:
            triu_rows, triu_columns = np.triu_indices(self.__order, k=1)

            result = np.zeros((self.__order, self.__order), dtype=int)
            result[triu_rows, triu_columns] = self.__flattened_row_major
            result[triu_columns, triu_rows] = self.__flattened_row_major

            # Update the adjacency matrix format representation to make it available for further
            # use, so that the same conversion does not have to be performed twice.
            self.__adjacency_matrix = result

            return self.__adjacency_matrix

        # Otherwise, the bitmask format representation must be known, hence it can be used to
        # find the adjacency matrix representation.
        masks = (1 << np.arange(self.__order, dtype=int)).reshape(-1, 1, 1)
        temp = (self.__bitmask & masks).transpose(1, 2, 0)
        temp = (temp != 0).astype(int)

        # If the number of rows of the bitmask format representation matrix matches the number of
        # available edge colors, then this means that the standard (non-reduced) bitmask format is
        # being used.
        if self.__bitmask.shape[0] == self.__edge_colors:
            result = np.full((self.__order, self.__order), self.__edge_colors, dtype=int)
            np.fill_diagonal(result, 0)
            weights = np.arange(-self.__edge_colors, 0, dtype=int)
            result += np.sum(temp * weights.reshape(-1, 1, 1), axis=0)

            # Update the adjacency matrix format representation to make it available for further
            # use, so that the same conversion does not have to be performed twice.
            self.__adjacency_matrix = result

            return self.__adjacency_matrix

        # Otherwise, the reduced bitmask format is being used.
        weights = np.arange(1, self.__edge_colors, dtype=int)
        result = np.sum(temp * weights.reshape(-1, 1, 1), axis=0)

        # Update the adjacency matrix format representation to make it available for further use,
        # so that the same conversion does not have to be performed twice.
        self.__adjacency_matrix = result

        return self.__adjacency_matrix

    @property
    def flattened_column_first(self) -> np.ndarray:
        """
        This property returns the `np.ndarray` that represents the given graph in the flattened
        column-first format (`GraphFormat.FLATTENED_COLUMN_FIRST`).
        """

        # If the output `np.ndarray` is already known, then just return it.
        if self.__flattened_column_major is not None:
            return self.__flattened_column_major

        # Otherwise, compute the output `np.ndarray` by using the adjacency matrix format
        # representation. If the adjacency matrix format representation is also unknown, then it
        # will first get computed by using one of the remaining two format representations, which
        # is surely known.
        tril_indices = np.tril_indices(self.__order, k=-1)
        result = self.adjacency_matrix[tril_indices]

        # Update the flattened column-first format representation to make it available for
        # further use, so that the same conversion does not have to be performed twice.
        self.__flattened_column_major = result

        return self.__flattened_column_major

    @property
    def flattened_row_first(self) -> np.ndarray:
        """
        This property returns the `np.ndarray` that represents the given graph in the flattened
        row-first format (`GraphFormat.FLATTENED_ROW_FIRST`).
        """

        # If the output `np.ndarray` is already known, then just return it.
        if self.__flattened_row_major is not None:
            return self.__flattened_row_major

        # Otherwise, compute the output `np.ndarray` by using the adjacency matrix format
        # representation. If the adjacency matrix format representation is also unknown, then it
        # will first get computed by using one of the remaining two format representations, which
        # is surely known.
        triu_indices = np.triu_indices(self.__order, k=1)
        result = self.adjacency_matrix[triu_indices]

        # Update the flattened row-first format representation to make it available for further
        # use, so that the same conversion does not have to be performed twice.
        self.__flattened_row_major = result

        return self.__flattened_row_major


class GraphBatch:
    r"""
    This class encapsulates the concept of a batch of edge-colored complete graphs of the same
    order. The graphs are of order $n$, and each of their edges can be colored in one of the colors
    from the set $\\{ 0, 1, 2, \\ldots c - 1 \\}$, where $c$ is the number of available edge
    colors. The positive integer $c$ must be at least two. An edge can also be uncolored, which is
    represented by the de facto auxiliary edge color $c$, as explained in the `GraphFormat`
    enumeration.

    The batch of graphs can be represented in any of the four graph formats from the `GraphFormat`
    enumeration. The representation and initialization is done in the same way as in the `Graph`
    class, the only difference being that all the `np.ndarray` objects used are of one dimension
    higher. More precisely, if ``a`` is the `np.ndarray` used to describe a given batch of graphs,
    then ``a[i]`` is the representation of the $i$-th graph from the batch in the corresponding
    format.

    :ivar __batch_size: A positive integer that determines the number of graphs in the given batch.
    :ivar __edge_colors: The number of available edge colors, i.e., $c$.
    :ivar __order: The graph order, i.e., $n$.
    :ivar __bitmask_batch: The `np.ndarray` from the bitmask format (`GraphFormat.BITMASK`)
        representation of the given batch of graphs, if it was used to initialize the batch of
        graphs or computed afterwards, and otherwise, `None`.
    :ivar __adjacency_matrix_batch: The `np.ndarray` from the adjacency matrix format
        (`GraphFormat.ADJACENCY_MATRIX`) representation of the given batch of graphs, if it was
        used to initialize the batch of graphs or computed afterwards, and otherwise, `None`.
    :ivar __flattened_column_first_batch: The `np.ndarray` from the flattened column-first format
        (`GraphFormat.FLATTENED_COLUMN_FIRST`) representation of the given batch of graphs, if it
        was used to initialize the batch of graphs or computed afterwards, and otherwise, `None`.
    :ivar __flattened_row_first_batch: The `np.ndarray` from the flattened row-first format
        (`GraphFormat.FLATTENED_ROW_FIRST`) representation of the given batch of graphs, if it was
        used to initialize the batch of graphs or computed afterwards, and otherwise, `None`.

    :note: Although available, the constructor of this class is not intended to be directly used.
        Instead, the instances should be initialized by using the three class methods
        `from_bitmask_batch`, `from_adjacency_matrix_batch` and `from_flattened_batch`.
    """

    def __init__(
        self,
        graph_format: GraphFormat,
        edge_colors: int,
        bitmask_batch: Optional[np.ndarray] = None,
        adjacency_matrix_batch: Optional[np.ndarray] = None,
        flattened_column_first_batch: Optional[np.ndarray] = None,
        flattened_row_first_batch: Optional[np.ndarray] = None,
    ):
        """
        This constructor initializes an instance using the chosen graph format. In other words,
        one of the four graph formats is selected, and then the corresponding `np.ndarray` is used
        together with the provided number of available edge colors to initialize a batch of graphs
        with respect to this format.

        :param graph_format: The selected graph format that is used to initialize the batch of
            graphs, given as an item of the `GraphFormat` enumeration.
        :param edge_colors: A positive integer (not below two) that represents the number of
            available edge colors.
        :param bitmask_batch: Either `None`, or the `np.ndarray` that describes the given batch of
            graphs in the bitmask format (`GraphFormat.BITMASK`). If the argument ``graph_format``
            equals `GraphFormat.BITMASK`, then this argument cannot be `None`, and otherwise, it is
            ignored.
        :param adjacency_matrix_batch: Either `None`, or the `np.ndarray` that describes the given
            batch of graphs in the adjacency matrix format (`GraphFormat.ADJACENCY_MATRIX`). If the
            argument ``graph_format`` equals `GraphFormat.ADJACENCY_MATRIX`, then this argument
            cannot be `None`, and otherwise, it is ignored.
        :param flattened_column_first_batch: Either `None`, or the `np.ndarray` that describes the
            given batch of graphs in the flattened column-first format
            (`GraphFormat.FLATTENED_COLUMN_FIRST`). If the argument ``graph_format`` equals
            `GraphFormat.FLATTENED_COLUMN_FIRST`, then this argument cannot be `None`, and
            otherwise, it is ignored.
        :param flattened_row_first_batch: Either `None`, the `np.ndarray` that describes the given
            batch of graphs in the flattened row-first format (`GraphFormat.FLATTENED_ROW_FIRST`).
            If the argument ``graph_format`` equals `GraphFormat.FLATTENED_ROW_FIRST`, then this
            argument cannot be `None`, and otherwise, it is ignored.

        :note: This constructor is not intended to be directly used. The class methods
            `from_bitmask_batch`, `from_adjacency_matrix_batch` and `from_flattened_batch` should
            be used to initialize the instances of the `GraphBatch` class.
        """

        self.__edge_colors: int = edge_colors

        if graph_format == GraphFormat.BITMASK:
            self.__batch_size: int = bitmask_batch.shape[0]
            self.__order: int = bitmask_batch.shape[2]
            self.__bitmask_batch: Optional[np.ndarray] = bitmask_batch
            self.__adjacency_matrix_batch: Optional[np.ndarray] = None
            self.__flattened_column_first_batch: Optional[np.ndarray] = None
            self.__flattened_row_first_batch: Optional[np.ndarray] = None

        elif graph_format == GraphFormat.ADJACENCY_MATRIX:
            self.__batch_size: int = adjacency_matrix_batch.shape[0]
            self.__order: int = adjacency_matrix_batch.shape[1]
            self.__bitmask_batch: Optional[np.ndarray] = None
            self.__adjacency_matrix_batch: Optional[np.ndarray] = adjacency_matrix_batch
            self.__flattened_column_first_batch: Optional[np.ndarray] = None
            self.__flattened_row_first_batch: Optional[np.ndarray] = None

        elif graph_format == GraphFormat.FLATTENED_COLUMN_FIRST:
            self.__batch_size: int = flattened_column_first_batch.shape[0]
            # Given $\binom{n}{2}$, find $n$.
            self.__order: int = (isqrt(8 * flattened_column_first_batch.shape[1] + 1) + 1) // 2
            self.__bitmask_batch: Optional[np.ndarray] = None
            self.__adjacency_matrix_batch: Optional[np.ndarray] = None
            self.__flattened_column_first_batch: Optional[np.ndarray] = (
                flattened_column_first_batch
            )
            self.__flattened_row_first_batch: Optional[np.ndarray] = None

        else:
            self.__batch_size: int = flattened_row_first_batch.shape[0]
            # Given $\binom{n}{2}$, find $n$.
            self.__order: int = (isqrt(8 * flattened_row_first_batch.shape[1] + 1) + 1) // 2
            self.__bitmask_batch: Optional[np.ndarray] = None
            self.__adjacency_matrix_batch: Optional[np.ndarray] = None
            self.__flattened_column_first_batch: Optional[np.ndarray] = None
            self.__flattened_row_first_batch: Optional[np.ndarray] = flattened_row_first_batch

    @classmethod
    def from_bitmask_batch(cls, edge_colors: int, bitmask_batch: np.ndarray) -> GraphBatch:
        """
        This class method initializes a `GraphBatch` object by using the bitmask or reduced bitmask
        format (`GraphFormat.BITMASK`).

        :param edge_colors: A positive integer (not below two) that represents the number of
            available edge colors.
        :param bitmask_batch: The `np.ndarray` that describes the bitmask or reduced bitmask format
            (`GraphFormat.BITMASK`) of the batch of graphs that should be initialized.

        :return: The initialized `GraphBatch` object.
        """

        return cls(
            graph_format=GraphFormat.BITMASK, edge_colors=edge_colors, bitmask_batch=bitmask_batch
        )

    @classmethod
    def from_adjacency_matrix_batch(
        cls,
        edge_colors: int,
        adjacency_matrix_batch: np.ndarray,
    ) -> GraphBatch:
        """
        This class method initializes a `GraphBatch` object by using the adjacency matrix format
        (`GraphFormat.ADJACENCY_MATRIX`).

        :param edge_colors: A positive integer (not below two) that represents the number of
            available edge colors.
        :param adjacency_matrix_batch: The `np.ndarray` that describes the adjacency matrix format
            (`GraphFormat.ADJACENCY_MATRIX`) of the batch of graphs that should be initialized.

        :return: The initialized `GraphBatch` object.
        """

        return cls(
            graph_format=GraphFormat.ADJACENCY_MATRIX,
            edge_colors=edge_colors,
            adjacency_matrix_batch=adjacency_matrix_batch,
        )

    @classmethod
    def from_flattened_batch(
        cls,
        edge_colors: int,
        edge_ordering: EdgeOrdering,
        flattened_batch: np.ndarray,
    ) -> GraphBatch:
        """
        This class method initializes a `GraphBatch` object by using either the flattened
        column-first format (`GraphFormat.FLATTENED_COLUMN_FIRST`) or the flattened row-first
        format (`GraphFormat.FLATTENED_ROW_FIRST`).

        :param edge_colors: A positive integer (not below two) that represents the number of
            available edge colors.
        :param edge_ordering: An item of the `EdgeOrdering` enumeration that determines whether the
            flattened column-first format should be used or the flattened row-first format.
        :param flattened_batch: The `np.ndarray` that describes the flattened column-first format
            (`GraphFormat.ADJACENCY_MATRIX`) or the flattened row-first format
            (`GraphFormat.FLATTENED_ROW_FIRST`) of the batch of graphs that should be initialized.

        :return: The initialized `GraphBatch` object.
        """

        if edge_ordering == EdgeOrdering.COLUMN_FIRST:
            return cls(
                graph_format=GraphFormat.FLATTENED_COLUMN_FIRST,
                edge_colors=edge_colors,
                flattened_column_first_batch=flattened_batch,
            )
        else:
            return cls(
                graph_format=GraphFormat.FLATTENED_ROW_FIRST,
                edge_colors=edge_colors,
                flattened_row_first_batch=flattened_batch,
            )

    @property
    def batch_size(self) -> int:
        """
        This property returns the batch size of the given batch of graphs, i.e., the number of
        graphs in the batch.
        """

        return self.__batch_size

    @property
    def edge_colors(self) -> int:
        """
        This property returns the number of available edge colors for the given batch of graphs, as
        an `int`.
        """

        return self.__edge_colors

    @property
    def order(self) -> int:
        """
        This property returns the order of all the graphs from the given batch, as an `int`.
        """

        return self.__order

    @property
    def bitmask_batch(self) -> np.ndarray:
        """
        This property returns the `np.ndarray` that represents the given batch of graphs in the
        bitmask format (`GraphFormat.BITMASK`).
        """

        # If the output `np.ndarray` is already known, then just return it.
        if self.__bitmask_batch is not None:
            return self.__bitmask_batch

        # Otherwise, compute the output `np.ndarray` by using the adjacency matrix format
        # representation. If the adjacency matrix format representation is also unknown, then it
        # will first get computed by using one of the remaining two format representations, which
        # is surely known.
        color_indices = np.arange(self.__edge_colors, dtype=int)
        masks = 1 << np.arange(self.__order, dtype=int)

        # If there are uncolored edges, then the reduced bitmask format cannot be used.
        if np.max(self.adjacency_matrix_batch) == self.__edge_colors:
            temp = (
                self.__adjacency_matrix_batch[:, None, :, :] == color_indices[None, :, None, None]
            ).astype(int)
            np.einsum("bcii->bci", temp)[:] = 0
            result = temp @ masks
        # Otherwise, we use the reduced bitmask format.
        else:
            temp = (
                self.__adjacency_matrix_batch[:, None, :, :] == color_indices[None, 1:, None, None]
            ).astype(int)
            result = temp @ masks

        # Update the bitmask format representation to make it available for further use, so that
        # the same conversion does not have to be performed twice.
        self.__bitmask_batch = result

        return self.__bitmask_batch

    @property
    def adjacency_matrix_batch(self) -> np.ndarray:
        """
        This property returns the `np.ndarray` that represents the given batch of graphs in the
        adjacency matrix format (`GraphFormat.ADJACENCY_MATRIX`).
        """

        # If the output `np.ndarray` is already known, then just return it.
        if self.__adjacency_matrix_batch is not None:
            return self.__adjacency_matrix_batch

        # If the flattened column-first format representation is known, use it to obtain the
        # adjacency matrix format representation.
        if self.__flattened_column_first_batch is not None:
            tril_rows, tril_columns = np.tril_indices(self.__order, k=-1)

            result = np.zeros((self.__batch_size, self.__order, self.__order), dtype=int)
            result[:, tril_rows, tril_columns] = self.__flattened_column_first_batch
            result[:, tril_columns, tril_rows] = self.__flattened_column_first_batch

            # Update the adjacency matrix format representation to make it available for further
            # use, so that the same conversion does not have to be performed twice.
            self.__adjacency_matrix_batch = result

            return self.__adjacency_matrix_batch

        # If the flattened row-first format representation is known, use it to obtain the adjacency
        # matrix format representation.
        if self.__flattened_row_first_batch is not None:
            triu_rows, triu_columns = np.triu_indices(self.__order, k=1)

            result = np.zeros((self.__batch_size, self.__order, self.__order), dtype=int)
            result[:, triu_rows, triu_columns] = self.__flattened_row_first_batch
            result[:, triu_columns, triu_rows] = self.__flattened_row_first_batch

            # Update the adjacency matrix format representation to make it available for further
            # use, so that the same conversion does not have to be performed twice.
            self.__adjacency_matrix_batch = result

            return self.__adjacency_matrix_batch

        # Otherwise, the bitmask format representation must be known, hence it can be used to
        # find the adjacency matrix representation.
        masks = (1 << np.arange(self.__order, dtype=int)).reshape(-1, 1, 1, 1)
        temp = (self.__bitmask_batch & masks).transpose(1, 2, 3, 0)
        temp = (temp != 0).astype(int)

        # If the number of rows of the bitmask format representation matrices matches the number of
        # available edge colors, then this means that the standard (non-reduced) bitmask format is
        # being used.
        if self.__bitmask_batch.shape[1] == self.__edge_colors:
            result = np.full(
                (self.__batch_size, self.__order, self.__order), self.__edge_colors, dtype=int
            )
            np.einsum("bii->bi", result)[:] = 0
            weights = np.arange(-self.__edge_colors, 0, dtype=int)
            result += np.sum(temp * weights.reshape(1, -1, 1, 1), axis=1)

            # Update the adjacency matrix format representation to make it available for further
            # use, so that the same conversion does not have to be performed twice.
            self.__adjacency_matrix_batch = result

            return self.__adjacency_matrix_batch

        # Otherwise, the reduced bitmask format is being used.
        weights = np.arange(1, self.__edge_colors, dtype=int)
        result = np.sum(temp * weights.reshape(1, -1, 1, 1), axis=1)

        # Update the adjacency matrix format representation to make it available for further use,
        # so that the same conversion does not have to be performed twice.
        self.__adjacency_matrix_batch = result

        return self.__adjacency_matrix_batch

    @property
    def flattened_column_first_batch(self) -> np.ndarray:
        """
        This property returns the `np.ndarray` that represents the given batch of graphs in the
        flattened column-first format (`GraphFormat.FLATTENED_COLUMN_FIRST`).
        """

        # If the output `np.ndarray` is already known, then just return it.
        if self.__flattened_column_first_batch is not None:
            return self.__flattened_column_first_batch

        # Otherwise, compute the output `np.ndarray` by using the adjacency matrix format
        # representation. If the adjacency matrix format representation is also unknown, then it
        # will first get computed by using one of the remaining two format representations, which
        # is surely known.
        tril_rows, tril_columns = np.tril_indices(self.__order, k=-1)
        result = self.adjacency_matrix_batch[:, tril_rows, tril_columns]

        # Update the flattened column-first format representation to make it available for
        # further use, so that the same conversion does not have to be performed twice.
        self.__flattened_column_first_batch = result

        return self.__flattened_column_first_batch

    @property
    def flattened_row_first_batch(self) -> np.ndarray:
        """
        This property returns the `np.ndarray` that represents the given batch of graphs in the
        flattened row-first format (`GraphFormat.FLATTENED_ROW_FIRST`).
        """

        # If the output `np.ndarray` is already known, then just return it.
        if self.__flattened_row_first_batch is not None:
            return self.__flattened_row_first_batch

        # Otherwise, compute the output `np.ndarray` by using the adjacency matrix format
        # representation. If the adjacency matrix format representation is also unknown, then it
        # will first get computed by using one of the remaining two format representations, which
        # is surely known.
        triu_rows, triu_columns = np.triu_indices(self.__order, k=1)
        result = self.adjacency_matrix_batch[:, triu_rows, triu_columns]

        # Update the flattened row-first format representation to make it available for further
        # use, so that the same conversion does not have to be performed twice.
        self.__flattened_row_first_batch = result

        return self.__flattened_row_first_batch
