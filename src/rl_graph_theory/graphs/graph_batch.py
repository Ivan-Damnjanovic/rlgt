"""
#TODO
"""

from __future__ import annotations

from math import isqrt
from typing import Optional

import numpy as np

from .graph_format import BitmaskType, FlattenedOrdering, GraphFormat


class GraphBatch:
    r"""
    This class encapsulates the concept of a batch of $k$-edge-colored looped complete graphs of
    the same order, with $k$ being at least two. The graphs in the batch must be consistent: they
    should all be directed or all be undirected, and they should either all allow loops or none.
    Therefore, the batch of graphs can be represented as a quintuple ``(edge_colors, is_directed,
    allow_loops, graph_format, format_representation)``, analogously to the `Graph` class. The
    representation and initialization is done in the same way as in the `Graph` class, the only
    difference being that all the `np.ndarray` objects used are of one dimension higher. More
    precisely, if ``a`` is the `np.ndarray` used to describe the structure of a given batch of
    graphs, then ``a[i]`` is the structural representation of the $i$-th graph in the batch in the
    corresponding format.

    :ivar __edge_colors: The number of proper edge colors, i.e., $k$.
    :ivar __is_directed: A boolean that indicates whether each of the graphs in the given batch is
        a $k$-edge-colored looped complete directed graph or a $k$-edge-colored looped complete
        undirected graph.
    :ivar __allow_loops: A boolean that indicates whether each of the graphs in the given batch is
        allowed to have loops (if loops are not allowed, then all the loops are removed from the
        considered complete graphs and they simply do not exist).
    :ivar __batch_size: A positive integer that determines the number of graphs in the given batch.
    :ivar __order: The graph order, i.e., the number of vertices of each of the graphs in the
        batch.
    :ivar __bitmask_out: The `np.ndarray` from the out-neighborhoods bitmask format
        (`GraphFormat.BITMASK_OUT`) structural representation of the given batch of graphs, if it
        was used to initialize the batch of graphs or computed afterwards, and otherwise, `None`.
    :ivar __bitmask_in: The `np.ndarray` from the in-neighborhoods bitmask format
        (`GraphFormat.BITMASK_IN`) structural representation of the given batch of graphs, if it
        was used to initialize the batch of graphs or computed afterwards, and otherwise, `None`.
    :ivar __adjacency_matrix: The `np.ndarray` from the adjacency matrix format
        (`GraphFormat.ADJACENCY_MATRIX`) structural representation of the given batch of graphs, if
        it was used to initialize the batch of graphs or computed afterwards, and otherwise,
        `None`.
    :ivar __flattened_row_major: The `np.ndarray` from the flattened row-major format
        (`GraphFormat.FLATTENED_ROW_MAJOR`) structural representation of the given batch of graphs,
        if it was used to initialize the batch of graphs or computed afterwards, and otherwise,
        `None`.
    :ivar __flattened_clockwise: The `np.ndarray` from the flattened clockwise format
        (`GraphFormat.FLATTENED_CLOCKWISE`) structural representation of the given batch of graphs,
        if it was used to initialize the batch of graphs or computed afterwards, and otherwise,
        `None`.

    :note: Although available, the constructor of this class is not intended to be directly used.
        Instead, the instances should be initialized by using the three class methods
        `from_bitmask`, `from_adjacency_matrix` and `from_flattened`.
    """

    def __init__(
        self,
        graph_format: GraphFormat,
        bitmask_out: Optional[np.ndarray] = None,
        bitmask_in: Optional[np.ndarray] = None,
        adjacency_matrix: Optional[np.ndarray] = None,
        flattened_row_major: Optional[np.ndarray] = None,
        flattened_clockwise: Optional[np.ndarray] = None,
        edge_colors: int = 2,
        is_directed: bool = False,
        allow_loops: bool = False,
    ):
        """
        This constructor initializes an instance using the chosen graph format. In other words,
        one of the five graph formats is selected, and then the corresponding `np.ndarray` is used
        to initialize a batch of graphs with respect to this format.

        :param graph_format: The selected graph format that is used to initialize the batch of
            graphs, given as an item of the `GraphFormat` enumeration.
        :param bitmask_out: Either `None`, or the `np.ndarray` that describes the given batch of
            graphs in the bitmask format for the out-neighborhoods (`GraphFormat.BITMASK_OUT`). If
            the argument ``graph_format`` equals `GraphFormat.BITMASK_OUT`, then this argument
            cannot be `None`, and otherwise, it is ignored.
        :param bitmask_in: Either `None`, or the `np.ndarray` that describes the given batch of
            graphs in the bitmask format for the in-neighborhoods (`GraphFormat.BITMASK_IN`). If
            the argument ``graph_format`` equals `GraphFormat.BITMASK_IN`, then this argument
            cannot be `None`, and otherwise, it is ignored.
        :param adjacency_matrix: Either `None`, or the `np.ndarray` that describes the given batch
            of graphs in the adjacency matrix format (`GraphFormat.ADJACENCY_MATRIX`). If the
            argument ``graph_format`` equals `GraphFormat.ADJACENCY_MATRIX`, then this argument
            cannot be `None`, and otherwise, it is ignored.
        :param flattened_row_major: Either `None`, or the `np.ndarray` that describes the given
            batch of graphs in the flattened row-major format (`GraphFormat.FLATTENED_ROW_MAJOR`).
            If the argument ``graph_format`` equals `GraphFormat.FLATTENED_ROW_MAJOR`, then this
            argument cannot be `None`, and otherwise, it is ignored.
        :param flattened_clockwise: Either `None`, or the `np.ndarray` that describes the given
            batch of graphs in the flattened clockwise format (`GraphFormat.FLATTENED_CLOCKWISE`).
            If the argument ``graph_format`` equals `GraphFormat.FLATTENED_CLOCKWISE`, then this
            argument cannot be `None`, and otherwise, it is ignored.
        :param edge_colors: A positive integer (not below two) that represents the number of
            proper edge colors, i.e., $k$. The default value is two.
        :param is_directed: A boolean that indicates whether each of the graphs in the given batch
            is a $k$-edge-colored looped complete directed graph or a $k$-edge-colored looped
            complete undirected graph. The default value is `False`, i.e., the graphs are
            undirected by default.
        :param allow_loops: A boolean that indicates whether each of the graphs in the given batch
            is allowed to have loops (if loops are not allowed, then all the loops are removed from
            the considered complete graphs and they simply do not exist). The default value is
            `False`, i.e., the graphs are not allowed to have loops by default.

        :note: This constructor is not intended to be directly used. The class methods
            `from_bitmask`, `from_adjacency_matrix` and `from_flattened` should be used to
            initialize the instances of the `GraphBatch` class.
        """

        self.__edge_colors: int = edge_colors
        self.__is_directed: bool = is_directed
        self.__allow_loops: bool = allow_loops

        self.__bitmask_out: Optional[np.ndarray] = None
        self.__bitmask_in: Optional[np.ndarray] = None
        self.__adjacency_matrix: Optional[np.ndarray] = None
        self.__flattened_row_major: Optional[np.ndarray] = None
        self.__flattened_clockwise: Optional[np.ndarray] = None

        if graph_format == GraphFormat.BITMASK_OUT:
            self.__batch_size: int = bitmask_out.shape[0]
            self.__order: int = bitmask_out.shape[2]
            self.__bitmask_out = bitmask_out
            # If the graphs are undirected, then the two bitmask formats are the same.
            if not self.__is_directed:
                self.__bitmask_in = bitmask_out

        elif graph_format == GraphFormat.BITMASK_IN:
            self.__batch_size: int = bitmask_in.shape[0]
            self.__order: int = bitmask_in.shape[2]
            self.__bitmask_in = bitmask_in
            # If the graphs are undirected, then the two bitmask formats are the same.
            if not self.__is_directed:
                self.__bitmask_out = bitmask_in

        elif graph_format == GraphFormat.ADJACENCY_MATRIX:
            self.__batch_size: int = adjacency_matrix.shape[0]
            self.__order: int = adjacency_matrix.shape[1]
            self.__adjacency_matrix = adjacency_matrix

        elif graph_format == GraphFormat.FLATTENED_ROW_MAJOR:
            self.__batch_size: int = flattened_row_major.shape[0]

            if self.__is_directed:
                if self.__allow_loops:
                    # Given $n^2$, find $n$.
                    self.__order: int = isqrt(flattened_row_major.shape[1])
                else:
                    # Given $n^2 - n$, find $n$.
                    self.__order: int = (isqrt(4 * flattened_row_major.shape[1] + 1) + 1) // 2
            else:
                if self.__allow_loops:
                    # Given \binom{n + 1}{2}$, find $n$.
                    self.__order: int = (isqrt(8 * flattened_row_major.shape[1] + 1) - 1) // 2
                else:
                    # Given $\binom{n}{2}$, find $n$.
                    self.__order: int = (isqrt(8 * flattened_row_major.shape[1] + 1) + 1) // 2

            self.__flattened_row_major = flattened_row_major

        else:
            self.__batch_size: int = flattened_clockwise.shape[0]

            if self.__is_directed:
                if self.__allow_loops:
                    # Given $n^2$, find $n$.
                    self.__order: int = isqrt(flattened_clockwise.shape[1])
                else:
                    # Given $n^2 - n$, find $n$.
                    self.__order: int = (isqrt(4 * flattened_clockwise.shape[1] + 1) + 1) // 2
            else:
                if self.__allow_loops:
                    # Given \binom{n + 1}{2}$, find $n$.
                    self.__order: int = (isqrt(8 * flattened_clockwise.shape[1] + 1) - 1) // 2
                else:
                    # Given $\binom{n}{2}$, find $n$.
                    self.__order: int = (isqrt(8 * flattened_clockwise.shape[1] + 1) + 1) // 2

            self.__flattened_clockwise = flattened_clockwise

    @classmethod
    def from_bitmask(
        cls,
        bitmask: np.ndarray,
        bitmask_type: BitmaskType = BitmaskType.OUT_NEIGHBORS,
        edge_colors: int = 2,
        is_directed: bool = False,
        allow_loops: bool = False,
    ) -> GraphBatch:
        """
        This class method initializes a `GraphBatch` object by using the (potentially reduced)
        bitmask format for the out-neighborhoods (`GraphFormat.BITMASK_OUT`) or the
        in-neighborhoods (`GraphFormat.BITMASK_IN`).

        :param bitmask: The `np.ndarray` that describes the (potentially reduced) bitmask format
            for the out-neighborhoods (`GraphFormat.BITMASK_OUT`) or the in-neighborhoods
            (`GraphFormat.BITMASK_IN`) of the batch of graphs that should be initialized.
        :param bitmask_type: An item of the `BitmaskType` enumeration that determines whether the
            bitmask format for the out-neighborhoods should be used or the bitmask format for the
            in-neighborhoods. The default value is `BitmaskType.OUT_NEIGHBORS`, i.e., the bitmask
            format for the out-neighborhoods is used by default.
        :param edge_colors: A positive integer (not below two) that represents the number of proper
            edge colors, i.e., $k$. The default value is two.
        :param is_directed: A boolean that indicates whether each of the graphs in the given batch
            is a $k$-edge-colored looped complete directed graph or a $k$-edge-colored looped
            complete undirected graph. The default value is `False`, i.e., the graphs are
            undirected by default.
        :param allow_loops: A boolean that indicates whether each of the graphs in the given batch
            is allowed to have loops (if loops are not allowed, then all the loops are removed from
            the considered complete graphs and they simply do not exist). The default value is
            `False`, i.e., the graphs are not allowed to have loops by default.

        :return: The initialized `GraphBatch` object.
        """

        if bitmask_type == BitmaskType.OUT_NEIGHBORS:
            return cls(
                graph_format=GraphFormat.BITMASK_OUT,
                bitmask_out=bitmask,
                edge_colors=edge_colors,
                is_directed=is_directed,
                allow_loops=allow_loops,
            )
        else:
            return cls(
                graph_format=GraphFormat.BITMASK_IN,
                bitmask_in=bitmask,
                edge_colors=edge_colors,
                is_directed=is_directed,
                allow_loops=allow_loops,
            )

    @classmethod
    def from_adjacency_matrix(
        cls,
        adjacency_matrix: np.ndarray,
        edge_colors: int = 2,
        is_directed: bool = False,
        allow_loops: bool = False,
    ) -> GraphBatch:
        """
        This class method initializes a `GraphBatch` object by using the adjacency matrix format
        (`GraphFormat.ADJACENCY_MATRIX`).

        :param adjacency_matrix: The `np.ndarray` that describes the adjacency matrix format
            (`GraphFormat.ADJACENCY_MATRIX`) of the batch of graphs that should be initialized.
        :param edge_colors: A positive integer (not below two) that represents the number of proper
            edge colors, i.e., $k$. The default value is two.
        :param is_directed: A boolean that indicates whether each of the graphs in the given batch
            is a $k$-edge-colored looped complete directed graph or a $k$-edge-colored looped
            complete undirected graph. The default value is `False`, i.e., the graphs are
            undirected by default.
        :param allow_loops: A boolean that indicates whether each of the graphs in the given batch
            is allowed to have loops (if loops are not allowed, then all the loops are removed from
            the considered complete graphs and they simply do not exist). The default value is
            `False`, i.e., the graphs are not allowed to have loops by default.

        :return: The initialized `GraphBatch` object.
        """

        return cls(
            graph_format=GraphFormat.ADJACENCY_MATRIX,
            adjacency_matrix=adjacency_matrix,
            edge_colors=edge_colors,
            is_directed=is_directed,
            allow_loops=allow_loops,
        )

    @classmethod
    def from_flattened(
        cls,
        flattened: np.ndarray,
        flattened_ordering: FlattenedOrdering = FlattenedOrdering.ROW_MAJOR,
        edge_colors: int = 2,
        is_directed: bool = False,
        allow_loops: bool = False,
    ) -> GraphBatch:
        """
        This class method initializes a `GraphBatch` object by using either the flattened row-major
        format (`GraphFormat.FLATTENED_ROW_MAJOR`) or the flattened clockwise format
        (`GraphFormat.FLATTENED_CLOCKWISE`).

        :param flattened: The `np.ndarray` that describes the flattened row-major format
            (`GraphFormat.FLATTENED_ROW_MAJOR`) or the flattened clockwise format
            (`GraphFormat.FLATTENED_CLOCKWISE`) of the batch of graphs that should be initialized.
        :param flattened_ordering: An item of the `FlattenedOrdering` enumeration that determines
            whether the flattened row-major format should be used or the flattened clockwise
            format. The default value is `FlattenedOrdering.ROW_MAJOR`, i.e., the flattened
            row-major format is used by default.
        :param edge_colors: A positive integer (not below two) that represents the number of proper
            edge colors, i.e., $k$. The default value is two.
        :param is_directed: A boolean that indicates whether each of the graphs in the given batch
            is a $k$-edge-colored looped complete directed graph or a $k$-edge-colored looped
            complete undirected graph. The default value is `False`, i.e., the graphs are
            undirected by default.
        :param allow_loops: A boolean that indicates whether each of the graphs in the given batch
            is allowed to have loops (if loops are not allowed, then all the loops are removed from
            the considered complete graphs and they simply do not exist). The default value is
            `False`, i.e., the graphs are not allowed to have loops by default.

        :return: The initialized `GraphBatch` object.
        """

        if flattened_ordering == FlattenedOrdering.ROW_MAJOR:
            return cls(
                graph_format=GraphFormat.FLATTENED_ROW_MAJOR,
                flattened_row_major=flattened,
                edge_colors=edge_colors,
                is_directed=is_directed,
                allow_loops=allow_loops,
            )
        else:
            return cls(
                graph_format=GraphFormat.FLATTENED_CLOCKWISE,
                flattened_clockwise=flattened,
                edge_colors=edge_colors,
                is_directed=is_directed,
                allow_loops=allow_loops,
            )

    @property
    def edge_colors(self) -> int:
        """
        This property returns the number of proper edge colors for the given batch of graphs, i.e.,
        $k$, as an `int`.
        """

        return self.__edge_colors

    @property
    def is_directed(self) -> bool:
        """
        This property returns the `bool` that determines whether each of the graphs in the given
        batch is directed. The value `True` indicates that the graphs are directed.
        """

        return self.__is_directed

    @property
    def allow_loops(self) -> bool:
        """
        This property returns the `bool` that determines whether loops are allowed in each of the
        graphs in the given batch. The value `True` indicates that loops are allowed.
        """

        return self.__allow_loops

    @property
    def batch_size(self) -> int:
        """
        This property returns the batch size of the given batch of graphs, i.e., the number of
        graphs in the batch.
        """

        return self.__batch_size

    @property
    def order(self) -> int:
        """
        This property returns the order of all the graphs in the given batch, as an `int`.
        """

        return self.__order

    @property
    def bitmask_out(self) -> np.ndarray:
        """
        This property returns the `np.ndarray` that represents the given batch of graphs in the
        bitmask format for the out-neighborhoods (`GraphFormat.BITMASK_OUT`).
        """

        # If the output `np.ndarray` is already known, then just return it.
        if self.__bitmask_out is not None:
            return self.__bitmask_out

        # Otherwise, compute the output `np.ndarray` by using the adjacency matrix format
        # representation. If the adjacency matrix format representation is also unknown, then it
        # will first get computed by using one of the remaining three format representations, which
        # is surely known.
        color_indices = np.arange(self.__edge_colors, dtype=int)
        masks = 1 << np.arange(self.__order, dtype=int)

        # If not all of the graphs are fully colored, then the reduced bitmask format cannot be
        # used.
        if np.max(self.adjacency_matrix) == self.__edge_colors:
            temp = (
                self.__adjacency_matrix[:, None, :, :] == color_indices[None, :, None, None]
            ).astype(int)
            if not self.__allow_loops:
                np.einsum("bcii->bci", temp)[:] = 0
            result = temp @ masks
        # Otherwise, we use the reduced bitmask format.
        else:
            temp = (
                self.__adjacency_matrix[:, None, :, :] == color_indices[None, 1:, None, None]
            ).astype(int)
            result = temp @ masks

        # Update the out-neighborhoods bitmask format representation to make it available for
        # further use, so that the same conversion does not have to be performed twice.
        self.__bitmask_out = result

        return self.__bitmask_out

    @property
    def bitmask_in(self) -> np.ndarray:
        """
        This property returns the `np.ndarray` that represents the given batch of graphs in the
        bitmask format for the in-neighborhoods (`GraphFormat.BITMASK_IN`).
        """

        # If the output `np.ndarray` is already known, then just return it.
        if self.__bitmask_in is not None:
            return self.__bitmask_in

        # If the graph is undirected, then just use the bitmask format for the out-neighborhoods
        # and update the in-neighborhoods bitmask format representation to make it available for
        # further use.
        if not self.__is_directed:
            self.__bitmask_in = self.bitmask_out

            return self.__bitmask_in

        # Otherwise, compute the output `np.ndarray` by using the adjacency matrix format
        # representation. If the adjacency matrix format representation is also unknown, then it
        # will first get computed by using one of the remaining three format representations, which
        # is surely known.
        color_indices = np.arange(self.__edge_colors, dtype=int)
        masks = 1 << np.arange(self.__order, dtype=int)

        # If not all of the graphs are fully colored, then the reduced bitmask format cannot be
        # used.
        if np.max(self.adjacency_matrix) == self.__edge_colors:
            temp = (
                self.__adjacency_matrix.transpose(0, 2, 1)[:, None, :, :]
                == color_indices[None, :, None, None]
            ).astype(int)
            if not self.__allow_loops:
                np.einsum("bcii->bci", temp)[:] = 0
            result = temp @ masks
        # Otherwise, we use the reduced bitmask format.
        else:
            temp = (
                self.__adjacency_matrix.transpose(0, 2, 1)[:, None, :, :]
                == color_indices[None, 1:, None, None]
            ).astype(int)
            result = temp @ masks

        # Update the in-neighborhoods bitmask format representation to make it available for
        # further use, so that the same conversion does not have to be performed twice.
        self.__bitmask_in = result

        return self.__bitmask_in

    @property
    def adjacency_matrix(self) -> np.ndarray:
        """
        This property returns the `np.ndarray` that represents the given batch of graphs in the
        adjacency matrix format (`GraphFormat.ADJACENCY_MATRIX`).
        """

        # If the output `np.ndarray` is already known, then just return it.
        if self.__adjacency_matrix is not None:
            return self.__adjacency_matrix

        # If the flattened row-major format representation is known, use it to obtain the adjacency
        # matrix format representation.
        if self.__flattened_row_major is not None:
            if self.__is_directed:
                if self.__allow_loops:
                    result = self.__flattened_row_major.reshape(
                        self.__batch_size, self.__order, self.__order
                    )
                else:
                    result = np.zeros((self.__batch_size, self.__order, self.__order), dtype=int)
                    result[:, ~np.eye(self.__order, dtype=bool)] = self.__flattened_row_major

            else:
                result = np.zeros((self.__batch_size, self.__order, self.__order), dtype=int)

                if self.__allow_loops:
                    triu_rows, triu_columns = np.triu_indices(self.__order, k=0)
                else:
                    triu_rows, triu_columns = np.triu_indices(self.__order, k=1)

                result[:, triu_rows, triu_columns] = self.__flattened_row_major
                result[:, triu_columns, triu_rows] = self.__flattened_row_major

            # Update the adjacency matrix format representation to make it available for further
            # use, so that the same conversion does not have to be performed twice.
            self.__adjacency_matrix = result

            return self.__adjacency_matrix

        # If the flattened clockwise format representation is known, use it to obtain the adjacency
        # matrix format representation.
        if self.__flattened_clockwise is not None:
            result = np.zeros((self.__batch_size, self.__order, self.__order), dtype=int)

            if self.__is_directed:
                if self.__allow_loops:
                    result[:, 0, 0] = self.__flattened_clockwise[:, 0]

                    start = 1
                    for layer in range(1, self.__order):
                        result[:, : layer + 1, layer] = self.__flattened_clockwise[
                            :, start : start + layer + 1
                        ]
                        start += layer + 1

                        result[:, layer, :layer] = self.__flattened_clockwise[
                            :, start + layer - 1 : start - 1 : -1
                        ]
                        start += layer

                else:
                    start = 0
                    for layer in range(1, self.__order):
                        result[:, :layer, layer] = self.__flattened_clockwise[
                            :, start : start + layer
                        ]
                        start += layer

                        result[:, layer, :layer] = self.__flattened_clockwise[
                            :, start + layer - 1 : start - 1 : -1
                        ]
                        start += layer

            else:
                if self.__allow_loops:
                    tril_rows, tril_columns = np.tril_indices(self.__order, k=0)
                else:
                    tril_rows, tril_columns = np.tril_indices(self.__order, k=-1)

                result[:, tril_rows, tril_columns] = self.__flattened_clockwise
                result[:, tril_columns, tril_rows] = self.__flattened_clockwise

            # Update the adjacency matrix format representation to make it available for further
            # use, so that the same conversion does not have to be performed twice.
            self.__adjacency_matrix = result

            return self.__adjacency_matrix

        # Otherwise, at least one of the two bitmask format representation must be known, hence it
        # can be used to find the adjacency matrix representation.
        masks = 1 << np.arange(self.__order, dtype=int)
        if self.__bitmask_out is not None:
            temp = self.__bitmask_out[:, :, :, None] & masks
        else:
            temp = self.__bitmask_in[:, :, :, None] & masks

        temp = (temp != 0).astype(int)

        # If the number of rows of any of the two bitmask format representation matrices matches
        # the number of proper edge colors, then this means that a standard (non-reduced) bitmask
        # format is being used.
        if temp.shape[1] == self.__edge_colors:
            result = np.full(
                (self.__batch_size, self.__order, self.__order), self.__edge_colors, dtype=int
            )
            if not self.__allow_loops:
                np.einsum("bii->bi", result)[:] = 0
            weights = np.arange(-self.__edge_colors, 0, dtype=int)
            result += np.sum(temp * weights[None, :, None, None], axis=1)

        # Otherwise, a reduced bitmask format is being used.
        else:
            weights = np.arange(1, self.__edge_colors, dtype=int)
            result = np.sum(temp * weights[None, :, None, None], axis=1)

        if self.__bitmask_out is None:
            result = result.transpose(0, 2, 1)

        # Update the adjacency matrix format representation to make it available for further use,
        # so that the same conversion does not have to be performed twice.
        self.__adjacency_matrix = result

        return self.__adjacency_matrix

    @property
    def flattened_row_major(self) -> np.ndarray:
        """
        This property returns the `np.ndarray` that represents the given batch of graphs in the
        flattened row-major format (`GraphFormat.FLATTENED_ROW_MAJOR`).
        """

        # If the output `np.ndarray` is already known, then just return it.
        if self.__flattened_row_major is not None:
            return self.__flattened_row_major

        # Otherwise, compute the output `np.ndarray` by using the adjacency matrix format
        # representation. If the adjacency matrix format representation is also unknown, then it
        # will first get computed by using one of the remaining three format representations, which
        # is surely known.
        if self.__is_directed:
            if self.__allow_loops:
                result = self.adjacency_matrix.reshape(self.__batch_size, -1)
            else:
                result = self.adjacency_matrix[:, ~np.eye(self.__order, dtype=bool)]

        else:
            if self.__allow_loops:
                triu_rows, triu_columns = np.triu_indices(self.__order, k=0)
            else:
                triu_rows, triu_columns = np.triu_indices(self.__order, k=1)

            result = self.adjacency_matrix[:, triu_rows, triu_columns]

        # Update the flattened row-major format representation to make it available for further
        # use, so that the same conversion does not have to be performed twice.
        self.__flattened_row_major = result

        return self.__flattened_row_major

    @property
    def flattened_clockwise(self) -> np.ndarray:
        """
        This property returns the `np.ndarray` that represents the given batch of graphs in the
        flattened clockwise format (`GraphFormat.FLATTENED_CLOCKWISE`).
        """

        # If the output `np.ndarray` is already known, then just return it.
        if self.__flattened_clockwise is not None:
            return self.__flattened_clockwise

        # Otherwise, compute the output `np.ndarray` by using the adjacency matrix format
        # representation. If the adjacency matrix format representation is also unknown, then it
        # will first get computed by using one of the remaining two format representations, which
        # is surely known.
        if self.__is_directed:
            if self.__allow_loops:
                result = np.zeros((self.__batch_size, self.__order * self.__order), dtype=int)
                result[:, 0] = self.adjacency_matrix[:, 0, 0]

                start = 1
                for layer in range(1, self.__order):
                    result[:, start : start + layer + 1] = self.adjacency_matrix[
                        :, : layer + 1, layer
                    ]
                    start += layer + 1

                    result[:, start : start + layer] = self.adjacency_matrix[
                        :, layer, layer - 1 :: -1
                    ]
                    start += layer

            else:
                result = np.zeros(
                    (self.__batch_size, self.__order * (self.__order - 1)), dtype=int
                )

                start = 0
                for layer in range(1, self.__order):
                    result[:, start : start + layer] = self.adjacency_matrix[:, :layer, layer]
                    start += layer

                    result[:, start : start + layer] = self.adjacency_matrix[
                        :, layer, layer - 1 :: -1
                    ]
                    start += layer

        else:
            if self.__allow_loops:
                tril_rows, tril_columns = np.tril_indices(self.__order, k=0)
            else:
                tril_rows, tril_columns = np.tril_indices(self.__order, k=-1)

            result = self.adjacency_matrix[:, tril_rows, tril_columns]

        # Update the flattened clockwise format representation to make it available for further
        # use, so that the same conversion does not have to be performed twice.
        self.__flattened_clockwise = result

        return self.__flattened_clockwise
