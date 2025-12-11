"""
This file initializes the lists containing the test cases for the `tests.graphs.test_graph` testing
module.
"""

import numpy as np


GRAPH_TEST_CASES_BASIC = [
    (
        2,
        2,
        np.array([[0, 0]], dtype=int),
        np.array([[0, 0], [0, 0]], dtype=int),
        np.array([0], dtype=int),
        np.array([0], dtype=int),
    ),
    (
        2,
        2,
        np.array([[2, 1]], dtype=int),
        np.array([[0, 1], [1, 0]], dtype=int),
        np.array([1], dtype=int),
        np.array([1], dtype=int),
    ),
    (
        3,
        2,
        np.array([[0, 0], [0, 0]], dtype=int),
        np.array([[0, 0], [0, 0]], dtype=int),
        np.array([0], dtype=int),
        np.array([0], dtype=int),
    ),
    (
        3,
        2,
        np.array([[2, 1], [0, 0]], dtype=int),
        np.array([[0, 1], [1, 0]], dtype=int),
        np.array([1], dtype=int),
        np.array([1], dtype=int),
    ),
    (
        3,
        2,
        np.array([[0, 0], [2, 1]], dtype=int),
        np.array([[0, 2], [2, 0]], dtype=int),
        np.array([2], dtype=int),
        np.array([2], dtype=int),
    ),
    (
        2,
        3,
        np.array([[0, 0, 0]], dtype=int),
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=int),
        np.array([0, 0, 0], dtype=int),
        np.array([0, 0, 0], dtype=int),
    ),
    (
        2,
        3,
        np.array([[2, 1, 0]], dtype=int),
        np.array(
            [
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 0],
            ],
            dtype=int,
        ),
        np.array([1, 0, 0], dtype=int),
        np.array([1, 0, 0], dtype=int),
    ),
    (
        2,
        3,
        np.array([[4, 4, 3]], dtype=int),
        np.array(
            [
                [0, 0, 1],
                [0, 0, 1],
                [1, 1, 0],
            ],
            dtype=int,
        ),
        np.array([0, 1, 1], dtype=int),
        np.array([0, 1, 1], dtype=int),
    ),
    (
        2,
        4,
        np.array([[10, 1, 0, 1]], dtype=int),
        np.array(
            [
                [0, 1, 0, 1],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 0],
            ],
            dtype=int,
        ),
        np.array([1, 0, 0, 1, 0, 0], dtype=int),
        np.array([1, 0, 1, 0, 0, 0], dtype=int),
    ),
]

GRAPH_TEST_CASES_LOOPS = [
    (
        2,
        2,
        np.array([[0, 0]], dtype=int),
        np.array([[0, 0], [0, 0]], dtype=int),
        np.array([0, 0, 0], dtype=int),
        np.array([0, 0, 0], dtype=int),
    ),
    (
        2,
        2,
        np.array([[1, 0]], dtype=int),
        np.array([[1, 0], [0, 0]], dtype=int),
        np.array([1, 0, 0], dtype=int),
        np.array([1, 0, 0], dtype=int),
    ),
    (
        2,
        2,
        np.array([[1, 2]], dtype=int),
        np.array([[1, 0], [0, 1]], dtype=int),
        np.array([1, 0, 1], dtype=int),
        np.array([1, 0, 1], dtype=int),
    ),
    (
        2,
        3,
        np.array([[5, 2, 1]], dtype=int),
        np.array(
            [
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
            ],
            dtype=int,
        ),
        np.array([1, 0, 1, 1, 0, 0], dtype=int),
        np.array([1, 0, 1, 1, 0, 0], dtype=int),
    ),
]

GRAPH_TEST_CASES_DIRECTED = [
    (
        2,
        2,
        np.array([[0, 0]], dtype=int),
        np.array([[0, 0]], dtype=int),
        np.array([[0, 0], [0, 0]], dtype=int),
        np.array([0, 0], dtype=int),
        np.array([0, 0], dtype=int),
    ),
    (
        2,
        2,
        np.array([[0, 1]], dtype=int),
        np.array([[2, 0]], dtype=int),
        np.array(
            [
                [0, 1],
                [0, 0],
            ],
            dtype=int,
        ),
        np.array([1, 0], dtype=int),
        np.array([1, 0], dtype=int),
    ),
    (
        2,
        2,
        np.array([[2, 0]], dtype=int),
        np.array([[0, 1]], dtype=int),
        np.array(
            [
                [0, 0],
                [1, 0],
            ],
            dtype=int,
        ),
        np.array([0, 1], dtype=int),
        np.array([0, 1], dtype=int),
    ),
    (
        2,
        3,
        np.array([[0, 1, 0]], dtype=int),
        np.array([[2, 0, 0]], dtype=int),
        np.array(
            [
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=int,
        ),
        np.array([1, 0, 0, 0, 0, 0], dtype=int),
        np.array([1, 0, 0, 0, 0, 0], dtype=int),
    ),
    (
        2,
        3,
        np.array([[0, 0, 2]], dtype=int),
        np.array([[0, 4, 0]], dtype=int),
        np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 0],
            ],
            dtype=int,
        ),
        np.array([0, 0, 0, 1, 0, 0], dtype=int),
        np.array([0, 0, 0, 1, 0, 0], dtype=int),
    ),
    (
        2,
        3,
        np.array([[0, 4, 0]], dtype=int),
        np.array([[0, 0, 2]], dtype=int),
        np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 1, 0],
            ],
            dtype=int,
        ),
        np.array([0, 0, 0, 0, 1, 0], dtype=int),
        np.array([0, 0, 0, 0, 0, 1], dtype=int),
    ),
    (
        2,
        3,
        np.array([[0, 5, 1]], dtype=int),
        np.array([[6, 0, 2]], dtype=int),
        np.array(
            [
                [0, 1, 1],
                [0, 0, 0],
                [0, 1, 0],
            ],
            dtype=int,
        ),
        np.array([1, 0, 1, 0, 1, 0], dtype=int),
        np.array([1, 1, 0, 0, 0, 1], dtype=int),
    ),
    (
        3,
        3,
        np.array(
            [
                [0, 4, 1],
                [0, 1, 0],
            ],
            dtype=int,
        ),
        np.array(
            [
                [4, 0, 2],
                [2, 0, 0],
            ],
            dtype=int,
        ),
        np.array(
            [
                [0, 2, 1],
                [0, 0, 0],
                [0, 1, 0],
            ],
            dtype=int,
        ),
        np.array([2, 0, 1, 0, 1, 0], dtype=int),
        np.array([2, 1, 0, 0, 0, 1], dtype=int),
    ),
]


GRAPH_TEST_CASES_DIRECTED_LOOPS = [
    (
        2,
        2,
        np.array([[0, 0]], dtype=int),
        np.array([[0, 0]], dtype=int),
        np.array([[0, 0], [0, 0]], dtype=int),
        np.array([0, 0, 0, 0], dtype=int),
        np.array([0, 0, 0, 0], dtype=int),
    ),
    (
        2,
        2,
        np.array([[0, 1]], dtype=int),
        np.array([[2, 0]], dtype=int),
        np.array(
            [
                [0, 1],
                [0, 0],
            ],
            dtype=int,
        ),
        np.array([0, 1, 0, 0], dtype=int),
        np.array([0, 1, 0, 0], dtype=int),
    ),
    (
        2,
        2,
        np.array([[0, 3]], dtype=int),
        np.array([[2, 2]], dtype=int),
        np.array(
            [
                [0, 1],
                [0, 1],
            ],
            dtype=int,
        ),
        np.array([0, 1, 1, 0], dtype=int),
        np.array([0, 1, 0, 1], dtype=int),
    ),
    (
        3,
        2,
        np.array([[0, 1], [0, 2]], dtype=int),
        np.array([[2, 0], [0, 2]], dtype=int),
        np.array(
            [
                [0, 1],
                [0, 2],
            ],
            dtype=int,
        ),
        np.array([0, 1, 2, 0], dtype=int),
        np.array([0, 1, 0, 2], dtype=int),
    ),
    (
        3,
        3,
        np.array(
            [
                [0, 4, 1],
                [0, 3, 0],
            ],
            dtype=int,
        ),
        np.array(
            [
                [4, 0, 2],
                [2, 2, 0],
            ],
            dtype=int,
        ),
        np.array(
            [
                [0, 2, 1],
                [0, 2, 0],
                [0, 1, 0],
            ],
            dtype=int,
        ),
        np.array([0, 2, 2, 0, 1, 0, 0, 1, 0], dtype=int),
        np.array([0, 2, 1, 0, 2, 0, 0, 1, 0], dtype=int),
    ),
]


GRAPH_TEST_CASES = [
    (
        4,
        7,
        np.array(
            [
                [6, 21, 35, 0, 2, 4, 0],
                [32, 32, 0, 0, 0, 3, 0],
                [16, 8, 16, 18, 109, 16, 16],
                [72, 64, 0, 1, 0, 0, 3],
            ],
            dtype=int,
        ),
        np.array(
            [
                [0, 0, 0, 3, 2, 1, 3],
                [0, 0, 0, 2, 0, 1, 3],
                [0, 0, 0, 4, 2, 0, 4],
                [3, 2, 4, 0, 2, 4, 4],
                [2, 0, 2, 2, 0, 2, 2],
                [1, 1, 0, 4, 2, 0, 4],
                [3, 3, 4, 4, 2, 4, 0],
            ],
            dtype=int,
        ),
        np.array([0, 0, 0, 3, 2, 4, 2, 0, 2, 2, 1, 1, 0, 4, 2, 3, 3, 4, 4, 2, 4], dtype=int),
        np.array([0, 0, 3, 2, 1, 3, 0, 2, 0, 1, 3, 4, 2, 0, 4, 2, 4, 4, 2, 2, 4], dtype=int),
    ),
    (
        3,
        8,
        np.array(
            [
                [32, 0, 16, 32, 4, 9, 0, 0],
                [80, 128, 40, 68, 65, 68, 57, 2],
                [0, 16, 192, 144, 170, 16, 132, 92],
            ],
            dtype=int,
        ),
        np.array(
            [
                [0, 3, 3, 3, 1, 0, 1, 3],
                [3, 0, 3, 3, 2, 3, 3, 1],
                [3, 3, 0, 1, 0, 1, 2, 2],
                [3, 3, 1, 0, 2, 0, 1, 2],
                [1, 2, 0, 2, 0, 2, 1, 2],
                [0, 3, 1, 0, 2, 0, 1, 3],
                [1, 3, 2, 1, 1, 1, 0, 2],
                [3, 1, 2, 2, 2, 3, 2, 0],
            ],
            dtype=int,
        ),
        np.array(
            [
                3,
                3,
                3,
                3,
                3,
                1,
                1,
                2,
                0,
                2,
                0,
                3,
                1,
                0,
                2,
                1,
                3,
                2,
                1,
                1,
                1,
                3,
                1,
                2,
                2,
                2,
                3,
                2,
            ],
            dtype=int,
        ),
        np.array(
            [
                3,
                3,
                3,
                1,
                0,
                1,
                3,
                3,
                3,
                2,
                3,
                3,
                1,
                1,
                0,
                1,
                2,
                2,
                2,
                0,
                1,
                2,
                2,
                1,
                2,
                1,
                3,
                2,
            ],
            dtype=int,
        ),
    ),
    (
        4,
        6,
        np.array(
            [
                [0, 12, 42, 6, 0, 4],
                [14, 1, 17, 17, 44, 16],
                [32, 32, 0, 0, 0, 3],
            ],
            dtype=int,
        ),
        np.array(
            [
                [0, 2, 2, 2, 0, 3],
                [2, 0, 1, 1, 0, 3],
                [2, 1, 0, 1, 2, 1],
                [2, 1, 1, 0, 2, 0],
                [0, 0, 2, 2, 0, 2],
                [3, 3, 1, 0, 2, 0],
            ],
            dtype=int,
        ),
        np.array([2, 2, 1, 2, 1, 1, 0, 0, 2, 2, 3, 3, 1, 0, 2], dtype=int),
        np.array([2, 2, 2, 0, 3, 1, 1, 0, 3, 1, 2, 1, 2, 0, 2], dtype=int),
    ),
    (
        2,
        5,
        np.array([[22, 9, 25, 6, 5]], dtype=int),
        np.array(
            [
                [0, 1, 1, 0, 1],
                [1, 0, 0, 1, 0],
                [1, 0, 0, 1, 1],
                [0, 1, 1, 0, 0],
                [1, 0, 1, 0, 0],
            ],
            dtype=int,
        ),
        np.array([1, 1, 0, 0, 1, 1, 1, 0, 1, 0], dtype=int),
        np.array([1, 1, 0, 1, 0, 1, 0, 1, 1, 0], dtype=int),
    ),
    (
        3,
        1,
        np.array([[0], [0]], dtype=int),
        np.array([[0]], dtype=int),
        np.zeros((0,), dtype=int),
        np.zeros((0,), dtype=int),
    ),
]


GRAPH_BATCH_TEST_CASES = [
    (
        1,
        4,
        7,
        np.array(
            [
                [
                    [6, 21, 35, 0, 2, 4, 0],
                    [32, 32, 0, 0, 0, 3, 0],
                    [16, 8, 16, 18, 109, 16, 16],
                    [72, 64, 0, 1, 0, 0, 3],
                ]
            ],
            dtype=int,
        ),
        np.array(
            [
                [
                    [0, 0, 0, 3, 2, 1, 3],
                    [0, 0, 0, 2, 0, 1, 3],
                    [0, 0, 0, 4, 2, 0, 4],
                    [3, 2, 4, 0, 2, 4, 4],
                    [2, 0, 2, 2, 0, 2, 2],
                    [1, 1, 0, 4, 2, 0, 4],
                    [3, 3, 4, 4, 2, 4, 0],
                ]
            ],
            dtype=int,
        ),
        np.array([[0, 0, 0, 3, 2, 4, 2, 0, 2, 2, 1, 1, 0, 4, 2, 3, 3, 4, 4, 2, 4]], dtype=int),
        np.array([[0, 0, 3, 2, 1, 3, 0, 2, 0, 1, 3, 4, 2, 0, 4, 2, 4, 4, 2, 2, 4]], dtype=int),
    ),
    (
        1,
        3,
        8,
        np.array(
            [
                [
                    [32, 0, 16, 32, 4, 9, 0, 0],
                    [80, 128, 40, 68, 65, 68, 57, 2],
                    [0, 16, 192, 144, 170, 16, 132, 92],
                ]
            ],
            dtype=int,
        ),
        np.array(
            [
                [
                    [0, 3, 3, 3, 1, 0, 1, 3],
                    [3, 0, 3, 3, 2, 3, 3, 1],
                    [3, 3, 0, 1, 0, 1, 2, 2],
                    [3, 3, 1, 0, 2, 0, 1, 2],
                    [1, 2, 0, 2, 0, 2, 1, 2],
                    [0, 3, 1, 0, 2, 0, 1, 3],
                    [1, 3, 2, 1, 1, 1, 0, 2],
                    [3, 1, 2, 2, 2, 3, 2, 0],
                ]
            ],
            dtype=int,
        ),
        np.array(
            [
                [
                    3,
                    3,
                    3,
                    3,
                    3,
                    1,
                    1,
                    2,
                    0,
                    2,
                    0,
                    3,
                    1,
                    0,
                    2,
                    1,
                    3,
                    2,
                    1,
                    1,
                    1,
                    3,
                    1,
                    2,
                    2,
                    2,
                    3,
                    2,
                ]
            ],
            dtype=int,
        ),
        np.array(
            [
                [
                    3,
                    3,
                    3,
                    1,
                    0,
                    1,
                    3,
                    3,
                    3,
                    2,
                    3,
                    3,
                    1,
                    1,
                    0,
                    1,
                    2,
                    2,
                    2,
                    0,
                    1,
                    2,
                    2,
                    1,
                    2,
                    1,
                    3,
                    2,
                ]
            ],
            dtype=int,
        ),
    ),
    (
        1,
        4,
        6,
        np.array(
            [
                [
                    [0, 12, 42, 6, 0, 4],
                    [14, 1, 17, 17, 44, 16],
                    [32, 32, 0, 0, 0, 3],
                ]
            ],
            dtype=int,
        ),
        np.array(
            [
                [
                    [0, 2, 2, 2, 0, 3],
                    [2, 0, 1, 1, 0, 3],
                    [2, 1, 0, 1, 2, 1],
                    [2, 1, 1, 0, 2, 0],
                    [0, 0, 2, 2, 0, 2],
                    [3, 3, 1, 0, 2, 0],
                ]
            ],
            dtype=int,
        ),
        np.array([[2, 2, 1, 2, 1, 1, 0, 0, 2, 2, 3, 3, 1, 0, 2]], dtype=int),
        np.array([[2, 2, 2, 0, 3, 1, 1, 0, 3, 1, 2, 1, 2, 0, 2]], dtype=int),
    ),
    (
        1,
        2,
        5,
        np.array([[[22, 9, 25, 6, 5]]], dtype=int),
        np.array(
            [
                [
                    [0, 1, 1, 0, 1],
                    [1, 0, 0, 1, 0],
                    [1, 0, 0, 1, 1],
                    [0, 1, 1, 0, 0],
                    [1, 0, 1, 0, 0],
                ]
            ],
            dtype=int,
        ),
        np.array([[1, 1, 0, 0, 1, 1, 1, 0, 1, 0]], dtype=int),
        np.array([[1, 1, 0, 1, 0, 1, 0, 1, 1, 0]], dtype=int),
    ),
    (
        1,
        3,
        1,
        np.array([[[0], [0]]], dtype=int),
        np.array([[[0]]], dtype=int),
        np.zeros((1, 0), dtype=int),
        np.zeros((1, 0), dtype=int),
    ),
    (
        3,
        4,
        5,
        np.array(
            [
                [
                    [8, 0, 24, 5, 4],
                    [2, 1, 0, 16, 8],
                    [0, 16, 0, 0, 2],
                    [4, 4, 3, 0, 0],
                ],
                [
                    [12, 4, 11, 5, 0],
                    [16, 16, 0, 0, 3],
                    [0, 0, 0, 16, 8],
                    [2, 9, 16, 2, 4],
                ],
                [
                    [2, 13, 2, 2, 0],
                    [0, 0, 0, 0, 0],
                    [4, 16, 9, 4, 2],
                    [0, 0, 0, 16, 8],
                ],
            ],
            dtype=int,
        ),
        np.array(
            [
                [
                    [0, 1, 3, 0, 4],
                    [1, 0, 3, 4, 2],
                    [3, 3, 0, 0, 0],
                    [0, 4, 0, 0, 1],
                    [4, 2, 0, 1, 0],
                ],
                [
                    [0, 3, 0, 0, 1],
                    [3, 0, 0, 3, 1],
                    [0, 0, 0, 0, 3],
                    [0, 3, 0, 0, 2],
                    [1, 1, 3, 2, 0],
                ],
                [
                    [0, 0, 2, 4, 4],
                    [0, 0, 0, 0, 2],
                    [2, 0, 0, 2, 4],
                    [4, 0, 2, 0, 3],
                    [4, 2, 4, 3, 0],
                ],
            ],
            dtype=int,
        ),
        np.array(
            [
                [1, 3, 3, 0, 4, 0, 4, 2, 0, 1],
                [3, 0, 0, 0, 3, 0, 1, 1, 3, 2],
                [0, 2, 0, 4, 0, 2, 4, 2, 4, 3],
            ],
            dtype=int,
        ),
        np.array(
            [
                [1, 3, 0, 4, 3, 4, 2, 0, 0, 1],
                [3, 0, 0, 1, 0, 3, 1, 0, 3, 2],
                [0, 2, 4, 4, 0, 0, 2, 2, 4, 3],
            ],
            dtype=int,
        ),
    ),
    (
        2,
        6,
        7,
        np.array(
            [
                [
                    [0, 0, 8, 68, 0, 0, 8],
                    [12, 0, 1, 1, 0, 0, 0],
                    [0, 88, 16, 2, 102, 16, 18],
                    [34, 1, 64, 48, 8, 9, 4],
                    [80, 4, 34, 0, 1, 68, 33],
                ],
                [
                    [32, 36, 50, 0, 36, 87, 32],
                    [8, 0, 0, 1, 0, 0, 0],
                    [2, 1, 8, 20, 8, 0, 0],
                    [0, 64, 64, 32, 0, 8, 6],
                    [68, 8, 1, 2, 0, 0, 1],
                ],
            ],
            dtype=int,
        ),
        np.array(
            [
                [
                    [0, 4, 2, 2, 5, 4, 5],
                    [4, 0, 5, 3, 3, 0, 3],
                    [2, 5, 0, 1, 3, 5, 4],
                    [2, 3, 1, 0, 4, 4, 1],
                    [5, 3, 3, 4, 0, 3, 3],
                    [4, 0, 5, 4, 3, 0, 5],
                    [5, 3, 4, 1, 3, 5, 0],
                ],
                [
                    [0, 3, 5, 2, 0, 1, 5],
                    [3, 0, 1, 5, 0, 1, 4],
                    [5, 1, 0, 3, 1, 1, 4],
                    [2, 5, 3, 0, 3, 4, 0],
                    [0, 0, 1, 3, 0, 1, 0],
                    [1, 1, 1, 4, 1, 0, 1],
                    [5, 4, 4, 0, 0, 1, 0],
                ],
            ],
            dtype=int,
        ),
        np.array(
            [
                [4, 2, 5, 2, 3, 1, 5, 3, 3, 4, 4, 0, 5, 4, 3, 5, 3, 4, 1, 3, 5],
                [3, 5, 1, 2, 5, 3, 0, 0, 1, 3, 1, 1, 1, 4, 1, 5, 4, 4, 0, 0, 1],
            ],
            dtype=int,
        ),
        np.array(
            [
                [4, 2, 2, 5, 4, 5, 5, 3, 3, 0, 3, 1, 3, 5, 4, 4, 4, 1, 3, 3, 5],
                [3, 5, 2, 0, 1, 5, 1, 5, 0, 1, 4, 3, 1, 1, 4, 3, 4, 0, 1, 0, 1],
            ],
            dtype=int,
        ),
    ),
    (
        7,
        4,
        1,
        np.zeros((7, 3, 1), dtype=int),
        np.zeros((7, 1, 1), dtype=int),
        np.zeros((7, 0), dtype=int),
        np.zeros((7, 0), dtype=int),
    ),
]
