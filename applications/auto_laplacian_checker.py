import pickle

from sage.all import *


# The expressions used to construct the right-hand sides of the inequalities from
#     V. Brankov, P. Hansen and D. Stevanović, Automated conjectures on upper bounds for the
#     largest Laplacian eigenvalue of graphs, Linear Algebra Appl. 414 (2006), 407-424.
LAPLACIAN_EXPRESSIONS = {
    1: lambda d, m: sqrt(4 * d**3 / m),
    2: lambda d, m: 2 * m**2 / d,
    3: lambda d, m: m**2 / d + m,
    4: lambda d, m: 2 * d**2 / m,
    5: lambda d, m: d**2 / m + m,
    6: lambda d, m: sqrt(m**2 + 3 * d**2),
    7: lambda d, m: d**2 / m + d,
    8: lambda d, m: sqrt(d * (m + 3 * d)),
    9: lambda d, m: (m + 3 * d) / 2,
    10: lambda d, m: sqrt(d * (d + 3 * m)),
    11: lambda d, m: 2 * m**3 / d**2,
    12: lambda d, m: sqrt(2 * m**2 + 2 * d**2),
    13: lambda d, m: 2 * m**4 / d**3,
    14: lambda d, m: 2 * d**3 / m**2,
    15: lambda d, m: sqrt(4 * m**3 / d),
    16: lambda d, m: 2 * d**4 / m**3,
    17: lambda d, m: (5 * d**4 + 11 * m**4) ** 0.25,
    18: lambda d, m: sqrt(2 * m**3 / d + 2 * d**2),
    19: lambda d, m: (4 * d**4 + 12 * d * m**3) ** 0.25,
    20: lambda d, m: sqrt(7 * d**2 + 9 * m**2) / 2,
    21: lambda d, m: sqrt(d**3 / m + 3 * m**2),
    22: lambda d, m: (2 * d**4 + 14 * d**2 * m**2) ** 0.25,
    23: lambda d, m: sqrt(d**2 + 3 * d * m),
    24: lambda d, m: (6 * d**4 + 10 * m**4) ** 0.25,
    25: lambda d, m: (3 * d**4 + 13 * d**2 * m**2) ** 0.25,
    26: lambda d, m: sqrt(5 * d**2 + 11 * d * m) / 2,
    27: lambda d, m: sqrt((3 * d**2 + 5 * d * m) / 2),
    28: lambda d, m: sqrt(2 * m**4 / d**2 + 2 * d * m),
    29: lambda d, m: sqrt(m**2 + 3 * m**3 / d),
    30: lambda d, m: m**3 / d**2 + d**2 / m,
    31: lambda d, m: 4 * m**2 / (m + d),
    32: lambda d, m: sqrt(m**3 * (m + 3 * d)) / d,
    33: lambda du, mu, dv, mv: 2 * (du + dv) - (mu + mv),
    34: lambda du, mu, dv, mv: 2 * (du**2 + dv**2) / (du + dv),
    35: lambda du, mu, dv, mv: 2 * (du**2 + dv**2) / (mu + mv),
    36: lambda du, mu, dv, mv: 2 * (mu**2 + mv**2) / (du + dv),
    37: lambda du, mu, dv, mv: sqrt(2 * (du**2 + dv**2)),
    38: lambda du, mu, dv, mv: 2 + sqrt(2 * (du - 1) ** 2 + 2 * (dv - 1) ** 2),
    39: lambda du, mu, dv, mv: 2 + sqrt(2 * (du**2 + dv**2) - 4 * (mu + mv) + 4),
    40: lambda du, mu, dv, mv: 2
    + sqrt(2 * ((mu - 1) ** 2 + (mv - 1) ** 2) + (du**2 + dv**2) - (du * mu + dv * mv)),
    41: lambda du, mu, dv, mv: 2
    + (mu + mv)
    - (du + dv)
    + sqrt(2 * (du**2 + dv**2) - 4 * (mu + mv) + 4),
    42: lambda du, mu, dv, mv: sqrt(du**2 + dv**2 + 2 * mu * mv),
    43: lambda du, mu, dv, mv: 2 + sqrt(3 * (mu**2 + mv**2) - 2 * mu * mv - 4 * (du + dv) + 4),
    44: lambda du, mu, dv, mv: 2
    + sqrt(2 * ((du - 1) ** 2 + (dv - 1) ** 2 + mu * mv - du * dv)),
    45: lambda du, mu, dv, mv: 2
    + sqrt((du - dv) ** 2 + 2 * (du * mu + dv * mv) - 4 * (mu + mv) + 4),
    46: lambda du, mu, dv, mv: 2 + sqrt(2 * (du**2 + dv**2) - 16 * (du * dv) / (mu + mv) + 4),
    47: lambda du, mu, dv, mv: (2 * (du**2 + dv**2) - (mu - mv) ** 2) / (du + dv),
    48: lambda du, mu, dv, mv: 2
    * (du**2 + dv**2)
    / (2 + sqrt(2 * (du**2 + dv**2) - 4 * (mu + mv) + 4)),
    49: lambda du, mu, dv, mv: 2
    + sqrt(2 * (mu**2 + mv**2) + (du - dv) ** 2 - 4 * (du + dv) + 4),
    50: lambda du, mu, dv, mv: 2 * (du**2 + dv**2 + mu * mv - du * dv) / (du + dv),
    51: lambda du, mu, dv, mv: 2 * (mu + mv) - 4 * (mu * mv) / (du + dv),
    52: lambda du, mu, dv, mv: 2
    + sqrt(sqrt(8 * (mu**4 + mv**4) - 8 * (du**2 + dv**2) + 4) - 4 * (du + dv) + 6),
    53: lambda du, mu, dv, mv: 2
    + sqrt(sqrt(8 * (mu**4 + mv**4) - 8 * (du * mu + dv * mv) + 4) - 4 * (du + dv) + 6),
    54: lambda du, mu, dv, mv: 2
    + sqrt(2 * (mu**2 + mv**2) + (du * mu + dv * mv) - (du**2 + dv**2) - 4 * (du + dv) + 4),
    55: lambda du, mu, dv, mv: 2
    + sqrt(3 * (mu**2 + mv**2) - (du**2 + dv**2) - 4 * (mu + mv) + 4),
    56: lambda du, mu, dv, mv: ((du**2 + dv**2) * (mu + mv)) / (2 * du * dv),
    57: lambda du, mu, dv, mv: 2
    + sqrt(2 * (mu**2 + mv**2) - 8 * (du**2 + dv**2) / (mu + mv) + 4),
    58: lambda du, mu, dv, mv: 2
    + sqrt(2 * (mu**2 + mu * mv + mv**2) - (du * mu + dv * mv) - 4 * (du + dv) + 4),
    59: lambda du, mu, dv, mv: (2 * (mu**2 + mu * mv + mv**2) - (du**2 + dv**2)) / (mu + mv),
    60: lambda du, mu, dv, mv: 2
    + sqrt(2 * (mu**2 + mu * mv + mv**2) - (du**2 + dv**2) - 4 * (du + dv) + 4),
    61: lambda du, mu, dv, mv: (2 * (mu**2 + mv**2))
    / (2 + sqrt(2 * ((du - 1) ** 2 + (dv - 1) ** 2))),
    62: lambda du, mu, dv, mv: 2
    + sqrt(mu**2 + 4 * mu * mv + mv**2 - 2 * du * dv - 4 * (du + dv) + 4),
    63: lambda du, mu, dv, mv: du + dv + mu + mv - 4 * (du * dv) / (mu + mv),
    64: lambda du, mu, dv, mv: (mu * mv * (du + dv)) / (du * dv),
    65: lambda du, mu, dv, mv: ((mu + mv) * (du * mu + dv * mv)) / (2 * mu * mv),
    66: lambda du, mu, dv, mv: (mu**2 + 4 * mu * mv + mv**2 - (du * mu + dv * mv)) / (du + dv),
    67: lambda du, mu, dv, mv: ((mu + mv) * (du * mu + dv * mv)) / (2 * du * dv),
    68: lambda du, mu, dv, mv: 2 + sqrt((mu - mv) ** 2 + 4 * du * dv - 4 * (mu + mv) + 4),
}


RESOLUTIONS = {}


def check(adjacency_matrix, expression_index):
    g = Graph(matrix(ZZ, adjacency_matrix))

    assert g.order() >= 2 and g.is_connected()
    # g.show()

    d_values = vector([g.degree(v) for v in g.vertices()])
    temp_sums = g.adjacency_matrix() * d_values
    m_values = vector([temp_sum / degree for temp_sum, degree in zip(temp_sums, d_values)])

    left_hand_side = max(g.spectrum(laplacian=True))

    if expression_index <= 32:
        right_hand_values = [
            LAPLACIAN_EXPRESSIONS[expression_index](d, m) for d, m in zip(d_values, m_values)
        ]
    else:
        right_hand_values = [
            LAPLACIAN_EXPRESSIONS[expression_index](d_values[u], m_values[u], d_values[v], m_values[v]) for u, v, _ in g.edges()
        ]

    result = left_hand_side - max(right_hand_values)
    if float(result) > 0.0001:
        if expression_index not in RESOLUTIONS:
            RESOLUTIONS[expression_index] = []
        
        RESOLUTIONS[expression_index].append(adjacency_matrix)


if __name__ == "__main__":
    with open("applications/auto_laplacian_solutions.pkl", "rb") as opened_file:
        all_solutions = pickle.load(opened_file)

    for adjacency_matrix in all_solutions:
        for expression_index in range(1, 69):
            check(adjacency_matrix, expression_index)

    print(RESOLUTIONS)