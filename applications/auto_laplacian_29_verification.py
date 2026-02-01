import math

adj = [
    [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
]

g = Graph(matrix(ZZ, adj))

print(g.order(), g.size(), g.is_connected())
g.show()

dseq = vector([g.degree(v) for v in g.vertices()])
sum_nd = g.adjacency_matrix() * dseq
average_nd = vector([i / j for i, j in zip(sum_nd, dseq)])

thingy = [math.sqrt(m ** 2 + (3 * m ** 3) / d) for m, d in zip(average_nd, dseq)]
max_thingy = max(thingy)
print(max_thingy)

mu = g.spectrum(laplacian=True)[0]
result = mu - max_thingy
print(result)