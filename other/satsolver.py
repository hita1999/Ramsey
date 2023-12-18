#Quote "https://medium.com/@pravin.bezwada/fun-with-ramsey-numbers-dcc477f138a8"

import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import itertools as it

from pysat.formula import CNF
from pysat.solvers import Solver
from tqdm import tqdm

warnings.filterwarnings("ignore")


class RAGS:
    def __init__(self, N, C):
        self.N = N
        self.C = C
        self.R = None
        self.mask = None
        self.done = False

    def build(self, mask):
        self.mask = mask + '0'
        n = self.N
        c = self.C
        S = np.zeros((n, n))
        mask = np.fromstring(" ".join(mask), dtype=int, sep=' ')
        for i in range(0, len(mask)):
            S[-1-i, :len(mask) - i] = mask[-len(mask)+i:]

        self.R = S + S.T
        G1 = nx.from_numpy_array(self.R)
        G2 = nx.complement(G1)
        red_clique = nx.graph_clique_number(G1)
        blue_clique = nx.graph_clique_number(G2)
        if red_clique >= c or blue_clique >= c:
            self.done = False
        else:
            self.done = True

    def plot(self):
        self.validate()
        nx.draw_circular(nx.from_numpy_array(self.R))

    def plot_complete(self):
        self.validate()
        nx.draw_circular(nx.complement(nx.from_numpy_array(self.R)))

    def plot_twocolor(self):
        self.validate()
        G = self.R
        G[G == 0] = 2
        np.fill_diagonal(G, 0)
        I = nx.from_numpy_array(G)
        weights = nx.get_edge_attributes(I, 'weight')
        edge_colors = ['red' if weights[e] == 1 else 'blue' for e in I.edges]

        # グラフの描画
        plt.figure(figsize=(6, 6))
        pos = nx.circular_layout(I)
        nx.draw(I, pos, with_labels=True, node_color='lightblue',
                node_size=1000, edge_color=edge_colors, width=2.0)
        plt.show()

    def print_results(self):
        self.validate()
        R = self.R
        G1 = nx.from_numpy_array(R)
        G2 = nx.complement(G1)
        print("Nodes: ", R.shape[0], "Complete: ", (len(G1.edges()) + len(G2.edges)) == R.shape[0] * (R.shape[0] - 1) // 2,
              "Cliques :", (nx.graph_clique_number(G1), nx.graph_clique_number(G2)))

    def validate(self):
        if self.R is None:
            raise ValueError("Graph not built")


def Adj(NN, graph):
    combins = it.combinations(graph, 2)
    nodes = []
    for comb in combins:
        comb = list(comb)
        (x, y) = (comb[0], comb[1]) if comb[0] > comb[1] else (
            comb[1], comb[0])
        v = NN[x, y]
        nodes.append(v)
    return set(nodes)


def addKey(combin, NN, keys):
    key = Adj(NN, combin)
    key = list(key)
    key.sort()
    keys.add(tuple(key))


def GenerateRegularMask(N, C):
    keys = set()
    NN = np.zeros((N, N)).astype(int)
    mask = list(range(1, N))

    for i in range(1, N):
        NN[i, :i] = mask[-i:]

    combins = it.combinations(range(0, N), C)

    solver = Solver()

    for combin in tqdm(combins):
        addKey(combin, NN, keys)

    for key in keys:
        solver.add_clause([int(num) for num in key])
        solver.add_clause([-int(num) for num in key])

    all = range(1, N)
    solver.add_clause([int(num) for num in all])

    if not solver.solve():
        print("Algorithm failed")
        return None

    result = solver.get_model()
    mask = "".join(["1" if i > 0 else "0" for i in result])
    return mask


def GenerateMask(N, C):
    keys = set()
    NN = np.zeros((N, N)).astype(int)
    mask = list(range(1, N-1))

    for i in range(2, N):
        NN[i, :i-1] = mask[-i+1:]

    counter = np.max(NN) + 1

    for j in range(1, N):
        NN[j, j-1] = counter
        counter += 1

    combins = it.combinations(range(0, N), C)

    solver = Solver()

    from tqdm import tqdm

    for combin in tqdm(combins):
        addKey(combin, NN, keys)

    for key in keys:
        solver.add_clause([int(num) for num in key])
        solver.add_clause([-int(num) for num in key])

    all = range(1, N)
    solver.add_clause([int(num) for num in all])

    if not solver.solve():
        print("Algorithm failed")
        return None

    result = solver.get_model()
    mask = "".join(["1" if i > 0 else "0" for i in result])
    return mask, NN


# mask, NN = GenerateMask(42, 5)
# print(mask)

# K = 42

# R = np.zeros((K, K))

# for i in range(0, K):
#   for j in range(0, i):
#     R[i, j] = mask[NN[i, j]-1]
# R = R + R.T

# rags = RAGS(42, 6)
# rags.R = R
# rags.print_results()
# rags.plot_twocolor()

# mask2 = GenerateRegularMask(40, 5)
# rags2 = RAGS(40, 5)
# rags2.build(mask2)
# rags2.print_results()
# rags2.plot_twocolor()

mask3 = GenerateRegularMask(6, 3)
rags3 = RAGS(6, 3)
rags3.build(mask3)
rags3.print_results()
rags3.plot_twocolor()