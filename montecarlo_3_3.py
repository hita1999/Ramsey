import random
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def cc_energy(n, B, de):
    e = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if abs(B[i, j] + B[j, k] + B[k, i]) == 3:
                    e += de
    return e


def acceptance(beta, delta_energy):
    return min(1, np.exp(-beta * delta_energy))


def cc_r_b(n, B):
    red = 0
    blue = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if B[i, j] + B[j, k] + B[k, i] == 3:
                    red += 1
                elif B[i, j] + B[j, k] + B[k, i] == -3:
                    blue += 1
    return red, blue


def metropolis_algorithm(n, mc_steps, beta, de):
    # Set random seed for reproducibility (Optional)
    np.random.seed(42)
    random.seed(42)

    # # edges and nchoosek in Python
    edges = n * (n - 1) // 2

    # Preallocate energy array
    energy = np.empty(mc_steps)

    # Random initial Adjacency matrix (Zero trace & Symmetric)
    B = np.random.randint(2, size=(n, n)) * 2 - 1
    B = np.triu(B, 1)
    B = B + B.T

    first = cc_energy(n, B, de)  # Initial total energy

    # If initial energy is zero we terminate
    if first == 0:
        print("\n\n case 01: success \n\n")  # first scenario
        return

    # Otherwise, we proceed with the simulation
    for i in range(mc_steps):
        # Calculate energy of old state
        energy[i] = cc_energy(n, B, de)

        # We search in the upper triangular part of the adjacency matrix
        row = random.randint(0, n - 2)  # choose a random row
        col = random.randint(row + 1, n - 1)  # same for column
        B[row, col] = B[row, col] * (-1)  # flip random matrix entry
        B[col, row] = B[row, col]  # symmetrize adjacency matrix

        # Calculate energy of NEW state
        energy[i + 1] = cc_energy(n, B, de)

        # if new energy is zero, terminate
        if energy[i + 1] == 0:
            print("\n\n case 02: success \n\n")  # second scenario
            break
        # else perturb thermally the system
        else:
            # energy difference (new - old)
            delta_energy = energy[i + 1] - energy[i]

            # Boltzmann factor acceptance rule
            if random.random() > acceptance(beta, delta_energy):
                B[row, col] = B[row, col] * (-1)
                B[col, row] = B[row, col]
                energy[i + 1] = energy[i]

    [red, blue] = cc_r_b(n, B)  # count red & blue cliques
    print("\n\n case 03: failure \n\n")  # third scenario
    print("- For %d vertices we obtain:\n" % n)
    print("- final energy = %d\n\n" % energy[i + 1])

    # metropolis_algorithm 関数の最後に以下を追加
    print("\n--- Result ---")
    print("- For %d vertices we obtain:" % n)
    print("- final energy = %d\n" % energy[-1])
    print("Red cliques: %d" % red)
    print("Blue cliques: %d" % blue)

    return energy, B


# Input parameters
n = 5  # # vertices
mc_steps = 1000  # # Monte Carlo steps
beta = 10  # Inverse temperature
de = 1  # Energy of monochromatic clique

energy, B = metropolis_algorithm(n, mc_steps, beta, de)

# GRAPH
# Input graph weights: Bonds matrix will give us the weights
G = nx.Graph(B)
pos = nx.circular_layout(G)

# Set edge weights based on B matrix values
edge_weights = [B[i, j] for i, j in G.edges()]

# Normalize edge weights for better visualization
edge_norm = plt.Normalize(min(edge_weights), max(edge_weights))

# Draw the graph with edge colors and colorbar
plt.figure(2)
nx.draw(G, pos, node_color='k', edge_color=edge_weights, edge_cmap=plt.cm.cool,
        with_labels=True, width=2, edge_vmin=min(edge_weights), edge_vmax=max(edge_weights))
c = plt.colorbar(cm.ScalarMappable(norm=edge_norm, cmap=plt.cm.cool))
c.set_label("Edge Weight")
plt.show()
