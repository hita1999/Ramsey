import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm

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
    np.random.seed(42)
    random.seed(42)

    edges = n * (n - 1) // 2
    energy = np.empty(mc_steps)

    B = np.random.randint(2, size=(n, n)) * 2 - 1
    np.fill_diagonal(B, 0)  # Set diagonal elements to zero
    B = np.triu(B, 1)
    B = B + B.T

    first = cc_energy(n, B, de)

    if first == 0:
        print("\n\n case 01: success \n\n")
        return

    for i in range(mc_steps - 1):  # Update loop to range(mc_steps - 1)
        energy[i] = cc_energy(n, B, de)

        row = random.randint(0, n - 2)
        col = random.randint(row + 1, n - 1)
        B[row, col] = B[row, col] * (-1)
        B[col, row] = B[row, col]

        energy[i + 1] = cc_energy(n, B, de)

        if energy[i + 1] == 0:
            print("\n\n case 02: success \n\n")
            break
        else:
            delta_energy = energy[i + 1] - energy[i]

            if random.random() > acceptance(beta, delta_energy):
                B[row, col] = B[row, col] * (-1)
                B[col, row] = B[row, col]
                energy[i + 1] = energy[i]

    print("\n\n case 03: failure \n\n")
    print("- For %d vertices we obtain:\n" % n)
    print("- final energy = %d\n\n" % energy[-1])

    [red, blue] = cc_r_b(n, B)
    print("\n--- Result ---")
    print("- For %d vertices we obtain:" % n)
    print("- final energy = %d\n" % energy[-1])
    print("Red cliques: %d" % red)
    print("Blue cliques: %d" % blue)

    return energy, B

# Input parameters
n = 18  # # vertices
mc_steps = 10000  # # Monte Carlo steps
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
nx.draw_networkx(G, pos, node_color='y', edge_color=edge_weights, edge_cmap=cm.jet,
                 with_labels=True, width=2, edge_vmin=min(edge_weights), edge_vmax=max(edge_weights))
c = plt.colorbar(cm.ScalarMappable(norm=edge_norm, cmap=cm.jet))
c.set_label("Edge Weight")
plt.axis('off')
plt.box(False)
plt.show()
