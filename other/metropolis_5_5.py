import numpy as np
import networkx as nx
import random
import math
import matplotlib.pyplot as plt

def ee_energy(n, B, de):
    e = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for c in range(k + 1, n):
                    for w in range(c + 1, n):
                        if abs(B[i, j] + B[j, k] + B[k, c] + B[c, w] + B[w, i] +
                               B[i, k] + B[j, c] + B[k, w] + B[i, c] + B[j, w]) == 10:
                            e += de
    return e

def acceptance(beta, delta_energy):
    return min(1, math.exp(-beta * delta_energy))

def ee_r_b(n, B):
    red = 0
    blue = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for c in range(k + 1, n):
                    for w in range(c + 1, n):
                        if (B[i, j] + B[j, k] + B[k, c] + B[c, w] + B[w, i] +
                                B[i, k] + B[j, c] + B[k, w] + B[i, c] + B[j, w]) == 10:
                            red += 1
                        elif (B[i, j] + B[j, k] + B[k, c] + B[c, w] + B[w, i] +
                              B[i, k] + B[j, c] + B[k, w] + B[i, c] + B[j, w]) == -10:
                            blue += 1
    return red, blue


np.random.seed(0)
n = 28  # Number of vertices
mc_steps = 10000  # Number of Monte Carlo steps
beta = 10  # Inverse temperature
de = 1  # Energy of monochromatic clique

# Random initial adjacency matrix (Zero trace & Symmetric)
B = np.random.randint(2, size=(n, n)) * 2 - 1
B = np.triu(B, 1)
B = B + B.T

first = ee_energy(n, B, de)  # Initial total energy

if first == 0:
    print("\nCase 01: success\n")
else:
    energy = np.zeros(mc_steps + 1)  # Initialize the energy array
    for i in range(mc_steps):
        energy_i = ee_energy(n, B, de)
        energy[i] = energy_i

        row = random.randint(0, n - 2)
        col = random.randint(min(row + 1, n - 1), max(row + 1, n - 1))
        B[row, col] = -B[row, col]
        B[col, row] = B[row, col]

        energy_i_plus_1 = ee_energy(n, B, de)
        energy[i + 1] = energy_i_plus_1

        if energy_i_plus_1 == 0:
            print("\nCase 02: success\n")
            break
        else:
            delta_energy = energy_i_plus_1 - energy_i
            if random.random() > acceptance(beta, delta_energy):
                B[row, col] = -B[row, col]
                B[col, row] = B[row, col]
                energy[i + 1] = energy_i

    if energy[mc_steps] != 0:
        print("\nCase 03: failure\n")
        print("- For %d vertices we obtain:" % n)
        print("- Final energy =", energy[mc_steps - 1])
    else:
        # Plot the graph
        plt.figure(2)
        G = nx.Graph(B)
        pos = nx.spring_layout(G, seed=42)  # Positioning nodes
        nx.draw_networkx(G, pos, with_labels=True, node_color='blue', edge_color=[G[u][v]['weight'] for u, v in G.edges],
                         cmap='jet', width=2)
        sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=-1, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, ax=plt.gca(), orientation='horizontal', label='Edge Weights')
        plt.axis('off')
        plt.show()

