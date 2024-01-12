import networkx as nx
import numpy as np

def read_adjacency_matrix(file_path):
    adjacency_matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [int(x) for x in line.strip()]
            adjacency_matrix.append(row)
    return np.array(adjacency_matrix)

def find_clique(adjacency_matrix, clique_size):
    G = nx.Graph(adjacency_matrix)

    cliques = list(nx.find_cliques(G))
    print(cliques)

    target_cliques = [clique for clique in cliques if len(clique) == clique_size]

    return target_cliques

def flip_matrix(matrix):
    new_matrix = 1 - matrix
    np.fill_diagonal(new_matrix, 0)
    return new_matrix

matrix = read_adjacency_matrix('generatedMatrix/L2(4)_add_17476.txt')

matrix = flip_matrix(matrix)
print(matrix)

clique_size = 4

result = find_clique(matrix, clique_size)

print(f"Cliques of size {clique_size}: {result}")
