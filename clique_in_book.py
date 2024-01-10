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
    # グラフを作成
    G = nx.Graph(adjacency_matrix)

    # ノードをクリークに結びつける
    cliques = list(nx.find_cliques(G))
    print(cliques)

    # 指定されたサイズのクリークを見つける
    target_cliques = [clique for clique in cliques if len(clique) == clique_size]

    return target_cliques

def flip_matrix(matrix):
    new_matrix = 1 - matrix
    np.fill_diagonal(new_matrix, 0)
    return new_matrix

# 例として、適当な隣接行列を作成
# この行列は完全グラフ（5つのノードからなる）を持っています
matrix = read_adjacency_matrix('adjcencyMatrix/K9_9-I.txt')

matrix = flip_matrix(matrix)
print(matrix)

# クリークのサイズを指定
clique_size = 3

# クリークを検索
result = find_clique(matrix, clique_size)

print(f"Cliques of size {clique_size}: {result}")
