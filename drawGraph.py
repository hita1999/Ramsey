import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def read_adjacency_matrix(file_path):
    adjacency_matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [int(x) for x in line.strip()]
            adjacency_matrix.append(row)
    return adjacency_matrix

file_path = 'adjcencyMatrix/16-1.txt'
adjacency_matrix = read_adjacency_matrix(file_path)

# 行列の対角線以外の0を2に変更する
for i in range(len(adjacency_matrix)):
    for j in range(len(adjacency_matrix[i])):
        if i != j and adjacency_matrix[i][j] == 0:
            adjacency_matrix[i][j] = 2


# グラフを作成
G = nx.from_numpy_array(np.array(adjacency_matrix))

# グラフの描画
weights = nx.get_edge_attributes(G, 'weight')
colors = ['red' if weights[e] == 1 else 'blue' for e in G.edges]
#colors = ['red' if weights[e] == 1 else 'white' for e in G.edges]

plt.figure(figsize=(7, 7))
pos = nx.circular_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000, edge_color=colors, width=2.0)
plt.show()
