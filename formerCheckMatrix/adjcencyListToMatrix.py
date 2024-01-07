import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def read_adj_list(file_path):
    adj_list = {}
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            neighbors = list(map(int, line.strip().split()))
            adj_list[i] = neighbors
    return adj_list

def adj_list_to_adj_matrix(adj_list, num_nodes):
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            adj_matrix[node][neighbor] = 1
    return adj_matrix


def save_adj_matrix_to_txt(adj_matrix, file_path):
    with open(file_path, 'w') as file:
        for row in adj_matrix:
            file.write(''.join(map(str, row)) + '\n')

file_path = 'adjcencyMatrix/R(4_6_35)_list.txt'
adj_list = read_adj_list(file_path)
num_nodes = len(adj_list)
adj_matrix = adj_list_to_adj_matrix(adj_list, num_nodes)
print(adj_matrix)

output_file_path = 'adjcencyMatrix/R(4_6_35).txt'
save_adj_matrix_to_txt(adj_matrix, output_file_path)

# 行列の対角線以外の0を2に変更する
for i in range(len(adj_matrix)):
    for j in range(len(adj_matrix[i])):
        if i != j and adj_matrix[i][j] == 0:
            adj_matrix[i][j] = 2

# グラフを作成
G = nx.from_numpy_array(np.array(adj_matrix))

# グラフの描画
weights = nx.get_edge_attributes(G, 'weight')
colors = ['red' if weights[e] == 1 else 'blue' for e in G.edges]

plt.figure(figsize=(10, 10))
pos = nx.circular_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000, edge_color=colors, width=2.0)
plt.show()
