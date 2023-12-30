import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def modify_matrix(matrix):
    modified_matrix = matrix.copy()
    rows, cols = matrix.shape

    # 行列の対角線以外の0を2に変更する
    for i in range(rows):
        for j in range(cols):
            if i != j and modified_matrix[i, j] == 0:
                modified_matrix[i, j] = 2

    return modified_matrix

def draw_subgraph(file_path, selected_rows, selected_cols):
    adjacency_matrix = read_adjacency_matrix(file_path)

    # 対象部分の隣接行列を取得
    selected_matrix = np.array([row[selected_cols] for row in np.array(adjacency_matrix)[selected_rows]])
    print(selected_matrix)
    
    modified_matrix = modify_matrix(selected_matrix)

    # グラフを作成
    G = nx.from_numpy_array(modified_matrix)

    # グラフの描画
    weights = nx.get_edge_attributes(G, 'weight')
    colors = ['red' if weights[e] == 1 else 'blue' for e in G.edges]

    plt.figure(figsize=(10, 10))
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000, edge_color=colors, width=2.0)
    plt.show()
    

def read_adjacency_matrix(file_path):
    adjacency_matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [int(x) for x in line.strip()]
            adjacency_matrix.append(row)
    return adjacency_matrix

file_path = 'adjcencyMatrix/Paley/Paley37_2.txt'
adjacency_matrix = read_adjacency_matrix(file_path)

#selected_rows = [i for i in range(len(adjacency_matrix))]
selected_rows = list((0, 3, 4, 31))

# 関数を呼び出して描画
draw_subgraph(file_path, selected_rows, selected_rows)
