import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 隣接行列の読み込み
def read_adj_matrix(file_path):
    with open(file_path, 'r') as file:
        adj_matrix = []
        for line in file:
            row = [int(x) for x in line.strip()]
            adj_matrix.append(row)
    return np.array(adj_matrix)

# 固有値と固有ベクトルの計算
def compute_eigenvectors(adj_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(adj_matrix)
    return eigenvalues, eigenvectors

# 固有ベクトルの可視化
def visualize_eigenvectors(adj_matrix, eigenvectors):
    G = nx.from_numpy_array(adj_matrix)
    
    plt.figure(figsize=(10, 8))
    
    # 2番目と3番目の固有ベクトルを使ってノードを配置
    pos = {i: [eigenvectors[i, 1], eigenvectors[i, 2]] for i in range(len(G.nodes))}
    
    # ノードの大きさを次数に応じて調整
    node_sizes = [100 * G.degree(node) for node in G.nodes]
    
    nx.draw(G, pos, node_size=node_sizes, with_labels=True, font_size=10)
    
    plt.title("Visualization of Eigenvectors")
    plt.show()

# メイン処理
if __name__ == "__main__":
    file_path = "adjcencyMatrix/R(4_8_57).txt"
    adj_matrix = read_adj_matrix(file_path)
    eigenvalues, eigenvectors = compute_eigenvectors(adj_matrix)
    visualize_eigenvectors(adj_matrix, eigenvectors)
