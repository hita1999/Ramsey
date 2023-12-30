import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

# 固有ベクトルを2次元に削減する
def reduce_to_2d(eigenvectors):
    pca = PCA(n_components=2)
    return pca.fit_transform(eigenvectors)

# 固有ベクトルの可視化
def visualize_eigenvectors_2d(adj_matrix, eigenvectors_2d):
    G = nx.from_numpy_array(adj_matrix)
    
    plt.figure(figsize=(10, 8))
    
    # 固有ベクトルの2次元座標をノードの位置として設定
    pos = {i: eigenvectors_2d[i] for i in range(len(G.nodes))}
    
    # ノードの大きさを次数に応じて調整
    node_sizes = [100 * G.degree(node) for node in G.nodes]
    
    nx.draw(G, pos, node_size=node_sizes, with_labels=True, font_size=10)
    
    plt.title("Visualization of Eigenvectors (2D)")
    plt.show()

# メイン処理
if __name__ == "__main__":
    file_path = "adjcencyMatrix/R(4_8_57).txt"
    adj_matrix = read_adj_matrix(file_path)
    eigenvalues, eigenvectors = compute_eigenvectors(adj_matrix)
    
    # 固有ベクトルを2次元に削減
    eigenvectors_2d = reduce_to_2d(eigenvectors)
    print(eigenvectors_2d)
    
    visualize_eigenvectors_2d(adj_matrix, eigenvectors_2d)
