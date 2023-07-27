import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 隣接行列を定義
adj_matrix = np.asarray([[0, 2, 1, 1, 2],
                         [2, 0, 2, 1, 1],
                         [1, 2, 0, 2, 1],
                         [1, 1, 2, 0, 2],
                         [2, 1, 1, 2, 0]])

# グラフを作成
G = nx.from_numpy_array(adj_matrix)

# グラフの描画
weights = nx.get_edge_attributes(G, 'weight')
colors = ['red' if weights[e] == 1 else 'blue' for e in G.edges]

plt.figure(figsize=(6, 6))
pos = nx.circular_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000, edge_color=colors, width=2.0)
plt.show()
