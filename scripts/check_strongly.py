import networkx as nx

def read_txt_to_graph(file_path):
    with open(file_path, 'r') as file:
        # 各行をリストとして読み込む
        lines = file.readlines()

    # 空の無向グラフを作成
    G = nx.Graph()

    # 行と列の数を取得
    num_rows = len(lines)
    num_cols = len(lines[0].strip())

    # 隣接行列からエッジを追加
    for i in range(num_rows):
        for j in range(num_cols):
            if lines[i][j] == '1':
                G.add_edge(str(i), str(j))

    return G

# テキストファイルのパスを指定してグラフを作成
file_path = 'generatedMatrix/circulantBlock/cir38B9B10/C1C2C3_asymmetric_1113035722.txt'
graph = read_txt_to_graph(file_path)

# グラフの情報を表示
print("Nodes:", graph.nodes())
print("Edges:", graph.edges())

# 他の操作や可視化などもここで行えます
print('strongly regular graph:', nx.is_strongly_regular(graph))