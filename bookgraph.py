def save_adj_matrix_to_txt(adj_matrix, file_path):
    with open(file_path, 'w') as file:
        for row in adj_matrix:
            file.write(''.join(map(str, row)) + '\n')
            
            
# 隣接行列のサイズを定義
n = 16
matrix_size = n + 2

# 隣接行列を初期化
adjacency_matrix = [[0] * matrix_size for _ in range(matrix_size)]

# K2の頂点同士は隣接している
for i in range(2):
    for j in range(2):
        if i != j:
            adjacency_matrix[i][j] = 1

# ¯Knの頂点同士は隣接していない
for i in range(2, matrix_size):
    for j in range(2, matrix_size):
        adjacency_matrix[i][j] = 0

# K2の頂点と¯Knの頂点は隣接している
for i in range(2):
    for j in range(2, matrix_size):
        adjacency_matrix[i][j] = 1
        adjacency_matrix[j][i] = 1  # 逆向きも隣接している

# 隣接行列を表示
for row in adjacency_matrix:
    print(row)


save_adj_matrix_to_txt(adjacency_matrix, 'adjcencyMatrix/R(B_B_6).txt')