import numpy as np

def generate_adjacency_matrix_with_50_ones(size):
    adjacency_matrix = np.zeros((size, size), dtype=int)

    # 上三角行列に1を50個配置
    indices = np.triu_indices(size, 1)
    random_indices = np.random.choice(len(indices[0]), 50, replace=False)
    adjacency_matrix[indices[0][random_indices], indices[1][random_indices]] = 1
    
    # 対角成分を0に設定
    np.fill_diagonal(adjacency_matrix, 0)
    
    # 対称な行列にする
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T
    
    return adjacency_matrix

def generate_adjacency_matrix(size):
    # 対称な行列を作成
    symmetrical_matrix = np.triu(np.random.randint(2, size=(size, size)), 1)
    adjacency_matrix = symmetrical_matrix + symmetrical_matrix.T
    
    # 対角成分を0に設定
    np.fill_diagonal(adjacency_matrix, 0)
    
    return adjacency_matrix

def save_adjacency_matrix(matrix, file_path):
    np.savetxt(file_path, matrix, fmt='%d', delimiter='')

# ランダムな行列をいくつか生成してファイルに保存
num_matrices = 5  # 作成する行列の数
matrix_size = 18  # 行列のサイズ

for i in range(1, num_matrices + 1):
    random_adjacency_matrix = generate_adjacency_matrix(matrix_size)
    file_path = f'random18VertexGraph/random_adjacency_matrix_{i}.txt'
    save_adjacency_matrix(random_adjacency_matrix, file_path)
    print(f'Saved {file_path}')
    
    random_adjacency_matrix = generate_adjacency_matrix_with_50_ones(18)
    file_path = 'random18VertexGraph/random_adjacency_matrix_with_50_ones.txt'
    save_adjacency_matrix(random_adjacency_matrix, file_path)
    print(f'Saved {file_path}')