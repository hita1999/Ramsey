import itertools
import numpy as np


def euclidean_distance(matrix1, matrix2):
    flattend_matrix1 = matrix1.flatten()
    flattend_matrix2 = matrix2.flatten()
    distance = np.linalg.norm(flattend_matrix1 - flattend_matrix2)
    return distance

def read_adjacency_matrix(file_path):
    adjacency_matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [int(x) for x in line.strip()]
            adjacency_matrix.append(row)
    return np.array(adjacency_matrix)


def check_combination(matrix, indices, target_matrix):
    submatrix = matrix[np.ix_(indices, indices)]
    return np.array_equal(submatrix, target_matrix) or np.array_equal(submatrix, 1 - target_matrix)


def generate_combinations(total, size):
    for indices in itertools.combinations(range(total), size):
        yield indices

file_path = 'R(B3_B6_18).txt'
original_matrix = read_adjacency_matrix(file_path)

target_path = 'adjcencyMatrix/B3.txt'
target_matrix = read_adjacency_matrix(target_path)
    

print(target_matrix)

inverted_matrix = 1 - target_matrix
np.fill_diagonal(inverted_matrix, 0)
print(inverted_matrix)

# インデックスの組み合わせを生成
target_rows = target_matrix.shape[0]


count = 0
inverted_count = 0
bound_distance = 0
# 生成された組み合わせを順にチェック
indices_combinations = generate_combinations(len(original_matrix), target_rows)  # ジェネレータを生成
for idx, indices in enumerate(indices_combinations):
    if idx % 1000 == 0:
        print(f"組み合わせ {idx}番目")
    
    submatrix = original_matrix[np.ix_(indices, indices)]
    distance = euclidean_distance(submatrix, target_matrix)
    inverted_distance = euclidean_distance(submatrix, inverted_matrix)
    
    if distance <= bound_distance:
        count += 1
        print("ユークリッド距離:", distance)
        print("選ばれたインデックス:", list(indices))
        print("選ばれた行と列だけを取り出した部分行列:")
        print(submatrix)
        
    if inverted_distance <= bound_distance:
        inverted_count += 1
        print("ユークリッド距離:", inverted_distance)
        print("選ばれたインデックス:", list(indices))
        print("選ばれた反転行列:")
        print(submatrix)
    
print(f"distance {bound_distance}以下のカウント数", count)
print(f"inverted distance {bound_distance}以下のカウント数", inverted_count)