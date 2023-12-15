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

file_path = 'adjcencyMatrix/pappus18.txt'
original_matrix = read_adjacency_matrix(file_path)

target_path = 'adjcencyMatrix/B6.txt'
target_matrix = read_adjacency_matrix(target_path)
    

print(target_matrix)

inverted_matrix = 1 - target_matrix
np.fill_diagonal(inverted_matrix, 0)
print(inverted_matrix)

# インデックスの組み合わせを生成
target_rows = 8  # 適切なサイズに変更
# indices_combinations = generate_combinations(len(original_matrix), target_rows)
# total_combinations = sum(1 for _ in indices_combinations)
# print("\n総組み合わせ数:", total_combinations)  # 生成される組み合わせの数を確認

# 生成された組み合わせを順にチェック
indices_combinations = generate_combinations(len(original_matrix), target_rows)  # ジェネレータを再生成
for idx, indices in enumerate(indices_combinations):
    if idx % 100 == 0:
        print(f"組み合わせ {idx}番目")
    
    submatrix = original_matrix[np.ix_(indices, indices)]
    distance = euclidean_distance(submatrix, target_matrix)
    inverted_distance = euclidean_distance(submatrix, inverted_matrix)
    
    if distance < 4:
        print("ユークリッド距離:", distance)
        print("選ばれたインデックス:", indices)
        print("選ばれた行と列だけを取り出した部分行列:")
        print(submatrix)
        
    if inverted_distance < 4:
        print("ユークリッド距離:", inverted_distance)
        print("選ばれたインデックス:", indices)
        print("選ばれた反転行列:")
        print(submatrix)
    # if check_combination(original_matrix, indices, target_matrix):
    #     print("一致する組み合わせが見つかりました！")
    #     print("選ばれたインデックス:", indices)
    #     print("選ばれた行と列だけを取り出した部分行列:")
    #     print(original_matrix[np.ix_(indices, indices)])
    #     break
    # elif check_combination(original_matrix, indices, inverted_matrix):
    #     print("一致する組み合わせが見つかりました！")
    #     print("選ばれたインデックス:", indices)
    #     print("選ばれた反転行列:")
    #     print(original_matrix[np.ix_(indices, indices)])
    #     break
else:
    print("一致する組み合わせは見つかりませんでした。")