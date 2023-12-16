import itertools
import numpy as np
from multiprocessing import Pool

def euclidean_distance(matrix1, matrix2):
    flattened_matrix1 = matrix1.flatten()
    flattened_matrix2 = matrix2.flatten()
    distance = np.linalg.norm(flattened_matrix1 - flattened_matrix2)
    return distance

def read_adjacency_matrix(file_path):
    adjacency_matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [int(x) for x in line.strip()]
            adjacency_matrix.append(row)
    return np.array(adjacency_matrix)

def check_combination(args):
    matrix, indices, target_matrix, inverted_matrix, bound_distance = args
    submatrix = matrix[np.ix_(indices, indices)]
    distance = euclidean_distance(submatrix, target_matrix)
    inverted_distance = euclidean_distance(submatrix, inverted_matrix)
    
    return distance <= bound_distance, inverted_distance <= bound_distance

def generate_combinations(total, size):
    for indices in itertools.combinations(range(total), size):
        yield indices

def process_combinations(args):
    file_path, target_matrix, inverted_matrix, bound_distance = args
    original_matrix = read_adjacency_matrix(file_path)
    target_rows = target_matrix.shape[0]

    count = 0
    inverted_count = 0

    # ジェネレータを生成
    indices_combinations = generate_combinations(len(original_matrix), target_rows)

    with Pool() as pool:
        # マルチプロセスでジェネレータを逐次的に処理
        results = list(pool.map(check_combination, [(original_matrix, indices, target_matrix, inverted_matrix, bound_distance) for indices in indices_combinations]))

    for result in results:
        count += result[0]
        inverted_count += result[1]

    print(f"distance {bound_distance}以下のカウント数", count)
    print(f"inverted distance {bound_distance}以下のカウント数", inverted_count)

if __name__ == "__main__":
    file_path = 'adjcencyMatrix/Paley(17).txt'
    target_path = 'adjcencyMatrix/B3.txt'
    target_matrix = read_adjacency_matrix(target_path)
    inverted_matrix = 1 - target_matrix
    np.fill_diagonal(inverted_matrix, 0)
    
    bound_distance = 0  # 適切な値に変更
    
    process_combinations((file_path, target_matrix, inverted_matrix, bound_distance))
    