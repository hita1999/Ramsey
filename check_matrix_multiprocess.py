import itertools
from math import comb
import numpy as np
from multiprocessing import Pool

def read_adjacency_matrix(file_path):
    adjacency_matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [int(x) for x in line.strip()]
            adjacency_matrix.append(row)
    return np.array(adjacency_matrix)

def check_combination(matrix, indices, target_matrix):
    return np.array_equal(matrix[np.ix_(indices, indices)], target_matrix) or np.array_equal(matrix[np.ix_(indices, indices)], 1 - target_matrix)

def generate_combinations(total, size):
    return itertools.combinations(range(total), size)

def save_matrix_to_txt(matrix, file_path):
    np.savetxt(file_path, matrix, fmt='%d', delimiter='')

def find_satisfying_graph_parallel(args):
    matrix, target_matrix, inverted_matrix, target_rows = args

    for indices in generate_combinations(len(matrix), target_rows):
        if check_combination(matrix, indices, target_matrix):
            return "target_matrix"
        elif check_combination(matrix, indices, inverted_matrix):
            return "inverted_matrix"

    return False

def main():
    file_path = 'adjcencyMatrix/T9.txt'
    original_matrix = read_adjacency_matrix(file_path)

    first_target_path = 'targetAdjcencyMatrix/B6.txt'
    first_target_matrix = read_adjacency_matrix(first_target_path)
    first_inverted_matrix = 1 - first_target_matrix
    np.fill_diagonal(first_inverted_matrix, 0)
    first_target_rows = first_target_matrix.shape[0]

    second_target_path = 'targetAdjcencyMatrix/B4.txt'
    second_target_matrix = read_adjacency_matrix(second_target_path)
    second_inverted_matrix = 1 - second_target_matrix
    np.fill_diagonal(second_inverted_matrix, 0)
    second_target_rows = second_target_matrix.shape[0]

    args_list = [(original_matrix, first_target_matrix, first_inverted_matrix, first_target_rows),
                 (original_matrix, second_target_matrix, second_inverted_matrix, second_target_rows)]

    with Pool() as pool:
        results = list(pool.map(find_satisfying_graph_parallel, args_list))

    for idx, satisfying_graph_type in enumerate(results):
        if satisfying_graph_type == "target_matrix":
            print(f"{file_path}には{first_target_path if idx == 0 else second_target_path}が見つかりました")
            print(f"{idx + 1}_graphにて条件を満たすグラフが見つかりました ({'first' if idx == 0 else 'second'}_target_matrix)")
        elif satisfying_graph_type == "inverted_matrix":
            print(f"{idx + 1}_graphにて条件を満たすグラフが見つかりました ({'first' if idx == 0 else 'second'}_inverted_matrix)")
        else:
            print(f"{file_path}には{first_target_path if idx == 0 else second_target_path}は見つかりませんでした")

if __name__ == "__main__":
    main()
