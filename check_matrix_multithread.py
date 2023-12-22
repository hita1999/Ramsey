import itertools
from math import comb
import numpy as np
from multiprocessing import Pool

from tqdm import tqdm

def read_adjacency_matrix(file_path):
    adjacency_matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [int(x) for x in line.strip()]
            adjacency_matrix.append(row)
    return np.array(adjacency_matrix)

def generate_matrix_candidates(total, size):
    for indices in itertools.combinations(range(total), size):
        yield indices

def check_combination(matrix, indices, target_matrix):
    submatrix = matrix[np.ix_(indices, indices)]
    return np.array_equal(submatrix, target_matrix)

def generate_matrix_candidates_with_progress(total, size):
    total_combinations = comb(total, size)
    for idx, indices in enumerate(tqdm(itertools.combinations(range(total), size), total=total_combinations, desc="Checking Matrix Candidates")):
        yield indices

def find_satisfying_graph_parallel(args):
    matrix, target_matrix, target_rows = args

    for indices in generate_matrix_candidates_with_progress(len(matrix), target_rows):
        if check_combination(matrix, indices, target_matrix):
            return "target_matrix"

    return False

def main():
    file_path = 'adjcencyMatrix/T7.txt'
    original_matrix = read_adjacency_matrix(file_path)

    first_target_path = 'targetAdjcencyMatrix/B7.txt'
    first_target_matrix = read_adjacency_matrix(first_target_path)
    first_target_rows = first_target_matrix.shape[0]

    second_target_path = 'targetAdjcencyMatrix/B6.txt'
    second_target_matrix = read_adjacency_matrix(second_target_path)
    second_target_rows = second_target_matrix.shape[0]

    args_list = [(original_matrix, first_target_matrix, first_target_rows),
                 (original_matrix, second_target_matrix, second_target_rows)]

    with Pool(processes=8) as pool:
        results = list(pool.imap_unordered(find_satisfying_graph_parallel, args_list))

    for idx, satisfying_graph_type in enumerate(results):
        if satisfying_graph_type == "target_matrix":
            print(f"{file_path}には{first_target_path if idx == 0 else second_target_path}が見つかりました")
            print(f"{idx + 1}_graphにて条件を満たすグラフが見つかりました ({'first' if idx == 0 else 'second'}_target_matrix)")
        else:
            print(f"{file_path}には{first_target_path if idx == 0 else second_target_path}は見つかりませんでした")

if __name__ == "__main__":
    main()