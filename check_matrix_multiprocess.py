import itertools
import numpy as np
from multiprocessing import Pool, cpu_count
from math import comb
from tqdm import tqdm

def read_adjacency_matrix(file_path):
    adjacency_matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [int(x) for x in line.strip()]
            adjacency_matrix.append(row)
    return np.array(adjacency_matrix)

def check_combination(matrix, indices, target_matrix):
    submatrix = matrix[np.ix_(indices, indices)]
    return np.array_equal(submatrix, target_matrix)

def generate_matrix_candidates(total, size):
    return itertools.combinations(range(total), size)

def generate_matrix_candidates_with_progress(total, size):
    total_combinations = comb(total, size)
    chunk_size = total_combinations // cpu_count()
    start = 0
    end = chunk_size
    for idx, indices in enumerate(tqdm(itertools.islice(generate_matrix_candidates(total, size), start, end), total=chunk_size, desc="Checking Matrix Candidates")):
        yield indices
        start = end
        end += chunk_size

def find_satisfying_graph_parallel(args):
    matrix, target_matrix, target_rows = args

    for indices in generate_matrix_candidates_with_progress(len(matrix), target_rows):
        if check_combination(matrix, indices, target_matrix):
            return "target_matrix"

    return False

def main():
    file_path = 'adjcencyMatrix/T8.txt'
    original_matrix = read_adjacency_matrix(file_path)

    first_target_path = 'targetAdjcencyMatrix/B7.txt'
    first_target_matrix = read_adjacency_matrix(first_target_path)
    first_target_rows = first_target_matrix.shape[0]

    second_target_path = 'targetAdjcencyMatrix/B6.txt'
    second_target_matrix = read_adjacency_matrix(second_target_path)
    second_target_rows = second_target_matrix.shape[0]

    args_list = [(original_matrix, first_target_matrix, first_target_rows),
                 (original_matrix, second_target_matrix, second_target_rows)]

    with Pool(processes=cpu_count()) as pool:
        results = list(pool.imap_unordered(find_satisfying_graph_parallel, args_list))

    for idx, satisfying_graph_type in enumerate(results):
        if satisfying_graph_type == "target_matrix":
            print(f"{file_path}には{first_target_path if idx == 0 else second_target_path}が見つかりました")
            print(f"{idx + 1}_graphにて条件を満たすグラフが見つかりました ({'first' if idx == 0 else 'second'}_target_matrix)")
        else:
            print(f"{file_path}には{first_target_path if idx == 0 else second_target_path}は見つかりませんでした")

if __name__ == "__main__":
    main()
