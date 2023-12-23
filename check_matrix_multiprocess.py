import itertools
from math import comb
import os
import numpy as np
from multiprocessing import Pool
import time

from tqdm import tqdm

def read_adjacency_matrix(file_path):
    adjacency_matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [int(x) for x in line.strip()]
            adjacency_matrix.append(row)
    return np.array(adjacency_matrix)

def generate_matrix_candidates(total, size):
    return itertools.combinations(range(total), size)

def check_combination(matrix, indices, target_matrix):
    submatrix = matrix[np.ix_(indices, indices)]
    return np.array_equal(submatrix, target_matrix)

def find_satisfying_graph(args):
    matrix, target_matrix, target_rows, start, end = args

    found = False
    with tqdm(total=end - start) as t:
        for idx, indices in enumerate(itertools.islice(itertools.combinations(range(len(matrix)), target_rows), start, end)):
            t.update(1)
            if check_combination(matrix, indices, target_matrix):
                found = True
                break

    return found

def main():
    file_path = 'adjcencyMatrix/T9.txt'
    original_matrix = read_adjacency_matrix(file_path)

    first_target_path = 'targetAdjcencyMatrix/B6.txt'
    first_target_matrix = read_adjacency_matrix(first_target_path)
    first_target_rows = first_target_matrix.shape[0]

    total_combinations = comb(len(original_matrix), first_target_rows)
    
    # コア数 分割
    num_processes = os.cpu_count()
    chunk_size = total_combinations // num_processes
    
    pool = Pool(processes=num_processes)

    # 各プロセスに適切な範囲を割り当てる
    args_list = [
        (original_matrix, first_target_matrix, first_target_rows, start, start + chunk_size)
        for start in range(0, total_combinations, chunk_size)
    ]

    start_time = time.time()

    # Manually update the progress
    results = list(pool.imap_unordered(find_satisfying_graph, args_list))

    elapsed_time = time.time() - start_time

    # Check results
    if any(results):
        print(f"{file_path}には{first_target_path}が見つかりました")
        print("条件を満たすグラフが見つかりました (first_target_matrix)")
    else:
        print(f"{file_path}には{first_target_path}は見つかりませんでした")

    print(f"\nTotal elapsed time: {elapsed_time:.2f} seconds")
    print(f"Iteration speed: {total_combinations / elapsed_time:.2f} iterations per second")

if __name__ == "__main__":
    main()
