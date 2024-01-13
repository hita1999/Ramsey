import itertools
from math import comb
import time
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def read_adjacency_matrix(file_path):
    adjacency_matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [int(x) for x in line.strip()]
            adjacency_matrix.append(row)
    return np.array(adjacency_matrix)

def multi_generate_combinations(total, size, start, end):
    return itertools.islice(itertools.combinations(range(total), size), start, end)

def search_book(matrix, page, spine, condition_func):

    if condition_func(matrix, page, spine):
        return True

def condition_function_1(matrix, page, spine):
    return matrix[np.ix_(spine, spine)].sum() == 2 and matrix[np.ix_(spine, page)].sum() == len(page) * 2

def condition_function_2(matrix, page, spine):
    return matrix[np.ix_(spine, spine)].sum() == 0 and matrix[np.ix_(spine, page)].sum() == 0

def find_satisfying_graph_partial(args):
    original_matrix, target_size, condition_func, start, end = args
    progress_bar = tqdm(total=(end - start), desc="Finding satisfying graph")
    
    for page_indices in multi_generate_combinations(len(original_matrix), target_size, start, end):
        remaining_indices = set(range(len(original_matrix))) - set(page_indices)

        for spine_indices in itertools.combinations(remaining_indices, 2):
            progress_bar.update(1)
            if set(spine_indices).isdisjoint(set(page_indices)):
                if search_book(original_matrix, page_indices, spine_indices, condition_func):
                    progress_bar.close()
                    return 1, spine_indices, page_indices
    
    progress_bar.close()
    return 0, 0, 0

def find_satisfying_graph_parallel(original_matrix, target_size, condition_func):
    original_matrix_number_of_rows = original_matrix.shape[0]
    total_combinations = comb(original_matrix_number_of_rows, target_size) * comb(original_matrix_number_of_rows - target_size, 2)
    
    num_threads = cpu_count()
    chunk_size = total_combinations // num_threads
    
    args_list = [(original_matrix, target_size, condition_func, i * chunk_size, (i + 1) * chunk_size) for i in range(num_threads)]
    
    with Pool(num_threads) as pool:
        results = pool.map(find_satisfying_graph_partial, args_list)
    
    for result in results:
        if result[0] == 1:
            return result
    
    return 0, 0, 0

def print_results(file_path, target_path, result, ret_spine_indices, ret_page_indices, graph_type):
    if result == 1:
        print(f"{file_path}には{target_path}が見つかりました")
        print(f"{graph_type}_graphにて条件を満たすグラフが見つかりました ({target_path})")
        print("page_indices:", ret_page_indices)
        print("spine_indices:", ret_spine_indices)
        print("for drawing graph, please run drawGraph.py", ret_page_indices + ret_spine_indices)

def main():
    file_path = 'adjcencyMatrix/Paley/Paley25.txt'
    original_matrix = read_adjacency_matrix(file_path)

    print("original_matrix")
    first_target_size = int(input("first_target_book: "))
    second_target_size = int(input("second_target_book: "))

    first_target_path = f'targetAdjcencyMatrix/B{first_target_size}.txt'
    second_target_path = f'targetAdjcencyMatrix/B{second_target_size}.txt'

    result1, ret_spine_indices1, ret_page_indices1 = find_satisfying_graph_parallel(original_matrix, first_target_size, condition_function_1)
    
    time.sleep(1)
    
    result2, ret_spine_indices2, ret_page_indices2 = find_satisfying_graph_parallel(original_matrix, second_target_size, condition_function_2)

    print_results(file_path, first_target_path, result1, ret_spine_indices1, ret_page_indices1, "first")
    print_results(file_path, second_target_path, result2, ret_spine_indices2, ret_page_indices2, "second")

if __name__ == "__main__":
    main()
