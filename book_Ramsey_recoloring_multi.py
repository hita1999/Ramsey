import itertools
import os
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

def integer_to_binary(original_matrix_size, add_num):
    binary_str = bin(add_num)[2:]
    binary_list = [int(x) for x in binary_str.zfill(original_matrix_size)]
    return binary_list
    

def swap_edge(original_matrix, swap_target_index, idx):
    
    swap_list = integer_to_binary(swap_target_index, idx)
    swap_list = np.array([swap_list])
    print(swap_list)
    
    original_matrix[swap_target_index, 0:swap_target_index] = swap_list
    original_matrix[0:swap_target_index, swap_target_index] = swap_list
    
    print(original_matrix)
    
    return original_matrix

def generate_combinations(elements, size):
    return itertools.combinations(elements, size)

def condition_function_1(matrix, page, spine):
    return matrix[np.ix_(spine, spine)].sum() == 2 and matrix[np.ix_(spine, page)].sum() == len(page) * 2


def condition_function_2(matrix, page, spine):
    return matrix[np.ix_(spine, spine)].sum() == 0 and matrix[np.ix_(spine, page)].sum() == 0

def search_book(matrix, page, spine, condition_func):

    if condition_func(matrix, page, spine):
        #print("target_matrix found", page, spine)
        return True

    return False
    
def set_indices(target_matrix, target_size, condition_func):
    for page_indices in generate_combinations(range(len(target_matrix)), target_size):

        remaining_indices = set(range(len(target_matrix))) - set(page_indices)

        for spine_indices in generate_combinations(remaining_indices, 2):
            if set(spine_indices).isdisjoint(set(page_indices)):
                result = search_book(target_matrix, page_indices, spine_indices, condition_func)

                if result:
                    return [spine_indices, page_indices]

    return []


def process_search(args):
    original_matrix, first_target_size, second_target_size, swap_target_index , i = args
    target_matrix = swap_edge(original_matrix, swap_target_index , i)
        
    result1 = set_indices(target_matrix, first_target_size, condition_function_1)
    if result1:
        count_true_total_1 = 1
    else:
        count_true_total_1 = 0

    result2 = set_indices(target_matrix, second_target_size, condition_function_2)
    if result2:
        count_true_total_2 = 1
    else:
        count_true_total_2 = 0

    return result1, count_true_total_1, result2, count_true_total_2


def search_Ramsey_parallel(original_matrix, first_target_size, second_target_size, swap_target_index):
    total_iterations = 2 ** swap_target_index
    progressbar = tqdm(total=total_iterations, desc="Finding satisfying graph")

    args_list = [(original_matrix, first_target_size, second_target_size, swap_target_index , i) for i in range(total_iterations)]

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_search, args_list), total=total_iterations))

    res1, count_true_total_1, res2, count_true_total_2 = zip(*results)
    count_true_total_1 = sum(count_true_total_1)
    count_true_total_2 = sum(count_true_total_2)

    progressbar.close()
    
    return res1, count_true_total_1, res2, count_true_total_2


def main():
    file_path = 'generatedMatrix/Paley9_add_0.txt'
    original_matrix = read_adjacency_matrix(file_path)
    
    print("original_matrix")
    first_target_size = int(input("first_target_book: "))
    second_target_size = int(input("second_target_book: "))
    swap_target_index = int(input("Enter the index to swap rows and columns: "))
    
    res1, count_true_total_1, res2, count_true_total_2 = search_Ramsey_parallel(original_matrix, first_target_size, second_target_size, swap_target_index)
    
    
    file_name, _ = os.path.splitext(os.path.basename(file_path))
    # Open a file in write mode
    with open(f'searchResultTextFile/recoloring_{swap_target_index}_{file_name}_B{first_target_size}_B{second_target_size}.txt', 'w') as file:

        # Result 1
        file.write(f"Result 1 B{first_target_size}:\n")
        if count_true_total_1 != 0:
            file.write("Index, Spine, Page\n")
            for idx, result in enumerate(res1):
                if result:
                    spine_indices, page_indices = result
                    file.write(f"{idx}, {spine_indices}, {page_indices}\n")

        file.write(f"True Count 1: {count_true_total_1}\n")

        # Result 2
        file.write(f"\nResult 2 B{second_target_size}:\n")
        if count_true_total_2 != 0:
            file.write("Index, Spine, Page\n")
            for idx, result in enumerate(res2):
                if result:
                    spine_indices, page_indices = result
                    file.write(f"{idx}, {spine_indices}, {page_indices}\n")

        file.write(f"True Count 2: {count_true_total_2}\n")
        
if __name__ == "__main__":
    main()