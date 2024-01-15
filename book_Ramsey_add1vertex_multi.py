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
    

def add_vertex(original_matrix, add_num):
    add_list = integer_to_binary(original_matrix.shape[0], add_num)
    add_list = np.array([add_list])
    new_matrix = np.vstack((original_matrix, add_list))
    
    # 新しい頂点との接続関係を設定
    add_list = np.append(add_list, 0)
    
    # 行列に追加
    new_matrix = np.hstack((new_matrix, add_list.reshape(-1, 1)))
    
    #print(new_matrix)
    return new_matrix

def generate_combinations(elements, size):
    return itertools.combinations(elements, size)

    
def set_indices(target_matrix, target_size, condition):
    for spine_indices in generate_combinations(range(len(target_matrix)), 2):
        if target_matrix[np.ix_(spine_indices, spine_indices)].sum() == condition:
            remaining_indices = set(range(len(target_matrix))) - set(spine_indices)
            
            for page_indices in generate_combinations(remaining_indices, target_size):
                if set(spine_indices).isdisjoint(set(page_indices)):
                    if target_matrix[np.ix_(spine_indices, page_indices)].sum() == len(page_indices) * condition:
                        return [spine_indices, page_indices]
    return []


def process_search(args):
    original_matrix, first_target_size, second_target_size, i = args
    target_matrix = add_vertex(original_matrix, i)
        
    result1 = set_indices(target_matrix, first_target_size, 2)
    if result1:
        count_true_total_1 = 1
    else:
        count_true_total_1 = 0

    result2 = set_indices(target_matrix, second_target_size, 0)
    if result2:
        count_true_total_2 = 1
    else:
        count_true_total_2 = 0

    return result1, count_true_total_1, result2, count_true_total_2


def search_Ramsey_parallel(original_matrix, first_target_size, second_target_size):
    total_iterations = 2 ** original_matrix.shape[0]
    progressbar = tqdm(total=total_iterations, desc="Finding satisfying graph")

    args_list = [(original_matrix, first_target_size, second_target_size, i) for i in range(total_iterations)]

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_search, args_list), total=total_iterations))

    res1, count_true_total_1, res2, count_true_total_2 = zip(*results)
    count_true_total_1 = sum(count_true_total_1)
    count_true_total_2 = sum(count_true_total_2)

    progressbar.close()
    
    return res1, count_true_total_1, res2, count_true_total_2


def main():
    file_path = 'adjcencyMatrix/Paley/Paley9.txt'
    original_matrix = read_adjacency_matrix(file_path)
    
    print("original_matrix")
    first_target_size = int(input("first_target_book: "))
    second_target_size = int(input("second_target_book: "))
    
    res1, count_true_total_1, res2, count_true_total_2 = search_Ramsey_parallel(original_matrix, first_target_size, second_target_size)
    
    
    file_name, _ = os.path.splitext(os.path.basename(file_path))
    # Open a file in write mode
    with open(f'searchResultTextFile/add1vertex_{file_name}_B{first_target_size}_B{second_target_size}.txt', 'w') as file:

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