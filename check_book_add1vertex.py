import itertools
import os
import numpy as np
from tqdm import tqdm



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

def generate_combinations(total, size):
    return itertools.combinations(range(total), size)

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
    for page_indices in generate_combinations(len(target_matrix), target_size):

        remaining_indices = set(range(len(target_matrix))) - set(page_indices)

        for spine_indices in generate_combinations(len(remaining_indices), 2):
            if set(spine_indices).isdisjoint(set(page_indices)):
                result = search_book(target_matrix, page_indices, spine_indices, condition_func)

                if result:
                    return [spine_indices, page_indices]

    return []

def search_Ramsey(original_matrix, first_target_size, second_target_size):
    progressbar = tqdm(total=2 ** original_matrix.shape[0], desc="Finding satisfying graph")
    res1 = []
    res2 = []
    count_true_total_1 = 0  # True の総数をカウントする変数
    count_true_total_2 = 0  # True の総数をカウントする変数

    for i in range(2 ** original_matrix.shape[0]):   
        target_matrix = add_vertex(original_matrix, i)
        
        result1 = set_indices(target_matrix, first_target_size, condition_function_1)
        res1.append(result1)

        if result1:
            count_true_total_1 += 1  # True のときだけカウント

        result2 = set_indices(target_matrix, second_target_size, condition_function_2)
        res2.append(result2)

        if result2:
            count_true_total_2 += 1  # True のときだけカウント
        
        progressbar.update(1)
    progressbar.close()
    
    return res1, count_true_total_1, res2, count_true_total_2


def main():
    file_path = 'adjcencyMatrix/Paley/Paley13.txt'
    original_matrix = read_adjacency_matrix(file_path)
    
    print("original_matrix")
    first_target_size = int(input("first_target_book: "))
    second_target_size = int(input("second_target_book: "))
    
    res1, count_true_total_1, res2, count_true_total_2 = search_Ramsey(original_matrix, first_target_size, second_target_size)
    
    
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
