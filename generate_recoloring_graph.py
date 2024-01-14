import os
import numpy as np


def read_adjacency_matrix(file_path):
    adjacency_matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [int(x) for x in line.strip()]
            adjacency_matrix.append(row)
    return np.array(adjacency_matrix)


def save_matrix_to_txt(matrix, file_path):
    np.savetxt(file_path, matrix, fmt='%d', delimiter='')


def integer_to_binary(original_matrix_size, add_num):
    binary_str = bin(add_num)[2:]
    binary_list = [int(x) for x in binary_str.zfill(original_matrix_size)]
    return binary_list


def swap_edge(original_matrix, swap_target_index, idx):
    
    swap_list = integer_to_binary(swap_target_index, idx)
    swap_list = np.array([swap_list])
    
    original_matrix[swap_target_index, 0:swap_target_index] = swap_list
    original_matrix[0:swap_target_index, swap_target_index] = swap_list
    
    
    return original_matrix

file_path = 'generatedMatrix/Paley9_add_0.txt'
original_matrix = read_adjacency_matrix(file_path)

file_name, _ = os.path.splitext(os.path.basename(file_path))
print("File Name:", file_name)

swap_target = int(input('input swap index: '))
idx = int(input('input index: '))

target_matrix = swap_edge(original_matrix, swap_target, idx)
save_matrix_to_txt(target_matrix, f'generatedMatrix/{file_name}_recolor_target_{swap_target}_idx_{idx}.txt')