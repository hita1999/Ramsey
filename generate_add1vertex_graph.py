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

file_path = 'adjcencyMatrix/Paley/Paley9.txt'
original_matrix = read_adjacency_matrix(file_path)

file_name, _ = os.path.splitext(os.path.basename(file_path))
print("File Name:", file_name)

idx = int(input('input index: '))

target_matrix = add_vertex(original_matrix, idx)
save_matrix_to_txt(target_matrix, f'generatedMatrix/{file_name}_add_{idx}.txt')