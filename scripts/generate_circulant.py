from scipy.linalg import circulant
import numpy as np

def integer_to_binary(matrix_size, num):
    binary_str = bin(num)[2:]
    binary_list = [int(x) for x in binary_str.zfill(matrix_size)]
    return binary_list

def generate_matrix(matrix_size, i):
    binary_array = integer_to_binary(matrix_size, i)
    circulant_matrix = circulant(binary_array)
    circulant_matrix = np.triu(circulant_matrix) + np.triu(circulant_matrix, 1).T
    return circulant_matrix

def save_matrix_to_txt(matrix, file_path):
    np.savetxt(file_path, matrix, fmt='%d', delimiter='')


matrix_size = 17
idx = 11892
target_matrix = generate_matrix(matrix_size, idx)
print(target_matrix)

save_matrix_to_txt(target_matrix, f'generatedMatrix/circulant{matrix_size}_idx_{idx}.txt')