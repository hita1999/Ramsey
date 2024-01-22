from multiprocessing import Pool, Manager
from os import cpu_count
from tqdm import tqdm
import numpy as np
from scipy.linalg import circulant
from numba import jit

@jit(nopython=True)
def set_indices(target_matrix, target_size, condition):
    n = len(target_matrix)
    ret = -1

    for i in range(n - 1):
        for j in range(i + 1, n):
            spine_indices = np.array([i, j], dtype=np.int64)

            row_1 = target_matrix[spine_indices[0], :]
            row_2 = target_matrix[spine_indices[1], :]

            if condition == 2 and (target_matrix[i, j] == 1):
                common_vertices = np.where((row_1 == 1) & (row_2 == 1))[0]
                if len(common_vertices) < target_size:
                    ret = 0
                else:
                    return -1

            if condition == 0 and (target_matrix[i, j] == 0):
                common_vertices = np.where((row_1 == 0) & (row_2 == 0))[0]
                common_vertices = np.array([x for x in common_vertices if x not in spine_indices])
                if len(common_vertices) < target_size:
                    ret = 0
                else:
                    return -1

    return ret

@jit(nopython=True)
def integer_to_binary(cir_size, binary_index):
    binary_array = np.zeros(cir_size, dtype=np.uint8)
    for j in range(cir_size-1, -1, -1):
        binary_array[cir_size - 1 - j] = np.right_shift(binary_index, j) & 1
    return binary_array

def save_matrix_to_txt(matrix, file_path):
    np.savetxt(file_path, matrix, fmt='%d', delimiter='')


def calculate_A_batch(args):
    binary_index, matrix_size, first_target_size, second_target_size, found = args
    chunk_size = cpu_count() * 2
    counter = 0

    for i in range(binary_index, binary_index + chunk_size):
        if i >= 2**(matrix_size-1):
            break
        matrix = integer_to_binary(matrix_size, i)
        counter += calculate_A((matrix, first_target_size, second_target_size, found))

    return counter

def calculate_A(args):
    matrix, first_target_size, second_target_size, found = args
    C1 = circulant(matrix)
    C1 = np.triu(C1) + np.triu(C1, 1).T
    
    ret = set_indices(C1, first_target_size, 2)
    if ret == 0:
        ret2 = set_indices(C1, second_target_size, 0)
        if ret2 == 0:
            print('found!')
            decimal_value = int(''.join(map(str, matrix)), 2)
            save_matrix_to_txt(C1, f'generatedMatrix/circulantBlock/C1_{decimal_value}.txt')
            print(decimal_value)
            print(matrix)
            found.value = True
    return 1

def main():
    with Manager() as manager:
        found = manager.Value('b', False)

        first_target_size = int(input("first_target_book: "))
        second_target_size = int(input("second_target_book: "))
        matrix_size = int(input("matrix_size: "))
        
        with Pool() as pool:
            args_list = [(i, matrix_size, first_target_size, second_target_size, found) for i in range(0, 2**(matrix_size-1), cpu_count())]
            for _ in tqdm(pool.imap_unordered(calculate_A_batch, args_list), total=len(args_list), desc="Finding satisfying graph"):
                pass

        if not found.value:
            print('not found')

if __name__ == "__main__":
    main()