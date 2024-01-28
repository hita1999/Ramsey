from multiprocessing import Pool, Manager
from os import cpu_count
import time
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
                common_vertices = np.array(
                    [x for x in common_vertices if x not in spine_indices])
                if len(common_vertices) < target_size:
                    ret = 0
                else:
                    return -1

    return ret

def diagonal_integer_to_binary(matrix_size, i):
    cir_size = matrix_size // 2
    r = matrix_size % 2

    binary_array = np.zeros(cir_size, dtype=np.uint8)
    for j in range(cir_size-1, -1, -1):
        binary_array[cir_size - 1 - j] = np.right_shift(i, j) & 1

    reversed_binary_array = binary_array[::-1]

    if r == 0:
        combined_array = np.concatenate([[0], binary_array, reversed_binary_array[1:]])
        
    else:
        combined_array = np.concatenate([[0], binary_array, reversed_binary_array])

    return combined_array

@jit(nopython=True, cache=True)
def circulant_numba(c):
    c = np.asarray(c).ravel()
    L = len(c)
    result = np.zeros((L, L), dtype=c.dtype)

    for i in range(L):
        for j in range(L):
            result[i, j] = c[(i - j) % L]

    return result


def save_matrix_to_txt(matrix, file_path):
    np.savetxt(file_path, matrix, fmt='%d', delimiter='')


def calculate_A(args):
    matrix_size, matrix_index, first_target_size, second_target_size = args
    
    counter = 0
    vector = diagonal_integer_to_binary(matrix_size, matrix_index)
    C1 = circulant_numba(vector)
    
    counter += 1

    ret = set_indices(C1, first_target_size, 2)
    if ret == 0:
        ret2 = set_indices(C1, second_target_size, 0)
        if ret2 == 0:
            print('found!')
            decimal_value = matrix_index
            return C1, vector, decimal_value
    return None


def main():
    manager = Manager()
    found = manager.Value('b', False)  # 'b' stands for boolean

    first_target_size = int(input("first_target_book: "))
    second_target_size = int(input("second_target_book: "))
    matrix_size = int(input("matrix_size: "))

    print('Max', (2 ** (matrix_size // 2)))

    args_list = [(matrix_size, matrix_index, first_target_size, second_target_size)
                 for matrix_index in range(2 ** (matrix_size // 2))]

    start = time.time()
    with Pool() as pool:
        for result in tqdm(pool.imap_unordered(calculate_A, args_list), total=len(args_list), desc="Finding satisfying graph", position=0, leave=True):
            if result is not None:
                A, vector, decimal_value = result
                print('found!')
                save_matrix_to_txt(
                    A, f'generatedMatrix/circulantBlock/C1_{decimal_value}.txt')
                print(decimal_value)
                print(vector)
                found.value = True
                break

    end = time.time()
    if not found.value:
        print('not found')

    print('elapsed time: ', end - start)


if __name__ == "__main__":
    main()
