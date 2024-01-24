from multiprocessing import Pool, cpu_count, Manager
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
def integer_to_binary(cir_size, i):
    binary_array = np.zeros(cir_size, dtype=np.uint8)
    for j in range(cir_size-1, -1, -1):
        binary_array[cir_size - 1 - j] = np.right_shift(i, j) & 1
    return binary_array

def save_matrix_to_txt(matrix, file_path):
    np.savetxt(file_path, matrix, fmt='%d', delimiter='')

def calculate_A(args):
    matrix_size, matrix_index, first_target_size, second_target_size, found = args
    counter = 0
    vector = integer_to_binary(matrix_size // 2, matrix_index)
    C1 = circulant(vector)
    C1 = np.triu(C1) + np.triu(C1, 1).T
    for matrix2_index in range(2 ** (matrix_size // 2 - 1)):
        vector2 = integer_to_binary(matrix_size // 2, matrix2_index)
        C2 = circulant(vector2)
        C2 = np.triu(C2) + np.triu(C2, 1).T
        for matrix3_index in range(2 ** (matrix_size // 2 - 1)):
            vector3 = integer_to_binary(matrix_size // 2, matrix3_index)
            C3 = circulant(vector3)
            C3 = np.triu(C3) + np.triu(C3, 1).T
            B1 = np.hstack((C1, C2))
            B2 = np.hstack((C2.T, C3))
            A = np.vstack((B1, B2))
            counter += 1

            ret = set_indices(A, first_target_size, 2)
            if ret == 0:
                ret2 = set_indices(A, second_target_size, 0)
                if ret2 == 0:
                    decimal_value = sum(int(''.join(map(str, vec)), 2) * (2 ** (matrix_size // 2 - 1)) ** exp for vec, exp in zip([vector, vector2, vector3], [2, 1, 0]))
                    return A, vector, vector2, vector3, decimal_value

    return None

def main():
    manager = Manager()
    found = manager.Value('b', False)  # 'b' stands for boolean

    first_target_size = int(input("first_target_book: "))
    second_target_size = int(input("second_target_book: "))
    matrix_size = int(input("matrix_size: "))

    print('Max', (2 ** (matrix_size // 2 - 1))**3)

    args_list = [(matrix_size, matrix_index, first_target_size, second_target_size, found) for matrix_index in range(2 ** (matrix_size // 2 - 1))]

    with Pool() as pool:
        for result in tqdm(pool.imap_unordered(calculate_A, args_list), total=len(args_list), desc="Finding satisfying graph", position=0, leave=True):
            if result is not None:
                A, vector, vector2, vector3, decimal_value = result
                print('found!')
                save_matrix_to_txt(A, f'generatedMatrix/circulantBlock/C1C2C3_{decimal_value}.txt')
                print(decimal_value)
                print(vector)
                print(vector2)
                print(vector3)
                found.value = True
                break

    if not found.value:
        print('not found')

if __name__ == "__main__":
    main()
