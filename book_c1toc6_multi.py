from calendar import c
import datetime
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm
import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
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

@jit(nopython=True, cache=True)
def integer_to_binary(cir_size, i):
    binary_array = np.zeros(cir_size, dtype=np.uint8)
    for j in range(cir_size-1, -1, -1):
        binary_array[cir_size - 1 - j] = np.right_shift(i, j) & 1
    return binary_array

@jit(nopython=True, cache=True)
def triu_numba(matrix):
    m, n = matrix.shape
    result = np.zeros_like(matrix)

    for i in range(m):
        for j in range(i, n):
            result[i, j] = matrix[i, j]

    return result

@jit(nopython=True, cache=True)
def circulant_numba(c):
    c = np.asarray(c).ravel()
    L = len(c)
    result = np.zeros((L, L), dtype=c.dtype)

    for i in range(L):
        for j in range(L):
            result[i, j] = c[(i - j) % L]

    return result

@jit(nopython=True, cache=True)
def assign_matrix_to_A(A, matrix, row_start, row_end, col_start, col_end):
    A[row_start:row_end, col_start:col_end] = matrix

def save_matrix_to_txt(matrix, file_path):
    np.savetxt(file_path, matrix, fmt='%d', delimiter='')

def calculate_A(args):
    matrix_size, matrix_index, first_target_size, second_target_size, found = args
    
    A = np.zeros((matrix_size, matrix_size), dtype=np.uint8)
    
    
    counter = 0
    vector = integer_to_binary(matrix_size // 3, matrix_index)
    C1 = circulant_numba(vector)
    C1 = triu_numba(C1) + triu_numba(C1).T

    assign_matrix_to_A(A, C1, 0, matrix_size//3, 0, matrix_size//3)
    
    
    for matrix2_index in range(2 ** (matrix_size // 3 - 1)):
        vector2 = integer_to_binary(matrix_size // 3, matrix2_index)
        C2 = circulant_numba(vector2)
        C2 = triu_numba(C2) + triu_numba(C2).T

        assign_matrix_to_A(A, C2, 0, matrix_size//3, matrix_size//3, 2*matrix_size//3)
        assign_matrix_to_A(A, C2.T, matrix_size//3, 2*matrix_size//3, 0, matrix_size//3)


        
        for matrix3_index in range(2 ** (matrix_size // 3 - 1)):
            vector3 = integer_to_binary(matrix_size // 3, matrix3_index)
            C3 = circulant_numba(vector3)
            C3 = triu_numba(C3) + triu_numba(C3).T


            assign_matrix_to_A(A, C3, 0, matrix_size//3, 2*matrix_size//3, matrix_size)
            assign_matrix_to_A(A, C3.T, 2*matrix_size//3, matrix_size, 0, matrix_size//3)
            
            for matrix4_index in range(2 ** (matrix_size // 3 - 1)):
                vector4 = integer_to_binary(matrix_size // 3, matrix4_index)
                C4 = circulant_numba(vector4)
                C4 = triu_numba(C4) + triu_numba(C4).T

                assign_matrix_to_A(A, C4, matrix_size//3, 2*matrix_size//3, matrix_size//3, 2*matrix_size//3)
                
                for matrix5_index in range(2 ** (matrix_size // 3 - 1)):
                    vector5 = integer_to_binary(matrix_size // 3, matrix5_index)
                    C5 = circulant_numba(vector5)
                    C5 = triu_numba(C5) + triu_numba(C5).T


                    assign_matrix_to_A(A, C5, matrix_size//3, 2*matrix_size//3, 2*matrix_size//3, matrix_size)
                    assign_matrix_to_A(A, C5.T, 2*matrix_size//3, matrix_size, matrix_size//3, 2*matrix_size//3)
                    
                    for matrix6_index in range(2 ** (matrix_size // 3 - 1)):
                        vector6 = integer_to_binary(matrix_size // 3, matrix6_index)
                        C6 = circulant_numba(vector6)
                        C6 = triu_numba(C6) + triu_numba(C6).T

                        assign_matrix_to_A(A, C6, 2*matrix_size//3, matrix_size, 2*matrix_size//3, matrix_size)
                        counter += 1

                        ret = set_indices(A, first_target_size, 2)
                        if ret == 0:
                            ret2 = set_indices(A, second_target_size, 0)
                            if ret2 == 0:
                                decimal_value = sum(int(''.join(map(str, vec)), 2) * (2 ** (matrix_size // 3 - 1)) ** exp for vec, exp in zip([vector, vector2, vector3, vector4, vector5, vector6], [5, 4, 3, 2, 1, 0]))
                                return A, vector, vector2, vector3, vector4, vector5, vector6, decimal_value

    return None

def main():
    manager = Manager()
    found = manager.Value('b', False)  # 'b' stands for boolean

    first_target_size = int(input("first_target_book: "))
    second_target_size = int(input("second_target_book: "))
    matrix_size = int(input("matrix_size: "))

    print('Max', (2 ** (matrix_size // 3 - 1))**6)

    args_list = [(matrix_size, matrix_index, first_target_size, second_target_size, found) for matrix_index in range(2 ** (matrix_size // 3 - 1))]

    with Pool() as pool:
        for result in tqdm(pool.imap_unordered(calculate_A, args_list), total=len(args_list), desc="Finding satisfying graph", position=0, leave=True):
            if result is not None:
                A, vector, vector2, vector3, vector4, vector5, vector6, decimal_value = result
                print('found!')
                save_matrix_to_txt(A, f'generatedMatrix/circulantBlock/C1toC6_{decimal_value}.txt')
                print(decimal_value)
                print(vector)
                print(vector2)
                print(vector3)
                print(vector4)
                print(vector5)
                print(vector6)
                found.value = True
                break

    if not found.value:
        print('not found')

if __name__ == "__main__":
    print(datetime.datetime.now().time())
    main()
