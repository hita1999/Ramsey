from tqdm import tqdm
import numpy as np
from scipy.linalg import circulant
from numba import jit


@jit(nopython=True)
def generate_combinations(elements, size):
    result = np.empty((0, size), dtype=np.int64)
    current = np.zeros(size, dtype=np.int64)
    index = 0
    start = 0

    while index >= 0:
        if index == size:
            result = np.concatenate((result, current.reshape(1, -1)))
            index -= 1
        else:
            for i in range(start, len(elements)):
                current[index] = elements[i]
                index += 1
                start = i + 1
                break
            else:
                index -= 1
                if index >= 0:
                    start = current[index] + 1

    return result

@jit(nopython=True)
def numba_ix(matrix, rows, cols):
    one_d_index = np.zeros(len(rows) * len(cols), dtype=np.int32)
    for i, r in enumerate(rows):
        start = i * len(cols)
        one_d_index[start: start + len(cols)] = cols + matrix.shape[1] * r

    arr_1d = matrix.reshape((matrix.shape[0] * matrix.shape[1], 1))
    slice_1d = np.take(arr_1d, one_d_index)
    return slice_1d.reshape((len(rows), len(cols)))

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
def integer_to_binary(cir_size):
    binary_list = []
    for i in range(2 ** (cir_size-1)):
        binary_array = np.zeros(cir_size, dtype=np.uint8)
        for j in range(cir_size-1, -1, -1):
            binary_array[cir_size - 1 - j] = np.right_shift(i, j) & 1
        binary_list.append(binary_array)
    return binary_list

def save_matrix_to_txt(matrix, file_path):
    np.savetxt(file_path, matrix, fmt='%d', delimiter='')


def main():

    first_target_size = int(input("first_target_book: "))
    second_target_size = int(input("second_target_book: "))
    matrix_size = 14
    
    matrix_list = integer_to_binary(int(matrix_size/2))
    print('Max',len(matrix_list)**3)
    
    counter = 0
    
    progressbar = tqdm(total=len(matrix_list)**3, desc="Finding satisfying graph")
    
    for matrix in matrix_list:
        C1 = circulant(matrix)
        C1 = np.triu(C1) + np.triu(C1, 1).T
        for matrix2 in matrix_list:
            C2 = circulant(matrix2)
            C2 = np.triu(C2) + np.triu(C2, 1).T
            for matrix3 in matrix_list:
                C3 = circulant(matrix3)
                C3 = np.triu(C3) + np.triu(C3, 1).T
                B1 = np.hstack((C1, C2))
                B2 = np.hstack((C2.T, C3))
                A = np.vstack((B1, B2))
                counter += 1
                progressbar.update(1)
                
                ret = set_indices(A, first_target_size, 2)
                if ret == 0:
                    ret2 = set_indices(A, second_target_size, 0)
                    if ret2 == 0:
                        print('found!')
                        print(A)
                        save_matrix_to_txt(A, 'generatedMatrix/circulantBlock/circulantBlock.txt')
                        print(matrix)
                        print(matrix2)
                        print(matrix3)
                        break
            if ret == 0:
                break
        if ret == 0:
            break
    progressbar.close()
    print(counter)
    if counter == len(matrix_list)**3:
        print('not found')

if __name__ == "__main__":
    main()
