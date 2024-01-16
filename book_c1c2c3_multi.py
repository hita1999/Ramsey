from multiprocessing import Pool, cpu_count
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
    for spine_indices in generate_combinations(np.arange(len(target_matrix)), 2):
        if numba_ix(target_matrix, spine_indices, spine_indices).sum() == condition:
            all_indices = np.concatenate((np.arange(len(target_matrix)), spine_indices))
            remaining_indices = np.array([idx for idx in all_indices if idx not in spine_indices])
            
            for page_indices in generate_combinations(remaining_indices, target_size):
                if numba_ix(target_matrix, spine_indices, page_indices).sum() == len(page_indices) * condition:
                    return 0
    
    return 1


@jit(nopython=True)
def integer_to_binary(cir_size):
    binary_list = []
    for i in range(2 ** (cir_size-1)):
        binary_array = np.zeros(cir_size, dtype=np.uint8)
        for j in range(cir_size-1, -1, -1):
            binary_array[cir_size - 1 - j] = np.right_shift(i, j) & 1
        binary_list.append(binary_array)
    return binary_list


def main():

    first_target_size = int(input("first_target_book: "))
    second_target_size = int(input("second_target_book: "))
    matrix_size = 10
    
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
                ret2 = set_indices(A, second_target_size, 0)
                if ret + ret2 == 2:
                    print('found!')
                    print(ret)
                    print(A)
                    print(matrix)
                    print(matrix2)
                    print(matrix3)
                    break
            if ret + ret2 == 2:
                break
        if ret + ret2 == 2:
            break
    progressbar.close()
    print(counter)

if __name__ == "__main__":
    main()