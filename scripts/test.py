import numpy as np
from numba import jit


def diagonal_integer_to_binary(matrix_size, i):
    cir_size = matrix_size // 6
    r = matrix_size % 6

    binary_array = np.zeros(cir_size, dtype=np.uint8)
    for j in range(cir_size-1, -1, -1):
        binary_array[cir_size - 1 - j] = np.right_shift(i, j) & 1

    reversed_binary_array = binary_array[::-1]

    if r == 0:
        combined_array = np.zeros(2 * cir_size, dtype=np.uint8)
        combined_array[:cir_size] = binary_array
        combined_array[cir_size:] = reversed_binary_array
    else:
        combined_array = np.zeros(2 * cir_size + 1, dtype=np.uint8)
        combined_array[1:cir_size+1] = binary_array
        combined_array[cir_size+1:] = reversed_binary_array

    return combined_array

def circulant_numba(c):
    c = np.asarray(c).ravel()
    L = len(c)
    result = np.zeros((L, L), dtype=c.dtype)

    for i in range(L):
        for j in range(L):
            result[i, j] = c[(i - j) % L]

    return result

def integer_to_binary(cir_size, i):
    binary_array = np.zeros(cir_size, dtype=np.uint8)
    for j in range(cir_size-1, -1, -1):
        binary_array[cir_size - 1 - j] = np.right_shift(i, j) & 1
    
    return binary_array

def triu_numba(matrix):
    m, n = matrix.shape
    result = np.zeros_like(matrix)

    for i in range(m):
        for j in range(i, n):
            result[i, j] = matrix[i, j]

    return result

cir_size = 27
i = 35

result = diagonal_integer_to_binary(cir_size, i)
print(result)
res2 = circulant_numba(result)
print(res2)


# res3 = integer_to_binary(11, 2**10-21)
# print(res3)

# res4 = circulant_numba(res3)
# print(res4)

# print(res4.T)
