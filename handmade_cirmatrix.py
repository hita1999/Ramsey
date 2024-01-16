import numpy as np

def circulant_matrix(size, vector):
    result = np.zeros((size, size), dtype=vector.dtype)
    for i in range(size):
        result[i, :] = np.roll(vector, i)
    
    result = np.triu(result) + np.triu(result, 1).T
    return result

def integer_to_binary(matrix_size, num):
    binary_str = bin(num)[2:]
    binary_list = [int(x) for x in binary_str.zfill(matrix_size)]
    return np.array(binary_list)

def save_matrix_to_txt(matrix, file_path):
    np.savetxt(file_path, matrix, fmt='%d', delimiter='')

# 例
vector1 = np.array([0, 0, 1,1,0])
vector2 = np.array([0, 0,1,1,0])
vector3 = np.array([0,1, 0,0,1])

# for i in range(2**3):
C1 = circulant_matrix(len(vector1), vector1)
C2 = circulant_matrix(len(vector2), vector2)
C3 = circulant_matrix(len(vector3), vector3)
B1 = np.hstack((C1, C2))
B2 = np.hstack((C2.T, C3))
A = np.vstack((B1, B2))
print(A)
save_matrix_to_txt(A, 'generatedMatrix/circulantBlock/circulantBlock.txt')
