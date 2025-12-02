import cProfile
from multiprocessing import Pool, cpu_count, Manager
import pstats
from tqdm import tqdm
import numpy as np
from scipy.linalg import circulant
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
def assign_matrix_to_A(A, matrix, row_start, row_end, col_start, col_end):
    A[row_start:row_end, col_start:col_end] = matrix

@jit(nopython=True, cache=True)
def diagonal_integer_to_binary(matrix_size, i):
    cir_size = matrix_size // 4
    r = matrix_size % 4

    binary_array = np.zeros(cir_size, dtype=np.uint8)
    for j in range(cir_size-1, -1, -1):
        binary_array[cir_size - 1 - j] = np.right_shift(i, j) & 1

    reversed_binary_array = binary_array[::-1]

    if r == 0:
        combined_array = np.zeros(2 * cir_size, dtype=np.uint8)
        combined_array[1:cir_size+1] = binary_array
        combined_array[cir_size+1:] = reversed_binary_array[1:]
    else:
        combined_array = np.zeros(2 * cir_size + 1, dtype=np.uint8)
        combined_array[1:cir_size+1] = binary_array
        combined_array[cir_size+1:] = reversed_binary_array

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


def calculate_A_and_profile(args):
    matrix_size, matrix_index, first_target_size, second_target_size = args
    if matrix_index == 3:
        # プロファイリング用のファイル名
        profile_filename = f"res_profile/profile_results_B11B12_3900X.txt"

        # プロファイリングを開始
        profile = cProfile.Profile()
        profile.enable()

        # 実際の処理を実行
        calculate_A(args)

        # プロファイリングを停止
        profile.disable()

        # プロファイリング結果を保存
        with open(profile_filename, "w") as f:
            stats = pstats.Stats(profile, stream=f)
            stats.sort_stats("cumulative")
            stats.print_stats()
    else:
        result = calculate_A(args)
        if result is not None:
            return result
    return None

def calculate_A(args):
    matrix_size, matrix_index, first_target_size, second_target_size = args

    A = np.zeros((matrix_size, matrix_size), dtype=np.uint8)

    counter = 0
    vector = diagonal_integer_to_binary(matrix_size, matrix_index)
    C1 = circulant_numba(vector)

    assign_matrix_to_A(A, C1, 0, matrix_size//2, 0, matrix_size//2)

    for matrix2_index in range(2 ** (matrix_size // 4)):
        vector2 = diagonal_integer_to_binary(matrix_size, matrix2_index)

        C2 = circulant_numba(vector2)

        assign_matrix_to_A(A, C2, 0, matrix_size//2,
                           matrix_size//2, matrix_size)
        assign_matrix_to_A(A, C2.T, matrix_size//2,
                           matrix_size, 0, matrix_size//2)

        for matrix3_index in range(2 ** (matrix_size // 4)):
            vector3 = diagonal_integer_to_binary(
                matrix_size, matrix3_index)
            C3 = circulant_numba(vector3)

            assign_matrix_to_A(A, C3, matrix_size//2,
                               matrix_size, matrix_size//2, matrix_size)

            counter += 1

            ret = set_indices(A, first_target_size, 2)
            if ret == 0:
                ret2 = set_indices(A, second_target_size, 0)
                if ret2 == 0:
                    print('found!')
                    decimal_value = matrix_index * matrix2_index * matrix3_index
                    return A, vector, vector2, vector3, decimal_value

    return None

def main():
    manager = Manager()
    found = manager.Value('b', False)  # 'b' stands for boolean

    first_target_size = int(input("first_target_book: "))
    second_target_size = int(input("second_target_book: "))
    matrix_size = int(input("matrix_size: "))

    print('Max', (2 ** (2 * (matrix_size // 4) + (matrix_size // 4 + 1))))

    args_list = [(matrix_size, matrix_index, first_target_size, second_target_size)
                 for matrix_index in range(2 ** (matrix_size // 4))]

    with Pool() as pool:
        # 各プロセスでプロファイリングを実行し、結果を取得
        results = list(pool.imap_unordered(calculate_A_and_profile, args_list))
        for result in results:
            if result is not None:
                A, vector, vector2, vector3, decimal_value = result
                print('A:', A)
                print('vector:', vector)
                print('vector2:', vector2)
                print('vector3:', vector3)
                print('decimal_value:', decimal_value)
                found.value = True
                break

    if not found.value:
        print('not found')

if __name__ == "__main__":
    main()
