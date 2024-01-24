from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm
import numpy as np
from numba import jit
import cProfile
import pstats

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

@jit(nopython=True)
def triu_numba(matrix, k=0):
    m, n = matrix.shape
    result = np.zeros_like(matrix, dtype=np.int8)

    for i in range(m):
        for j in range(max(i - k, 0), n):
            result[i, j] = matrix[i, j]

    return result

@jit(nopython=True)
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
    matrix_size, matrix_index, first_target_size, second_target_size, found = args
    if matrix_index == 0:
        # プロファイリング用のファイル名
        profile_filename = f"res_profile/profile_results_B2B5_0.txt"

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
        calculate_A(args)
    return matrix_index

def calculate_A(args):
    matrix_size, matrix_index, first_target_size, second_target_size, found = args
    counter = 0
    vector = integer_to_binary(matrix_size // 3, matrix_index)
    C1 = circulant_numba(vector)
    triu_C1 = triu_numba(C1)
    
    for matrix2_index in range(2 ** (matrix_size // 3 - 1)):
        vector2 = integer_to_binary(matrix_size // 3, matrix2_index)
        C2 = circulant_numba(vector2)
        triu_C2 = triu_numba(C2)
        
        for matrix3_index in range(2 ** (matrix_size // 3 - 1)):
            vector3 = integer_to_binary(matrix_size // 3, matrix3_index)
            C3 = circulant_numba(vector3)
            triu_C3 = triu_numba(C3)
            
            for matrix4_index in range(2 ** (matrix_size // 3 - 1)):
                vector4 = integer_to_binary(matrix_size // 3, matrix4_index)
                C4 = circulant_numba(vector4)
                triu_C4 = triu_numba(C4)

                for matrix5_index in range(2 ** (matrix_size // 3 - 1)):
                    vector5 = integer_to_binary(matrix_size // 3, matrix5_index)
                    C5 = circulant_numba(vector5)
                    triu_C5 = triu_numba(C5)

                    for matrix6_index in range(2 ** (matrix_size // 3 - 1)):
                        vector6 = integer_to_binary(matrix_size // 3, matrix6_index)
                        C6 = circulant_numba(vector6)
                        triu_C6 = triu_numba(C6)

                        # Pre-allocate memory for B1, B2, B3
                        B1 = np.empty((C1.shape[0], triu_C1.shape[1] + triu_C2.shape[1] + triu_C3.shape[1]), dtype=triu_C1.dtype)
                        B2 = np.empty((C2.shape[0], triu_C2.shape[1] + triu_C4.shape[1] + triu_C5.shape[1]), dtype=triu_C2.dtype)
                        B3 = np.empty((C3.shape[0], triu_C3.shape[1] + triu_C5.shape[1] + triu_C6.shape[1]), dtype=triu_C3.dtype)

                        # Fill in the values
                        B1[:, :triu_C1.shape[1]] = triu_C1
                        B1[:, triu_C1.shape[1]:triu_C1.shape[1] + triu_C2.shape[1]] = triu_C2
                        B1[:, triu_C1.shape[1] + triu_C2.shape[1]:] = triu_C3

                        B2[:, :triu_C2.shape[1]] = triu_C2
                        B2[:, triu_C2.shape[1]:triu_C2.shape[1] + triu_C4.shape[1]] = triu_C4
                        B2[:, triu_C2.shape[1] + triu_C4.shape[1]:] = triu_C5

                        B3[:, :triu_C3.shape[1]] = triu_C3
                        B3[:, triu_C3.shape[1]:triu_C3.shape[1] + triu_C5.shape[1]] = triu_C5
                        B3[:, triu_C3.shape[1] + triu_C5.shape[1]:] = triu_C6

                        A = np.vstack((B1, B2, B3))
                        counter += 1

                        ret = set_indices(A, first_target_size, 2)
                        if ret == 0:
                            ret2 = set_indices(A, second_target_size, 0)
                            if ret2 == 0:
                                decimal_value = int(''.join(map(str, np.concatenate([vector, vector2, vector3, vector4, vector5, vector6], 0))), 2)
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
        # 各プロセスでプロファイリングを実行し、結果を取得
        results = list(pool.imap_unordered(calculate_A_and_profile, args_list))


    if not found.value:
        print('not found')

if __name__ == "__main__":
    profile = cProfile.Profile()
    profile.enable()

    main()

    profile.disable()

    # メインプロセスのプロファイリング結果を保存
    with open("res_profile/main_profile_results.txt", "w") as f:
        stats = pstats.Stats(profile, stream=f)
        stats.sort_stats("tottime")
        stats.print_stats()
