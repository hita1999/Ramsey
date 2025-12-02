import cProfile, pstats
from multiprocessing import Pool
import numpy as np
from numba import jit

# ここで set_indices_optimized をインポート
#@jit(nopython=True)
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

def main():
    # 例えば、適切なデータや条件をセットアップ
    n = 100000
    target_matrix = np.random.randint(2, size=(n, n))
    target_size = 12
    condition = 2

    # cProfile を使用して関数をプロファイリング
    profiler = cProfile.Profile()
    profiler.enable()

    # ここで set_indices_optimized を呼び出す
    set_indices(target_matrix, target_size, condition)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats()
if __name__ == "__main__":
    main()
