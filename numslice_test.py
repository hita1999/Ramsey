import numpy as np
from numba import njit

# Numbaジット関数を使って高速化
@njit
def assign_array(array1, array2):
    array1[:array2.shape[0], :array2.shape[1]] = array2

# 2x3のゼロ行列を用意
array1 = np.zeros((4, 4), dtype=np.int8)

# 別の行列を用意（例として3x3の行列）
array2 = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]], dtype=np.int8)

# 高速化された関数で代入
assign_array(array1, array2)

# 結果の表示
print(array1)

print(array1[1:2, :3])
