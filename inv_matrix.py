import numpy as np

# サンプルのNumPy array
original_array = np.array([
    [0, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 0]
])

# 0と1を反転させる
inverted_array = original_array.copy()
inverted_array[np.triu_indices(inverted_array.shape[0], k=1)] = 1 - inverted_array[np.triu_indices(inverted_array.shape[0], k=1)]

# 結果の表示
print("Original Array:")
print(original_array)

print("\nInverted Array:")
print(inverted_array)
