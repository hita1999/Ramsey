import numpy as np

def check_submatrix(adjacency_matrix, target_submatrix):
    for i in range(len(adjacency_matrix) - len(target_submatrix) + 1):
        for j in range(len(adjacency_matrix[0]) - len(target_submatrix[0]) + 1):
            submatrix = adjacency_matrix[i:i+len(target_submatrix), j:j+len(target_submatrix[0])]
            if np.array_equal(submatrix, target_submatrix):
                return True, i, j
    return False, -1, -1

# 指定された部分行列
target_submatrix = np.array([
    [0, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0]
])

# ランダムな16x16の隣接行列の生成
random_adjacency_matrix = np.random.randint(2, size=(16, 16))

# 指定された部分行列が存在するかどうかを確認
found, row, col = check_submatrix(random_adjacency_matrix, target_submatrix)

if found:
    print("指定された部分行列が見つかりました。")
    print("行:", row, "列:", col)
else:
    print("指定された部分行列は見つかりませんでした。")

print("\nランダムな隣接行列:")
print(random_adjacency_matrix)
