import numpy as np

def read_matrix_from_file(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [int(x) for x in line.strip()]
            matrix.append(row)
    return np.array(matrix)

if __name__ == '__main__':
    file_path = 'generatedMatrix/Paley17_add_0.txt'
    matrix = read_matrix_from_file(file_path)

    # 上三角部分を抽出
    upper_triangle = np.triu(matrix)

    # 辺の総数を計算
    total_edges = np.count_nonzero(upper_triangle == 1)

    print("辺の総数:", total_edges)
