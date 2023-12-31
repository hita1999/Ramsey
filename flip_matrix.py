from operator import index
from os import read
import re
import sys

from check_matrix import read_adjacency_matrix, save_matrix_to_txt

def flip_matrix(matrix, i, j):
    matrix[i][j] = '1' if matrix[i][j] == '0' else '0'
    matrix[j][i] = '1' if matrix[j][i] == '0' else '0'

def print_matrix(matrix):
    for row in matrix:
        print(''.join(map(str, row)))

if __name__ == "__main__":
    # ファイルから行列を読み込むなどのコードが必要な場合はここに追加
    matrix = read_adjacency_matrix('adjcencyMatrix/Paley/Paley37_2.txt')

    index_1 = int(input("Enter the first index: "))
    index_2 = int(input("Enter the second index: "))

    # 行列の反転前の表示
    print("Original Matrix:")
    print_matrix(matrix)

    # 指定されたインデックスの数値を反転
    flip_matrix(matrix, index_1, index_2)

    # 反転後の表示
    print("\nMatrix after flipping at indices {} and {}:".format(index_1, index_2))
    print_matrix(matrix)

    save_matrix_to_txt(matrix, 'adjcencyMatrix/Paley/Paley37_2.txt')
