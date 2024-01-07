import itertools
from math import comb
import numpy as np
from tqdm import tqdm

def read_adjacency_matrix(file_path):
    adjacency_matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [int(x) for x in line.strip()]
            adjacency_matrix.append(row)
    return np.array(adjacency_matrix)

def check_combination(matrix, indices, target_matrix):
    return np.array_equal(matrix[np.ix_(indices, indices)], target_matrix) or np.array_equal(matrix[np.ix_(indices, indices)], 1 - target_matrix)

def generate_combinations(total, size):
    return itertools.combinations(range(total), size)

def save_matrix_to_txt(matrix, file_path):
    np.savetxt(file_path, matrix, fmt='%d', delimiter='')

def find_satisfying_graph(matrix, target_matrix, inverted_matrix, target_rows):
    original_matrix_number_of_rows = matrix.shape[0]
    total_combinations = comb(original_matrix_number_of_rows, target_rows)
    progress_bar = tqdm(total=total_combinations, desc="Finding satisfying graph")
    
    for indices in generate_combinations(len(matrix), target_rows):
        progress_bar.update(1)  # 進捗を更新
        if check_combination(matrix, indices, target_matrix):
            progress_bar.close()  # 進捗バーを閉じる
            return "target_matrix", indices
        elif check_combination(matrix, indices, inverted_matrix):
            progress_bar.close()  # 進捗バーを閉じる
            return "inverted_matrix", indices
    progress_bar.close()  # 進捗バーを閉じる
    return False, 0

def main():
    file_path = 'adjcencyMatrix/K9_9-I.txt'
    original_matrix = read_adjacency_matrix(file_path)

    first_target_path = 'targetAdjcencyMatrix/B6.txt'
    first_target_matrix = read_adjacency_matrix(first_target_path)
    first_inverted_matrix = 1 - first_target_matrix
    np.fill_diagonal(first_inverted_matrix, 0)
    first_target_rows = first_target_matrix.shape[0]

    second_target_path = 'targetAdjcencyMatrix/B3.txt'
    second_target_matrix = read_adjacency_matrix(second_target_path)
    second_inverted_matrix = 1 - second_target_matrix
    np.fill_diagonal(second_inverted_matrix, 0)
    second_target_rows = second_target_matrix.shape[0]
    
    satisfying_graph_type1, retIndices = find_satisfying_graph(original_matrix, first_target_matrix, first_inverted_matrix, first_target_rows)
    satisfying_graph_type2, retIndices2 = find_satisfying_graph(original_matrix, second_target_matrix, second_inverted_matrix, second_target_rows)

    if satisfying_graph_type1 == "target_matrix":
        print(f"{file_path}には{first_target_path}が見つかりました")
        print("first_graphにて条件を満たすグラフが見つかりました (first_target_matrix)")
        print(f"見つかったグラフのインデックス: {retIndices}")
    elif satisfying_graph_type1 == "inverted_matrix":
        print("first_graphにて条件を満たすグラフが見つかりました (first_inverted_matrix)")
        print(f"見つかったグラフのインデックス: {retIndices}")
    else:
        print(f"{file_path}には{first_target_path}は見つかりませんでした")
        
    if satisfying_graph_type2 == "target_matrix":
        print(f"{file_path}には{second_target_path}が見つかりました")
        print("second_graphにて条件を満たすグラフが見つかりました (second_target_matrix)")
        print(f"見つかったグラフのインデックス: {retIndices2}")
    elif satisfying_graph_type2 == "inverted_matrix":
        print("second_graphにて条件を満たすグラフが見つかりました (second_inverted_matrix)")
        print(f"見つかったグラフのインデックス: {retIndices2}")
    else:
        print(f"{file_path}には{second_target_path}は見つかりませんでした")
        

if __name__ == "__main__":
    main()
