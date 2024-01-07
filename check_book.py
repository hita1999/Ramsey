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

def generate_combinations(total, size):
    return itertools.combinations(range(total), size)

def search_book(matrix, page, spine):
    if matrix[np.ix_(page, page)].sum() == 0:
    
        if matrix[np.ix_(spine, spine)].sum() == 2:
            if matrix[np.ix_(spine, page)].sum() == len(page)*2:
                print("target_matrix found", page, spine)
                return True

def find_satisfying_graph(original_matrix, target_size):
    original_matrix_number_of_rows = original_matrix.shape[0]
    total_combinations = comb(original_matrix_number_of_rows, target_size) * comb(original_matrix_number_of_rows-target_size, 2)
    
    progress_bar = tqdm(total=total_combinations, desc="Finding satisfying graph")
    
    for page_indices in generate_combinations(len(original_matrix), target_size):
        remaining_indices = set(range(len(original_matrix))) - set(page_indices)
        
        for spine_indices in generate_combinations(len(remaining_indices), 2):
            progress_bar.update(1)  # 進捗を更新
            if search_book(original_matrix, page_indices, spine_indices):
                progress_bar.close()  # 進捗バーを閉じる
                return 1, spine_indices, page_indices
        progress_bar.close()  # 進捗バーを閉じる
    return 0, 0, 0

def main():
    file_path = 'adjcencyMatrix/Paley/Paley17.txt'
    original_matrix = read_adjacency_matrix(file_path)
    
    print("original_matrix")
    first_target_size = int(input("first_target_book: "))
    second_target_size = int(input("second_target_book: "))
    
    first_target_path = 'targetAdjcencyMatrix/B' + str(first_target_size) + '.txt'
    second_target_path = 'targetAdjcencyMatrix/B' + str(second_target_size) + '.txt'
    
    result1, ret_spine_indices1, ret_page_indices1 = find_satisfying_graph(original_matrix, first_target_size)
    
    result2, ret_spine_indices2, ret_page_indices2 = find_satisfying_graph(original_matrix, second_target_size)
    
    if result1 == 1:
        print(f"{file_path}には{first_target_path}が見つかりました")
        print("first_graphにて条件を満たすグラフが見つかりました (first_target_matrix)")
        print("page_indices:", ret_page_indices1)
        print("spine_indices:", ret_spine_indices1)
        print("for drawing graph, please run drawGraph.py", ret_page_indices1+ret_spine_indices1)
        
    if result2 == 1:
        print(f"{file_path}には{second_target_path}が見つかりました")
        print("second_graphにて条件を満たすグラフが見つかりました (second_target_matrix)")
        print("page_indices:", ret_page_indices2)
        print("spine_indices:", ret_spine_indices2)
        print("for drawing graph, please run drawGraph.py", ret_page_indices2+ret_spine_indices2)


if __name__ == "__main__":
    main()