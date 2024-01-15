import itertools
from math import comb
import numpy as np
from tqdm import tqdm
from numba import jit


def read_adjacency_matrix(file_path):
    adjacency_matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [int(x) for x in line.strip()]
            adjacency_matrix.append(row)
    return np.array(adjacency_matrix)


def generate_combinations(elements, size):
    return itertools.combinations(elements, size)



def find_satisfying_graph(original_matrix, target_size, condition):
    original_matrix_number_of_rows = original_matrix.shape[0]
    total_combinations = comb(original_matrix_number_of_rows, 2) * comb(original_matrix_number_of_rows - 2, target_size)

    progress_bar = tqdm(total=total_combinations, desc="Finding satisfying graph")

    for spine_indices in generate_combinations(range(len(original_matrix)), 2):
        if original_matrix[np.ix_(spine_indices, spine_indices)].sum() == condition:
            remaining_indices = set(range(len(original_matrix))) - set(spine_indices)
            
            for page_indices in generate_combinations(remaining_indices, target_size):
                progress_bar.update(1)
                if set(spine_indices).isdisjoint(set(page_indices)):
                    if original_matrix[np.ix_(spine_indices, page_indices)].sum() == len(page_indices) * condition:
                        progress_bar.close()
                        return 1, spine_indices, page_indices
                else:
                    # Update progress bar even if conditions are not met
                    progress_bar.update(1)
        else:
            progress_bar.update(comb(original_matrix_number_of_rows - 2, target_size))
    progress_bar.close()
    return 0, 0, 0


def print_results(file_path, target_path, result, ret_spine_indices, ret_page_indices, graph_type):
    if result == 1:
        print(f"{file_path}には{target_path}が見つかりました")
        print(f"{graph_type}_graphにて条件を満たすグラフが見つかりました ({target_path})")
        print("page_indices:", ret_page_indices)
        print("spine_indices:", ret_spine_indices)
        print("for drawing graph, please run drawGraph.py", ret_page_indices + ret_spine_indices)
    else:
        print(f"{file_path}には{target_path}が見つかりませんでした")
        

def flip_matrix(matrix):
    new_matrix = 1 - matrix
    np.fill_diagonal(new_matrix, 0)
    return new_matrix


def main():
    file_path = 'adjcencyMatrix/Paley/Paley9_double18.txt'
    original_matrix = read_adjacency_matrix(file_path)


    #original_matrix = flip_matrix(original_matrix)
    print("matrix", original_matrix)
    first_target_size = int(input("first_target_book: "))
    second_target_size = int(input("second_target_book: "))

    first_target_path = f'targetAdjcencyMatrix/B{first_target_size}.txt'
    second_target_path = f'targetAdjcencyMatrix/B{second_target_size}.txt'

    result1, ret_spine_indices1, ret_page_indices1 = find_satisfying_graph(original_matrix, first_target_size, 2)
    result2, ret_spine_indices2, ret_page_indices2 = find_satisfying_graph(original_matrix, second_target_size, 0)

    print_results(file_path, first_target_path, result1, ret_spine_indices1, ret_page_indices1, "first")
    print_results(file_path, second_target_path, result2, ret_spine_indices2, ret_page_indices2, "second")


if __name__ == "__main__":
    main()
