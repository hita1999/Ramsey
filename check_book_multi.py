import cProfile
import itertools
from math import comb
from multiprocessing import Pool
import pstats
import numpy as np
from tqdm import tqdm


def read_adjacency_matrix(file_path):
    adjacency_matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [int(x) for x in line.strip()]
            adjacency_matrix.append(row)
    return np.array(adjacency_matrix)


def generate_combinations(elements, size):
    return itertools.combinations(elements, size)

def worker(args):
    spine_indices, original_matrix, target_size, condition = args
    if original_matrix[np.ix_(spine_indices, spine_indices)].sum() == condition:
        remaining_indices = set(range(len(original_matrix))) - set(spine_indices)

        for page_indices in generate_combinations(remaining_indices, target_size):
            if set(spine_indices).isdisjoint(set(page_indices)):
                if original_matrix[np.ix_(spine_indices, page_indices)].sum() == len(page_indices) * condition:
                    return 1, spine_indices, page_indices
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

def find_satisfying_graph_parallel(original_matrix, target_size, condition, args_list):
    with Pool() as pool, tqdm(total=len(args_list), desc="Finding satisfying graph") as pbar:
        results = list(tqdm(pool.imap_unordered(worker, [(args, original_matrix, target_size, condition) for args in args_list]), total=len(args_list)))

    for result, ret_spine_indices, ret_page_indices in results:
        if result == 1:
            return result, ret_spine_indices, ret_page_indices
        pbar.update(1)

    return 0, 0, 0


def main():
    file_path = 'generatedMatrix/circulantBlock/cir18B3B6/C1C2C3_1332.txt'
    original_matrix = read_adjacency_matrix(file_path)


    #original_matrix = flip_matrix(original_matrix)
    print("matrix", original_matrix)
    first_target_size = int(input("first_target_book: "))
    second_target_size = int(input("second_target_book: "))

    first_target_path = f'targetAdjcencyMatrix/B{first_target_size}.txt'
    second_target_path = f'targetAdjcencyMatrix/B{second_target_size}.txt'
    
    args_list = [spine_indices for spine_indices in generate_combinations(range(len(original_matrix)), 2)]

    result1, ret_spine_indices1, ret_page_indices1 = find_satisfying_graph_parallel(original_matrix, first_target_size, 2, args_list)
    result2, ret_spine_indices2, ret_page_indices2 = find_satisfying_graph_parallel(original_matrix, second_target_size, 0, args_list)

    print_results(file_path, first_target_path, result1, ret_spine_indices1, ret_page_indices1, "first")
    print_results(file_path, second_target_path, result2, ret_spine_indices2, ret_page_indices2, "second")


if __name__ == "__main__":
    profile_filename = f"res_profile/profile_check_B4B5_1_m1.txt"

    # プロファイリングを開始
    profile = cProfile.Profile()
    profile.enable()

    # 実際の処理を実行
    main()

    # プロファイリングを停止
    profile.disable()

    # プロファイリング結果を保存
    with open(profile_filename, "w") as f:
        stats = pstats.Stats(profile, stream=f)
        stats.sort_stats("cumulative")
        stats.print_stats()
