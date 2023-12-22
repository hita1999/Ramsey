import itertools
from math import comb
import numpy as np
from multiprocessing import Pool, Manager
from tqdm import tqdm

def read_adjacency_matrix(file_path):
    adjacency_matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [int(x) for x in line.strip()]
            adjacency_matrix.append(row)
    return np.array(adjacency_matrix)

def generate_matrix_candidates(total, size):
    for indices in itertools.combinations(range(total), size):
        yield indices

def check_combination(matrix, indices, target_matrix):
    submatrix = matrix[np.ix_(indices, indices)]
    return np.array_equal(submatrix, target_matrix)

def find_satisfying_graph(args):
    matrix, target_matrix, target_rows, progress_queue = args

    total_combinations = comb(len(matrix), target_rows)
    progress_bar = tqdm(total=total_combinations, desc="Checking Matrix Candidates")

    count = 0  # 進捗を数える変数を追加

    for indices in generate_matrix_candidates(len(matrix), target_rows):
        if check_combination(matrix, indices, target_matrix):
            count += 1  # 進捗をインクリメント
        progress_bar.update(1)  # 進捗バーを更新

    progress_bar.close()  # 進捗バーを閉じる

    # 進捗をキューに格納
    progress_queue.put(count)

    if count > 0:
        return "target_matrix"
    else:
        return False

def main():
    file_path = 'adjcencyMatrix/T8.txt'
    original_matrix = read_adjacency_matrix(file_path)

    first_target_path = 'targetAdjcencyMatrix/B7.txt'
    first_target_matrix = read_adjacency_matrix(first_target_path)
    first_target_rows = first_target_matrix.shape[0]

    second_target_path = 'targetAdjcencyMatrix/B6.txt'
    second_target_matrix = read_adjacency_matrix(second_target_path)
    second_target_rows = second_target_matrix.shape[0]

    with Manager() as manager:
        progress_queue = manager.Queue()  # マネージャーを使用して進捗を格納するキューを生成

        args_list = [
            (original_matrix, first_target_matrix, first_target_rows, progress_queue),
            (original_matrix, second_target_matrix, second_target_rows, progress_queue)
        ]

        with Pool(processes=24) as pool:
            results = list(pool.imap_unordered(find_satisfying_graph, args_list))

        # 各プロセスの進捗を取得
        counts = [progress_queue.get() for _ in range(len(args_list))]

    for idx, (satisfying_graph_type, count) in enumerate(zip(results, counts)):
        if satisfying_graph_type == "target_matrix":
            print(f"{file_path}には{first_target_path if idx == 0 else second_target_path}が見つかりました")
            print(f"{idx + 1}_graphにて条件を満たすグラフが見つかりました ({'first' if idx == 0 else 'second'}_target_matrix)")
            print(f"進捗: {count} / {comb(len(original_matrix), first_target_rows)}")
        else:
            print(f"{file_path}には{first_target_path if idx == 0 else second_target_path}は見つかりませんでした")

if __name__ == "__main__":
    main()
