import itertools
import numpy as np
import multiprocessing

def read_adjacency_matrix(file_path):
    adjacency_matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [int(x) for x in line.strip()]
            adjacency_matrix.append(row)
    return np.array(adjacency_matrix)

def generate_combinations(total, size):
    for indices in itertools.combinations(range(total), size):
        yield indices

def check_combination(args):
    original_matrix, target_matrix, inverted_matrix, target_rows, bound_distance = args
    count = 0
    inverted_count = 0
    
    indices_combinations = generate_combinations(len(original_matrix), target_rows)
    
    for idx, indices in enumerate(indices_combinations):
        submatrix = original_matrix[np.ix_(indices, indices)]
        
        if np.array_equal(submatrix, target_matrix):
            count += 1
        elif np.array_equal(submatrix, inverted_matrix):
            inverted_count += 1

    return count, inverted_count

def main():
    file_path = 'adjcencyMatrix/T8.txt'
    original_matrix = read_adjacency_matrix(file_path)

    target_path = 'targetAdjcencyMatrix/B2.txt'
    target_matrix = read_adjacency_matrix(target_path)

    inverted_matrix = 1 - target_matrix
    np.fill_diagonal(inverted_matrix, 0)

    # インデックスの組み合わせを生成
    target_rows = target_matrix.shape[0]

    # CPUのコア数を取得
    num_cores = multiprocessing.cpu_count()

    # 引数リストを生成
    args_list = [(original_matrix, target_matrix, inverted_matrix, target_rows, 0) for _ in range(num_cores)]

    # プロセスプールを作成
    with multiprocessing.Pool(processes=num_cores) as pool:
        # 各プロセスに引数リストを渡して処理を実行
        results = pool.map(check_combination, args_list)

    # 結果を合算
    total_count = sum(result[0] for result in results)
    total_inverted_count = sum(result[1] for result in results)

    print(f"distance 0以下のカウント数: {total_count}")
    print(f"inverted distance 0以下のカウント数: {total_inverted_count}")

if __name__ == "__main__":
    main()
