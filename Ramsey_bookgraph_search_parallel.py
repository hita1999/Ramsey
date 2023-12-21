import itertools
import random
import numpy as np
from multiprocessing import Pool, cpu_count


def read_adjacency_matrix(file_path):
    adjacency_matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [int(x) for x in line.strip()]
            adjacency_matrix.append(row)
    return np.array(adjacency_matrix)


def check_combination(args):
    matrix, indices, target_matrix, inverted_matrix = args
    submatrix = matrix[np.ix_(indices, indices)]
    return (np.array_equal(submatrix, target_matrix), np.array_equal(submatrix, inverted_matrix))


def generate_combinations(total, size):
    return itertools.combinations(range(total), size)


def save_matrix_to_txt(matrix, file_path):
    np.savetxt(file_path, matrix, fmt='%d', delimiter='')


def check_combinations_parallel(original_matrix, target_matrix, inverted_matrix, target_rows):
    indices_combinations = generate_combinations(len(original_matrix), target_rows)

    # Use multiprocessing Pool for parallel processing
    with Pool(processes=cpu_count()) as pool:
        args = [(original_matrix, indices, target_matrix, inverted_matrix) for indices in indices_combinations]
        results = pool.map(check_combination, args)

    count = sum(result[0] for result in results)
    inverted_count = sum(result[1] for result in results)

    print(f"一致する組み合わせのカウント数", count)
    print(f"反転した組み合わせのカウント数", inverted_count)

    return count, inverted_count


def generate_random_binary_sequence(binary_sequence_length):
    random_sequence = np.random.choice([0, 1], size=binary_sequence_length)
    print(random_sequence)
    return random_sequence


def swap_rows_and_columns(original_matrix, random_sequence):
    idx = len(random_sequence)
    original_length = original_matrix.shape[0]

    original_matrix[original_length-idx-1, original_length-idx:] = random_sequence
    original_matrix[original_length-idx:, original_length-idx-1] = random_sequence

    return original_matrix


def main():
    file_path = 'adjcencyMatrix/K9_9-I.txt'
    original_matrix = read_adjacency_matrix(file_path)

    first_target_path = 'adjcencyMatrix/B6.txt'
    first_target_matrix = read_adjacency_matrix(first_target_path)
    first_inverted_matrix = 1 - first_target_matrix
    np.fill_diagonal(first_inverted_matrix, 0)
    first_target_rows = first_target_matrix.shape[0]

    second_target_path = 'adjcencyMatrix/B3.txt'
    second_target_matrix = read_adjacency_matrix(second_target_path)
    second_inverted_matrix = 1 - second_target_matrix
    np.fill_diagonal(second_inverted_matrix, 0)
    second_target_rows = second_target_matrix.shape[0]

    count_sum = 1

    while count_sum > 0:
        idx = random.randint(0, len(original_matrix)-1)
        new_sequence = generate_random_binary_sequence(len(original_matrix[idx][idx+1:]))
        new_matrix = swap_rows_and_columns(original_matrix, new_sequence)
        fisrt_count, first_inverted_count = check_combinations_parallel(new_matrix, first_target_matrix, first_inverted_matrix, first_target_rows)
        first_count_sum = min(fisrt_count, first_inverted_count)

        if first_count_sum == 0:
            second_count, second_inverted_count = check_combinations_parallel(new_matrix, second_target_matrix, second_inverted_matrix, second_target_rows)

            if second_count == 0 and second_inverted_count == 0:
                print("条件を満たすグラフが見つかりました")
                break
            else:
                print("条件を満たすグラフは見つかりませんでした")
                original_matrix = new_matrix
                continue

    save_matrix_to_txt(new_matrix, 'R(B3_B6_18).txt')


if __name__ == "__main__":
    main()
