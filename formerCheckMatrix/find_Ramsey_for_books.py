import itertools
import random
import numpy as np


def euclidean_distance(matrix1, matrix2):
    flattened_matrix1 = matrix1.flatten()
    flattened_matrix2 = matrix2.flatten()
    distance = np.linalg.norm(flattened_matrix1 - flattened_matrix2)
    return distance


def read_adjacency_matrix(file_path):
    adjacency_matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [int(x) for x in line.strip()]
            adjacency_matrix.append(row)
    return np.array(adjacency_matrix)


def check_combination(matrix, indices, target_matrix):
    submatrix = matrix[np.ix_(indices, indices)]
    return np.array_equal(submatrix, target_matrix) or np.array_equal(submatrix, 1 - target_matrix)


def generate_combinations(total, size):
    return itertools.combinations(range(total), size)


def save_matrix_to_txt(matrix, file_path):
    np.savetxt(file_path, matrix, fmt='%d', delimiter='')


def check_combinations(original_matrix, target_matrix, inverted_matrix, target_rows, bound_distance):
    count = 0
    inverted_count = 0

    indices_combinations = generate_combinations(len(original_matrix), target_rows)
    
    for idx, indices in enumerate(indices_combinations):
        # if idx % 1000 == 0:
        #     print(f"組み合わせ {idx}番目")

        submatrix = original_matrix[np.ix_(indices, indices)]
        distance = euclidean_distance(submatrix, target_matrix)
        inverted_distance = euclidean_distance(submatrix, inverted_matrix)

        if distance <= bound_distance:
            count += 1

        if inverted_distance <= bound_distance:
            inverted_count += 1

    print(f"distance {bound_distance}以下のカウント数", count)
    print(f"inverted distance {bound_distance}以下のカウント数", inverted_count)
    
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
    
    bound_distance = 0
    count_sum = 1
    
    while count_sum > 0:
        idx = random.randint(0, len(original_matrix)-1)
        new_sequence = generate_random_binary_sequence(len(original_matrix[idx][idx+1:]))
        new_matrix = swap_rows_and_columns(original_matrix, new_sequence)
        #print(new_matrix)
        fisrt_count, first_inverted_count = check_combinations(new_matrix, first_target_matrix, first_inverted_matrix, first_target_rows, bound_distance)
        first_count_sum = min(fisrt_count, first_inverted_count)
        
        if first_count_sum == 0:
            second_count, second_inverted_count = check_combinations(new_matrix, second_target_matrix, second_inverted_matrix, second_target_rows, bound_distance)
            
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
