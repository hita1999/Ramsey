import itertools
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
        if idx % 1000 == 0:
            print(f"組み合わせ {idx}番目")

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
    file_path = 'R(B3_B6_18).txt'
    original_matrix = read_adjacency_matrix(file_path)

    target_path = 'adjcencyMatrix/B6.txt'
    target_matrix = read_adjacency_matrix(target_path)

    inverted_matrix = 1 - target_matrix
    np.fill_diagonal(inverted_matrix, 0)

    target_rows = target_matrix.shape[0]
    bound_distance = 0
    count_sum = 1
    idx = 17
    while count_sum > 0:
        new_sequence = generate_random_binary_sequence(len(original_matrix[idx][idx+1:]))
        new_matrix = swap_rows_and_columns(original_matrix, new_sequence)
        print(new_matrix)
        count, inverted_count = check_combinations(new_matrix, target_matrix, inverted_matrix, target_rows, bound_distance)
        count_sum = min(count, inverted_count)
        
    save_matrix_to_txt(new_matrix, 'R(B3_B6_18).txt')

if __name__ == "__main__":
    main()
