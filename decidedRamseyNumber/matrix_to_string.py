def read_matrix_from_file(file_path):
    with open(file_path, 'r') as file:
        matrix = []
        for line in file:
            line = line.strip()
            row = []
            for char in line:
                row.append(int(char))
            matrix.append(row)
    return matrix

def matrix_to_string(matrix):
    rows = []
    for row in matrix:
        row_str = ','.join(str(x) for x in row) + ','
        rows.append(row_str)
    return '\n'.join(rows)


file_path = 'decidedRamseyNumber/R(B6_B8_28).txt'
matrix = read_matrix_from_file(file_path)
formatted_matrix = matrix_to_string(matrix)
print(formatted_matrix)