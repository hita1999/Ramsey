import numpy as np
from scipy.linalg import circulant

def generate_circulant_adjacency(n):
    arr = np.random.randint(2, size=n)
    # Generate a circulant matrix with 1s and 0s
    circulant_matrix = circulant( arr)

    # Make the circulant matrix symmetric
    circulant_matrix = np.triu(circulant_matrix) + np.triu(circulant_matrix, 1).T

    return circulant_matrix

# Example usage
n = 18  # Adjust the size of the matrix as needed
adjacency_matrix = generate_circulant_adjacency(n)
print("Circulant and Adjacency Matrix:")
print(adjacency_matrix)
