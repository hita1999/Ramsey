import itertools

def find_clique(adj_matrix, k):
    n = len(adj_matrix)
    for clique_size in range(k, n + 1):
        for combination in itertools.combinations(range(n), clique_size):
            is_clique = True
            for i, node1 in enumerate(combination):
                for j, node2 in enumerate(combination):
                    if i < j and not adj_matrix[node1][node2]:
                        is_clique = False
                        break
                if not is_clique:
                    break
            if is_clique:
                return combination
    return None

# Example adjacency matrix (0 represents no edge, 1 represents an edge)
adj_matrix = [
    [0, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 0]
]

k = 4
clique = find_clique(adj_matrix, k)
if clique:
    print(f"Clique of size {k}: {clique}")
else:
    print(f"No clique of size {k} found.")
