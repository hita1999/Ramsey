import random

def steepest_descent_tabu_search(V, r, L):
    # Set elements of V to random colors
    for i in range(len(V)):
        V[i] = random.randint(1, r)
    
    H = []
    while True:
        W = []
        for i in range(len(V)):
            for c in range(1, r+1):
                if (i, c) not in H:
                    Vi_c = V.copy()
                    Vi_c[i] = c
                    score = compute_score(Vi_c, r)
                    W.append((Vi_c, score))
        
        M = [Vi_c for Vi_c, score in W if score == min([score for Vi_c, score in W])]
        if not M:
            break
        
        # Randomly choose new V in M
        Vi_c = random.choice(M)
        H.append((Vi_c.index(0), 0))
        H.append((Vi_c.index(1), 1))
        H.append((Vi_c.index(2), 2))
        if len(H) > L:
            H.pop(0)
    
    return V

def compute_score(V, r):
    score = 0
    for c in range(1, r+1):
        fc = 0
        for i in range(len(V)):
            if V[i] == c:
                for j in range(i+1, len(V)):
                    if V[j] == c:
                        fc += 1
                        if fc > score:
                            score = fc
    return score


def main():
    # Specify the number of vertices and the number of colors for the graph
    num_vertices = 10  # Change to an appropriate value
    num_colors = 3     # Change to an appropriate value
    tabu_list_size = 5 # Change to an appropriate value

    # Create the graph's vertices and compute the Ramsey number
    graph = [0] * num_vertices
    result = steepest_descent_tabu_search(graph, num_colors, tabu_list_size)
    print("Ramsey Number Computation Result:", compute_score(result, num_colors))

if __name__ == "__main__":
    main()
