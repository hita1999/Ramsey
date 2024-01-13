import itertools

elements = range(18)
combination_size = 5

for combination in itertools.combinations(elements, combination_size):
    print(combination)
