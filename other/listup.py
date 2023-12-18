import itertools

total_columns = 18
combinations = list(itertools.combinations(range(total_columns), 5))

# 生成された組み合わせを表示
for combination in combinations:
    if 0 in combination and 1 in combination:
        print(combination)
print(len(combinations))