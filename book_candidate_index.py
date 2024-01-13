all_elements = set(range(2**9))

result1_set = set()
result2_set = set()

current_result_set = None

with open('searchResultTextFile/add1vertex_Paley9_B2_B3.txt', 'r') as file:
    for line in file:
        line = line.strip()
        if line.startswith('Result 1 B2:'):
            current_result_set = result1_set
        elif line.startswith('Result 2 B3:'):
            current_result_set = result2_set
        else:
            values = line.split(',')
            if len(values) > 0:
                index_value = values[0].strip()
                if index_value.isdigit():
                    current_result_set.add(int(index_value))

print("Result 1 Set:", result1_set)
print("True Count 1:", len(result1_set))

print("\nResult 2 Set:", result2_set)
print("True Count 2:", len(result2_set))

# set1およびset2に属さない要素を抽出
not_in_either = all_elements - (result1_set.union(result2_set))

print(not_in_either)


