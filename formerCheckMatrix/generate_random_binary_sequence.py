import random

def generate_random_binary_sequence(binary_sequence):
    shuffled_sequence = list(binary_sequence)
    random.shuffle(shuffled_sequence)
    return ''.join(shuffled_sequence)

if __name__ == "__main__":
    # 01の数列
    binary_sequence = "1010101010101010"

    # ランダムに入れ替えた01の数列を生成
    randomized_sequence = generate_random_binary_sequence(binary_sequence)

    print("Original sequence:", binary_sequence)
    print("Randomized sequence:", randomized_sequence)
