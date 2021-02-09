import numpy as np

k = int(input())
number_objects = int(input())

expected_square_value = 0
square_expected_value = 0
qty = np.zeros((k, 2), dtype=float)
matrix = np.zeros((number_objects, 2), dtype=int)
for i in range(number_objects):
    matrix[i] = input().split(' ')
    qty[matrix[i][0] - 1][0] += matrix[i][1] / number_objects
    qty[matrix[i][0] - 1][1] += 1 / number_objects
    expected_square_value += matrix[i][1] / number_objects * matrix[i][1]
square_expected_value = sum([0 if qty[i][1] == 0 else qty[i][0] / qty[i][1] * qty[i][0] for i in range(k)])
print(expected_square_value - square_expected_value)
