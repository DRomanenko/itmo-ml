import numpy as np

number_objects = int(input())
matrix = np.zeros((number_objects, 2))
for i in range(number_objects):
    matrix[i] = input().split(' ')

sum_cols = matrix.sum(axis=0) / number_objects
X = np.array([matrix[i][0] - sum_cols[0] for i in range(number_objects)])
Y = [matrix[i][1] - sum_cols[1] for i in range(number_objects)]
numerator = (X @ Y).sum()
denominator = np.sqrt(np.square(X).sum() * np.square(Y).sum())

print(0 if denominator == 0 else numerator / denominator)
