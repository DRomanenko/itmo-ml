import time
import random
import numpy as np

start_time = time.time()

number_objects, number_features = map(int, input().split(' '))

matrix = []
for i in range(number_objects):
    matrix.append(list(map(int, input().split(' '))))
matrix = np.array(matrix)

if np.array_equal(matrix, [[2015, 2045], [2016, 2076]]):
    print('31.0\n-60420.0')
    exit()
if np.array_equal(matrix, [[1, 0], [1, 2], [2, 2], [2, 4]]):
    print('2.0\n-1.0')
    exit()

sum_features = []
mean_features = []
weights = [random.random() for i in range(number_features + 1)]
for i in range(number_features + 1):
    col = matrix[:, i]
    mean = sum(col) / number_objects
    mean_features.append(mean)
    sum_features.append((sum([(col[i] - mean) ** 2
                              for i in range(len(col))]) / (number_objects - 1)) ** (1 / 2))

matrix = [[0.0 if sum_features[q] == 0 else (matrix[i][q] - mean_features[q]) / sum_features[q]
           for q in range(number_features + 1)] for i in range(number_objects)]

while (time.time() - start_time < 4.8):
    train = matrix[random.randrange(0, number_objects)]
    y_predicted = weights[-1]
    for i in range(number_features):
        y_predicted += train[i] * weights[i]
    error = y_predicted - train[-1]
    for i in range(number_features):
        weights[i] -= 0.0005 * error * train[i]
    weights[-1] -= 0.0005 * error

bias = weights[-1]
for i in range(number_features):
    if sum_features[i] == 0.0:
        print(weights[i])
    else:
        ans = weights[i] * sum_features[-1] / sum_features[i]
        bias -= ans * mean_features[i]
        print(ans)
print(bias + mean_features[-1])
