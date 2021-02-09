import numpy as np
import math as m


# Metrics
def distance_euclidean(row1, row2):
    distance = 0.0
    for i in range(len(row2)):
        distance += (row1[i] - row2[i]) ** 2
    return np.sqrt(distance)


def distance_manhattan(row1, row2):
    distance = 0.0
    for i in range(len(row2)):
        distance += abs(row1[i] - row2[i])
    return distance


def distance_chebyshev(row1, row2):
    distance = 0.0
    for i in range(len(row2)):
        distance = max(distance, abs(row1[i] - row2[i]))
    return distance


metrics = {
    'euclidean': distance_euclidean,
    'chebyshev': distance_chebyshev,
    'manhattan': distance_manhattan,
}


def safe_division(a, b):
    return 0 if b == 0 else a / b


# Kernels
def kernel_uniform(u):
    return 0.5 if abs(u) < 1 else 0.0


def kernel_triangular(u):
    return max(0, 1 - abs(u))


def kernel_epanechnikov(u):
    return max(0, 0.75 * (1 - u * u))


def kernel_quartic(u):
    return max(0, 15 / 16 * ((1 - u * u) ** 2))


def kernel_triweight(u):
    return max(0, 35 / 32 * (1 - u * u) ** 3)


def kernel_tricube(u):
    return max(0, 70 / 81 * (1 - abs(u) ** 3) ** 3)


def kernel_gaussian(u):
    return 1 / m.sqrt(2 * m.pi) * m.e ** (-0.5 * u * u)


def kernel_cosine(u):
    return max(0.0, m.pi / 4 * m.cos(m.pi / 2 * u))


def kernel_logistic(u):
    return 1 / (m.e ** u + 2 + m.e ** (-u))


def kernel_sigmoid(u):
    return (2 / m.pi) * (1 / (m.e ** u + m.e ** (-u)))


kernels = {
    'uniform': kernel_uniform,
    'triangular': kernel_triangular,
    'epanechnikov': kernel_epanechnikov,
    'quartic': kernel_quartic,
    'triweight': kernel_triweight,
    'tricube': kernel_tricube,
    'gaussian': kernel_gaussian,
    'cosine': kernel_cosine,
    'logistic': kernel_logistic,
    'sigmoid': kernel_sigmoid
}

number_objects, number_features = map(int, input().split(' '))

matrix = []
for i in range(number_objects):
    matrix.append(list(map(int, input().split(' '))))

query = list(map(int, input().split(' ')))
metric_name = input()
kernel_name = input()
window_name = input()
h = int(input())

calc_metric = []
for row in matrix:
    calc_metric.append((metrics[metric_name](row[:-1], query), row[-1]))

calc_metric = sorted(calc_metric, key=lambda x: x[0])

res_kernels = 0
mean_kernels = 0
for cur_res_metric in calc_metric:
    res_kernel = 0 if (h == 0 or (window_name == 'variable' and calc_metric[h][0] == 0)) and cur_res_metric[0] != 0 \
        else kernels[kernel_name](safe_division(cur_res_metric[0], calc_metric[h][0]) if window_name == 'variable'
                                  else safe_division(cur_res_metric[0], h))
    res_kernels += res_kernel
    mean_kernels += res_kernel * cur_res_metric[1]

if res_kernels != 0:
    print(mean_kernels / res_kernels)
else:
    print(sum([row[-1] for row in matrix]) / number_objects)
