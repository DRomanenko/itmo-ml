import numpy as np

# Metrics
def distance_euclidean(row1, row2):
    distance = 0.0
    for i in range(len(row2) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return np.sqrt(distance)


def distance_manhattan(row1, row2):
    distance = 0.0
    for i in range(len(row2) - 1):
        distance += abs(row1[i] - row2[i])
    return distance


def distance_chebyshev(row1, row2):
    distance = 0.0
    for i in range(len(row2) - 1):
        distance = max(distance, abs(row1[i] - row2[i]))
    return distance

metrics = {
    "euclidean": distance_euclidean,
    "chebyshev": distance_chebyshev,
    "manhattan": distance_manhattan,
}
