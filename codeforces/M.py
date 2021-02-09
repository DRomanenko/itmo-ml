import numpy as np


def calc_rang(features):
    indexes = [(val, index) for index, val in enumerate(features)]
    indexes.sort()
    rangs = np.zeros(len(features))
    rangs[indexes[0][1]] = 0
    cur_rang = 0
    for q in range(1, len(features)):
        if indexes[q - 1][0] != indexes[q][0]:
            cur_rang += 1
        rangs[indexes[q][1]] = cur_rang
    return rangs


number_objects = int(input())
matrix = np.zeros((number_objects, 2), dtype=float)
for i in range(number_objects):
    matrix[i] = input().split(' ')

rang_x = calc_rang(matrix[:, 0])
rang_y = calc_rang(matrix[:, 1])

print(0 if number_objects < 2 else 1 - 6 * sum([np.square(rang_x[i] - rang_y[i]) for i in range(number_objects)]) / (
        number_objects ** 3 - number_objects))
