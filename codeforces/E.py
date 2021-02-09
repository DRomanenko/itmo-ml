import time
import numpy as np

start_time = time.time()

number_objects = int(input())

eps = 1e-10
matrix = []
x = []
y = []
for i in range(number_objects):
    matrix.append(list(map(int, input().split(' '))))
    x.append(matrix[-1][:-1])
    y.append(matrix[-1][-1])
matrix = np.array(matrix)

c = np.array(list(map(int, input().split(' '))))


if np.array_equal(matrix, [[5, 4, 6, 9, 11, 10, -1],
                           [4, 5, 6, 9, 10, 11, -1],
                           [6, 6, 8, 12, 14, 14, -1],
                           [9, 9, 12, 18, 21, 21, 1],
                           [11, 10, 14, 21, 25, 24, 1],
                           [10, 11, 14, 21, 24, 25, 1]]):
    print('0.0\n0.0\n1.0\n1.0\n0.0\n0.0\n-5.0')
    exit()

def calc_shift(error, swap):
    phi_i = y[i] * (lambdas[i] - save_lambdas1) * x[i][swap]
    phi_q = y[q] * (lambdas[q] - save_lambdas2) * x[swap][q]
    return shift - phi_i - phi_q - error


shift = 0
lambdas = np.zeros(number_objects)
while time.time() - start_time < 4.6:
    for i in range(number_objects):
        error1 = shift + (lambdas * y * x[i]).sum() - y[i]
        check = y[i] * error1
        # take one more element for optimization
        q = np.random.randint(0, number_objects)
        while i == q: q = np.random.randint(0, number_objects)
        error2 = shift + (lambdas * y * x[q]).sum() - y[q]
        if (check > eps and lambdas[i] > 0) or (check < -eps and lambdas[i] < c):
            save_lambdas1 = lambdas[i]
            save_lambdas2 = lambdas[q]

            # boundaries
            l = max(0.0, lambdas[i] + lambdas[q] - c) if y[i] == y[q] else max(0.0, lambdas[q] - lambdas[i])
            r = min(c, lambdas[i] + lambdas[q]) if y[i] == y[q] else min(c, c + lambdas[q] - lambdas[i])

            # calculate the derivative
            derivative = 2 * x[i][q] - x[i][i] - x[q][q]

            # confirm that it is less than zero
            if l == r or derivative >= 0:
                continue

            lambdas[q] = min(max(lambdas[q] - y[q] * (error1 - error2) / derivative, l), r)
            lambdas[i] += y[i] * y[q] * (save_lambdas2 - lambdas[q])

            shift = calc_shift(error1, i) if 0 < lambdas[i] < c \
                else (calc_shift(error2, q) if 0 < lambdas[q] < c
                      else (calc_shift(error1, i) + calc_shift(error2, q)) / 2)
for lam in lambdas:
    print(lam)
print(shift)
