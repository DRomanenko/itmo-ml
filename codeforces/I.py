import math
import numpy as np


def print_node(node):
    print('\n'.join(map(lambda x: ' '.join(map(str, x)), node)))


def tnh(matrix, type=True):
    return np.array([[math.tanh(matrix[i][q]) if type else 1 / (math.cosh(matrix[i][q]) ** 2)
                      for q in range(len(matrix[0]))]
                     for i in range(len(matrix))])


def rlu(matrix, alpha, type=True):
    return np.array([[(matrix[i][q] if type else 1.0) / (alpha if matrix[i][q] < 0 else 1.0)
                      for q in range(len(matrix[0]))]
                     for i in range(len(matrix))])


def mul(matrix1, matrix2):
    return matrix1 @ matrix2


def summ(matrixes):
    matrixes = np.array(matrixes)
    return np.array([[matrixes[:, i, q].sum() for q in range(len(matrixes[0][0]))]
                     for i in range(len(matrixes[0]))])


def had(matrixes):
    matrixes = np.array(matrixes)
    return np.array([[matrixes[:, i, q].prod() for q in range(len(matrixes[0][0]))]
                     for i in range(len(matrixes[0]))])


def calc_on_type(nodes_description, nodes, i=None, nodes_diff=None):
    node_type = nodes_description[0]
    if node_type == 'tnh':
        index = nodes_description[1][0] - 1
        if i is None:
            return tnh(nodes[index])
        else:
            nodes_diff[index] = summ([nodes_diff[index],
                                      had([tnh(nodes[index], False), nodes_diff[i]])])
    elif node_type == 'rlu':
        index = nodes_description[1][1] - 1
        if i is None:
            return rlu(nodes[index], nodes_description[1][0])
        else:
            nodes_diff[index] = summ([nodes_diff[index],
                                      had([rlu(nodes[index], nodes_description[1][0], False), nodes_diff[i]])])
    elif node_type == 'mul':
        index1, index2 = nodes_description[1][0] - 1, nodes_description[1][1] - 1
        if i is None:
            return mul(nodes[index1], nodes[index2])
        else:
            nodes_diff[index1] = summ([nodes_diff[index1],
                                       mul(nodes_diff[i], nodes[index2].transpose())])
            nodes_diff[index2] = summ([nodes_diff[index2],
                                       mul(nodes[index1].transpose(), nodes_diff[i])])
    elif node_type == 'sum':
        lt = nodes_description[1][1:]
        if i is None:
            return summ([nodes[i - 1] for i in lt])
        else:
            for index in lt: nodes_diff[index - 1] = summ([nodes_diff[index - 1], nodes_diff[i]])
    elif node_type == 'had':
        lt = nodes_description[1][1:]
        if i is None:
            return had([nodes[i - 1] for i in lt])
        else:
            if len(lt) < 2:
                nodes_diff[lt[0] - 1] = summ([nodes_diff[lt[0] - 1], nodes_diff[i]])
            else:
                for num, index in enumerate(lt, 0):
                    lt = np.concatenate((lt[:num], lt[num + 1:]))
                    nodes_diff[index - 1] = summ(
                        [nodes_diff[index - 1], had([nodes[cur_index - 1] for cur_index in lt] + [nodes_diff[i]])])
    return None


n, m, k = map(int, input().split(' '))
nodes_description = []
for i in range(n):
    line = input().split()
    nodes_description.append([line[0], np.array(list(map(int, line[1:])))])

nodes = []
nodes_diff = []
for i in range(m):
    size = nodes_description[i][1][0]
    nodes.append([])
    nodes_diff.append(np.array([np.zeros(nodes_description[i][1][1]) for _ in range(size)]))
    for row in range(size):
        nodes[i].append(np.array(list(map(int, input().split()))))
    nodes[i] = np.array(nodes[i])

for i in range((n - m - k) + 1, n):
    nodes.append(calc_on_type(nodes_description[i], nodes))
    nodes_diff.append(np.array([np.zeros(len(nodes[-1][0])) for _ in range(len(nodes[-1]))] if i < n - k
                               else [list(map(float, input().split())) for _ in range(len(nodes[-1]))]))

for i in range(n - 1, m - 1, -1):
    calc_on_type(nodes_description[i], nodes, i, nodes_diff)

for i in range(k):
    print_node(nodes[n - 1 - i])

for i in range(m):
    print_node(nodes_diff[i])
