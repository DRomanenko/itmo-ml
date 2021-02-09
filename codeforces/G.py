import collections as col


class Node:
    num = l = r = next_num = shift = None

    def __init__(self, *args):
        if len(args) == 5:
            self.num, self.l, self.r, self.next_num, self.shift = args
        if len(args) == 2:
            self.num, self.next_num = args

    def print(self):
        if self.l is None and self.r is None:
            print('C ' + str(self.next_num))
        else:
            print('Q ' + str(self.next_num + 1)
                  + ' ' + str(self.shift)
                  + ' ' + str(self.l.num)
                  + ' ' + str(self.r.num))
            self.l.print()
            self.r.print()


def build(depth, l, r):
    global num
    num += 1
    if depth > max_depth:
        return Node(num, col.Counter(list(map(lambda u: matrix[u][-1], range(l, r)))).most_common()[0][0])
    y_l = matrix[l][-1]
    check_class = True
    for i in range(l, r):
        if y_l != matrix[i][-1]:
            check_class = False
            break
    if check_class:
        return Node(num, y_l)

    best = [0, 1000, 0]
    for i in range(number_features):
        matrix[l:r] = sorted(matrix[l:r], key=lambda x: x[i])
        sum_l, qty_l = 0, [0 for _ in range(number_classes + 1)]
        sum_r, qty_r = 0, [0 for _ in range(number_classes + 1)]
        for q in range(l, r):
            sum_r += 2 * qty_r[matrix[q][-1]] + 1
            qty_r[matrix[q][-1]] += 1
        for q in range(l, r):
            if l != q:
                score = (1 - sum_l) / (q - l) + (1 - sum_r) / (r - q)
                if score < best[1]:
                    best = i, score, q
            sum_l += 2 * qty_l[matrix[q][-1]] + 1
            qty_l[matrix[q][-1]] += 1
            sum_r += -2 * qty_r[matrix[q][-1]] + 1
            qty_r[matrix[q][-1]] -= 1
    matrix[l:r] = sorted(matrix[l:r], key=lambda x: x[best[0]])
    if l == best[2] or r == best[2]:
        return Node(num, col.Counter(list(map(lambda u: matrix[u][-1], range(l, r)))).most_common()[0][0])
    else:
        return Node(num
                    , build(depth + 1, l, best[2])
                    , build(depth + 1, best[2], r)
                    , best[0], 1 / 2 * (matrix[best[2]][best[0]] + matrix[best[2] - 1][best[0]]))


num = 0
number_features, number_classes, max_depth = map(int, input().split(' '))
number_objects = int(input())
matrix = []

for i in range(number_objects):
    matrix.append(list(map(int, input().split())))

tree = build(1, 0, number_objects)
print(num)
tree.print()
