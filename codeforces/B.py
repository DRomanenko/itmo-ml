import numpy as np

number_classes = int(input())
confusion_matrix = np.zeros((number_classes, number_classes))
for i in range(number_classes):
    line = list(map(int, input().split(' ')))
    for q in range(number_classes):
        confusion_matrix[i][q] = line[q]


def safe_division(a, b):
    return 0 if b == 0 else a / b


def calc_f_measure(precision, recall):
    return safe_division(2 * precision * recall, precision + recall)


all_sum = confusion_matrix.sum()
# macro measure
precision = sum([safe_division(confusion_matrix[i][i] * confusion_matrix.sum(axis=1)[i],
                               confusion_matrix.sum(axis=0)[i])
                 for i in range(number_classes)]) / all_sum
recall = confusion_matrix.diagonal().sum() / all_sum
print(calc_f_measure(precision, recall))

# micro measure
print(sum([calc_f_measure(safe_division(confusion_matrix[i][i], confusion_matrix.sum(axis=0)[i]),
                          safe_division(confusion_matrix[i][i], confusion_matrix.sum(axis=1)[i]))
           * confusion_matrix.sum(axis=1)[i]
           for i in range(number_classes)]) / all_sum)
