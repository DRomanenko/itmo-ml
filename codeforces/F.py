import math
import numpy as np

number_classes = int(input())
penalties = np.array(list(map(int, input().split(' '))))
smoothing = int(input())
number_messages = int(input())

words = set()
num_qty = np.zeros(number_classes)
num_words = []
probabilities = []
for i in range(number_classes):
    num_words.append({})
    probabilities.append({})

for i in range(number_messages):
    line = input().split(' ')
    num = int(line[0]) - 1
    num_qty[num] += 1
    for word in set(line[2:]):
        num_words[num][word] = 1 if num_words[num].get(word) is None else num_words[num][word] + 1
        words.add(word)

m = int(input())

for num in range(number_classes):
    for word in words:
        if num_words[num].get(word) is None:
            num_words[num][word] = 0
        probabilities[num][word] = (smoothing + num_words[num][word]) / (2 * smoothing + num_qty[num])

for i in range(m):
    line = set(input().split(' ')[1:])
    scores = np.zeros(number_classes)
    for num in range(number_classes):
        if num_qty[num] == 0:
            scores[num] = 0.0
        else:
            for word in words:
                if word in line:
                    if probabilities[num][word] == 0:
                        scores[num] = 0
                        break
                    else:
                        scores[num] += math.log(probabilities[num][word])
                else:
                    scores[num] = 0.0 if probabilities[num][word] == 1 \
                        else scores[num] + math.log(1 - probabilities[num][word])
            scores[num] += math.log(num_qty[num] / number_messages)
    max_score = scores.max()
    for num in range(number_classes):
        scores[num] = scores[num] if scores[num] == 0 else penalties[num] * math.exp(scores[num] - max_score)

    norm = scores.sum()
    print(' '.join(str(score / norm) for score in scores))