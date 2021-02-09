N, M, K = map(int, input().split(' '))
arr = list(map(int, input().split(' ')))
d = sorted([[arr[i], i] for i in range(len(arr))])

ans = [[] for i in range(K)]
for i in range(len(arr)):
    ans[i % K].append(d[i][1] + 1)

for row in ans:
    print(len(row), *row, sep=' ')