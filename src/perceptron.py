# 感知机
# 假设：trainingDatas的L2等于1，η等于1
# trainingDatas：[(向量, 结果)]，训练集
# testDatas：[向量集]，测试集
# return：  iterable((testData, y))

import numpy as np
from functools import partial


def perceptron(trainingDatas, testDatas, showLogs = False):
    (w, b) = perceptronTraining(trainingDatas, showLogs)
    print("f(x) = (%d, %d)x + %s"  %(w[0], w[1], "(" + str(b) + ")" if b < 0 else str(b)))
    print("test datas:")
    for testData in testDatas:
        yield (testData, perceptronResult(testData, w, b))


def perceptronResult(x, w, b):
    y = w.dot(x) + b
    return 1 if y > 0 else (0 if y == 0 else -1)


# 对偶形式算法
# trainingDatas：[(向量, 结果)]，训练集
# return：  (w, b)
def perceptronTraining(trainingDatas, showLogs):
    count = len(trainingDatas)
    a, b = [0] * count, 0

    marix = []
    for x in trainingDatas:
        tmpMarix = []
        for y in trainingDatas:
            tmpMarix.append(x[0].dot(y[0]))
        marix.append(list(enumerate(tmpMarix)))

    newDatas = [(x[1], x[0]) for x in enumerate(trainingDatas)]
    result = [False] * count

    while False in result:
        for data in newDatas:
            i = data[1]
            y = data[0][1] * (sum([a[x[0]] * newDatas[x[0]][0][1] * x[1] for x in marix[i]]) + b)
            if y <= 0:
                a[i] = a[i] + 1
                b = b + data[0][1]
                result[data[1]] = False
                if showLogs:
                    print((data[0], a, b))
                break
            else:
                result[data[1]] = True

    w = sum(a[x[1]] * x[0][0] * x[0][1] for x in newDatas)
    return w, b


# 普通算法
# trainingDatas：[(向量, 结果)]，训练集
# return：  (w, b)
def perceptronTraining_V1(trainingDatas, showLogs):
    w, b = np.array([0, 0]), 0
    newDatas = [(x[1], x[0]) for x in enumerate(trainingDatas)]
    result = [False] * len(trainingDatas)
    while False in result:
        for data in newDatas:
            y = (perceptronResult(data[0][0], w, b))
            if y != data[0][1]:
                w = w + data[0][1] * data[0][0]
                b = b + data[0][1]
                result[data[1]] = False
                if showLogs:
                    print((data[0], w, b))
                break
            else:
                result[data[1]] = True
    return w, b


perceptronWithLogs = partial(perceptron, showLogs = True)
test_results  = perceptronWithLogs([(np.array([3, 3]), 1), (np.array([4, 3]), 1), (np.array([1, 1]), -1)], [np.array([2, 2]), np.array([-2, 0])])
for test_result in test_results:
    print(test_result)


