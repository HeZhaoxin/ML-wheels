import math
import numpy as np
import random

def sigmoid(x):
    try:
        return 1 / ( 1 + math.exp(-x) )
    except OverflowError:
        return 0.0


class bin_logistic_regression_model:
    def __init__(self, eta=0.03):
        self.__eta = eta

    def fit(self, train, target, epoch = 100):
        x = np.array(train)
        x = np.concatenate((x, [[1]] * x.shape[0]), axis=1)
        y = np.array(target)
        if len(train) != len(target):
            raise Exception
        dimension = x.shape[1]
        self.__w = [0] * dimension
        for n in range(epoch):
            for xi, yi in zip(x, y):
                z = np.dot(xi, self.__w)
                h = sigmoid(z)
                grad = np.dot(xi, h-yi)
                self.__w = self.__w - self.__eta * grad
            print(self.__w)

    def predict(self, data):
        x = np.array(data)
        x = np.append(x, [1])
        z = np.dot(x, self.__w)
        h = sigmoid(z)
        if h > 0.5:
            return 1
        else:
            return 0

    def get_parameter(self):
        return self.__w


if __name__ == '__main__':
    import time
    time_start = time.time()
    w = [2.5, -1, 5]
    x = [[random.uniform(-10,10), random.uniform(-20,30)] for _ in range(1000)]
    xb = np.array(x)
    xb = np.concatenate((xb, [[1]] * xb.shape[0]), axis=1)
    y = [1 if sum(xij * wj for xij, wj in zip(xi, w)) >= 0 else 0 for xi in xb]
    model = bin_logistic_regression_model()
    model.fit(x, y, epoch=1000)
    print(model.get_parameter())
    print(model.predict([2, 12]))
    print(model.predict([0, 0]))
    print(model.predict([0, 4.5]))
    print(model.predict([0, 5.5]))
    time_end = time.time()
    print(time_end-time_start)