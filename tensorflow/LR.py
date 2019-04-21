import numpy as np
import matplotlib as plt
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn import metrics

class LR_BGD(object):
    def __init__(self):
        pass

    def _build_graph(self):
        self.X = tf.placeholder(tf.float64)
        self.y = tf.placeholder(tf.float64)

        self._wight = tf.Variable(tf.zeros([self.feature_num, 1], dtype=tf.float64))
        self._bias = tf.Variable(tf.zeros([1, 1], dtype=tf.float64))

        db = tf.matmul(self.X, self._wight) + self._bias
        hyp = tf.sigmoid(db)

        cost0 = tf.multiply(self.y, tf.reshape(tf.clip_by_value(tf.log(hyp), 1e-20, (1.0 - (1e-20))), [-1]))
        cost1 = tf.multiply((1 - self.y), tf.reshape(tf.log(tf.clip_by_value((1 - hyp), 1e-10, (1.0 - (1e-10)))), [-1]))
        cost = cost0 + cost1
        loss = tf.reduce_mean(cost) / -self.sample_num
        return hyp, loss

    def fit(self, data, target, epoch=100, learning_rate = 0.001):
        self.sample_num, self.feature_num = data.shape
        self.proba, self.loss = self._build_graph()
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        for step in range(epoch):
            self.sess.run(train, {self.X: data, self.y: target})

    def predict(self, data):
        res = self.sess.run(self.proba, {self.X: data}).flatten()
        return res

    def get_parameter(self):
        w = self.sess.run(self._wight).flatten()
        b = self.sess.run(self._bias).flatten()
        return w, b

if __name__ == '__main__':
    cancer = load_breast_cancer()
    X = np.array(scale(cancer.data))
    y = np.array(cancer.target)

    train_X, test_X, train_Y, test_Y = X, X, y, y

    model = LR_BGD()
    model.fit(train_X, train_Y, epoch=200)
    res = model.predict(test_X)
    label_train = list(map(lambda e: 1 if e > 0.5 else 0, res))
    print(confusion_matrix(label_train, test_Y))
    w, b = model.get_parameter()
    print(w)
    print(b)