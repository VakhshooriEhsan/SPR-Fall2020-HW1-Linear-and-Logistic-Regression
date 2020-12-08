import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import e

def readData(addr):
    _data = pd.read_csv(addr, sep=",", header=None)
    _data.columns = ["x", "y", "e1", "e2", "class"]
    _data = _data[["x", "y", "class"]]
    _data = _data[np.logical_or(_data["class"]=='Iris-setosa', _data["class"]=='Iris-virginica')]
    _data[["x", "y"]] -= _data[["x", "y"]].mean()
    _data[["x", "y"]] /= np.sqrt(_data[["x", "y"]].var())
    _data = _data.values
    return _data

def g(z): # Sigmoid
    return 1.0 / ( 1 + e**(-z))

def LR(_x, _y, _theta, alpha, iterations): # Logistic Regression
    J = np.arange(0)
    mul = np.matmul
    _h = g(mul(_x, _theta))
    J = np.append(J, np.sum(1.0 / len(_x) * ( -_y * np.log(_h.astype(float)) - (1-_y) * np.log(1-_h.astype(float)))))
    for _ in range(iterations):
        _h = g(mul(_x, _theta))
        _theta = _theta - (alpha/len(_x)) * mul( (_h-_y).transpose(), _x).transpose()
        J = np.append(J, np.sum(1.0 / len(_x) * ( -_y * np.log(_h.astype(float)) - (1-_y) * np.log(1-_h.astype(float)))))
    return _theta, J

# ------------------------------ Read Datas ------------------------------
_data = readData('Datas/iris.data')

_data0 = _data[ _data[:, 2] == 'Iris-setosa']
_data1 = _data[ _data[:, 2] == 'Iris-virginica']

m0 = int(len(_data0)*0.8)
mt0 = int(len(_data1)*0.2)
m1 = int(len(_data1)*0.8)
mt1 = int(len(_data1)*0.2)

_x = np.concatenate((_data0[0:m0][:, [0,1]], _data1[0:m1][:, [0,1]]))
_y = np.concatenate((np.zeros((m0, 1)), np.ones((m1, 1))))
_1x = np.insert(_x, 0, np.ones(m0+m1), axis=1)

_xt = np.concatenate((_data0[m0:][:, [0,1]], _data1[m1:][:, [0,1]]))
_yt = np.concatenate((np.zeros((mt0, 1)), np.ones((mt1, 1))))
_1xt = np.insert(_xt, 0, np.ones(mt0+mt1), axis=1)

# --------------------------------- LR ---------------------------------

iterations = 10000
alpha = 0.001
_theta = np.zeros((len(_1x[0]), 1))
_theta, J = LR(_1x, _y, _theta, alpha, iterations)
print('GDA:')
for i in range(len(_theta)):
    print('Theta', i, ':', _theta[i][0])

mul = np.matmul

_h = g(mul(_1x, _theta))
ERR = np.sum(1.0 / len(_x) * ( -_y * np.log(_h.astype(float)) - (1-_y) * np.log(1-_h.astype(float))))
print('train data error:', ERR)

_h = g(mul(_1xt, _theta))
ERR = np.sum(1.0 / len(_1xt) * ( -_yt * np.log(_h.astype(float)) - (1-_yt) * np.log(1-_h.astype(float))))
print('test data error:', ERR)

print('Equation of the decision boundary: x2 =', -(_theta[0][0]/_theta[2][0]), '+ x1 *', -(_theta[1][0]/_theta[2][0]))

# -------------------------------- Plots --------------------------------
plt.figure(1)

k = np.arange(-3, 3, 0.1)

plt.subplot(2, 2, 1)
plt.plot(_x[:m0, [0]], _x[:m0, [1]], '.', label='Iris-setosa')
plt.plot(_x[m1:, [0]], _x[m1:, [1]], '.', label='Iris-virginica')
plt.plot(k, -(_theta[0][0]/_theta[2][0])-k*(_theta[1][0]/_theta[2][0]), '-', label='decision boundary')
plt.title('Train datasets and decision boundary')
plt.ylabel('x2')
plt.xlabel('x1')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(_xt[:mt0, [0]], _xt[:mt0, [1]], '.', label='Iris-setosa')
plt.plot(_xt[mt1:, [0]], _xt[mt1:, [1]], '.', label='Iris-virginica')
plt.plot(k, -(_theta[0][0]/_theta[2][0])-k*(_theta[1][0]/_theta[2][0]), '-', label='decision boundary')
plt.title('Test datasets and decision boundary')
plt.ylabel('x2')
plt.xlabel('x1')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(np.arange(len(J)), J, '-')
plt.title('Cost function')
plt.ylabel('J')
plt.xlabel('Iterations')
plt.legend()

plt.show()
