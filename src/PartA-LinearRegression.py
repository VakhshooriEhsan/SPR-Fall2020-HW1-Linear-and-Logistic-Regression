import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def readData(addr):
    _data = pd.read_csv(addr)
    _data -= _data.mean()
    _data /= np.sqrt(_data.var())
    _data = _data.values
    _data = _data[np.argsort(_data[:, 0])]
    return _data

def CFS(_x, _y): # closed form solution
    mul = np.matmul
    inv = np.linalg.pinv
    _xt = _x.transpose()
    _theta = mul( mul( inv( mul(_xt, _x)), _xt), _y)
    return _theta

def GDA(_x, _y, _theta, alpha, iterations): # Gradient Descent algorithm
    J = np.arange(0)
    mul = np.matmul
    J = np.append(J, 1.0 / (2*len(_x)) * mul( (mul( _x, _theta) - _y).transpose(), (mul(_x, _theta) - _y)))
    for _ in range(iterations):
        _h = mul(_x, _theta)
        _theta = _theta - (alpha/len(_x)) * mul( (_h-_y).transpose(), _x).transpose()
        J = np.append(J, 1.0 / (2*len(_x)) * mul( (mul( _x, _theta) - _y).transpose(), (mul(_x, _theta) - _y)))
    return _theta, J


# ------------------------------ Read Datas ------------------------------
_data = readData('Datas/Data-Train.csv')
_x = _data[:, [0]]
_y = _data[:, [1]]
m = len(_x)
_1x = np.insert(_x, 0, np.ones(m), axis=1)

_dataTest = readData('Datas/Data-Test.csv')
_xt = _dataTest[:, [0]]
_yt = _dataTest[:, [1]]
mt = len(_xt)
_1xt = np.insert(_xt, 0, np.ones(mt), axis=1)

# --------------------------------- CFS ---------------------------------
_theta1 = CFS(_1x, _y)
print('CFS:')
for i in range(len(_theta1)):
    print('Theta', i, ':', _theta1[i][0])

MSE1 = 1.0 / (m) * np.matmul( (np.matmul( _1x, _theta1) - _y).transpose(), (np.matmul(_1x, _theta1) - _y))
print('CFS MSE for train data:', MSE1[0][0])

MSEt1 = 1.0 / (mt) * np.matmul( (np.matmul( _1xt, _theta1) - _yt).transpose(), (np.matmul(_1xt, _theta1) - _yt))
print('CFS MSE for test data:', MSEt1[0][0])

print()

# --------------------------------- GDA ---------------------------------
iterations = 2000
alpha = 0.001
_theta2 = np.zeros((len(_1x[0]), 1))
_theta2, J = GDA(_1x, _y, _theta2, alpha, iterations)
print('GDA:')
for i in range(len(_theta2)):
    print('Theta', i, ':', _theta2[i][0])

MSE2 = 1.0 / (m) * np.matmul( (np.matmul( _1x, _theta2) - _y).transpose(), (np.matmul(_1x, _theta2) - _y))
print('GDA MSE for train data:', MSE2[0][0])

MSEt2 = 1.0 / (mt) * np.matmul( (np.matmul( _1xt, _theta2) - _yt).transpose(), (np.matmul(_1xt, _theta2) - _yt))
print('GDA MSE for test data:', MSEt2[0][0])

# -------------------------------- Plots --------------------------------
plt.figure(1)

plt.subplot(2, 2, 1)
plt.plot(_x, _y, '.', label='Train datasets')
plt.plot(_x, np.matmul(_1x, _theta1), '-', label='closed form solution')
plt.plot(_x, np.matmul(_1x, _theta2), '-', label='Gradient Descent algorithm')
plt.title('Train datasets and regression lines')
plt.ylabel('y')
plt.xlabel('x')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(_xt, _yt, '.', label='Test datasets')
plt.plot(_xt, np.matmul(_1xt, _theta1), '-', label='closed form solution')
plt.plot(_xt, np.matmul(_1xt, _theta2), '-', label='Gradient Descent algorithm')
plt.title('Test datasets and regression lines')
plt.ylabel('y')
plt.xlabel('x')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(np.arange(len(J)), J, '-')
plt.title('Cost function')
plt.ylabel('J')
plt.xlabel('Iterations')
plt.legend()

plt.show()
