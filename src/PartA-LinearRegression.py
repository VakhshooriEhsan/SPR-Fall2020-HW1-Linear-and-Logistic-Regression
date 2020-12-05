import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def readData(add):
    _data = pd.read_csv(add)
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

_theta1 = CFS(_1x, _y)
print('CFS Theta:')
print(_theta1)

MSE = 1.0 / (2*m) * np.matmul( (np.matmul( _1x, _theta1) - _y).transpose(), (np.matmul(_1x, _theta1) - _y))
print('CFS MSE for train data:')
print(MSE)

MSEt = 1.0 / (2*mt) * np.matmul( (np.matmul( _1xt, _theta1) - _yt).transpose(), (np.matmul(_1xt, _theta1) - _yt))
print('CFS MSE for test data:')
print(MSEt)


plt.figure(1)

plt.subplot(1, 2, 1)
plt.plot(_x, _y, '.')
plt.plot(_x, np.matmul(_1x, _theta1), '-')
plt.title('')
plt.ylabel('y')
plt.xlabel('x')

plt.subplot(1, 2, 2)
plt.plot(_xt, _yt, '.')
plt.plot(_xt, np.matmul(_1xt, _theta1), '-')
plt.title('')
plt.ylabel('y')
plt.xlabel('x')

plt.show()
