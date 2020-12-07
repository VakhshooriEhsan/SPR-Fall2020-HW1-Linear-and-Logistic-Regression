import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def readData(addr):
    _data = pd.read_csv(addr, sep=",", header=None)
    _data.columns = ["x", "y", "e1", "e2", "class"]
    _data = _data[["x", "y", "class"]]
    _data = _data[np.logical_or(_data["class"]=='Iris-setosa', _data["class"]=='Iris-virginica')]
    print(_data[["x", "y"]].mean())
    _data[["x", "y"]] -= _data[["x", "y"]].mean()
    _data[["x", "y"]] /= np.sqrt(_data[["x", "y"]].var())
    _data = _data.values
    return _data


# ------------------------------ Read Datas ------------------------------
_data = readData('Datas/iris.data')

_data0 = _data[ _data[:, 2] == 'Iris-setosa']
_data1 = _data[ _data[:, 2] == 'Iris-virginica']
m0 = int(len(_data0)*0.8)
mt0 = int(len(_data1)*0.2)
m1 = int(len(_data0)*0.8)
mt1 = int(len(_data1)*0.2)

_x0 = _data0[0:m0][:, [0]]
_y0 = _data0[0:m0][:, [1]]
_xt0 = _data0[m0:][:, [0]]
_yt0 = _data0[m0:][:, [1]]

_x1 = _data1[0:m1][:, [0]]
_y1 = _data1[0:m1][:, [1]]
_xt1 = _data1[m1:][:, [0]]
_yt1 = _data1[m1:][:, [1]]

# -------------------------------- Plots --------------------------------
plt.figure(1)

plt.subplot(1, 2, 1)
plt.plot(_x0, _y0, '.', label='Iris-setosa')
plt.plot(_x1, _y1, '.', label='Iris-virginica')
plt.title('Train datasets and decision boundary')
plt.ylabel('y')
plt.xlabel('x')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(_xt0, _yt0, '.', label='Iris-setosa')
plt.plot(_xt1, _yt1, '.', label='Iris-virginica')
plt.title('Test datasets and decision boundary')
plt.ylabel('y')
plt.xlabel('x')
plt.legend()

plt.show()
