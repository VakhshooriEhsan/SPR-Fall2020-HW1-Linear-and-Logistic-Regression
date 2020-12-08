# Linear and logistic regression

## Setup and run:

* Install python3

* Install python library:
```bash
    $ pip install pandas
    $ pip install numpy
    $ pip install matplotlib
```

## A. Linear regression

### 1. Reading datas:

* Reading datas by `readData(addr)` function from `./Datas/Data-Train.csv` & `./Datas/Data-Test.csv` and saving them as a numpy array.
* Finding X, Y of train and test datas.

### 2. Closed Form Solution(CFS):

* This formula was used to calculating theta:

![f1](https://github.com/VakhshooriEhsan/SPR-Fall2020-HW1-Linear-and-Logistic-Regression/blob/master/docs/imgs/fig1.PNG?raw=true)

* This formula was used to calculating MSE error on the train and test datas:

![f2](https://github.com/VakhshooriEhsan/SPR-Fall2020-HW1-Linear-and-Logistic-Regression/blob/master/docs/imgs/fig2.PNG?raw=true)

### 3. Gradient Descent Algorithm(GDA):

* This formula was used to calculating theta:

![f3](https://github.com/VakhshooriEhsan/SPR-Fall2020-HW1-Linear-and-Logistic-Regression/blob/master/docs/imgs/fig3.PNG?raw=true)

![f3](https://github.com/VakhshooriEhsan/SPR-Fall2020-HW1-Linear-and-Logistic-Regression/blob/master/docs/imgs/fig4.PNG?raw=true)

* This formula was used to calculating cost function:

![f3](https://github.com/VakhshooriEhsan/SPR-Fall2020-HW1-Linear-and-Logistic-Regression/blob/master/docs/imgs/fig7.PNG?raw=true)

* This formula was used to calculating MSE error on the train and test datas:

![f2](https://github.com/VakhshooriEhsan/SPR-Fall2020-HW1-Linear-and-Logistic-Regression/blob/master/docs/imgs/fig2.PNG?raw=true)

### 4. results:

```
CFS:
Theta 0 : 5.585809592645319e-16
Theta 1 : 0.995059336172486
CFS MSE for train data: 0.009847060578475559
CFS MSE for test data: 0.010843636745466137

GDA:
Theta 0 : -1.1102430086395994e-16
Theta 1 : 0.8602577852788518
GDA MSE for train data: 0.028000347243681556
GDA MSE for test data: 0.028816400545735362
```

![f4](https://github.com/VakhshooriEhsan/SPR-Fall2020-HW1-Linear-and-Logistic-Regression/blob/master/docs/imgs/Figure_1.png?raw=true)

## B. Logistic regression

### 1. Reading datas:

* Reading datas by `readData(addr)` function from `./Datas/iris.data` and saving them as a numpy array.
* Finding X, Y of train and test datas.

### 2. Closed Form Solution(CFS):

* This formula was used to calculating theta:

![f1](https://github.com/VakhshooriEhsan/SPR-Fall2020-HW1-Linear-and-Logistic-Regression/blob/master/docs/imgs/fig3.PNG?raw=true)

![f3](https://github.com/VakhshooriEhsan/SPR-Fall2020-HW1-Linear-and-Logistic-Regression/blob/master/docs/imgs/fig5.PNG?raw=true)

![f3](https://github.com/VakhshooriEhsan/SPR-Fall2020-HW1-Linear-and-Logistic-Regression/blob/master/docs/imgs/fig6.PNG?raw=true)

* This formula was used to calculating cost function:

![f3](https://github.com/VakhshooriEhsan/SPR-Fall2020-HW1-Linear-and-Logistic-Regression/blob/master/docs/imgs/fig8.PNG?raw=true)

* This formula was used to calculating error on the train and test datas:

![f2](https://github.com/VakhshooriEhsan/SPR-Fall2020-HW1-Linear-and-Logistic-Regression/blob/master/docs/imgs/fig9.PNG?raw=true)

### 3. results:

```
GDA:
Theta 0 : 0.023796282928132278
Theta 1 : 1.7636998542401963
Theta 2 : -1.1139672821655044
train data error: 0.1506591952614027
test data error: 0.1879910317326229
Equation of the decision boundary: x2 = 0.02136174312217979 + x1 * 1.5832600135361603
```

![f4](https://github.com/VakhshooriEhsan/SPR-Fall2020-HW1-Linear-and-Logistic-Regression/blob/master/docs/imgs/Figure_2.png?raw=true)
