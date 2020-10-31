# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import regression
from numpy import *
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    xArr, yArr = regression.loadDataSet('ex0.txt')
    # print(regression.lwlr(xArr[0],xArr,yArr,1.0))
    # print(regression.lwlr(xArr[0], xArr, yArr, 0.001))
    # 得到数据集所有点的估计
    yHat = regression.lwlrTest(xArr, xArr, yArr, 0.003)
    xMat = mat(xArr)
    srdInd = xMat[:, 1].argsort(0)
    xSort = xMat[srdInd][:, 0, :]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srdInd])
    plt.show()
