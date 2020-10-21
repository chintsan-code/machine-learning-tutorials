# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import svmMLiA
from numpy import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
    b, alphas = svmMLiA.smoPK(dataArr, labelArr, 0.6, 0.001, 40)
    print(b)
    print(alphas[alphas > 0])
    ws = svmMLiA.calcWs(alphas, dataArr, labelArr)
    print(ws)
    dataMat = mat(dataArr)
    print(dataMat[0] * mat(ws) + b)
