# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import svmMLiA
from numpy import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
    b, alphas = svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    print(b)
    print(alphas[alphas > 0])
    for i in range(100):
        if alphas[i] > 0.0:
            print(dataArr[i])
            print(labelArr[i])
