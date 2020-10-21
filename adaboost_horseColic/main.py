# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import adaboost
from numpy import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    datArr, labelArr = adaboost.loadDataSet('horseColicTraining2.txt')
    classifierArray, aggClassEst = adaboost.adaBoostTrainDS(datArr, labelArr, 10)

    testArr, testLabelArr = adaboost.loadDataSet('horseColicTest2.txt')
    predixtion10 = adaboost.adaClassify(testArr, classifierArray)
    testSize = shape(predixtion10)[0]
    errArr = mat(ones((testSize, 1)))
    esum = errArr[predixtion10 != mat(testLabelArr).T].sum()
    print("error sum =", esum)
