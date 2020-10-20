# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import adaboost

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    datMat, classLabels = adaboost.loadSimpData()
    classifierArray, aggClassEst = adaboost.adaBoostTrainDS(datMat, classLabels, 30)
    # adaboost.adaClassify([0, 0], classifierArray) #一个数据
    result = adaboost.adaClassify([[5, 5], [0, 0]], classifierArray)  # 多个数据
    print(result)
