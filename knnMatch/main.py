# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import knn
import matplotlib
import matplotlib.pyplot as plt
from array import array
from numpy import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 数据加载
    datingDataMat, datingLabels = knn.file2matrix('datingTestSet.txt')
    print(datingDataMat)
    print(datingLabels)

    # 显示
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 看不到任何有用的模式信息
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
    # 标注上色彩
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    plt.show()
    plt.close()

    # 归一化数据
    normMat, ranges, minVals = knn.autoNorm(datingDataMat)
    print('norm mat:')
    print(normMat)
    print('range:')
    print(ranges)
    print('norm mat:')
    print(minVals)

    # 测试分类器,使用数据集前hoRatio比例做测试集
    hoRatio = 0.10
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = knn.classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)

    # 预测分类
    resultList = ['not at all', 'in small doses', 'in large doxes']
    ffMiles = float(input('frequent flier miles earned per year?'))
    percentTats = float(input("percentage of time spent playing video games?"))
    iceCream = float(input('liters of ice cream consumed per year?'))
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = knn.classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print('You will probably like this person:', resultList[classifierResult - 1])
