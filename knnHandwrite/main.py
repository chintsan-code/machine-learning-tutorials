# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from numpy import *
import operator
from os import listdir
import knn


# 把32*32分辨率的二进制图像存储到1*1024向量中
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    hwLabels = []
    # 获取训练文件目录列表（一个文件对应一张图像）
    trainingFileList = listdir('./digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        # 找分类标签
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('./digits/trainingDigits/%s' % fileNameStr)

    # 获取测试文件列表
    testFileList = listdir('./digits/testDigits')
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        # 找分类标签
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('./digits/testDigits/%s' % fileNameStr)
        # 计算分类结果
        classfierResult = knn.classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('file:' + fileStr + ' the classifier came back with:%d, the real answer is:%d' % (
        classfierResult, classNumStr))
        if (classfierResult != classNumStr):
            errorCount += 1
    print('the total number of errors is:%d' % errorCount)
    print('the total error rate is:%f' % (errorCount / float(mTest)))
