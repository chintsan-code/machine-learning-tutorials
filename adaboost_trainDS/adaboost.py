from numpy import *


def loadSimpData():
    datMat = matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def loadDataSet(fileName):  # general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t'))  # get number of fields
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


# 用于测试是否有某个值小于或者大于我们正在测试的阈值
# 通过阈值比较对数据进行分类，所有在阈值一边的数据会分到类别-1，而在另外一边的数据分到类别+1
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):  # just classify the data
    retArray = ones((shape(dataMatrix)[0], 1))  # 每个数据点类别的累加估计值
    # 通过数组过滤来实现，首先将返回数组retArray的全部元素设置为1，然后将所有不满足不等式要求的元素设置为-1
    if threshIneq == 'lt':  # less than
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:  # greater than
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


# 在一个加权数据集中循环，并找到具有最低错误率的单层决策树(决策树桩)
def buildStump(dataArr, classLabels, D):  # D为计算时的权重向量
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)  # m:数据集样本数, n:数据集特征数
    numSteps = 10.0
    bestStump = {}  # 用于存储给定权重向量D时所得到的最佳单层决策树的相关信息
    bestClasEst = mat(zeros((m, 1)))
    minError = inf  # init error sum, to +infinity 最小误差率，将其初始值设置为正无穷大，之后用于寻找可能的最小错误率
    for i in range(n):  # loop over all dimensions  # 对所有特征进行遍历
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):  # loop over all range in current dimension
            for inequal in ['lt', 'gt']:  # go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal,
                                              inequal)  # call stump classify with i, j, lessThan
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0  # 预测值等于样本标签，将其置为0
                weightedError = D.T * errArr  # calc total error multiplied by D  计算加权错误率
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:  # 如果错误率低于minError，则将当前单层决策树设为最佳单层决策树
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst  # bestClasEst为估计的类别向量


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)  # init D to all equal
    aggClassEst = mat(zeros((m, 1)))  # 样本的估计累加值
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)  # build Stump
        print("D:", D.T)
        alpha = float(
            0.5 * log((1.0 - error) / max(error, 1e-16)))  # calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)  # store Stump Params in Array
        print("classEst: ", classEst.T)  # 当前弱分类器的分类结果
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)  # exponent for D calc, getting messy
        D = multiply(D, exp(expon))  # Calc New D for next iteration 为下一次迭代更新样本权重D
        D = D / D.sum()
        # calc training error of all classifiers, if this is 0 quit for loop early (use break)
        # 错误率累加计算
        aggClassEst += alpha * classEst  # 记录每个数据点的类别估计累计值
        print("aggClassEst: ", aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst
