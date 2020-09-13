# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import trees

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    myDat, labels = trees.createDataSet()
    # 划分数据集
    retDataSet = trees.splitDataSet(myDat, 0, 5)
    print(retDataSet)
    bestFeature = trees.chooseBestFeatureToSplit(myDat)
    print("best feature: %d" % bestFeature)
