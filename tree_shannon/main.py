# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import trees

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    myDat, labels = trees.createDataSet()
    sh = trees.calcShannonEnt(myDat)
    print(sh)
    # 熵越高，说明混合的数据也越多
    # 修改数据集，添加类型'maybe'
    myDat[0][-1] = 'maybe'
    sh = trees.calcShannonEnt(myDat)
    print(sh)
