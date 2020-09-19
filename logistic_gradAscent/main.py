# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import logRegres

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataArr, labelMat = logRegres.loadDataSet()
    weights = logRegres.gradAscent(dataArr, labelMat)
    # print(weights)
    logRegres.plotBestFit(weights.getA())
