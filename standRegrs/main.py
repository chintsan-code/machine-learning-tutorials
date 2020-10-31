# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import regression

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    xArr, yArr = regression.loadDataSet('ex0.txt')
    print(xArr[0:2])
    ws = regression.standRegres(xArr, yArr)
    print(ws)
    regression.plotBestFit(xArr, yArr, ws)
    corrcoef = regression.calcCorrcoef(xArr, yArr, ws)
    print(corrcoef)
