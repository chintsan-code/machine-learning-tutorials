# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import logRegres
from numpy import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataArr, lableMat = logRegres.loadDataSet()
    weights = logRegres.stocGradAscentInprove(array(dataArr), lableMat)
    logRegres.plotBestFit(weights)
