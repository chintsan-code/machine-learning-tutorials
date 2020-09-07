# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import knn
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    group, labels = knn.createDataSet()
    print('group:')
    print(group)
    print('labels:')
    print(labels)
    plt.figure()
    for i in range(len(group)):
        plt.scatter(group[i][0], group[i][1])

    #k值不能是4,不然找不到距离最小的点集类型，实际上A，B类型数目各是2
    class00 = knn.classify0([0, 0], group, labels, 3)
    print('[0, 0] class:' + class00)

    plt.show()