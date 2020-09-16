# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import bayes

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    listOPosts, listClasses = bayes.loadDataSet()
    myvocabList = bayes.createVocabList(listOPosts)
    print(myvocabList)
    returnVec0 = bayes.setOfWords2Vec(myvocabList, listOPosts[0])
    print(returnVec0)
    returnVec3 = bayes.setOfWords2Vec(myvocabList, listOPosts[3])
    print(returnVec3)
    # 训练
    trainMat = []
    for postionDoc in listOPosts:
        trainMat.append(bayes.setOfWords2Vec(myvocabList, postionDoc))
    p0v, p1v, pAb = bayes.trainNB0(trainMat, listClasses)
    print(pAb)
    print(p0v)
    print(p1v)
