#!/usr/bin/env python
# -*- coding: utf-8  -*-

import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from numpy import *

# # =====NB=========#
# def NbClass(x_train, y_train):
#     from sklearn.naive_bayes import MultinomialNB
#     clf = MultinomialNB(alpha=0.01).fit(x_train, y_train)
#     return clf


# ========SVM========#
def SvmClass(x_train, y_train):
    clf = SVC(kernel='linear', probability=True)  # default with 'rbf'
    clf.fit(x_train, y_train)  # 训练，对于监督模型来说是 fit(X, y)，对于非监督模型是 fit(X)
    return clf


# ========Logistic Regression========#
def LogisticClass(x_train, y_train):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty='l2')
    clf.fit(x_train, y_train)
    return clf


# ========KNN========#
def KnnClass(x_train, y_train):
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)
    return clf


# ========Decision Tree ========#
def DccisionClass(x_train, y_train):
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    return clf


# ========Random Forest Classifier ========#
def random_forest_class(x_train, y_train):
    clf = RandomForestClassifier(n_estimators=8)  # 参数n_estimators设置弱分类器的数量
    clf.fit(x_train, y_train)
    return clf


# ========准确率召回率 ========#
def Precision(clf):
    doc_class_predicted = clf.predict(x_test)
    print(np.mean(doc_class_predicted == y_test))  # 预测结果和真实标签
    #准确率与召回率
    precision, recall, thresholds = precision_recall_curve(y_test, clf.predict(x_test))
    answer = clf.predict_proba(x_test)[:, 1]
    report = answer > 0.5
    print(classification_report(y_test, report, target_names=['neg', 'pos']))
    print("--------------------")
    from sklearn.metrics import accuracy_score
    print('准确率: %.2f' % accuracy_score(y_test, doc_class_predicted))


if __name__ == '__main__':
    # 获取数据 [5996 rows x 302 columns]
    df = pd.read_csv('6000_word_vector.csv')
    y = df.iloc[:, 1]
    x = df.iloc[:, 2:]

    # SVM (RBF)
    # using training data with 100 dimensions

    # 划分训练、测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    print('**************支持向量机************  ')
    Precision(SvmClass(x_train, y_train))
    # print('**************朴素贝叶斯************  ')
    # Precision(NbClass(x_train, y_train))
    print('**************最近邻ＫＮＮ************  ')
    Precision(KnnClass(x_train, y_train))
    print('**************逻辑回归************  ')
    Precision(LogisticClass(x_train, y_train))
    print('**************决策树************  ')
    Precision(DccisionClass(x_train, y_train))
    print('**************逻辑森林************  ')
    Precision(random_forest_class(x_train, y_train))

    


