import numpy as np
from numpy import mean,std
import pandas as pd
from sklearn.model_selection import KFold

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC


from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import ShuffleSplit
from sklearn import tree


def get_z_score(data):
    score_list = []
    # num_features = (len(data[0])-2)/24

    #不对第一列求zscore
    for i in range(data.shape[0]):
        score_list.append(data[i][0])

    for j in range(1,data.shape[1]-1):
        standard = std(data[:, j])
        my_mean = mean(data[:,j])
        for i in range(data.shape[0]):
            if standard !=0:
                z_score = round((data[i, j] - my_mean) / standard + 0.001,2)
            else:
                z_score = standard
            score_list.append(z_score)

    #将最后一列的label加入list
    for i in range(data.shape[0]):
        score_list.append(data[i][data.shape[1]-1])

    score_array = np.array(score_list)
    score_array = np.reshape(score_array,(len(data[0]),-1))
    score_array = score_array.T

    return score_array
def get_dataset(path,myK,random_state=0,need_zscore=True):
    pd_data = pd.read_csv(path, sep='\t', header=None)
    wb_data = pd_data.as_matrix()
    if need_zscore:
        data = get_z_score(wb_data)
    else:
        data = wb_data
    myX = data[:, 1:-1]
    myy = data[:, -1]

    kf = ShuffleSplit(n_splits=5, random_state=random_state)
    train_index = []
    test_index = []
    for train, test in kf.split(myX):
        train_index.append(train)
        test_index.append(test)
    Xtrain, Xtest, ytrain, ytest = myX[train_index[myK]], myX[test_index[myK]], myy[train_index[myK]], myy[test_index[myK]]
    return Xtrain, Xtest, ytrain, ytest
def get_stacking(clf, x_train, y_train, x_test, n_folds=5):
    """
    这个函数是stacking的核心，使用交叉验证的方法得到次级训练集
    x_train, y_train, x_test 的值应该为numpy里面的数组类型 numpy.ndarray .
    如果输入为pandas的DataFrame类型则会把报错"""
    train_num, test_num = x_train.shape[0], x_test.shape[0]
    second_level_train_set = np.zeros((train_num,))
    second_level_test_set = np.zeros((test_num,))
    test_nfolds_sets = np.zeros((test_num, n_folds))
    kf = KFold(n_splits=n_folds)

    for i,(train_index, test_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[train_index], y_train[train_index]
        x_tst, y_tst =  x_train[test_index], y_train[test_index]

        clf.fit(x_tra, y_tra)

        second_level_train_set[test_index] = clf.predict(x_tst)
        test_nfolds_sets[:,i] = clf.predict(x_test)


    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)

    return second_level_train_set, second_level_test_set
def get_result(predict,ytest):
    tp = getTP(predict,ytest)
    fp = getFP(predict,ytest)
    tn = getTN(predict,ytest)
    fn = getFN(predict,ytest)
    print('tn:',tn,'fp:',fp,'fn:',fn,'tp:',tp)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = (2*precision*recall)/(precision+recall)

    return (accuracy,precision,recall,f1_score)

def getTP(predictions,input_y):
    count = 0
    for pre, real in zip(predictions,input_y):
        if pre == real and pre == 1:
            count += 1
    return count
def getFP(predictions,input_y):
    count = 0
    for pre, real in zip(predictions, input_y):
        if pre == 1 and real == 0:
            count += 1
    return count
def getTN(predictions,input_y):
    count = 0
    for pre, real in zip(predictions, input_y):
        if pre == real and real == 0:
            count += 1
    return count
def getFN(predictions,input_y):
    count = 0
    for pre, real in zip(predictions, input_y):
        if pre == 0 and real == 1:
            count += 1
    return count


if __name__ == "__main__":

    rf_model = RandomForestClassifier()
    adb_model = AdaBoostClassifier()
    gdbc_model = GradientBoostingClassifier()
    et_model = ExtraTreesClassifier()
    svc_model = SVC()


    for myK in range(5):
        train_x, test_x, train_y, test_y = get_dataset('../Features/features2.txt',myK=myK)

        train_sets = []
        test_sets = []
        for clf in [rf_model, adb_model, gdbc_model, et_model, svc_model]:
            train_set, test_set = get_stacking(clf, train_x, train_y, test_x)
            train_sets.append(train_set)
            test_sets.append(test_set)

        meta_train = np.concatenate([result_set.reshape(-1,1) for result_set in train_sets], axis=1)
        meta_test = np.concatenate([y_test_set.reshape(-1,1) for y_test_set in test_sets], axis=1)

        #使用决策树作为我们的次级分类器

        dt_model = DecisionTreeClassifier()
        dt_model.fit(meta_train, train_y)
        df_predict = dt_model.predict(meta_test)


        result = get_result(df_predict,test_y)
        print(result)