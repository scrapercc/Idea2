import pandas as pd
from numpy import mean,std
import numpy as np
import random
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
    return Xtrain, Xtest, ytrain, ytest

def random_sample(XTrain,YTrain):
    sample = []
    for i in range(XTrain.shape[0]):
        sample.append(random.randint(0,XTrain.shape[0]-1))
    print(sample)
    return XTrain[sample],YTrain[sample]

def get_result(predict,ytest):
    tp = getTP(predict,ytest)
    fp = getFP(predict,ytest)
    tn = getTN(predict,ytest)
    fn = getFN(predict,ytest)
    print('tn:',tn,'fp:',fp,'fn:',fn,'tp:',tp)
    accuracy = (tp+fn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = (2*precision*recall)/(precision+recall)

    return (accuracy,precision,recall,f1_score)
    # precision = getTP(predict, ytest) / (getTP(predict, ytest) + getFP(predict, ytest))
    # recall = getTP(predict, ytest) / (getTP(predict, ytest) + getFN(predict, ytest))
    # f1_score = 2 * getTP(predict, ytest) / (2 * getTP(predict, ytest) + getFP(predict, ytest) + getFN(predict, ytest))
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
def get_yfit(sub_fits,type='vote'):
    label_1_count = np.sum(sub_fits, axis=0)
    y_fit = []
    if type == 'vote':
        for count_1 in label_1_count:
            if count_1 >= 3:
                y_fit.append(1)
            else:
                y_fit.append(0)
    return y_fit

if __name__ == "__main__":
    model_num = 5
    for myK in range(1):
        Xtrain, Xtest, ytrain, ytest = get_dataset('../Features/features2.txt',myK=myK)

        y_pred = []
        for i in range(model_num):
            X_train, y_train = random_sample(Xtrain, ytrain)
            base_model = tree.DecisionTreeClassifier()
            base_model.fit(X_train, y_train)
            sub_fit = base_model.predict(Xtest)
            print(type(sub_fit))
            y_pred.append(sub_fit)

        y_preds = np.array(y_pred)

        y_fit = get_yfit(sub_fits=y_preds,type='vote')

        result = get_result(y_fit,ytest)
        print(result)