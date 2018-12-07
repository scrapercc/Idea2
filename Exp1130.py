import pandas as pd
import json
from treelib import Node,Tree
import codecs
from sklearn import svm
from sklearn.model_selection import ShuffleSplit
import numpy as np
from numpy import mean,std
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation,metrics
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
class Info(object):
    def __init__(self,data):
        self.reposts_count = data.get('reposts_count')
        self.uid = data.get('uid')
        self.bi_followers_count = data.get('bi_followers_count')
        self.text = data.get('text')
        self.user_description = data.get('user_description')
        self.friends_count = data.get('friends_count')
        self.mid = data.get('mid')
        self.attitudes_count = data.get('attitudes_count')
        self.followers_count = data.get('followers_count')
        self.statuses_count = data.get('statuses_count')
        self.verified = data.get('verified')
        self.user_created_at = data.get('user_created_at')
        self.favourites_count = data.get('favourites_count')
        self.gender = data.get('gender')
        self.comments_count = data.get('comments_count')
        self.t = data.get('t')

def modelfit(alg, XTrain, XTest,yTrain,yTest,useTrainCV=True, cv_folds=5, early_stopping_rounds=100):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(XTrain, label=yTrain)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, metrics='auc', early_stopping_rounds=early_stopping_rounds, show_stdv=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    #Fit the algorithm on the data
    alg.fit(XTrain, yTrain,eval_metric='auc')
    dtrain_predictions = alg.predict(XTest)
    y_fit = [round(value) for value in dtrain_predictions]
    #Print model report:
    print ("\nModel Report")
    # print ("Accuracy : %.4g" % metrics.accuracy_score(yTest,y_fit))
    res = get_result(y_fit, y_test)
    print(res)
    return res
    # feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')

def extract_features(eid,tree,label):
    features = {}
    # 初始化
    for i in range(1, 11):
        features[i] = { 'count': 0,
                        # '社交网络特征'
                        'rep_count': 0,
                        'comments_count': 0,
                        # 文本特征
                        # 'text_length': 0,
                        # 'text_NN_rat': 0,
                        #  'text_verb_rat': 0,
                        #  'text_adj_rat': 0,
                        #
                        #  'pos_count': 0,
                        #  'neg_count': 0,
                        #  'neu_count': 0,
                        #  '@_count': 0,
                        #  'stopword_count': 0,

                         # 用户特征
                         'bi_followers_count': 0,
                         # 'user_des_len': 0,
                         'friends_count': 0,
                         'verified_count': 0,
                         'followers_count': 0,
                         'statuses_count': 0,
                         'male_count': 0,  # m男 f女
                         'female_count': 0,
                         'favourites_count': 0,
                    }

    #提取特征，从第一层到第十层
    for node in tree.all_nodes_itr():
        level = tree.depth(node=node)
        if level <= 10 and level > 0: #只统计1到10层的个数
            features[level]['count'] += 1
            features[level]['rep_count'] += node.data.reposts_count
            features[level]['comments_count'] += node.data.comments_count

            # features[level]['pos_count'] += node.data.reposts_count
            # features[level]['neg_count'] += node.data.reposts_count
            # features[level]['neu_count'] += node.data.reposts_count

            features[level]['bi_followers_count'] += node.data.bi_followers_count
            features[level]['friends_count'] += node.data.friends_count
            if node.data.verified == True:
                features[level]['verified_count'] += node.data.verified
            features[level]['followers_count'] += node.data.followers_count
            features[level]['statuses_count'] += node.data.statuses_count
            if node.data.gender == 'm':
                features[level]['male_count'] += 1
            elif node.data.gender == 'f':
                features[level]['female_count'] += 1
            features[level]['favourites_count'] += node.data.favourites_count

    #对第一层到第十层的特征求平均值
    for i in range(1,11):
        if features[i]['count'] != 0:
            features[i].update({
                'rep_count': round(features[i]['rep_count'] / features[i]['count'],2),
                'comments_count': round(features[i]['comments_count'] / features[i]['count'], 2),

                'bi_followers_count': round(features[i]['bi_followers_count'] / features[i]['count'],2),
                'friends_count': round(features[i]['friends_count'] / features[i]['count'],2),
                'verified_count': round(features[i]['verified_count'] / features[i]['count'],2),
                'followers_count': round(features[i]['followers_count'] / features[i]['count'], 2),
                'statuses_count': round(features[i]['statuses_count'] / features[i]['count'], 2),
                'male_count': round(features[i]['male_count'] / features[i]['count'], 2),
                'female_count': round(features[i]['female_count'] / features[i]['count'], 2),
                'favourites_count': round(features[i]['favourites_count'] / features[i]['count'],2),
            })

    return features

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
def get_result(predict,ytest):
    res = {}
    tp = getTP(predict,ytest)
    fp = getFP(predict,ytest)
    tn = getTN(predict,ytest)
    fn = getFN(predict,ytest)
    # print('tn:',tn,'fp:',fp,'fn:',fn,'tp:',tp)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = (2*precision*recall)/(precision+recall)

    res['acc'] = accuracy
    res['pre'] = precision
    res['recall'] = recall
    res['f1'] = f1_score
    return res

def classify(X,y,method='svm'):
    clf = None
    if method == 'svm' or method == 'SVM':
        clf = svm.SVC(gamma='scale')
    elif method == 'random_forest' or method=='RandomForest':
        clf = RandomForestClassifier(n_estimators=10)
    elif method == 'decision_tree' or method=='DecisionTree':
        clf = tree.DecisionTreeClassifier()
    elif method == 'logic_regresion' or method == 'LogicRegresion':
        clf = LogisticRegression(C=1000.0, random_state=0)
    elif method=='xgboost':
        clf = XGBClassifier()
    elif method == 'xgboost2':
        param = {'max_depth': 6, 'eta': 1, 'silent': 0, 'objective': 'binary:logistic'}
        num_round = 10

    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f1_score = 0.0
    kf = ShuffleSplit(n_splits=5, random_state=0)
    for train, test in kf.split(X):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        if method == 'xgboost2':
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test,label=y_test)
            param = {'max_depth': 6, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
            num_round = 2
            bst = xgb.train(param, dtrain, num_round)
            preds = bst.predict(dtest)
            y_fit = [round(value) for value in preds]
            res = get_result(y_fit,y_test)
            print(res)
            accuracy += res['acc']
            precision += res['pre']
            recall += res['recall']
            f1_score += res['f1']
        else:
            # print('train0: %d, train1: %d' % (np.sum(y[train] == 0), np.sum(y[train] == 1)))
            # print('test0: %s, test1: %s' % (np.sum(y[test] == 0), np.sum(y[test] == 1)))
            clf.fit(X_train, y_train.ravel())
            y_fit = clf.predict(X_test)
            res = get_result(y_fit, y_test)
            print(res)
            accuracy += res['acc']
            precision += res['pre']
            recall += res['recall']
            f1_score += res['f1']
    print(accuracy/5,precision/5,recall/5,f1_score/5)

def get_dataset(path):
    pd_data = pd.read_csv(path, sep='\t', header=None)
    wb_data = pd_data.as_matrix()
    data = get_z_score(wb_data)
    X = data[:, 1:-1]
    y = data[:, -1]
    return X,y

def write_infos_to_file(path,info,eid,label):
    with codecs.open(path,'a+',encoding='utf-8') as info_file:
        info_file.write(eid+'\t')
        for key in info:
            info_file.write(str(info[key])+'\t')
        info_file.write(label+'\n')

def write_infos_to_file2(path,infos,eid,label):
    with codecs.open(path, 'a+', encoding='utf-8') as info_file:
        info_file.write(eid+'\t')
        for key1 in infos:#key1-->[1,10]
            info = infos[key1]
            for key2 in info:#key2-->[count,rep_count,comments_count...]
                info_file.write(str(info[key2])+'\t')
        info_file.write(label+'\n')

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


#根据参数类型计算传播树结构各层节点个数的平均值
def cal_node_level_count(type):

    data = pd.read_csv('D:/chenjiao/SinaWeibo/datasets2/Weibo.txt', sep='\t', header=None)
    if type=='fake':
        data = data.loc[data[1]=='label:1']
    elif type == 'real':
        data = data.loc[data[1] == 'label:0']

    data_array = data.as_matrix()

    tree_dict_list = []
    max_depth = 0
    all_infos = []
    for i in range(data_array.shape[0]):
        eid = str(data_array[i][0]).replace('eid:', '')
        label = str(data_array[i][1].replace('label:', ''))
        load_f = open('D:/chenjiao/SinaWeibo/datasets2/Weibo/{}.json'.format(eid), 'r', encoding='utf-8')
        json_data = json.load(load_f)
        print('-----',eid)
        tree = Tree()
        tree.create_node(json_data[0].get("mid"),json_data[0].get("mid"))

        for j in range(1,len(json_data)):
            try:
                tree.create_node(json_data[j].get("mid"),json_data[j].get("mid"),parent=json_data[j].get("parent"))
            except:
                pass
        # tree.show()

        tree_depth = tree.depth()
        if tree_depth > max_depth:
            max_depth = tree_depth

        #统计各层节点个数
        tree_dict = {}
        for node in tree.all_nodes_itr():
            level = tree.depth(node=node)
            if level not in tree_dict:
                tree_dict[level] = 1
            else:
                tree_dict[level] += 1
        #统计完

        tree_dict_list.append(tree_dict)

    tree_levels_count_list = {}
    for dict in tree_dict_list:
        for i in range(max_depth+1):
            if i in dict:
                if i in tree_levels_count_list:
                    tree_levels_count_list[i] += dict[i]
                else:
                    tree_levels_count_list[i] = dict[i]

    print(tree_levels_count_list)

    for key in tree_levels_count_list:
        print(key,tree_levels_count_list[key]/data_array.shape[0])


if __name__ == "__main__":
    # data = pd.read_csv('D:/chenjiao/SinaWeibo/datasets2/Weibo.txt', sep='\t', header=None)
    # data_array = data.as_matrix()
    #
    #
    # for i in range(data_array.shape[0]):
    #     eid = str(data_array[i][0]).replace('eid:', '')
    #     label = str(data_array[i][1].replace('label:', ''))
    #     load_f = open('D:/chenjiao/SinaWeibo/datasets2/Weibo/{}.json'.format(eid), 'r', encoding='utf-8')
    #     json_data = json.load(load_f)
    #     print('-----',eid)
    #     tree = Tree()
    #     tree.create_node(json_data[0].get("mid"),json_data[0].get("mid"))
    #
    #     for j in range(1,len(json_data)):
    #         try:
    #             tree.create_node(tag=json_data[j].get("mid"),identifier=json_data[j].get("mid"),parent=json_data[j].get("parent"),data=Info(json_data[j]))
    #         except:
    #             pass
    #     # tree.show()
    #
    #     features = extract_features(eid,tree,label)
    #     print(features)
    #     # write_infos_to_file('./Features/features1.txt',tree_dict,eid,label)
    #     write_infos_to_file2('./Features/features2.txt', features, eid, label)


    X,y = get_dataset('./Features/features2.txt')
    # classify(X,y,'xgboost2')





    # xgb1 = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8,
    #                      colsample_bytree=0.8, objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)

    # param_test1 = {'max_depth': range(3, 10, 2), 'min_child_weight': range(1, 6, 2)}
    # gsearch1 = GridSearchCV(
    #     estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5, min_child_weight=1, gamma=0,
    #                             subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', nthread=4,
    #                             scale_pos_weight=1, seed=27), param_grid=param_test1, scoring='roc_auc', n_jobs=4,
    #     iid=False, cv=5)

    # param_test2 = {'max_depth': [4, 5, 6, 7,8, 9], 'min_child_weight': [1]}
    # gsearch1 = GridSearchCV(
    #     estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5, min_child_weight=2, gamma=0,
    #                             subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', nthread=4,
    #                             scale_pos_weight=1, seed=27), param_grid=param_test2, scoring='roc_auc', n_jobs=4,
    #     iid=False, cv=5)

    # param_test3 = {'gamma': [i / 10.0 for i in range(0, 5)]}
    # gsearch1 = GridSearchCV(
    #     estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=8, min_child_weight=1, gamma=0,
    #                             subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', nthread=4,
    #                             scale_pos_weight=1, seed=27), param_grid=param_test3, scoring='roc_auc', n_jobs=4,
    #     iid=False, cv=5)

    xgb2 = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=8, min_child_weight=1, gamma=0, subsample=0.9,
                         colsample_bytree=0.9, objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)

    # param_test4 = {'subsample': [i / 10.0 for i in range(6, 10)], 'colsample_bytree': [i / 10.0 for i in range(6, 10)]}
    # gsearch1 = GridSearchCV(
    #     estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=6, min_child_weight=1, gamma=0.1,
    #                             subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', nthread=4,
    #                             scale_pos_weight=1, seed=27), param_grid=param_test4, scoring='roc_auc', n_jobs=4,
    #     iid=False, cv=5)

    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f1_score = 0.0
    kf = ShuffleSplit(n_splits=5, random_state=1)
    for train, test in kf.split(X):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        res = modelfit(xgb2, X_train, X_test,y_train,y_test)
        accuracy += res['acc']
        precision += res['pre']
        recall += res['recall']
        f1_score += res['f1']
    print(accuracy/5,precision/5,recall/5,f1_score/5)
        # gsearch1.fit(X_train, y_train)
        # print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

