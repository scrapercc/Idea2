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

def extract_features(eid,tree,label):

    tree_dict = {}
    #初始化
    for i in range(1,11):
        tree_dict[str(i)] = 0

    for node in tree.all_nodes_itr():
        level = tree.depth(node=node)
        if level <= 10 and level > 0: #只统计1到10层的个数
            tree_dict[str(level)] += 1

    return tree_dict

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
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f1_score = 0.0
    kf = ShuffleSplit(n_splits=5, random_state=0)
    for train, test in kf.split(X):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
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


    X = wb_data[:, 1:-1]
    y = wb_data[:, -1]
    return X,y

def write_infos_to_file(path,info,eid,label):
    with codecs.open(path,'a+',encoding='utf-8') as info_file:
        info_file.write(eid+'\t')
        for key in info:
            info_file.write(str(info[key])+'\t')
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
    data = pd.read_csv('D:/chenjiao/SinaWeibo/datasets2/Weibo.txt', sep='\t', header=None)
    data_array = data.as_matrix()


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

        tree_dict = extract_features(eid,tree,label)

        # write_infos_to_file('./Features/features1.txt',tree_dict,eid,label)

    # data = pd.read_csv('./Features/features1.txt',header=None,sep='\t').as_matrix()
    # zscores = get_z_score(data)
    # X= zscores[:,1:-1]
    # y =zscores[:,-1]
    # classify(X,y,'logic_regresion')





