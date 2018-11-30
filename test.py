import pandas as pd
import json
data = pd.read_csv('D:/chenjiao/SinaWeibo/datasets2/Weibo.txt',sep='\t',header=None)
data_array = data.as_matrix()

for i in range(data_array.shape[0]):
        eid = str(data_array[i][0]).replace('eid:', '')
        load_f = open('D:/chenjiao/SinaWeibo/datasets2/Weibo/{}.json'.format(eid), 'r', encoding='utf-8')
        json_data = json.load(load_f)
        parents_set = []
        for j in range(1,len(json_data)):
            parent = json_data[j].get('parent')
            parents_set.append(parent)
        parents_set = set(parents_set)
        print(eid,len(parents_set),len(json_data)-1,parents_set)