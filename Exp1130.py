import pandas as pd
import json
from treelib import Node,Tree

# class Node:
#
#     def __init__(self,mid):
#         self.mid = mid
#         self.uid = None
#         self.parent = None
#         self.childs = None
#         self.level = None
#         self.text = None
#         self.time = None
#
if __name__ == "__main__":
    data = pd.read_csv('D:/chenjiao/SinaWeibo/datasets2/Weibo.txt', sep='\t', header=None)
    data_array = data.as_matrix()

    # for i in range(1):
    #     eid = str(data_array[i][0]).replace('eid:', '')
    #     load_f = open('D:/chenjiao/SinaWeibo/datasets2/Weibo/{}.json'.format(eid), 'r', encoding='utf-8')
    #     json_data = json.load(load_f)
    #     tree = Tree()
    #     tree.create_node(json_data[0].get("mid"),json_data[0].get("mid"))
    #
    #     for j in range(1,len(json_data)):
    #         tree.create_node(json_data[j].get("mid"),json_data[j].get("mid"),parent=json_data[j].get("parent"))
    #     tree.show()


    load_f = open('D:/chenjiao/SinaWeibo/datasets2/Weibo/3467820220106487.json', 'r', encoding='utf-8')
    json_data = json.load(load_f)
    tree = Tree()
    tree.create_node(json_data[0].get("mid"), json_data[0].get("mid"))

    for j in range(1, len(json_data)):
        tree.create_node(json_data[j].get("mid"), json_data[j].get("mid"), parent=json_data[j].get("parent"))
    tree.show()

