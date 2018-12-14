
import gensim
import codecs
from gensim import corpora
from sklearn.model_selection import ShuffleSplit
import os

def get_docs(path):
    data_segment = codecs.open(path, 'r', encoding='utf-8').readlines()
    docs = []
    for line in data_segment:
        text = line.split('\t')[1]
        doc = text.replace('\n','').split(' ')
        docs.append(doc)
    # print(docs)
    return docs

def build_ldaModel(train_docs,num_topics=2,passes=50):
    dictionary = corpora.Dictionary(train_docs)
    # print(dictionary.token2id)

    doc_term_matrix = [dictionary.doc2bow(doc) for doc in train_docs]
    ldamodel = gensim.models.ldamodel.LdaModel(doc_term_matrix, num_topics=num_topics, id2word=dictionary, passes=passes)

    return dictionary,ldamodel

if __name__ == "__main__":
    docs = get_docs('text_segment4.txt')

    kf = ShuffleSplit(n_splits=5, random_state=0)
    train_index = []
    test_index = []
    for train, test in kf.split(docs):
        train_index.append(train)
        test_index.append(test)

    num_topics = 3
    passes = 50
    dir = './Text_lda_pro/Num_topics_{}/'.format(str(num_topics))
    if not os.path.exists(dir):
        os.makedirs(dir)

    for i in range(5):
        print("==============第",i+1,"次交叉")
        train_docs = []

        for index in train_index[i].tolist():
            train_docs.append(docs[index])


        dictionary,ldamodel = build_ldaModel(train_docs,num_topics=num_topics,passes=passes)

        data_segment = codecs.open('text_segment4.txt', 'r', encoding='utf-8').readlines()

        text_ldaPro = codecs.open(dir+'lda_pro_{}'.format(str(i)),'a+',encoding='utf-8')
        for line in data_segment:
            eid = line.split('\t')[0]
            doc = line.split('\t')[1].replace('\n','').split(' ')

            doc_bow = dictionary.doc2bow(doc)  # 文档转换成bow
            doc_lda = ldamodel[doc_bow]  # 得到新文档的主题分布
            # 输出新文档的主题分布

            text_ldaPro.write(eid+'\t')
            for index,topic in enumerate(doc_lda):
                if index < len(doc_lda)-1:
                    text_ldaPro.write(str(topic[1])+'\t')
                else:
                    text_ldaPro.write(str(topic[1]) + '\n')

            #     print(ldamodel.print_topic(topic[0]))
                print(topic[1])


