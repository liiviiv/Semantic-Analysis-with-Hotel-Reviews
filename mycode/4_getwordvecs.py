#!/usr/bin/env python
# -*- coding: utf-8  -*-

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')# 忽略警告
import logging
import os.path
import codecs,sys
import numpy as np
import pandas as pd
import gensim
from gensim.models import Word2Vec


# 返回特征词向量
def getWordVecs(wordList, model):
    vecs = []
    for word in wordList:
        word = word.replace('\n', '')
        try:
            vecs.append(model[word])
        except KeyError:
            continue
    return np.array(vecs, dtype='float')
    

# 构建文档词向量 
def buildVecs(filename, model):
    fileVecs = []
    with codecs.open(filename, 'rb', encoding='gbk') as contents:
        for line in contents:
            # logger.info("Start line: " + line)
            line = line.strip('\r\n')  # 去除句子首尾的换行符号
            wordList = line.split(' ')
            vecs = getWordVecs(wordList, model)
            # for each sentence, the mean vector of all its vectors is used to represent this sentence
            if len(vecs) > 0:
                vecsArray = sum(np.array(vecs))/len(vecs) # mean
                fileVecs.append(vecsArray)
    return fileVecs   


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    
    # load word2vec model
    # model = Word2Vec.load('Word2vec_model.pkl')
    model = gensim.models.KeyedVectors.load_word2vec_format('Word2vec_model2.vector', binary=False)  # 加载单词向量

    posInput = buildVecs('..\dataset\ChnSentiCorp_htl_ba_6000\\6000_pos_cut.txt', model)
    negInput = buildVecs('..\dataset\ChnSentiCorp_htl_ba_6000\\6000_neg_cut.txt', model)

    # use 1 for positive sentiment， 0 for negative
    Y = np.concatenate((np.ones(len(posInput)), np.zeros(len(negInput))))

    X = posInput[:]
    for neg in negInput:
        X.append(neg)
    X = np.array(X)

    # write in file   
    df_x = pd.DataFrame(X)
    df_y = pd.DataFrame(Y)
    data = pd.concat([df_y, df_x], axis=1)
    data.to_csv('6000_word_vector.csv')
    

    


