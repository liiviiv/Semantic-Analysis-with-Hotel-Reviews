#!/usr/bin/env python
# -*- coding: utf-8  -*-

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')  # 忽略警告
import logging
import os.path
import sys
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    
    # load word2vec model
    inp = '..\dataset\ChnSentiCorp_htl_ba_6000\\6000_all_cut.txt'
    output1 = 'word2vec.model'
    output2 = 'word2vec.vector'

    # size:生成的词向量的维度;
    # min_count:可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5;
    # window：即词向量上下文最大距离，skip-gram和cbow算法是基于滑动窗口来做预测,默认值为5，对于一般的语料推荐在[5,10]之间。
    model = Word2Vec(LineSentence(inp), size=300, min_count=5, window=5, workers=multiprocessing.cpu_count(), iter=1)
    # 生成word2vec词典
    model.build_vocab(inp)
    # 训练word2vec模型
    model.train(inp, total_examples=model.corpus_count, epochs=50)
    model.save('Word2vec_model.pkl')
    model.wv.save_word2vec_format('Word2vec_model.vector', binary=False)
    

    


