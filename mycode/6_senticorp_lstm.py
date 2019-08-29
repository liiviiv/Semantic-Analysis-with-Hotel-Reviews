import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from numpy import *
from gensim.models import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.models import model_from_yaml
import yaml
import gensim
import numpy as np
import codecs

word_vector_dim = 300  # 词编码的维度
max_sentence_len = 100  # 最大句子长度
batch_size = 32
epoch = 10


# 构造并训练lstm网络
def train_lstm(w2idx, w2vec, x_train, y_train, x_test, y_test):

    # 构造word2vec模型训练后生成的索引词典和词向量词典之间的映射矩阵，用来初始化情感分类网络模型的Ensemble层的权重：
    word_num = len(w2idx) + 1 # 计算词典所含词的个数，加1是因为有未出现在词典中的词语（位置索引为0）
    embedding_weights = np.zeros((word_num, word_vector_dim))
    for word, index in w2idx.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = w2vec[word]

    # 定义lstm网络结构：
    model = Sequential()
    model.add(Embedding(input_dim=word_num, output_dim=word_vector_dim, mask_zero=True, weights=[embedding_weights],
                        input_length=max_sentence_len))
    model.add(LSTM(output_dim=50, activation='sigmoid', inner_activation='hard_sigmoid', dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(30))
    #model.add(Dropout(0.5))
    model.add(Dense(1))

    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # 训练网络模型：
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch, verbose=2, validation_data=(x_test, y_test))

    # 测试训练好的网络模型在测试集上的的性能：
    score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    print('\ntest_loss:', score[0], 'test_acc:', score[1])  # 返回损失值和准确率

    # 计算查准率和查全率：
    TP = 0  # 标签为1，模型预测值为1
    FP = 0  # 标签为0，模型预测值为1
    FN = 0  # 标签为1，模型预测值为0
    TN = 0  # 标签为0，模型预测值为0
    for i in range(len(x_test)):
        x_vec = x_test[i].reshape(1, 100)
        y_vec = y_test[i].reshape(1, 1)
        if model.predict_classes(x_vec)[0][0] == 1 and y_vec == 1:
            TP += 1
        elif model.predict_classes(x_vec)[0][0] == 1 and y_vec == 0:
            FP += 1
        elif model.predict_classes(x_vec)[0][0] == 0 and y_vec == 1:
            FN += 1
        elif model.predict_classes(x_vec)[0][0] == 0 and y_vec == 0:
            TN += 1
    print('TP=', TP, 'FP=', FP, 'FN=', FN, 'TN=', TN)
    print('查准率(pos)=', TP / (TP + FP))
    print('查全率(pos)=', TP / (TP + FN))
    print('查准率(neg)=', TN / (FN + TN))
    print('查全率(neg)=', TN / (FP + TN))

    yaml_string = model.to_yaml()
    with open('lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights('lstm.h5')


# 生成每个词的索引及编码向量词典
def create_dictionaries(combined=None):
    # 加载word2vec模型:
    model = gensim.models.KeyedVectors.load_word2vec_format('Word2vec_model2.vector', binary=False)  # 加载单词向量
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
        # 生成词的索引词典
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}
        # 生成词向量矩阵词典
        w2vec = {word: model[word] for word in w2indx.keys()}

        # 对于经过分词后的每个句子，将句子中的每个特征词使用word2vec模型转化为对应的词典中的索引，若某个词没有编码向量，则索引设为0
        # 通过这种方式将每个句子/样本转化为向量，向量的维度就是该句子中特征词的个数。
        x_idx = []
        for sentence in combined:
            new_txt = []
            sentence = sentence.strip('\r\n')  # 去除句子首尾的换行符号
            wordList = sentence.split(' ')  # 将句子切割成词语
            for word in wordList:
                try:
                    new_txt.append(w2indx[word])
                except:
                    new_txt.append(0)
            x_idx.append(new_txt)

        # 句子长度归一化
        x_idx = sequence.pad_sequences(x_idx, maxlen=max_sentence_len)
        return w2indx, w2vec, x_idx
    else:
        print('Errors when transfer the x_words to x_idx')


if __name__ == '__main__':
    filename_pos = '..\dataset\ChnSentiCorp_htl_ba_6000\\6000_pos_cut.txt'
    filename_neg = '..\dataset\ChnSentiCorp_htl_ba_6000\\6000_neg_cut.txt'
    with codecs.open(filename_pos, 'rb', encoding='gbk') as file_pos:
        pos_words = file_pos.readlines()
    with codecs.open(filename_neg, 'rb', encoding='gbk') as file_neg:
        neg_words = file_neg.readlines()
    x_words = np.concatenate((pos_words, neg_words))
    # use 1 for positive sentiment， 0 for negative
    y = np.concatenate((np.ones(len(pos_words)), np.zeros(len(neg_words))))

    # 生成每个词的索引及编码向量词典，并将样本转化为索引向量
    w2idx, w2vec, x_idx = create_dictionaries(combined=x_words)

    # 划分训练、测试集:
    x_train, x_test, y_train, y_test = train_test_split(x_idx, y, test_size=0.2, random_state=1)
    # print(x_train.shape, y_train.shape)
    train_lstm(w2idx, w2vec, x_train, y_train, x_test, y_test)


