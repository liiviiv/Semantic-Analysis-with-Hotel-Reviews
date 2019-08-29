#!/usr/bin/env python
# -*- coding: utf-8  -*-
# 逐行读取文件数据进行jieba分词
import jieba
import jieba.analyse
import codecs, sys, string, re


# 文本分词
def prepareData(sourceFile, targetFile, stopwords):
    f = codecs.open(sourceFile, 'r', encoding='gbk')
    target = codecs.open(targetFile, 'w', encoding='gbk')
    lineNum = 1
    line = f.readline()
    while line:
        line = clearTxt(line)
        seg_line = sent2word(line, stopwords)
        target.write(seg_line + '\r\n')
        lineNum = lineNum + 1
        line = f.readline()
    print(sourceFile, ', done.')
    f.close()
    target.close()


# 清洗文本
def clearTxt(line):
    if line != '':
        line = line.strip()  # 去除句子首尾的空格
        # intab = ""
        # outtab = ""
        # trantab = str.maketrans(intab, outtab)
        # pun_num = string.punctuation + string.digits
        # line = line.encode('utf-8')
        # line = line.translate(trantab, pun_num)
        # line = line.decode("utf8")
        # 去除文本中的英文和数字:
        line = re.sub("[a-zA-Z0-9]", "", line)
        # 去除文本中的中文符号和英文符号:
        line = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）]+", "", line)
    return line


# 文本切割
def sent2word(line, stopwords):
    segList = jieba.cut(line, cut_all=False)
    segSentence = ''
    for word in segList:
        if word != '\t' and (word not in stopwords):
            segSentence += (word + " ")
    return segSentence.strip()


# 将数据清洗后的正负语料特征文本合并到一个文件中
def combine_pos_and_neg(targetfile, negfile, posfile):
    target = codecs.open(targetfile, 'w', encoding='utf-8')
    # copy 6000_neg_cut.txt to 6000_all_cut.txt:
    f1 = codecs.open(negfile, 'r', encoding='gbk')
    line = f1.readline()
    while line:
        target.write(line + '\r\n')
        line = f1.readline()
    f1.close()
    # copy 6000_pos_cut.txt to 6000_all_cut.txt:
    f2 = codecs.open(posfile, 'r', encoding='gbk')
    line = f2.readline()
    while line:
        target.write(line + '\r\n')
        line = f2.readline()
    f2.close()
    target.close()
    print('done')


if __name__ == '__main__':
    stopwords = [w.strip() for w in codecs.open('stopWord.txt', 'r', encoding='utf-8')]

    sourceFile1 = '..\dataset\ChnSentiCorp_htl_ba_6000\\6000_neg.txt'
    targetFile1 = '..\dataset\ChnSentiCorp_htl_ba_6000\\6000_neg_cut.txt'
    prepareData(sourceFile1, targetFile1, stopwords)

    sourceFile2 = '..\dataset\ChnSentiCorp_htl_ba_6000\\6000_pos.txt'
    targetFile2 = '..\dataset\ChnSentiCorp_htl_ba_6000\\6000_pos_cut.txt'
    prepareData(sourceFile2, targetFile2, stopwords)

    targetFile3 = '..\dataset\ChnSentiCorp_htl_ba_6000\\6000_all_cut.txt'
    combine_pos_and_neg(targetFile3, targetFile1, targetFile2)
