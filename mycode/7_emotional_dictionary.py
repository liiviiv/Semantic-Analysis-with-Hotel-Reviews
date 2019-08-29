from collections import defaultdict
import codecs


def classify_words(word_dict, sen_dict, not_word_list, degree_dic):
    """词语分类,找出情感词、否定词、程度副词"""
    # 分类结果，词语的index作为key,词语的分值作为value，否定词分值设为-1
    sen_word = dict()
    not_word = dict()
    degree_word = dict()
    # 分类
    for word in word_dict.keys():
        if word in sen_dict.keys() and word not in not_word_list and word not in degree_dic.keys():
            # 找出分词结果中在情感字典中的词
            sen_word[word_dict[word]] = sen_dict[word]
        elif word in not_word_list and word not in degree_dic.keys():
            # 分词结果中在否定词列表中的词
            not_word[word_dict[word]] = -1
        elif word in degree_dic.keys():
            # 分词结果中在程度副词中的词
            degree_word[word_dict[word]] = degree_dic[word]

    # 将分类结果返回
    return sen_word, not_word, degree_word


def list_to_dict(word_list):
    """将分词后的列表转为字典，key为单词，value为单词在列表中的索引，索引相当于词语在文档中出现的位置"""
    data = {}
    for x in range(0, len(word_list)):
        data[word_list[x]] = x
    return data


def get_init_weight(sen_word, not_word, degree_word):
    # 权重初始化为1
    W = 1
    # 将情感字典的key转为list
    sen_word_index_list = list(sen_word.keys())
    if len(sen_word_index_list) == 0:
        return W
    # 获取第一个情感词的下标，遍历从0到此位置之间的所有词，找出程度词和否定词
    for i in range(0, sen_word_index_list[0]):
        if i in not_word.keys():
            W *= -1
        elif i in degree_word.keys():
            # 更新权重，如果有程度副词，分值乘以程度副词的程度分值
            W *= float(degree_word[i])
    return W


def socre_sentiment(sen_word, not_word, degree_word, seg_result):
    """计算得分"""
    # 权重初始化为1
    W = 1
    score = 0
    # 情感词下标初始化
    sentiment_index = -1
    # 情感词的位置下标集合
    sentiment_index_list = list(sen_word.keys())
    # 遍历分词结果(遍历分词结果是为了定位两个情感词之间的程度副词和否定词)
    for i in range(0, len(seg_result)):
        # 如果是情感词（根据下标是否在情感词分类结果中判断）
        if i in sen_word.keys():
            # 权重*情感词得分
            score += W * float(sen_word[i])
            W = 1 + i/len(seg_result) #越接近句尾的情感词权重更大
            # 情感词下标加1，获取下一个情感词的位置
            sentiment_index += 1
            if sentiment_index < len(sentiment_index_list) - 1:
                # 判断当前的情感词与下一个情感词之间是否有程度副词或否定词
                for j in range(sentiment_index_list[sentiment_index], sentiment_index_list[sentiment_index + 1]):
                    # 更新权重，如果有否定词，取反
                    if j in not_word.keys():
                        W *= -1
                    elif j in degree_word.keys():
                        # 更新权重，如果有程度副词，分值乘以程度副词的程度分值
                        W *= float(degree_word[j])
        # 定位到下一个情感词
        if sentiment_index < len(sentiment_index_list) - 1:
            i = sentiment_index_list[sentiment_index + 1]
    return score


# 计算得分
def setiment_score(seg_list, sen_dict, not_word_list, degree_dic):
    # 1.将分词结果列表转为dic，然后找出情感词、否定词、程度副词
    sen_word, not_word, degree_word = classify_words(list_to_dict(seg_list), sen_dict, not_word_list, degree_dic)
    # 2.计算得分
    score = socre_sentiment(sen_word, not_word, degree_word, seg_list)
    return score


# 测试
if __name__ == '__main__':
    # get emotional dictionary:
    # 读取情感字典文件
    sen_file = open('..\dataset\宾馆情感词典.txt', 'r+', encoding='utf-8')
    # 获取字典文件内容
    sen_list = sen_file.readlines()
    # 创建情感字典
    sen_dict = defaultdict()
    # 读取字典文件每一行内容，将其转换为字典对象，key为情感词，value为对应的分值
    for s in sen_list:
        # 每一行内容根据空格分割，索引0是情感词，索引01是情感分值
        sen_dict[s.split(' ')[0]] = s.split(' ')[1]

    # 读取否定词文件
    not_word_file = open('..\dataset\否定词.txt', 'r+', encoding='utf-8')
    # 由于否定词只有词，没有分值，使用list即可
    not_word_list = not_word_file.readlines()

    # 读取程度副词文件
    degree_file = open('..\dataset\程度词.txt', 'r+', encoding='utf-8')
    degree_list = degree_file.readlines()
    degree_dic = defaultdict()
    # 程度副词与情感词处理方式一样，转为程度副词字典对象，key为程度副词，value为对应的程度值
    for d in degree_list:
        degree_dic[d.split(',')[0]] = d.split(',')[1]

    # test the positive samples:
    filename1 = '..\dataset\ChnSentiCorp_htl_ba_6000\\6000_pos_cut.txt'
    pos_cnt1 = 0
    neg_cnt1 = 0
    nul1 = 0
    with codecs.open(filename1, 'rb', encoding='gbk') as contents:
        for line in contents:
            # logger.info("Start line: " + line)
            line = line.strip('\r\n')  # 去除句子首尾的换行符号
            wordList = line.split(' ')
            score = setiment_score(wordList, sen_dict, not_word_list, degree_dic)
            # for each sentence, predict it's positive or negative using the emotional score,
            # and compare the score with its real label:
            if score > 0:
                pos_cnt1 += 1
            elif score == 0:
                # print(wordList)
                nul1 += 1
            else:
                neg_cnt1 += 1
    print('Positive samples: pos:', pos_cnt1, ',neg:', neg_cnt1, ',not find', nul1)

    # test the negative samples:
    filename2 = '..\dataset\ChnSentiCorp_htl_ba_6000\\6000_neg_cut.txt'
    pos_cnt2 = 0
    neg_cnt2 = 0
    nul2 = 0
    with codecs.open(filename2, 'rb', encoding='gbk') as contents:
        for line in contents:
            # logger.info("Start line: " + line)
            line = line.strip('\r\n')  # 去除句子首尾的换行符号
            wordList = line.split(' ')
            score = setiment_score(wordList, sen_dict, not_word_list, degree_dic)
            # for each sentence, predict it's positive or negative using the emotional score,
            # and compare the score with its real label:
            if score > 0:
                # print(wordList)
                pos_cnt2 += 1
            elif score == 0:
                nul2 += 1
            else:
                neg_cnt2 += 1
    print('Negative samples: pos:', pos_cnt2, ',neg:', neg_cnt2, ',not find', nul2)

    # close these dictionary files:
    sen_file.close()
    degree_file.close()
    not_word_file.close()