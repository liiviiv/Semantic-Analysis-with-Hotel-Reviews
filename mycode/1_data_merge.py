#!/usr/bin/env python
# -*- coding: utf-8  -*-
import logging
import os.path
import codecs
import sys

# get the content of one single file, i.e. a sample:
def getContent(fullname):
    f = codecs.open(fullname, 'r', encoding='gbk', errors="ignore")
    # content = f.readline()
    content = []
    for eachline in f:
        # eachline = eachline.decode('gbk','ignore').strip()
        eachline = eachline.strip()
        if eachline:  # 很多空行
            content.append(eachline)
    f.close()
    return content
    

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])#得到文件名
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)

    inp = '..\dataset\ChnSentiCorp_htl_ba_6000'
    folders = ['neg', 'pos']
    for foldername in folders:
        logger.info("running " + foldername +" files.")
        outp = inp + '\\' + '6000_' + foldername +'.txt' # 输出文件
        output = codecs.open(outp, 'w')
        i = 0
        rootdir = inp + '\\' + foldername
        # parent:father directory, e.g. '..\dataset\ChnSentiCorp_htl_ba_2000\neg'
        # dirnames:the names of folders(without file names), if there is no folder in rootdir, dirnames is empty
        # filenames:all file names in rootdir
        for parent, dirnames, filenames in os.walk(rootdir):
            for filename in filenames:
                content = getContent(rootdir + '\\' + filename)
                # every file/sample is written in one line:
                # output.writelines(content)
                output.write(''.join(content) + '\n')
                i = i+1
        output.close()
        logger.info("Saved "+str(i)+" files.")
