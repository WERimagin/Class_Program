#SQuADのデータ処理
#必要条件:CoreNLP
#Tools/core...で
#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

import os
import sys
sys.path.append("../")
import json
import gzip
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize,sent_tokenize
import pickle
import collections
import random

def answer_find(context_text,answer_start,answer_end):
    context=sent_tokenize(context_text)
    start_p=0

    #start_p:対象となる文の文字レベルでの始まりの位置
    #end_p:対象となる文の文字レベルでの終端の位置
    #answer_startがstart_pからend_pの間にあるかを確認。answer_endも同様
    for i,sentence in enumerate(context):
        end_p=start_p+len(sentence)
        if start_p<=answer_start and answer_start<=end_p:
            sentence_start_id=i
        if start_p<=answer_end and answer_end<=end_p:
            sentence_end_id=i
        #スペースが消えている分の追加、end_pの計算のところでするべきかは不明
        start_p+=len(sentence)+1

    #得られた文を結合する（大抵は文は一つ）
    answer_sentence=" ".join(context[sentence_start_id:sentence_end_id+1])

    return answer_sentence

#sentenceを受け取り、tokenizeして返す
def tokenize(sent):
    return [token.replace('``','"').replace("''",'"') for token in word_tokenize(sent)]

#単語のリストとgloveのword2vecからword2idとid2vecを生成し、保存
def vec_process(source_word,target_word):

    vec_size=300
    source_word=sorted(source_word.items(),key=lambda x:-x[1])
    target_word=sorted(target_word.items(),key=lambda x:-x[1])

    s_word2id={w:i for i,(w,count) in enumerate(source_word,6) if count>=0}
    t_word2id={w:i for i,(w,count) in enumerate(target_word,6) if count>=0}
    s_word2id["<PAD>"]=0
    s_word2id["<UNK>"]=1
    s_word2id["<SOS>"]=2
    s_word2id["<EOS>"]=3
    s_word2id["<SEP>"]=4
    s_word2id["<SEP2>"]=5

    t_word2id["<PAD>"]=0
    t_word2id["<UNK>"]=1
    t_word2id["<SOS>"]=2
    t_word2id["<EOS>"]=3
    t_word2id["<SEP>"]=4
    t_word2id["<SEP2>"]=5

    print(len(list(s_word2id.items())))
    print(len(list(t_word2id.items())))

    with open("data/word2id.json","w")as f:
        t={"s_word2id":s_word2id,
            "t_word2id":t_word2id}
        json.dump(t,f)

def data_process(input_path,output_path,train=False):
    data=[]
    with open(input_path,"r") as f:
        for line in f:
            data.append(line.strip())



    source=[]
    target=[]
    source_word=collections.defaultdict(int)
    target_word=collections.defaultdict(int)

    for line in tqdm(data[0:]):
        line=line.lower()
        line=line.split("\t")
        s=tokenize(line[1])
        t=tokenize(line[0])
        for w in s:
            source_word[w]+=1
        for w in t:
            target_word[w]+=1
        source.append(" ".join(s))
        target.append(" ".join(t))

    print(len(source))
    print(len(target))


    with open(output_path,"w")as f:
        mydict={"source":source,
                "target":target}
        json.dump(mydict,f)

    if train==True:
        vec_process(source_word,target_word)


if __name__ == "__main__":
    #main
    random.seed(0)


    data_process(input_path="data/eng-fra.txt",
                output_path="data/processed_data.json",
                train=True
                )
