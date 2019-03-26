import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append("../")
from tqdm import tqdm
import nltk
import pickle
import json

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt

from model.seq2seq import Seq2Seq
from func.utils import BatchMaker,make_vec,to_var,logger,data_loader
from func.predict import loss_calc,predict_calc,predict_sentence
from func import constants
from func.parser import get_args
import nltk

#attention_result:(tgt_len,src_len)
def show_result(source,target,predict,attention_result):
    print(source)
    print(target)
    print(predict)
    print(attention_result)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax=ax.matshow(attention_result.tolist())
    fig.colorbar(cax)

    ax.set_xticklabels(['']+source.split())
    ax.set_yticklabels(['']+predict.split())
    plt.show()


#epochあたりの学習
def model_handler(args,data,train=True):
    start=time.time()
    sources=data["sources"]
    targets=data["targets"]
    s_id2word=data["s_id2word"]
    t_id2word=data["t_id2word"]
    data_size=len(sources)

    batch_size=args.test_batch_size
    model.eval()
    #batchをランダムな配列で指定する
    batchmaker=BatchMaker(data_size,batch_size,True)
    batches=batchmaker()
    predict_rate=0
    loss_sum=0
    #生成した文を保存するリスト
    predicts=[]
    targets_list=[]
    for i_batch,batch in enumerate(batches):
        if i_batch>=5:
            break
        #これからそれぞれを取り出し処理してモデルへ
        input_words=make_vec(args,[sources[i] for i in batch])
        output_words=make_vec(args,[targets[i] for i in batch])#(batch,seq_len)
        #modelにデータを渡してpredictする
        predict,attention_result=model(input_words,output_words,train)#(batch,seq_len,vocab_size)
        predict,target=predict_sentence(args,predict,output_words[:,1:],t_id2word)#(batch,seq_len)
        source=[" ".join([s_id2word[id] for id in sources[i]]) for i in batch]#idから単語へ戻す
        show_result(source[0],target[0],predict[0],attention_result)

##start main
args=get_args()
train_data,test_data=data_loader(args,"data/processed_data.json",first=True)
test_data=test_data if args.use_train_data==False else train_data

device_kind="cuda:{}".format(args.cuda_number) if torch.cuda.is_available() else "cpu"
args.device=torch.device(device_kind)
args.test_batch_size=1

model=Seq2Seq(args)
model.to(args.device)

if args.model_name!="":
    param = torch.load("model_data/{}".format(args.model_name),map_location=device_kind)
    model.load_state_dict(param)
#start_epochが0なら最初から、指定されていたら学習済みのものをロードする
elif args.start_epoch>=1:
    param = torch.load("model_data/epoch_{}_model.pth".format(args.start_epoch-1))
    model.load_state_dict(param)
else:
    args.start_epoch=0

#pytorch0.4より、OpenNMT参考
device=torch.device("cuda:{}".format(args.cuda_number) if torch.cuda.is_available() else "cpu")
model.to(device)

model_handler(args,test_data,train=False)
