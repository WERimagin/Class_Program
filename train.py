#流れ
#0.データの処理->prepro.shで実行、dataから必要なデータを取り出しpickle化、word2id,id2vecの処理
#1.contexts,questionsを取り出しid化
#2.dataloaderからbatchを取り出し(ただのshuffleされたid列)、それに従いbatchを作成してtorch化
#3.モデルに入れてp1,p2(スタート位置、エンド位置を出力)
#4.predictはp1,p2それぞれのargmaxを取り、それと正解の位置を比較して出力する

import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append("../")
from tqdm import tqdm
import nltk
import pickle
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
import time
from model.seq2seq import Seq2Seq

from func.utils import BatchMaker,make_vec,to_var,logger,data_loader
from func.predict import loss_calc,predict_calc,predict_sentence
from func import constants
from func.parser import get_args
import nltk
import datetime
import os

#epochあたりの学習
def model_handler(args,data,train=True,data_kind="train"):
    start=time.time()
    sources=data["sources"]
    targets=data["targets"]
    t_id2word=data["t_id2word"]
    data_size=len(sources)
    batch_size=0
    if train:
        batch_size=args.train_batch_size
        model.train()
    else:
        batch_size=args.test_batch_size
        model.eval()
    #batchをランダムな配列で指定する
    batchmaker=BatchMaker(data_size,batch_size,True)
    batches=batchmaker()
    predict_rate=0
    loss_sum=0
    for i_batch,batch in enumerate(batches):
        #これからそれぞれを取り出し処理してモデルへ
        input_words=make_vec(args,[sources[i] for i in batch])
        output_words=make_vec(args,[targets[i] for i in batch])#(batch,seq_len)
        if train:
            optimizer.zero_grad()
        #modelにデータを渡してpredictする
        predict,_=model(input_words,output_words,train)#(batch,seq_len,vocab_size)
        #trainの場合はパラメータの更新を行う
        if train==True:
            #predictもoutput_wordsも<SOS>を除く
            loss=loss_calc(predict,output_words[:,1:])#batch*seq_lenをして内部で計算
            loss.backward()
            optimizer.step()
            loss_sum+=loss.data
            if i_batch%args.print_iter==0:
                now=time.time()
                logger(args,"epoch,{}\tbatch\t{}\tloss:{}\ttime:{}".format(epoch,i_batch,loss.data/args.train_batch_size,now-start))
                predict,target=predict_sentence(args,predict,output_words[:,1:],t_id2word)#(batch,seq_len)
        else:
            predict_rate+=predict_calc(predict,output_words[:,1:])
            predict,target=predict_sentence(args,predict,output_words[:,1:],t_id2word)#(batch,seq_len)
            if i_batch==0:
                for i in range(3):
                    logger(args,predict[i])
                    logger(args,target[i])
                    logger(args,"")

    #epochの記録
    if train:
        logger(args,"epoch:{}\ttime:{}\tloss:{}".format(epoch,time.time()-start,loss_sum/data_size))

    else:
        predict_rate=predict_rate/data_size
        logger(args,"predict_rate:{}".format(predict_rate))

        #テストデータにおいて、predict_rateが上回った時のみモデルを保存
        if data_kind=="train" and args.high_score<predict_rate:
            args.high_score=predict_rate
            args.high_epoch=epoch
            torch.save(model.state_dict(), "model_data/model.pth"\
                        .format(args.start_time,round(predict_rate,3),epoch))
            logger(args,"save model")

##start main
args=get_args()
train_data,test_data=data_loader(args,"data/processed_data.json",first=True)

device_kind="cuda:{}".format(args.cuda_number) if torch.cuda.is_available() else "cpu"
args.device=torch.device(device_kind)

model=Seq2Seq(args)
model.to(args.device)

#start_epochが0なら最初から、指定されていたら学習済みのものをロードする
if args.start_epoch>=1:
    param = torch.load("model_data/epoch_{}_model.pth".format(args.start_epoch-1))
    model.load_state_dict(param)
else:
    args.start_epoch=0


optimizer = optim.Adam(model.parameters(),lr=args.lr)

logger(args,"use {}".format(device_kind))

for epoch in range(args.start_epoch,args.epoch_num):
    logger(args,"start_epoch:{}".format(epoch))
    #学習を行わない。主にcpu上でのテスト
    if args.not_train==False:
        model_handler(args,train_data,train=True)
        model_handler(args,train_data,train=False)
    model_handler(args,test_data,data_kind="test",train=False)
    logger(args,"")
