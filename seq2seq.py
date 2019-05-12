import os
import sys
import json
import gzip
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize,sent_tokenize
import pickle
import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

PAD=0
UNK=1
SOS=2

device_kind="cuda" if torch.cuda.is_available() else "cpu"
device=torch.device(device_kind)

CUDA_LAUNCH_BLOCKING=1

def word2idmaker(sentences,vocab_size):
    words=collections.defaultdict(int)
    for sentence in sentences:
        for w in sentence:
            words[w]+=1

    words=sorted(words.items(),key=lambda x:-x[1])
    word2id={w:i for i,(w,count) in enumerate(words[0:vocab_size],5) if count>=0}
    word2id["<PAD>"]=0
    word2id["<UNK>"]=1
    word2id["<SOS>"]=2
    word2id["<EOS>"]=3
    word2id["<SEP>"]=4

    id2word={i:w for w,i in list(word2id.items())}

    return word2id,id2word

def process_word2id(sentences,word2id_dict,tgt):
    sentences=[[word2id_dict[w] if w in word2id_dict else word2id_dict["<UNK>"]  for w in sentence]  for sentence in sentences]
    if tgt:
        sentences=[[word2id_dict["<SOS>"]] + sentence + [word2id_dict["<EOS>"]] for sentence in sentences]
    return sentences

def data_loader(src_path,tgt_path,train):
    src=[]
    tgt=[]
    with open(src_path)as f:
        for i,line in enumerate(f):
            src.append(line.strip().split())

    with open(tgt_path)as f:
        for line in f:
            tgt.append(line.strip().split())

    src_r=[]
    tgt_r=[]
    for i in range(len(src)):
      if len(src[i])<=max_src_size and len(tgt[i])<=max_tgt_size:
        src_r.append(src[i])
        tgt_r.append(tgt[i])
    src=src_r
    tgt=tgt_r

    print(len(src),len(tgt))

    src_word2id,src_id2word=word2idmaker(src,src_vocab_size)
    src_id=process_word2id(src,src_word2id,tgt=False)

    tgt_word2id,tgt_id2word=word2idmaker(tgt,tgt_vocab_size)
    tgt_id=process_word2id(tgt,tgt_word2id,tgt=True)

    data={}
    data["src_id"]=src_id
    data["tgt_id"]=tgt_id
    if train==True:
        data["src_id2word"]=src_id2word
        data["tgt_id2word"]=tgt_id2word

    return data

def batchmaker(data_size,batch_size,shuffle=True):
  data=list(range(data_size))
  if shuffle:
      random.shuffle(data)
  batches=[]
  batch=[]
  for i in range(data_size):
      batch.append(data[i])
      if len(batch)==batch_size:
          batches.append(batch)
          batch=[]
  if len(batch)>0:
      batches.append(batch)
  return batches

def make_vec(sentences):
    maxsize=max([len(sentence) for sentence in sentences])
    sentences=[sentence+[0]*(maxsize-len(sentence)) for sentence in sentences]
    #return torch.tensor(sentences,dtype=torch.long).to(device)
    return torch.from_numpy(np.array(sentences,dtype="long")).to(device)

def model_handler(data,train):
    sources=data["src_id"]
    targets=data["tgt_id"]
    #t_id2word=data["t_id2word"]
    data_size=len(sources)
    if train:
        model.train()
    else:
        model.eval()
    batches=batchmaker(data_size,batch_size,train)
    predict_rate=0
    loss_sum=0
    for i_batch,batch in enumerate(batches):
        input_words=make_vec([sources[i] for i in batch])
        output_words=make_vec([targets[i] for i in batch])#(batch,seq_len)
        if train:
            optimizer.zero_grad()
        predict,_=model(input_words,output_words,train)#(batch,seq_len,vocab_size)
        #trainの場合はパラメータの更新を行う
        if train==True:
            loss=loss_calc(predict,output_words[:,1:])#batch*seq_lenをして内部で計算
            loss.backward()
            optimizer.step()
            loss_sum+=loss.data
            if i_batch%print_iter==0:
              print("batch:{} loss:{}".format(i_batch,loss.data))
        else:
            predict_rate+=predict_calc(predict,output_words[:,1:])
            predict,target=predict_sentence(args,predict,output_words[:,1:],t_id2word)#(batch,seq_len)

    #epochの記録
    #if train:
    #    logger(args,"epoch:{}\ttime:{}\tloss:{}".format(epoch,time.time()-start,loss_sum/data_size))

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

class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder=Encoder()
        self.decoder=Decoder()

    def forward(self, input_words,output_words,train=True):
        encoder_outputs, encoder_hidden = self.encoder(input_words)#(batch,seq_len,hidden_size*2)
        output=self.decoder(encoder_outputs,encoder_hidden,output_words,train)
        return output

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.word_embed=nn.Embedding(src_vocab_size, embed_size,padding_idx=PAD)
        self.rnn=nn.LSTM(embed_size,hidden_size,num_layers=layer_size,bidirectional=True,dropout=dropout_rate,batch_first=True)
        self.dropout=nn.Dropout(dropout_rate)

    def forward(self,input_words):#input:(batch,seq_len)
        embed = self.word_embed(input_words)#(batch,seq_len,embed_size)
        output, hidden=self.rnn(embed) #(batch,seq_len,hidden_size*direction),(direction*layer_size,batch,hidden_size)
        return output, hidden

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.word_embed=nn.Embedding(tgt_vocab_size,embed_size,padding_idx=PAD)
        self.rnn=nn.LSTM(embed_size,hidden_size,num_layers=layer_size,bidirectional=False,dropout=dropout_rate,batch_first=True)
        self.attention=Attention()
        self.attention_wight=nn.Linear(hidden_size*3,hidden_size*3)
        self.out=nn.Linear(hidden_size*3,tgt_vocab_size)
        self.dropout=nn.Dropout(dropout_rate)
        self.activate=nn.Tanh()
        #self.out=nn.Linear(self.hidden_size*1,self.vocab_size)

    #decoderでのタイムステップ（単語ごと）の処理
    #input:(batch,1)
    #encoder_output:(batch,seq_len,hidden_size*direction)
    def decode_step(self,decoder_input,decoder_hidden,encoder_output):
        decoder_input=torch.unsqueeze(decoder_input,1)#(batch,1)
        embed=self.word_embed(decoder_input)#(batch,1,embed_size)
        embed=self.dropout(embed)


        output,decoder_hidden=self.rnn(embed,decoder_hidden)#(batch,1,hidden_size),(2,batch,hidden_size)
        #output=self.dropout(output)
        output=torch.squeeze(output,1)#(batch,hidden_size)

        use_attention=True
        #attentionの計算
        if use_attention:
            #encoderの出力と合わせてアテンションを計算
            attention_output,attention_result=self.attention(output,encoder_output)#(batch,hidden_size*2)
            output=self.attention_wight(torch.cat((output,attention_output),dim=-1))#(batch,hidden_size*3)
            output=self.activate(output)
            output=self.dropout(output)

        #単語辞書のサイズに変換する
        output=self.out(output)#(batch,vocab_size)

        #outputの中で最大値（実際に出力する単語）を返す
        predict=torch.argmax(output,dim=-1) #(batch)

        return output,decoder_hidden,predict,attention_result

    #encoder_output:(batch,seq_len,hidden_size*direction)
    #encoder_hidden:(direction*layer_size,batch,hidden_size)
    #output_words:(batch,output_seq_len)
    def forward(self,encoder_output,encoder_hidden,output_words,train=True):
        batch_size=output_words.size(0)
        output_seq_len=output_words.size(1)-1
        src_seq_len=encoder_output.size(1)

        #初期隠れベクトル、batch_first=Trueでも(1,batch,hidden_size)の順番、正直無くても良い
        #hidden,cell
        encoder_hidden,encoder_cell=encoder_hidden[0],encoder_hidden[1]
        encoder_hidden=encoder_hidden.view(2,layer_size,batch_size,hidden_size)
        encoder_cell=encoder_cell.view(2,layer_size,batch_size,hidden_size)
        decoder_hidden=torch.add(encoder_hidden[0],encoder_hidden[1])
        decoder_cell=torch.add(encoder_cell[0],encoder_cell[1])
        decoder_hidden=(decoder_hidden,decoder_cell)


        source = output_words[:, :-1]
        target = output_words[:, 1:]


        output_maxlen=output_seq_len
        teacher_forcing_ratio=1

        #decoderからの出力結果
        outputs=torch.from_numpy(np.zeros((output_seq_len,batch_size,tgt_vocab_size))).to(device)
        predict=torch.from_numpy(np.array([SOS]*batch_size,dtype="long")).to(device) #(batch_size)
        attention_result=torch.zeros(output_seq_len,src_seq_len)

        for i in range(output_maxlen):
            #使用する入力。
            current_input=source[:,i] if random.random()<teacher_forcing_ratio else predict.view(-1)#(batch)
            output,decoder_hidden,predict,result=self.decode_step(current_input,decoder_hidden,encoder_output)#(batch,vocab_size),(batch)
            outputs[i]=output#outputsにdecoderの各ステップから出力されたベクトルを入力
            #if batch_size==1:
            #   attention_result[i]=result

        outputs=torch.transpose(outputs,0,1)#(batch,seq_len,vocab_size)
        return outputs,attention_result

class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()

        self.W=nn.Linear(hidden_size,hidden_size*2)
        self.attention_wight=nn.Linear(hidden_size,hidden_size*2)


    #input*W*encoder_outputでscoreを計算(general)
    #decoder_hidden:(batch,hidden_size)
    #encoder_output:(batch,seq_len,hidden_size*2)
    #return:(batch,hidden_size*2)
    def forward(self,decoder_hidden,encoder_output):
        decoder_hidden=torch.unsqueeze(decoder_hidden,dim=1)#(batch,1,hidden_size)
        encoder_output_transpose=torch.transpose(encoder_output,1,2)#(batch,hidden_size*2,seq_len)

        output=self.W(decoder_hidden)#(batch,1,hidden_size*2)
        output=torch.bmm(output,encoder_output_transpose)#(batch,1,seq_len)
        output=F.softmax(output,dim=-1)#(batch,1,seq_len)
        attention_result=output#(batch,1,seq_len)

        output=torch.bmm(output,encoder_output)#(batch,1,hidden_size*2)
        output=torch.squeeze(output,dim=1)#(batch,hidden_size*2)
        return output,attention_result

def loss_calc(predict,target):
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    batch=predict.size(0)
    seq_len=predict.size(1)
    predict=predict.contiguous().view(batch*seq_len,-1)#(batch*seq_len,vocab_size)
    target=target.contiguous().view(-1)#(batch*seq_len)
    loss=criterion(predict,target)
    return loss

cuda_number=0
epoch_num=5
batch_size=1
embed_size=300
dropout_rate=0.3
hidden_size=50
layer_size=2
max_src_size=10
max_tgt_size=10
src_vocab_size=20000
tgt_vocab_size=20000
print_iter=100

tok_path="kftt-data-1.0/data/tok"
train_data=data_loader(src_path="{}/kyoto-train.ja".format(tok_path),tgt_path="{}/kyoto-train.en".format(tok_path),train=True)
#test_data=data_loader(src_path="kyoto-dev.ja",tgt_path="kyoto-dev.en",train=False)

model=Seq2Seq()
print(device)
print(model)
model.to(device)
optimizer = optim.Adam(model.parameters())

for epoch in range(0,epoch_num):
    model_handler(train_data,train=True)
    #model_handler(test_data,train=False)
