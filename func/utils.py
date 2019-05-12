import random
import numpy as np
import torch
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from func import constants
import datetime
import json
import platform

#batchのidを返す
class BatchMaker:
    def __init__(self,data_size,batch_size,shuffle=True):
        self.data_size=data_size
        self.batch_size=batch_size
        self.data=list(range(self.data_size))
        self.shuffle=shuffle
    def __call__(self):
        if self.shuffle:
            random.shuffle(self.data)
        batches=[]
        batch=[]
        for i in range(self.data_size):
            batch.append(self.data[i])
            if len(batch)==self.batch_size:
                batches.append(batch)
                batch=[]
        if len(batch)>0:
            batches.append(batch)
        return batches

def to_var(args,x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

#渡されたデータをpytorchのためにto_varで変換する
def make_tensor(id_number):
    return to_var(torch.from_numpy(np.array(id_number,dtype="long")))

#渡されたデータをpytorchのためにto_varで変換する
def make_vec(args,sentences):
    maxsize=max([len(sentence) for sentence in sentences])
    sentences_cp=[]
    for sentence in sentences:
        sentences_cp.append(sentence+[constants.PAD]*(maxsize-len(sentence)))
    return torch.from_numpy(np.array(sentences_cp,dtype="long")).to(args.device)

def logger(args,text):
    print(text)
    #サーバーの時のみ、logを記録
    if args.system=="Linux":
        with open("log.txt","a")as f:
            f.write("{}\t{}\n".format(str(datetime.datetime.today()).replace(" ","-"),text))

#ファイルから文、質問文、word2idなどを読み込み、辞書形式で返す
def data_loader(args,path,first=True):
    with open(path,"r")as f:
        t=json.load(f)
        sources=t["source"]
        targets=t["target"]
    with open("data/word2id.json","r")as f:
        t=json.load(f)#numpy(vocab_size*embed_size)
        s_word2id=t["s_word2id"]
        t_word2id=t["t_word2id"]

    s_word2id={w:i for w,i in s_word2id.items() if i<args.vocab_size}
    s_id2word={i:w for w,i in s_word2id.items()}

    t_word2id={w:i for w,i in t_word2id.items() if i<args.vocab_size}
    t_id2word={i:w for w,i in t_word2id.items()}

    random.seed(0)
    data_size=int(len(sources)*args.data_rate)
    pairs=[[s,t] for s,t in zip(sources,targets)]
    random.shuffle(pairs)

    sources_rm=[]
    targets_rm=[]

    for s,t in pairs[0:data_size]:
        if len(s.split())<=args.src_length and len(t.split())<=args.tgt_length:
            sources_rm.append(s)
            targets_rm.append(t)

    logger(args,"data_size:{}".format(len(sources_rm)))
    logger(args,sources_rm[0])
    logger(args,targets_rm[0])

    sources_id=[[s_word2id[w] if w in s_word2id else s_word2id["<UNK>"] for w in sent.split()] for sent in sources_rm]
    targets_id=[[t_word2id[w] if w in t_word2id else t_word2id["<UNK>"] for w in sent.split()] for sent in targets_rm]
    targets_id=[[t_word2id["<SOS>"]] + sent + [t_word2id["<EOS>"]] for sent in targets_id]

    #データをシャッフルし、9割をtrain,1割をtest
    train_data_size=int(len(sources_id)*0.9)
    pairs=[sources_id,targets_id]

    train_sources=pairs[0][0:train_data_size]
    train_targets=pairs[1][0:train_data_size]
    test_sources=pairs[0][train_data_size:]
    test_targets=pairs[1][train_data_size:]

    train_data={"sources":train_sources,
        "targets":train_targets,
        "s_id2word":s_id2word,
        "t_id2word":t_id2word}

    test_data={"sources":test_sources,
        "targets":test_targets,
        "s_id2word":s_id2word,
        "t_id2word":t_id2word}


    return train_data,test_data
