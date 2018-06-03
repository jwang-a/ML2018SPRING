import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.functional import pad
from torch.nn.functional import avg_pool3d
import torch.optim as optim
import torchvision
from torchvision import transforms
import numpy as np
import pandas as pd
import sys
import re
import gensim
from gensim.models.word2vec import Word2Vec

#self defined function
class Swish(nn.Module):
    def __init__(self):
        super(Swish,self).__init__()
    def forward(self,x):
        return x*functional.sigmoid(x)

#get dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self,df,loader=torchvision.datasets.folder.default_loader,task=None,maxlength=20,usingw2v=False,trainw2v=False,w2vwindow=2,embed_size=512,dictionary=None):
        self.task = task
        self.loader = loader
        self.maxlength = maxlength
        self.usingw2v = usingw2v
        self.trainw2v = trainw2v
        self.w2vwindow = w2vwindow
        self.embed_size = embed_size
        self.dictionary = dictionary
        if self.task is 'train':
            self.label = []
            self.df = []
            for i in df:
                i = i.split(b' +++$+++ ')
                self.label.append(int(i[0].decode('utf-8')))
                self.df.append(i[1])
            self.cleandata()
            if self.usingw2v is False:
                self.makedict()
                self.dictionary = self.Tok2Ind
                self.substitute(self.dictionary)
        elif self.task is 'vald':
            self.label = []
            self.df = []
            for i in df:
                i = i.split(b' +++$+++ ')
                self.label.append(int(i[0].decode('utf-8')))
                self.df.append(i[1])
            self.cleandata()
            if self.usingw2v is False:
                self.substitute(self.dictionary)
        elif self.task is 'extra':
            self.df = []
            for i in df:
                if len(i)==0:
                    continue
                self.df.append(i)
            self.cleandata()
            if self.usingw2v is False:
                self.substitute(self.dictionary)
        elif self.task is 'test':
            self.df = []
            for i in df:
                self.df.append(i[i.find(b',')+1:])
            self.cleandata()
            if self.usingw2v is False:
                self.substitute(self.dictionary)
        elif self.task is None:
            raise Exception('Error : No task specified for dataloader')
        else:
            raise Exception('Error : Unknown Task')

    def __getitem__(self,index):
        line = list(self.df[index])
        if self.usingw2v is False:
            line = torch.LongTensor(line)
        if self.usingw2v is True:
            length = len(line)
            for i,word in enumerate(line):
                if word in self.dictionary.vocab:
                    line[i] = self.dictionary[word].tolist()
                else:
                    line[i] = [0 for leng in range(self.embed_size)]
            line = line[max(0,len(line)-self.maxlength):]
            line = [[0 for leng in range(self.embed_size)] for padnum in range(self.maxlength-length)]+line
            line = torch.FloatTensor(line)
        if self.task is 'train' or self.task is 'vald':
            return line,self.label[index]
        elif self.task is 'extra':
            return line
        elif self.task is 'test':
            return line
        elif self.task is None:
            raise Exception('Error : No task specified for dataloader')
        else:
            raise Exception('Error : Unknown Task')

    def __len__(self):
        n=len(self.df)
        return n

    def cleandata(self):
        for i in range(len(self.df)):
            cleaned = chr(self.df[i][0])
            for j in range(1,len(self.df[i])):
                    temp = chr(self.df[i][j])
                    #if cleaned[-1] != temp:
                    #    cleaned+=temp
                    cleaned+=temp
            cleaned = cleaned.replace("i ' m", "im").replace("you ' re","youre").replace("didn ' t","didnt")    .replace("can ' t","cant").replace("haven ' t", "havent").replace("won ' t", "wont")    .replace("isn ' t","isnt").replace("don ' t", "dont").replace("doesn ' t", "doesnt")    .replace("aren ' t", "arent").replace("weren ' t", "werent").replace("wouldn ' t","wouldnt")    .replace("ain ' t","aint").replace("shouldn ' t","shouldnt").replace("wasn ' t","wasnt")    .replace(" ' s","s").replace("wudn ' t","wouldnt").replace(" .. "," ... ")    .replace("couldn ' t","couldnt")
            for repeated in re.findall(r'((\w)\2{2,})',cleaned):
                cleaned = cleaned.replace(repeated[0],repeated[1])
            cleaned = re.sub(r'(\d+)','1',cleaned)
            for punct in re.findall(r'([-#$%!=*:;~`/\\\\()!"+,&?\'.])',cleaned):
                if punct[0:2] is '..':
                    cleaned = cleaned.replace(punct,'...')
                else:
                    cleaned = cleaned.replace(punct,punct[0])
            stemmer = gensim.parsing.porter.PorterStemmer()
            cleaned = stemmer.stem_sentence(cleaned)
            #cleaned = re.sub(r'(\b[\w{1}]\b)',' ',' '+cleaned+' ')
            #cleaned = re.sub(r'([\W])',' ',cleaned)
            #cleaned = (' '+cleaned+' ').replace(" in "," ").replace(" or "," ").replace(" if "," ").replace(" of "," ").replace(" with "," ").replace(" the "," ").replace(" this "," ").replace(" that "," ").replace(" these "," ").replace(" those "," ").replace(" d "," ").replace(" ur ","  ").replace(" rt "," ").replace(" a "," ").replace(" at "," ").replace(" and "," ").replace(" to "," ").replace(" be "," ").replace(" me "," ").replace(" did "," do ").replace(" dam "," damn ").replace(" coz "," cause ").replace(" because "," cause ").replace(" im "," i ").replace(" would "," ").replace(" could "," ").replace(" is "," ").replace(" are "," ").replace(" were "," ").replace(" was "," ").replace(" am "," ").replace(" i ' m ","  ").replace(" you ' re "," you ").replace(" didn ' t "," no ").replace(" can ' t "," no ").replace(" canot "," no ").replace(" haven ' t "," no ").replace(" hadn ' t "," no ").replace(" won ' t "," no ").replace(" isn ' t "," no ").replace(" don ' t "," no ").replace(" doesn ' t "," no ").replace(" aren ' t "," no ").replace(" weren ' t "," no ").replace(" wouldn ' t "," no ").replace(" ain ' t "," no ").replace(" shouldn ' t "," no ").replace(" wasn ' t "," no ").replace(" ' s "," ").replace(" wudn ' t "," no ").replace(" couldn ' t "," no ").replace(" have "," ").replace(" had "," ").replace(" wil "," ").replace(" can "," ").replace(" ' d "," ").replace(" it "," ").replace(" idk "," no know ").replace(" wtf "," fuck ").replace(" btw "," ").replace(" what "," ").replace(" lmao "," laugh ").replace(" i "," ").replace(" you "," ").replace(" he "," ").replace(" she "," ").replace(" his "," ").replace(" her "," ").replace(" my "," ").replace(" your "," ").replace(" does "," ").replace(" do "," ").replace(" dos "," ").replace(" did "," ").replace(" would "," ").replace(" have "," ").replace(" had "," ").replace(" has "," ").replace(" can "," ").replace(" could "," ").replace(" should "," ").replace(" ' l "," ").replace(" a "," ").replace("ing "," ").replace("s "," ").replace("ed ","e ")
            #cleaned = re.sub(r'([\n\t])',' ',' '+cleaned+' ')
            #cleaned = re.sub(r'(\d)','1',cleaned)
            #cleaned = re.sub(r'([-@#$%!=*:;~`/\\\\()!"+,&?\'.])','',cleaned)
            #cleaned = re.sub(r'(\b[\w{1}]\b)',' ',cleaned)
            self.df[i] = cleaned.strip().split()
            #print(cleaned)
        #exit()

    def makedict(self):
        self.Tok2Ind = {'<PAD>':0,'<BOS>':1,'<EOS>':2,'<UNK>':3}
        self.Ind2Tok = {0:'<PAD>',1:'<BOS>',2:'<EOS>',3:'<UNK>'}
        self.Count = {}
        for line in self.df:
            for word in line:
                if word not in self.Count.keys():
                    self.Count[word] = 0
                self.Count[word]+=1
        tmp = sorted(self.Count.items(),key=lambda x:x[1],reverse=True)
        collect_words = len(tmp)
        for item,index in zip(tmp,list(range(4,collect_words+4))):
            self.Tok2Ind[item[0]] = index
            self.Ind2Tok[index] = item[0]

    def substitute(self,dictionary):
        for i,line in enumerate(self.df):
            line = line[max(0,len(line)-self.maxlength):]
            for j,word in enumerate(line):
                if word in dictionary.keys():
                    line[j] = dictionary[word]
                else:
                    line[j] = 3
            temp = len(line)
            self.df[i] = [0 for i in range(self.maxlength-temp)]+line

#NN Model
class LSTMencoder(nn.Module):
    def __init__(self,hidden_size,batch_size,embed_size,dict_size,usingw2v,maxlength,usingswish):
        super(LSTMencoder,self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.dict_size = dict_size
        self.usingw2v = usingw2v
        self.layers = 1
        self.embed = nn.Embedding(self.dict_size,self.embed_size)
        self.lstm = nn.LSTM(self.embed_size,self.hidden_size,self.layers,batch_first=True,dropout=0.2)

    def create_hidden(self):
        return Variable(torch.zeros(self.layers,self.batch_size,self.hidden_size)).cuda(),Variable(torch.zeros(self.layers,self.batch_size,self.hidden_size)).cuda()

    def forward(self,seq):
        if self.usingw2v is False:
            seq = self.embed(seq)
        hid = self.create_hidden()
        out,hid = self.lstm(seq,hid)
        out = out[:,-1,:].view(self.batch_size,self.hidden_size)
        return out

class GRUencoder(nn.Module):
    def __init__(self,hidden_size,batch_size,embed_size,dict_size,usingw2v,maxlength,usingswish):
        super(GRUencoder,self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.dict_size = dict_size
        self.usingw2v = usingw2v
        self.layers = 2
        self.embed = nn.Embedding(self.dict_size,self.embed_size)
        self.gru = nn.GRU(self.embed_size,self.hidden_size,self.layers,batch_first=True,dropout=0.2)

    def forward(self,seq):
        if self.usingw2v is False:
            seq = self.embed(seq)
        out,hid = self.gru(seq)
        out = out[:,-1,:].view(self.batch_size,self.hidden_size)
        return out

class Bidir_LSTMencoder(nn.Module):
    def __init__(self,hidden_size,batch_size,embed_size,dict_size,usingw2v,maxlength,usingswish):
        super(Bidir_LSTMencoder,self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.dict_size = dict_size
        self.usingw2v = usingw2v
        self.usingswish = usingswish
        self.layers = 2
        self.embed = nn.Embedding(self.dict_size,self.embed_size)
        self.lstm = nn.LSTM(self.embed_size,self.hidden_size,self.layers,batch_first=True,dropout=0.2,bidirectional=True)
        self.reform = nn.Linear(self.hidden_size*2,self.hidden_size)
        self.dropout = nn.Dropout(0.5)
        if self.usingswish is True:
            self.activation = Swish()
        else:
            self.activation = nn.RReLU()
        self.batchnorm = nn.BatchNorm1d(self.hidden_size)

    def create_hidden(self):
        return Variable(torch.zeros(self.layers*2,self.batch_size,self.hidden_size)).cuda(),Variable(torch.zeros(self.layers*2,self.batch_size,self.hidden_size)).cuda()

    def forward(self,seq):
        if self.usingw2v is False:
            seq = self.embed(seq)
        hid = self.create_hidden()
        out,hid = self.lstm(seq,hid)
        out = out[:,-1,:].view(self.batch_size,self.hidden_size*2)
        out = self.reform(out)
        out = self.dropout(out)
        out = self.activation(out)
        out = self.batchnorm(out)
        return out

class LSTM_CNNencoder(nn.Module):
    def __init__(self,hidden_size,batch_size,embed_size,dict_size,usingw2v,maxlength,usingswish):
        super(LSTM_CNNencoder,self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.dict_size = dict_size
        self.usingw2v = usingw2v
        self.maxlength = maxlength
        self.layers = 2
        self.embed = nn.Embedding(self.dict_size,self.embed_size)
        self.lstm = nn.LSTM(self.embed_size,self.hidden_size,self.layers,batch_first=True,dropout=0.2)
        self.conv1 = nn.Conv2d(in_channels=self.maxlength,
                               out_channels=16,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=4,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.rrelu = nn.RReLU()
        self.dropout = nn.Dropout(0.5)
        self.batchnorm = nn.BatchNorm1d(self.hidden_size)

    def create_hidden(self):
        return Variable(torch.zeros(self.layers,self.batch_size,self.hidden_size)).cuda(),Variable(torch.zeros(self.layers,self.batch_size,self.hidden_size)).cuda()

    def forward(self,seq):
        if self.usingw2v is False:
            seq = self.embed(seq)
        hid = self.create_hidden()
        out,hid = self.lstm(seq,hid)
        out = out.contiguous().view(self.batch_size,self.maxlength,int(self.hidden_size**(1/2)),int(self.hidden_size**(1/2)))
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.rrelu(out)
        out = self.pool1(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.rrelu(out)
        out = out.view(self.batch_size,self.hidden_size)
        out = self.batchnorm(out)
        out = self.dropout(out)
        return out

class GRU_CNNencoder(nn.Module):
    def __init__(self,hidden_size,batch_size,embed_size,dict_size,usingw2v,maxlength,usingswish):
        super(GRU_CNNencoder,self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.dict_size = dict_size
        self.usingw2v = usingw2v
        self.maxlength = maxlength
        self.layers = 2
        self.embed = nn.Embedding(self.dict_size,self.embed_size)
        self.gru = nn.GRU(self.embed_size,self.hidden_size,self.layers,batch_first=True,dropout=0.5)
        self.conv1 = nn.Conv2d(in_channels=self.maxlength,
                               out_channels=16,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=4,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.rrelu = nn.RReLU()
        self.dropout = nn.Dropout(0.5)
        self.batchnorm = nn.BatchNorm1d(self.hidden_size)

    def forward(self,seq):
        if self.usingw2v is False:
            seq = self.embed(seq)
        out,hid = self.gru(seq)
        out = out.contiguous().view(self.batch_size,self.maxlength,int(self.hidden_size**(1/2)),int(self.hidden_size**(1/2)))
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.rrelu(out)
        out = self.pool1(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.rrelu(out)
        out = out.view(self.batch_size,self.hidden_size)
        out = self.batchnorm(out)
        out = self.dropout(out)
        return out

class Classifier(nn.Module):
    def __init__(self,encoder,hidden_size,batch_size,embed_size,dict_size,maxlength,usingw2v,usingswish):
        super(Classifier,self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.dict_size = dict_size
        self.maxlength = maxlength
        self.usingw2v = usingw2v
        self.usingswish = usingswish
        self.seq2vec = encoder(self.hidden_size,self.batch_size,self.embed_size,self.dict_size,usingw2v,self.maxlength,usingswish)
        self.fc1 = nn.Linear(self.hidden_size,64)
        self.fc2 = nn.Linear(64,2)
        if self.usingswish is True:
            self.activation = Swish()
        else:
            self.activation = nn.RReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self,seq):
        encoded = self.seq2vec(seq)
        out = self.fc1(encoded)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = functional.softmax(out,dim=1)
        return out

class Frame():
    def __init__(self,traindata=None,extradata=None,valddata=None,testdata=None,maxlength=41,batch_size=100,hidden_size=512,embed_size=128,num_epochs=20,learning_rate=1e-4,usingw2v=False,trainw2v=False,w2vwindow=2,usingswish=False,cuda=False,Log=None,loaddictpath=None,savedictpath=None,resultpath=None):
        self.maxlength = maxlength
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.usingw2v = usingw2v
        self.trainw2v = trainw2v
        self.w2vwindow = w2vwindow
        self.usingswish= usingswish
        self.cuda = cuda
        if Log is None:
            self.Log = sys.stdout
        else:
            self.Log = open(Log,'w')
        self.traindata = traindata
        self.extradata = extradata
        self.valddata = valddata
        self.testdata = testdata
        self.savedictpath = savedictpath
        self.loaddictpath = loaddictpath
        if resultpath is None:
            raise Exception('Error : Resultpath not specified')
        self.result = open(resultpath,'w')

    def loadtraindata(self):
        self.train_dataset, self.trainloader = self.loaddata(self.traindata,'train',True,None)

    def loadextradata(self,dictionary):
        self.extra_dataset, self.extraloader = self.loaddata(self.extradata,'extra',False,dictionary)

    def loadvalddata(self,dictionary):
        self.vald_dataset, self.valdloader = self.loaddata(self.valddata,'vald',False,dictionary)

    def loadtestdata(self,dictionary):
        self.test_dataset, self.testloader = self.loaddata(self.testdata,'test',False,dictionary)

    def loaddata(self,path=None,task=None,Shuffle=False,dictionary=None):
        if path is None:
            raise Exception('Error : Datapath not specified')
        fetch = open(path,'rb').read().split(b'\n')[:-1]
        if task is 'test':
            fetch = fetch[1:]
        dataset = Dataset(df=fetch,
                          task=task,
                          maxlength = self.maxlength,
                          usingw2v = self.usingw2v,
                          trainw2v = self.trainw2v,
                          w2vwindow = self.w2vwindow,
                          embed_size = self.embed_size,
                          dictionary = dictionary)
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=self.batch_size,
                                             shuffle=Shuffle,
                                             num_workers=0)
        return dataset,loader

    def trainword2vec(self,data):
        word2vec = Word2Vec(data,size=self.embed_size,window=self.w2vwindow,min_count=0,workers=8,seed=0)
        word2vec.save('WORD2VECDICT')
        del word2vec

    def loadword2vec(self):
        self.dictionary = Word2Vec.load('WORD2VECDICT').wv
        if self.traindata is not None:
            self.train_dataset.dictionary = self.dictionary
        if self.extradata is not None:
            self.train_dataset.dictionary = self.dictionary
        if self.valddata is not None:
            self.vald_dataset.dictionary = self.dictionary
        if self.testdata is not None:
            self.test_dataset.dictionary = self.dictionary

    def init_model(self,encoder=None):
        if self.usingw2v is False:
            self.model = Classifier(encoder,self.hidden_size,self.batch_size,self.embed_size,len(self.train_dataset.Tok2Ind),self.maxlength,False,self.usingswish)
        elif self.usingw2v is True:
            self.model = Classifier(encoder,self.hidden_size,self.batch_size,self.embed_size,len(self.dictionary.vocab),self.maxlength,True,self.usingswish)
        if self.loaddictpath is not None:
            self.model.load_state_dict(torch.load(self.loaddictpath))
        if self.cuda is True:
            self.model.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)

    def train(self):
        for epoch in range(self.num_epochs):
            total, correct, avgloss = self.train_util()
            print('Epoch [%d/%d], Loss: %.4f, Acc: %.4f'%(epoch+1,self.num_epochs,avgloss/total,correct/total),file=self.Log)
            if self.valddata is not None:
                self.vald()
            self.Log.flush()
            if epoch%20==19 or epoch==self.num_epochs-1:
                self.checkpoint()

    def train_util(self):
        self.model.train()
        total = 0
        correct = 0
        avgloss = 0
        for i,(inputs,labels) in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            inputs = Variable(inputs)
            if self.cuda is True:
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data,1)
            total+=labels.size(0)
            correct+=int((predicted==labels).sum())
            loss = self.criterion(outputs,Variable(labels))
            avgloss+=float(loss.data.cpu())*labels.size(0)
            loss.backward()
            self.optimizer.step()
            if i%100==99:
                print('STEP %d, Loss: %.4f, Acc: %.4f'%(i+1,loss.data.cpu(),int((predicted==labels).sum())/labels.size(0)*100),file=self.Log)
            self.Log.flush()
        return total,correct,avgloss

    def checkpoint(self):
        if self.savedictpath is not None:
            torch.save(self.model.state_dict(),self.savedictpath)

    def vald(self):
        self.model.eval()
        total = 0
        correct = 0
        for inputs,labels in self.valdloader:
            inputs = Variable(inputs)
            if self.cuda is True:
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data,1)
            total+=labels.size(0)
            correct+=int((predicted==labels).sum())
        print('-----------VALDACC : %.4f-----------'%(correct/total*100),file=self.Log)

    def test(self):
        self.model.eval()
        ans = []
        for inputs in self.testloader:
            inputs = Variable(inputs)
            if self.cuda is True:
                inputs = inputs.cuda()
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data,1)
            ans+=predicted.cpu().numpy().tolist()
        ans = np.asarray(ans)
        print('id,label',file=self.result)
        for i in range(len(ans)):
            print('%d,%d'%(i,ans[i]),file=self.result)


def main():
    Model = Frame(traindata=sys.argv[1],
                  extradata=sys.argv[2],
                  valddata=None,
                  testdata=None,
                  maxlength=30,
                  batch_size=100,
                  hidden_size=100,
                  embed_size=50,
                  num_epochs=20,
                  learning_rate=1e-4,
                  usingw2v = True,
                  trainw2v = True,
                  w2vwindow=5,
                  usingswish=True,
                  cuda=True,
                  Log='LSTMACC',
                  loaddictpath=None,
                  savedictpath='MODEL.plk',
                  resultpath='ans.csv')
    if Model.traindata is not None:
        Model.loadtraindata()
    if Model.extradata is not None:
        Model.loadextradata(Model.train_dataset.dictionary)
    if Model.valddata is not None:
        Model.loadvalddata(Model.train_dataset.dictionary)
    if Model.testdata is not None:
        Model.loadtestdata(Model.train_dataset.dictionary)

    if Model.trainw2v is not False:
        Model.trainword2vec(Model.train_dataset.df+Model.extra_dataset.df)
    if Model.usingw2v is not False:
        Model.loadword2vec()

    Model.init_model(Bidir_LSTMencoder)

    if Model.traindata is not None:
        Model.train()
    if Model.testdata is not None:
        Model.test()

main()
