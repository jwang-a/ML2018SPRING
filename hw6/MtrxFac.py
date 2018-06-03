import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torchvision
from torchvision import transforms
import numpy as np
import pandas as pd
import sys
import os
import math

#get dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self,df,loader=torchvision.datasets.folder.default_loader,mode='train',userdict=None,moviedict=None):
        self.mode = mode
        self.userdict = userdict
        self.moviedict = moviedict
        self.loader = loader
        df = open(df,errors='replace').read().split('\n')[1:-1]
        length = len(df)
        if self.mode is 'train' or self.mode is 'vald':
            self.user = []
            self.movie = []
            self.label = []
            for i in range(length):
                df[i] = df[i].split(',')
                self.user.append(self.userdict[df[i][1]])
                self.movie.append(self.moviedict[df[i][2]])
                self.label.append(int(df[i][3]))
        elif self.mode is 'test':
            self.user = []
            self.movie = []
            for i in range(length):
                df[i] = df[i].split(',')
                self.user.append(self.userdict[df[i][1]])
                self.movie.append(self.moviedict[df[i][2]])
        else:
            raise Exception('Error : UNDEFINED MODE')

    def __getitem__(self,index):
        if self.mode is 'train' or self.mode is 'vald':
            return self.user[index],self.movie[index],self.label[index]
        elif self.mode is 'test':
            return self.user[index],self.movie[index]
        else:
            raise Exception('Error : UNDEFINED MODE')

    def __len__(self):
        n=len(self.user)
        return n

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.usersemb = nn.Embedding(6040,15)
        self.usersbia = nn.Embedding(6040,1)
        self.movieemb = nn.Embedding(3883,15)
        self.moviebia = nn.Embedding(3883,1)
        self.dropout = nn.Dropout(0.2)
        self.rrelu = nn.RReLU()
        nn.init.xavier_normal(self.usersemb.weight.data)
        nn.init.xavier_normal(self.usersbia.weight.data)
        nn.init.xavier_normal(self.movieemb.weight.data)
        nn.init.xavier_normal(self.moviebia.weight.data)


    def forward(self,users,movie):
        out1 = self.usersemb(users)
        #out1 = self.rrelu(out1)
        #out1 = self.dropout(out1)
        out2 = self.movieemb(movie)
        #out2 = self.rrelu(out2)
        #out2 = self.dropout(out2)
        out = self.usersbia(users)+self.moviebia(movie)
        out+=(out1*out2).sum(1).view(out.size())
        return out

class Frame():
    def __init__(self,traindata=None,testdata=None,valddata=None,movie=None,users=None,batch_size=100,num_epochs=20,learning_rate=1e-4,cuda=False,Log=None,loaddictpath=None,savedictpath=None,resultpath=None):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.cuda = cuda
        if Log is None:
            self.Log = sys.stdout
        else:
            self.Log = open(Log,'w')
        self.traindata = traindata
        self.testdata = testdata
        self.valddata = valddata
        self.movie = movie
        self.users = users
        self.savedictpath = savedictpath
        self.loaddictpath = loaddictpath
        if resultpath is None:
            raise Exception('Error : Resultpath not specified')
        self.resultpath = open(resultpath,'w')
        self.makedict()
        if self.traindata is not None:
            self.loadtraindata()
        if self.testdata is not None:
            self.loadtestdata()
        if self.valddata is not None:
            self.loadvalddata()
        self.init_model()

    def loadtraindata(self):
        self.traindataset,self.trainloader = self.loaddata('train',self.traindata,True)

    def loadtestdata(self):
        self.testdataset,self.testloader = self.loaddata('test',self.testdata,False)

    def loadvalddata(self):
        self.valddataset, self.valdloader = self.loaddata('vald',self.valddata,False)

    def loaddata(self,mode,data,shuffle=False):
        if data is None:
            raise Exception('Error : Datapath not specified')
        dataset = Dataset(df=data,mode=mode,userdict=self.userdict,moviedict=self.moviedict)
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=self.batch_size,
                                             shuffle=shuffle,
                                             num_workers=0)
        return dataset,dataloader

    def makedict(self):
        self.movie2genre = {}
        self.genre2movie = {}
        self.userdict = {}
        self.moviedict = {}
        movie = open(self.movie,errors='replace').read().split('\n')[1:-1]
        for i in movie:
            i = i.split('::')
            self.movie2genre[i[0]] = i[2]
            if i[2] not in self.genre2movie.keys():
                self.genre2movie[i[2]] = [i[0]]
            else:
                self.genre2movie[i[2]].append(i[0])
        traind = open(self.traindata,errors='replace').read().split('\n')[1:-1]
        for i in traind:
            i = i.split(',')
            if i[1] not in self.userdict.keys():
                self.userdict[i[1]] = len(self.userdict.keys())
            if i[2] not in self.moviedict.keys():
                self.moviedict[i[2]] = len(self.moviedict.keys())
        testd = open(self.testdata,errors='replace').read().split('\n')[1:-1]
        for i in testd:
            i = i.split(',')
            if i[2] not in self.moviedict.keys():
                for j in self.genre2movie[self.movie2genre[i[2]]]:
                    if j!=i[2]:
                        self.moviedict[i[2]] = int(j)
                        break

    def init_model(self):
        self.model = Classifier()
        if self.loaddictpath is not None:
            self.model.load_state_dict(torch.load(self.loaddictpath))
        if self.cuda is True:
            self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def train(self):
        for epoch in range(self.num_epochs):
            total, avgloss = self.train_util()
            print('Epoch [%d/%d], Loss: %.4f'%(epoch+1,self.num_epochs,math.sqrt(avgloss/total)),file=self.Log)
            self.Log.flush()
            if self.valddata is not None:
                total, avgloss = self.vald_util()
                print('------VALD Loss: %.4f------'%(math.sqrt(avgloss/total)),file=self.Log)
                self.Log.flush()

            if epoch%20==19 or epoch==self.num_epochs-1:
                self.checkpoint()

    def train_util(self):
        total = 0
        avgloss = 0
        self.model.train()
        for i,(users,movie,labels) in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            users = users.long()
            movie = movie.long()
            labels = labels.float()
            if self.cuda is True:
                users = users.cuda()
                movie = movie.cuda()
                labels = labels.cuda()
            users = Variable(users)
            movie = Variable(movie)
            outputs = self.model(users,movie)
            loss = self.criterion(outputs,Variable(labels))
            loss.backward()
            self.optimizer.step()

            total+=labels.size(0)
            avgloss+=float(loss.data.cpu())*labels.size(0)
            if i%100==99:
                print('STEP %d, Loss: %.4f'%(i+1,math.sqrt(loss.data.cpu())),file=self.Log)
            self.Log.flush()
        return total,avgloss

    def checkpoint(self):
        if self.savedictpath is not None:
            torch.save(self.model.state_dict(),self.savedictpath)

    def vald_util(self):
        total = 0
        avgloss = 0
        for (users,movie,labels) in self.valdloader:
            users = users.long()
            movie = movie.long()
            labels = labels.float()
            if self.cuda is True:
                users = users.cuda()
                movie = movie.cuda()
                labels = labels.cuda()
            users = Variable(users)
            movie = Variable(movie)
            outputs = self.model(users,movie)
            outputs = outputs.clamp(1.0,5.0)
            loss = self.criterion(outputs,Variable(labels))

            total+=len(labels)
            avgloss+=float(loss.data.cpu())*labels.size(0)
        return total,avgloss

    def test(self):
        print('TestDataID,Rating',file=self.resultpath)
        tot = 1
        for (users,movie) in self.testloader:
            users = users.long()
            movie = movie.long()
            if self.cuda is True:
                users = users.cuda()
                movie = movie.cuda()
            users = Variable(users)
            movie = Variable(movie)
            outputs = self.model(users,movie)
            outputs = outputs.clamp(1.0,5.0)
            outputs = outputs.data.cpu().numpy()
            for i in outputs:
                print('%d,%f'%(tot,i),file=self.resultpath)
                tot+=1

def main():
    Model = Frame(traindata='datamapper',
                  testdata=sys.argv[1],
                  valddata=None,
                  movie=sys.argv[3],
                  users=sys.argv[4],
                  batch_size=128,
                  num_epochs=10,
                  learning_rate=1e-3,
                  cuda=True,
                  Log=None,
                  loaddictpath='MTRXFAC.plk',
                  savedictpath=None,
                  resultpath=sys.argv[2])
    #Model.train()
    Model.test()

main()
