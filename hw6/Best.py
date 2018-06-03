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
    def __init__(self,df,loader=torchvision.datasets.folder.default_loader,mode='train',users=None,movie=None,usersmap=None,moviemap=None):
        self.mode = mode
        self.loader = loader
        self.usersdict = users
        self.moviedict = movie
        self.usersmap = usersmap
        self.moviemap = moviemap
        df = open(df,errors='replace').read().split('\n')[1:-1]
        length = len(df)
        if self.mode is 'train' or self.mode is 'vald':
            self.user = []
            self.movie = []
            self.label = []
            for i in range(length):
                df[i] = df[i].split(',')
                self.user.append(int(df[i][1]))
                self.movie.append(int(df[i][2]))
                self.label.append(int(df[i][3]))
        elif self.mode is 'test':
            self.user = []
            self.movie = []
            for i in range(length):
                df[i] = df[i].split(',')
                self.user.append(int(df[i][1]))
                self.movie.append(int(df[i][2]))
        else:
            raise Exception('Error : UNDEFINED MODE')

    def __getitem__(self,index):
        if self.mode is 'train' or self.mode is 'vald':
            return self.usersdict[self.user[index]],self.usersmap[self.user[index]],self.moviedict[self.movie[index]],self.moviemap[self.movie[index]],self.label[index]
        elif self.mode is 'test':
            return self.usersdict[self.user[index]],self.usersmap[self.user[index]],self.moviedict[self.movie[index]],self.moviemap[self.movie[index]]
        else:
            raise Exception('Error : UNDEFINED MODE')

    def __len__(self):
        n=len(self.user)
        return n

class UsersEncoder(nn.Module):
    def __init__(self):
        super(UsersEncoder,self).__init__()
        self.fc1 = nn.Linear(30,64)
        self.fc2 = nn.Linear(64,15)
        self.dropout = nn.Dropout(0.2)
        self.rrelu = nn.RReLU()
        nn.init.xavier_normal(self.fc1.weight.data)
        nn.init.xavier_normal(self.fc2.weight.data)

    def forward(self,x):
        out = self.fc1(x)
        out = self.rrelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class MovieEncoder(nn.Module):
    def __init__(self):
        super(MovieEncoder,self).__init__()
        self.fc1 = nn.Linear(18,64)
        self.fc2 = nn.Linear(64,15)
        self.dropout = nn.Dropout(0.2)
        self.rrelu = nn.RReLU()
        nn.init.xavier_normal(self.fc1.weight.data)
        nn.init.xavier_normal(self.fc2.weight.data)

    def forward(self,x):
        out = self.fc1(x)
        out = self.rrelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class Classifier(nn.Module):
    def __init__(self,usernum,movienum):
        super(Classifier,self).__init__()
        self.usersencoder = UsersEncoder()
        self.movieencoder = MovieEncoder()
        self.usersemb = nn.Embedding(usernum,15)
        self.movieemb = nn.Embedding(movienum,15)
        self.fcusers = nn.Linear(30,15)
        self.fcmovie = nn.Linear(30,15)
        self.fc = nn.Linear(15,1)
        self.dropout = nn.Dropout(0.2)
        nn.init.xavier_normal(self.usersemb.weight.data)
        nn.init.xavier_normal(self.movieemb.weight.data)
        nn.init.xavier_normal(self.fcusers.weight.data)
        nn.init.xavier_normal(self.fcmovie.weight.data)
        nn.init.xavier_normal(self.fc.weight.data)

    def forward(self,users,useridx,movie,movieidx):
        usersemb =  self.usersemb(useridx)
        userscode = self.usersencoder(users)
        users = torch.cat((usersemb,userscode),1)
        movieemb = self.movieemb(movieidx)
        moviecode = self.movieencoder(movie)
        movie = torch.cat((movieemb,moviecode),1)
        users = self.fcusers(users)
        movie = self.fcmovie(movie)
        out = torch.mul(usersemb,movieemb)
        out = self.fc(out)
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
        self.embedding_dict()
        if self.traindata is not None:
            self.loadtraindata()
        if self.testdata is not None:
            self.loadtestdata()
        if self.valddata is not None:
            self.loadvalddata()

    def embedding_dict(self):
        if self.movie is None or self.users is None:
            raise Exception('Error : CRITICAL DATA NOT SUPPLIED')
        self.moviedict = {}
        self.movietype = {}
        self.movieidx = {}
        self.usersdict = {}
        self.occuptype = {}
        self.usersidx = {}
        self.agedict = dict([(1,0),(18,1),(25,2),(35,3),(45,4),(50,5),(56,6)])
        users = open(self.users,errors='replace').read().split('\n')[1:-1]
        length = len(users)
        for i in range(length):
            users[i] = users[i].split('::')
            if users[i][1] is 'F':
                users[i][1] = [0,1]
            else:
                users[i][1] = [1,0]
            users[i][0] = int(users[i][0])
            users[i][2] = int(users[i][2])
            users[i][3] = int(users[i][3])
            users[i][2] = self.agedict[users[i][2]]
            if users[i][0] not in self.usersidx.keys():
                self.usersidx[users[i][0]] = len(self.usersidx.keys())
            if users[i][3] not in self.occuptype.keys():
                self.occuptype[users[i][3]] = len(self.occuptype.keys())
            users[i][3] = self.occuptype[users[i][3]]
        types = len(self.occuptype.keys())
        for i in range(length):
            users[i][3] = [1 if j==users[i][3] else 0 for j in range(types)]
            users[i][2] = [1 if j==users[i][2] else 0 for j in range(7)]
            users[i][1] = users[i][1]+users[i][2]+users[i][3]
            self.usersdict[users[i][0]] = np.asarray(users[i][1])
        movie = open(self.movie,errors='replace').read().split('\n')[1:-1]
        length = len(movie)
        for i in range(length):
            movie[i] = movie[i].split('::')
            movie[i][0] = int(movie[i][0])
            if movie[i][0] not in self.movieidx.keys():
                self.movieidx[movie[i][0]] = len(self.movieidx.keys())
            movie[i][2] = movie[i][2].split('|')
            types = len(movie[i][2])
            for j in range(types):
                if movie[i][2][j] not in self.movietype.keys():
                    self.movietype[movie[i][2][j]] = len(self.movietype.keys())
                movie[i][2][j] = self.movietype[movie[i][2][j]]
        types = len(self.movietype.keys())
        for i in range(length):
            movie[i][2] = [1 if j in movie[i][2] else 0 for j in range(types)]
            self.moviedict[movie[i][0]] = np.asarray(movie[i][2])

    def loadtraindata(self):
        self.traindataset,self.trainloader = self.loaddata('train',self.traindata,True)

    def loadtestdata(self):
        self.testdataset,self.testloader = self.loaddata('test',self.testdata,False)

    def loadvalddata(self):
        self.valddataset, self.valdloader = self.loaddata('vald',self.valddata,False)

    def loaddata(self,mode,data,shuffle=False):
        if data is None:
            raise Exception('Error : Datapath not specified')
        dataset = Dataset(df=data,mode=mode,users=self.usersdict,movie=self.moviedict,usersmap=self.usersidx,moviemap=self.movieidx)
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=self.batch_size,
                                             shuffle=shuffle,
                                             num_workers=0)
        return dataset,dataloader

    def init_model(self):
        self.model = Classifier(len(self.usersidx.keys()),len(self.movieidx.keys()))
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
        self.model.train()
        total = 0
        avgloss = 0
        for i,(users,usersidx,movie,movieidx,labels) in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            users = users.float()
            usersidx = usersidx.long()
            movie = movie.float()
            movieidx = movieidx.long()
            labels = labels.float()
            if self.cuda is True:
                users = users.cuda()
                usersidx = usersidx.cuda()
                movie = movie.cuda()
                movieidx = movieidx.cuda()
                labels = labels.cuda()
            users = Variable(users)
            usersidx = Variable(usersidx)
            movie = Variable(movie)
            movieidx = Variable(movieidx)
            outputs = self.model(users,usersidx,movie,movieidx)
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
        self.model.eval()
        total = 0
        avgloss = 0
        for (users,usersidx,movie,movieidx,labels) in self.valdloader:
            users = users.float()
            usersidx = usersidx.long()
            movie = movie.float()
            movieidx = movieidx.long()
            labels = labels.float()
            if self.cuda is True:
                users = users.cuda()
                usersidx = usersidx.cuda()
                movie = movie.cuda()
                movieidx = movieidx.cuda()
                labels = labels.cuda()
            users = Variable(users)
            usersidx = Variable(usersidx)
            movie = Variable(movie)
            movieidx = Variable(movieidx)
            outputs = self.model(users,usersidx,movie,movieidx)
            outputs = torch.clamp(outputs,1.0,5.0)
            loss = self.criterion(outputs,Variable(labels))

            total+=labels.size(0)
            avgloss+=float(loss.data.cpu())*labels.size(0)
        return total,avgloss

    def test(self):
        self.model.eval()
        print('TestDataID,Rating',file=self.resultpath)
        tot = 1
        for (users,usersidx,movie,movieidx) in self.testloader:
            users = users.float()
            usersidx = usersidx.long()
            movie = movie.float()
            movieidx = movieidx.long()
            if self.cuda is True:
                users = users.cuda()
                usersidx = usersidx.cuda()
                movie = movie.cuda()
                movieidx = movieidx.cuda()
            users = Variable(users)
            usersidx = Variable(usersidx)
            movie = Variable(movie)
            movieidx = Variable(movieidx)
            outputs = self.model(users,usersidx,movie,movieidx)
            outputs = torch.clamp(outputs,1.0,5.0)
            outputs = outputs.data.cpu().numpy()
            for i in outputs:
                print('%d,%f'%(tot,i),file=self.resultpath)
                tot+=1

def main():
    Model = Frame(traindata=None,
                  testdata=sys.argv[1],
                  valddata=None,
                  movie=sys.argv[3],
                  users=sys.argv[4],
                  batch_size=128,
                  num_epochs=10,
                  learning_rate=1e-3,
                  cuda=True,
                  Log=None,
                  loaddictpath='BEST.plk',
                  savedictpath=None,
                  resultpath=sys.argv[2])
    Model.init_model()
    Model.test()

main()
