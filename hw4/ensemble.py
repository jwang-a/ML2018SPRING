import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import math
from sklearn.model_selection import train_test_split
from random import shuffle

#Hyper Parameters
hidden_size1 = 1000
hidden_size2 = 1000
hidden_size3 = 200
hidden_size4 = 100
hidden_size5 = 50
output_size = 2
num_epochs = 1000
learning_rate = 0.0001
dropout = 0.1
batch_size = 50
batch_num = 16
treenum = 99

threshold = 0.000
output = open('DRFtest.csv','w')
#####load_importance#####
pick = np.loadtxt(open('importance').read().split('\n'))

#####load_traindata#####
fetch = open('./data/train_X').read().split('\n')
labels = fetch[0].split(',')
labellength = len(pick)
choose = []
for i in range(labellength):
	if labels[i]=='fnlwgt':
		continue
	if pick[i]>threshold:
		choose.append(i)
chooselen = len(choose)

#####load_testdata#####
fetch = open('./data/test_X').read().split('\n')[1:]
xtest = np.loadtxt(fetch,delimiter=',')
xtest = xtest[:,choose]
xtest = torch.from_numpy(xtest).float()
testlength = len(xtest)
fetch = open('./label').read().split('\n')[1:-1]
ytest = np.loadtxt(fetch,delimiter=',')[:,1].astype(int)
ytest = torch.from_numpy(ytest)
#NN Model
class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size,hidden_size1)
        self.fc2 = nn.Linear(hidden_size1,num_classes)
        self.fc3 = nn.Linear(hidden_size2,hidden_size3)
        self.fc4 = nn.Linear(hidden_size3,hidden_size4)
        self.fc5 = nn.Linear(hidden_size4,hidden_size5)
        self.fc6 = nn.Linear(hidden_size5,num_classes)
        self.activation = nn.RReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        return out



finputs = Variable(xtest.view(testlength,chooselen).cuda())

#predict
result = []
for _ in range(treenum):
    print(_)
    tree = 'fctree'+str(_)+'.plk'
    net = Net(chooselen,hidden_size1,hidden_size2,hidden_size3,hidden_size4,hidden_size5,output_size)
    net.load_state_dict(torch.load(tree))
    net.cuda()
    a,outputs = torch.max(net(finputs),1)
    result.append(outputs)

#ensemble
print('id,label',file=output)
for i in range(16281):
    a = 0
    for j in range(treenum):
        a+=int(result[j][i])
    if a>treenum//2:
        print('%d,1'%(i+1),file=output)
    else:
        print('%d,0'%(i+1),file=output)
