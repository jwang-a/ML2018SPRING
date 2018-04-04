import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import math
from random import shuffle
import sys 

#Hyper Parameters
hidden_size1 = 1000
hidden_size2 = 1000
hidden_size3 = 200
hidden_size4 = 100
hidden_size5 = 50
output_size = 2
batch_size = 50
batch_num = 16
treenum = 99

threshold = 0.000
output = open(sys.argv[4],'w')

#####load_importance#####
pick = np.loadtxt(open('importance').read().split('\n'))

#####load_traindata#####
fetch = open(sys.argv[1]).read().split('\n')
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
fetch = open(sys.argv[3]).read().split('\n')[1:]
xtest = np.loadtxt(fetch,delimiter=',')
xtest = xtest[:,choose]
xtest = torch.from_numpy(xtest).float()
testlength = len(xtest)

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

result = []
for _ in range(treenum):
    print(_)
    tree = 'fctree'+str(_)+'.plk'
    net = Net(chooselen,hidden_size1,hidden_size2,hidden_size3,hidden_size4,hidden_size5,output_size)
    net.load_state_dict(torch.load(tree))
    net.cuda()
    a,outputs = torch.max(net(finputs),1)
    result.append(outputs)

print('id,label',file=output)
for i in range(16281):
    a = 0
    for j in range(treenum):
        a+=int(result[j][i])
    if a>treenum//2:
        print('%d,1'%(i+1),file=output)
    else:
        print('%d,0'%(i+1),file=output)
