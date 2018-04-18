from __future__ import division
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
from operator import mul
from operator import add
from PIL import Image, ImageOps,ImageEnhance, PILLOW_VERSION
import PIL
import pandas as pd
import random
import numbers
import math
try:
    import accimage
except ImportError:
    accimage = None
import sys

#Define Function
class GaussianNoise(nn.Module):
    def __init__(self,stddev):
        super().__init__()
        self.stddev = stddev
    def forward(self,din):
        return din+torch.autograd.Variable(torch.randn(din.size()).cuda()*self.stddev)
 
def local_response_norm(input,size,alpha=0.0001,beta=0.75,k=1):
	dim = input.dim()
	if dim<3:
		raise ValueError('Expected 3D or higher dimensionality \
				input (got {} dimensions)'.format(dim))
	div = input.mul(input).unsqueeze(1)
	if dim==3:
		div = pad(div,(0,0,size//2,(size-1)//2))
		div = avg_pool2d(div,(size,1),stride=1).squeeze(1)
	else:
		sizes = input.size()
		div = div.view(sizes[0], 1,sizes[1],sizes[2],-1)
		div = pad(div,(0,0,0,0,size//2,(size-1)//2))
		div = avg_pool3d(div,(size,1,1),stride=1).squeeze(1)
		div = div.view(sizes)
	div = div.mul(alpha).add(k).pow(beta)
	return input/div

#hyper parameters
batch_size = 100
output = open(sys.argv[2],'w')

#get dataset
totensor = transforms.ToTensor()

class ImagesDataset(torch.utils.data.Dataset):
	def __init__(self,df,loader=torchvision.datasets.folder.default_loader,train=False,transform=None):
                self.df = df
                self.loader = loader
                self.train = train
                self.transform = transform
                self.df = np.array(df['feature'].str.split(' ').values.tolist()).reshape(-1,48,48).astype(np.float)
	def __getitem__(self,index):
		img = self.df[index].astype('uint8')
		img = Image.fromarray(img,'L')
		img = totensor(img)
		return img
	def __len__(self):
		n=len(self.df)
		return n

fetch = pd.read_csv(sys.argv[1])
test_dataset = ImagesDataset(df=fetch,
			     train=True)
testloader = torch.utils.data.DataLoader(dataset=test_dataset,
					 batch_size=batch_size,
					 shuffle=False,
					 num_workers=0)

#NN Model
class FeatBlock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                                out_channels=64,
                                kernel_size=1,
                                padding=0,
                                stride=1)
        self.conv2 = nn.Conv2d(in_channels=64,
                                out_channels=out_channel,
                                kernel_size=3,
                                padding=1,
                                stride=1)
        self.conv3 = nn.Conv2d(in_channels=in_channel,
                                out_channels=out_channel,
                                kernel_size=1,
                                padding=0,
                                stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2,
                                    stride=2,
                                    padding=0)
        self.rrelu = nn.LeakyReLU(0.1)
    def forward(self,x,pool=False):
        out1 = self.conv1(x)
        out1 = self.rrelu(out1)
        out1 = self.conv2(out1)
        out1 = self.rrelu(out1)
        out2 = self.pool1(x)
        out2 = self.conv3(out2)
        out2 = self.rrelu(out2)
        out = out1+out2
        if pool==True:
            out = self.pool2(out)
        return out
#

class OTABlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.batchnorm = nn.BatchNorm2d(num_features=32)
        self.preconv = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=3,
                               padding=1,
                               stride=1)
        self.prepool = nn.MaxPool2d(kernel_size=2,
                                 stride=2,
                                 padding=0)
        self.feat1= FeatBlock(32,128)
        self.feat2= FeatBlock(128,128)

        self.fc1 = nn.Linear(128*36,2000)
        self.fc2 = nn.Linear(2000,500)
        self.fc3 = nn.Linear(500,7)
        self.dropout = nn.Dropout(0.2)
        self.dropoutfc = nn.Dropout(0.5)
        self.rrelu = nn.LeakyReLU(0.1)
        self.noise = GaussianNoise(0.1)

    def forward(self, x,Train=False):
        pre = self.preconv(x)
        pre = self.batchnorm(pre)
        pre = self.rrelu(pre)
        pre = self.prepool(pre)
        pre = local_response_norm(pre,2)

        out = self.feat1(pre,pool=True)
#        if Train==True:
#            bl1 = self.noise(bl1)
        out = self.dropout(out)
        out = self.feat2(out,pool=True)
        out = self.dropout(out)

        out = out.view(-1,128*36)
        out = self.fc1(out)
        out = self.rrelu(out)
        out = self.dropoutfc(out)
        out = self.fc2(out)
        out = self.rrelu(out)
        out = self.fc3(out)
        out = functional.softmax(out,1)
        return out

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.Blocks = nn.ModuleList([OTABlock() for i in range(3)])
        self.agg1 = nn.Linear(21,100)
        self.agg2 = nn.Linear(100,100)
        self.agg3 = nn.Linear(100,7)
        self.aggdropout = nn.Dropout(0.2)
        self.rrelu = nn.LeakyReLU(0.1)

    def forward(self,x,stage,Train=False,Agg=False):
        if Agg==False:
            out = self.Blocks[stage](x,Train=Train)
        else:
            out = self.agg1(x)
            out = self.rrelu(out)
#            out = self.aggdropout(out)
            out = self.agg2(out)
            out = self.rrelu(out)
#            out = self.aggdropout(out)
            out = self.agg3(out)
        return out

net = Net()
net.load_state_dict(torch.load('model.plk'))
net.cuda()

#validation
agg = [[] for i in range(7178)]
net.eval()
for stage in range(3):
    for i,inputs in enumerate(testloader):
        inputs = Variable(inputs.cuda())
        outputs = net(inputs,stage,Train=False)
        _, predicted = torch.max(outputs.data,1)
        for j in range(len(predicted)):
            agg[i*batch_size+j]+=outputs[j].data.cpu().numpy().tolist()
Agg = []
for i in range(71):
    Agg.append(torch.FloatTensor(agg[i*100:(i+1)*100]))
Agg.append(torch.FloatTensor(agg[7100:7178]))
print('id,label',file=output)
for i,inputs in enumerate(testloader):
    inputs = Variable(Agg[i].cuda())
    outputs = net(inputs,0,Agg=True)
    _, predicted = torch.max(outputs.data,1)
    for j in range(len(predicted)):
        print('%d,%d'%(i*batch_size+j,predicted[j]),file=output)
