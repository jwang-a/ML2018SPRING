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
import sys
try:
    import accimage
except ImportError:
    accimage = None

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

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def _get_inverse_affine_matrix(center, angle, translate, scale, shear):
    angle = math.radians(angle)
    shear = math.radians(shear)
    scale = 1.0 / scale
    d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
    matrix = [
        math.cos(angle + shear), math.sin(angle + shear), 0,
        -math.sin(angle), math.cos(angle), 0
    ]
    matrix = [scale / d * m for m in matrix]
    matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
    matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])
    matrix[2] += center[0]
    matrix[5] += center[1]
    return matrix


def affine(img, angle, translate, scale, shear, resample=0, fillcolor=0):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
        "Argument translate should be a list or tuple of length 2"

    assert scale > 0.0, "Argument scale should be positive"

    output_size = img.size
    center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)
    matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
    kwargs = {"fillcolor": fillcolor} if PILLOW_VERSION[0] == '5' else {}
    return img.transform(output_size, Image.AFFINE, matrix, resample,**kwargs)

__all__ = ["Compose", "ToTensor", "ToPILImage", "Normalize", "Resize", "Scale", "CenterCrop", "Pad", "Lambda", "RandomApply", "RandomChoice", "RandomOrder", "RandomCrop", "RandomHorizontalFlip", "RandomVerticalFlip", "RandomResizedCrop", "RandomSizedCrop", "FiveCrop", "TenCrop", "LinearTransformation", "ColorJitter", "RandomRotation", "RandomAffine", "Grayscale", "RandomGrayscale"]

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}


class RandomAffine(object):
    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, img):
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
        return affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor)

    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.resample > 0:
            s += ', resample={resample}'
        if self.fillcolor != 0:
            s += ', fillcolor={fillcolor}'
        s += ')'
        d = dict(self.__dict__)
        d['resample'] = _pil_interpolation_to_str[d['resample']]
        return s.format(name=self.__class__.__name__, **d)



#Hyper Parameters
num_epochs = 10000
learning_rate = 1e-4
batch_size = 100
output = open(sys.argv[1],'w')

#get dataset
mytransform = transforms.Compose([
	transforms.RandomHorizontalFlip(),
        RandomAffine(degrees=5,translate=(0.1,0.1),scale=(0.9,1.1),shear=0.1,resample=False,fillcolor=0),
	transforms.ToTensor()
])
totensor = transforms.ToTensor()


class ImagesDataset(torch.utils.data.Dataset):
        def __init__(self,df,loader=torchvision.datasets.folder.default_loader,train=False,transform=None):
                self.df = df
                self.loader = loader
                self.train = train
                self.transform = transform
                if train is True:
                    self.label = np.array(df['label'])
                self.df = np.array(df['feature'].str.split(' ').values.tolist()).reshape(-1,48,48).astype(np.float)
        def __getitem__(self,index):
                img = self.df[index].astype('uint8')
                img = Image.fromarray(img,'L')
                if self.transform is not None:
                    img = self.transform(img)
                else:
                    img = totensor(img)
                if self.train is True:
                    target = self.label[index]
                    return img,target
                else:
                    return img
        def __len__(self):
                n=len(self.df)
                return n

fetch = pd.read_csv('./data/train.csv')
train_dataset = ImagesDataset(df=fetch,
			      train=True,
			      transform=mytransform)
trainloader = torch.utils.data.DataLoader(dataset=train_dataset,
					 batch_size=batch_size,
					 shuffle=True,
					 num_workers=0)

fetch = pd.read_csv('./data/test.csv')
test_dataset = ImagesDataset(df=fetch,
			     train=False)
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
            out = self.aggdropout(out)
            out = self.agg2(out)
            out = self.rrelu(out)
            out = self.aggdropout(out)
            out = self.agg3(out)
        out = functional.softmax(out,1)
        return out

net = Net()
net.cuda()

#loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)

#train
for stage in range(3):
    for epoch in range(num_epochs):
        net.train()
        for i,(inputs,labels) in enumerate(trainloader):
            inputs = Variable(inputs.cuda())
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs,stage,Train=True)
            _, predicted = torch.max(outputs.data,1)
            total = labels.size(0)
            correct=int((predicted==labels).sum())
            loss = criterion(outputs,Variable(labels))
            loss.backward()
            optimizer.step()

###aggregate
agg = [[] for i in range(28709)]
net.eval()
for stage in range(3):
    for i,(inputs,labels) in enumerate(trainloader):
        inputs = Variable(inputs.cuda())
        labels = labels.cuda()
        outputs = net(inputs,stage,Train=False)
        _, predicted = torch.max(outputs.data,1)
        for j in range(len(predicted)):
            agg[i*batch_size+j]+=outputs[j].data.cpu().numpy().tolist()
Agg = []
for i in range(287):
    Agg.append(torch.FloatTensor(agg[i*100:(i+1)*100]))
Agg.append(torch.FloatTensor(agg[28700:28709]))
net.train()
for epoch in range(num_epochs):
    total = 0
    correct = 0
    for i,(inputs,labels) in enumerate(trainloader):
        inputs = Variable(Agg[i].cuda())
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs,0,Agg=True)
        _, predicted = torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=int((predicted==labels).sum())
        loss = criterion(outputs,Variable(labels))
        loss.backward()
        optimizer.step()

#result
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
