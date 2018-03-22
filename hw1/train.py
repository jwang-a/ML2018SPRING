import numpy as np
import math
import random

takelen = 8
upperbnd = 120
tupperbnd = 110
lowerbnd = 2
tlowerbnd = 2
malfunc = 3
rate = 0.0001
iters = 40
BOUND = [lowerbnd,5,25,50,53,55,57,60,63,65,67,70,upperbnd]
Blen = len(BOUND)-1

#####Deal traindata#####
fetch = open('./data/train.csv','rb')
traindata = fetch.read().split(b'\r\n')[1:]
xtemp = []
xtemp2 = []
ytemp = []
xtrain = []
ytrain = []
length = int(len(traindata)/18)
for i in range(length):
	xtemp+=(traindata[i*18+9].split(b',')[3:])
length = len(xtemp)-takelen
for i in range(length):
	if i%480>472:
		continue
	xtemp2.append(xtemp[i:i+takelen])
	ytemp.append(xtemp[i+takelen])
length = len(xtemp2)
for i in range(length):
	flag = 0
	for j in range(takelen):
		if float(xtemp2[i][j])>upperbnd or float(xtemp2[i][j])<lowerbnd:
			flag = 1
			break
	if float(ytemp[i])>upperbnd or float(ytemp[i])<lowerbnd:
		flag = 1
	if flag==1:
		continue
	xtrain.append(xtemp2[i])
	ytrain.append(ytemp[i])

xtrain = np.asarray(xtrain)
trainlength = len(xtrain)
xtrain = xtrain.reshape(trainlength,-1)
xtrain = np.concatenate((xtrain,np.ones((trainlength,1))),axis=1)
xtrain = xtrain.astype(float)
ytrain = np.asarray(ytrain).astype(float)

trainave = []
for i in range(trainlength):
	trainave.append(np.mean(xtrain[i][:-1]))
nxtrain = []
nytrain = []
for i in range(Blen):
	nxtrain.append([])
	nytrain.append([])
for i in range(trainlength):
	for j in range(Blen):
		if trainave[i]>=BOUND[j] and trainave[i]<BOUND[j+1]:
			nxtrain[j].append(xtrain[i])
			nytrain[j].append(ytrain[i])
xtrain = np.asarray(nxtrain)
ytrain = np.asarray(nytrain)
trainlength = []
for i in range(Blen):
	xtrain[i] = np.asarray(xtrain[i])
	ytrain[i] = np.asarray(ytrain[i])
	trainlength.append(len(xtrain[i]))

#####Deal testdata#####
fetch = open('./data/test.csv','rb')
testdata = fetch.read().split(b'\r\n')
xtest = []
length = int(len(testdata)/18)
for i in range(length):
	testdata[i*18+9] = testdata[i*18+9].split(b',')
	testdata[i*18+9] = testdata[i*18+9][11-takelen:11]
	xtest.append(testdata[i*18+9])
	for k in range(takelen):
		if float(xtest[i][takelen-1-k])<lowerbnd or float(xtest[i][takelen-1-k])>upperbnd:
			if(k!=0):
				xtest[i][takelen-1-k] = xtest[i][takelen-k]
			else:
				xtest[i][takelen-1-k] = xtest[i][takelen-1-k-malfunc]
		if float(xtest[i][takelen-1-k])<tlowerbnd:
			xtest[i][takelen-1-k] = tlowerbnd
		if float(xtest[i][takelen-1-k])>tupperbnd:
			xtest[i][takelen-1-k] = tupperbnd
xtest = np.asarray(xtest)
testlength = len(xtest)
xtest = xtest.reshape(testlength,-1)
xtest = np.concatenate((xtest,np.ones((testlength,1))),axis=1)
xtest = xtest.astype(float)

testave = []
for i in range(testlength):
	testave.append(np.mean(xtest[i][:-1]))


#####Linear regression#####
preresult = []
result = []
w = np.zeros(takelen+1)
w[-2] = 1
for i in range(Blen):
	for j in range(100*trainlength[i]):
		L = np.dot(xtrain[i],w)
		Lsum = math.sqrt(np.sum(np.power((L-ytrain[i]),2))/trainlength[i])
		w-=rate*(np.dot(xtrain[i].T,L-ytrain[i]))/trainlength[i]/Lsum
	preresult.append(np.dot(xtest,w))
print('id,value')
for i in range(testlength):
	for j in range(Blen):
		if testave[i]>=BOUND[j] and testave[i]<BOUND[j+1]:
			result.append(preresult[j][i])
	if testave[i]>=50 and testave[i]<53:
		result[i] = testave[i]
	print('id_%d,%f'%(i,result[i]))
