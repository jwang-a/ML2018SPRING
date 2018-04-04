import numpy as np
import math
import sys

threshold = 0.000
outthres = 0.5
ave = 10
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
	if '?' in labels[i]:
		continue
	if labels[i][:13]=='education_num':
		continue
	if pick[i]>threshold:
		choose.append(i)
chooselen = len(choose)
xtrain = np.loadtxt(fetch[1:],delimiter=',')
trainlength = len(xtrain)
xtrain = np.concatenate((xtrain[:,choose],np.ones((trainlength,1))),axis=1)
fetch = open(sys.argv[2]).read().split('\n')
ytrain = np.loadtxt(fetch,delimiter=',')

#####load_testdata#####
fetch = open(sys.argv[3]).read().split('\n')[1:]
xtest = np.loadtxt(fetch,delimiter=',')
testlength = len(xtest)
xtest = np.concatenate((xtest[:,choose],np.ones((testlength,1))),axis=1)

#####train#####
G1 = xtrain[ytrain==0]
G2 = xtrain[ytrain==1]
G1ave = np.average(G1,axis=0)
G2ave = np.average(G2,axis=0)
G1div = np.dot((G1-G1ave).T,(G1-G1ave))
G2div = np.dot((G2-G2ave).T,(G2-G2ave))
div = (G1div+G2div)/trainlength
PG1 = math.log(len(G1)/trainlength)
PG2 = math.log(len(G2)/trainlength)
print('id,label',file=output)
for i in range(testlength):
	PxG1 = (-np.dot(np.dot((xtest[i]-G1ave),np.linalg.pinv(div)),(xtest[i]-G1ave).T)/2)
	PxG2 = (-np.dot(np.dot((xtest[i]-G2ave),np.linalg.pinv(div)),(xtest[i]-G2ave).T)/2)
	P1 = PxG1+PG1
	P2 = PxG2+PG2
	if P1>P2:
		print('%d,0'%(i+1),file=output)
	else:
		print('%d,1'%(i+1),file=output)
