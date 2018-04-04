import numpy as np
import sys

threshold = 0
epoch = 3000
rate = 5e-7
lam = 5e-6
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
xtrain = np.loadtxt(fetch[1:],delimiter=',')
trainlength = len(xtrain)
xtrain = np.concatenate((np.ones((trainlength,1)),xtrain[:,choose]),axis=1)
fetch = open(sys.argv[2]).read().split('\n')
ytrain = np.loadtxt(fetch,delimiter=',')
ytrain[ytrain==0] = -1

#####load_testdata#####
fetch = open(sys.argv[3]).read().split('\n')[1:]
xtest = np.loadtxt(fetch,delimiter=',')
testlength = len(xtest)
xtest = np.concatenate((np.ones((testlength,1)),xtest[:,choose]),axis=1)

#####train#####
w = np.linalg.pinv(xtrain).dot(ytrain)
sgra = np.zeros(xtrain.shape[1])
for i in range(epoch):
	L = 1./(1.+np.exp(-ytrain*np.dot(xtrain,w)))
	predicted = np.sign(np.dot(xtrain,w))
	grad = -np.dot(ytrain*xtrain.T,L)/trainlength+lam*w
	sgra+=grad**2
	ada = np.sqrt(sgra)
	w-=rate*grad/ada
result = np.dot(xtest,w)
result[result>0] = 1
result[result<=0] = 0
print('id,label',file=output)
for i in range(testlength):
	print('%d,%d'%(i+1,int(result[i])),file = output)
