import numpy as np
import math
import sys

takelen = 8
upperbnd = 120
tupperbnd = 110
lowerbnd = 2
tlowerbnd = 2
malfunc = 3
BOUND = [lowerbnd,5,25,50,53,55,57,60,63,65,67,70,upperbnd]
Blen = len(BOUND)-1
output = open(sys.argv[2],'w')

#####Deal testdata#####
fetch = open(sys.argv[1],'rb')
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
w = np.load('hw1_best_w.npy')
preresult = []
result = []
for i in range(Blen):
	preresult.append(np.dot(xtest,w[i]))
print('id,value',file=output)
for i in range(testlength):
	for j in range(Blen):
		if testave[i]>=BOUND[j] and testave[i]<BOUND[j+1]:
			result.append(preresult[j][i])
	if testave[i]>=50 and testave[i]<53:
		result[i] = testave[i]-1
	if testave[i]>=55 and testave[i]<57:
		result[i] = result[i]*2-testave[i]
	print('id_%d,%f'%(i,result[i]),file=output)
