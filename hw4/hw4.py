import numpy as np
import sys

output = open(sys.argv[3],'w')

label = np.load('labels.npy')
test = open(sys.argv[2]).read().split('\n')[1:-1]
length = len(test)

print('ID,Ans',file=output)
for i in range(length):
	test[i] = test[i].split(',')
	lab1 = label[int(test[i][1])]
	lab2 = label[int(test[i][2])]
	if lab1==lab2:
		print('%d,1'%(i),file=output)
	else:
		print('%d,0'%(i),file=output)
