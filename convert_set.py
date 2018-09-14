#import train_set
import numpy as np
from test_set import test_set_str
	
def parse(data_str):
	'''
	Parse given data string into observations and labels

	:param data_str: string of observation, label examples in given format
	:type data_str: string
	:return: (X,Y) matrices where each line in X is an observation, each line in Y
		is corresponding label.
	'''
	X = []
	Y = []
	sep = data_str.split('----')
	for inst in sep:
		inst_lst = inst.split()
		if len(inst_lst) == 6:
			r = [float(elem) for elem in inst_lst[:3]]
			Y.append([float(elem) for elem in inst_lst[:3]])
			X.append([float(elem) for elem in inst_lst[3:5]])
	return (X,Y)

### SAMPLE STRING FOR TESTING ###

tester = """334.5179					
3210.784148					
19.962276					
1.169307					
0.165084					
514.749897					
----					
1570.48172					
3468.106238					
48.158829					
4.619207					
0.41717					
1174.068773					
----					
2270.654518					
759.693063					
3.842391					
3.628208					
0.134283					
655.865276					
----"""