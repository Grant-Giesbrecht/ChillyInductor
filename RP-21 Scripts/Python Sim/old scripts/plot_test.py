import matplotlib.pyplot as plt
import random

from base import *

N_list = list(range(10))

cm = CMap('Greens', data=N_list)

for i in N_list:
	
	data = []
	for x in range(15):
		data.append(random.random())
	
	plt.plot(data, color=cm(i))
	
plt.show()

# cm = plt.get_cmap('Greys')
# cmr = cm.resampled(len(N_list))

# for i in N_list:
	
# 	data = []
# 	for x in range(15):
# 		data.append(random.random())
	
# 	plt.plot(data, color=cmr(int(i)))
	
# plt.show()