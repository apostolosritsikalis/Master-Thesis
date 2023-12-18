from cProfile import label
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pylab import *

data = np.loadtxt('BaxterRand1.csv')
data1 = np.loadtxt('BaxterRand2.csv')
data2 = np.loadtxt('BaxterRand3.csv')
data3 = np.loadtxt('BaxterRand4.csv')

plt.title('BaxterRand_Dataset')
plt.xlabel('Batches')
plt.ylabel('Loss')

plt.plot(data, label= 'Neural_Network_1', alpha=0.5, color='blue')
# plt.plot(data1, label='Neural_Network_2', alpha=0.5, color='red')
# plt.plot(data2, label='Neural_Network_3', alpha=0.5, color='green')
# plt.plot(data3, label='Neural_Network_4', alpha=0.5, color='red')
plt.legend()
plt.savefig("BaxterRand1.pdf", format="pdf")
plt.show()
