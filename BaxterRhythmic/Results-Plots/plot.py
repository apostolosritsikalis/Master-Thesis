from cProfile import label
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pylab import *

data = np.loadtxt('BaxterRythmic1.csv')
data1 = np.loadtxt('BaxterRythmic2.csv')
data2 = np.loadtxt('BaxterRythmic3.csv')
data3 = np.loadtxt('BaxterRythmic4.csv')

plt.title('BaxterRythmic_Dataset')
plt.xlabel('Batches')
plt.ylabel('Loss')

plt.plot(data, label= 'Neural_Network_1', alpha=0.5, color='blue')
# plt.plot(data1, label='Neural_Network_2', alpha=0.5, color='red')
# plt.plot(data2, label='Neural_Network_3', alpha=0.5, color='green')
# plt.plot(data3, label='Neural_Network_4', alpha=0.5, color='red')
plt.legend()
plt.savefig("BaxterRythmic1.pdf", format="pdf")
plt.show()
