from cProfile import label
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pylab import *

data = np.loadtxt('urPicknPlace1.csv')
data1 = np.loadtxt('urPicknPlace2.csv')
data2 = np.loadtxt('urPicknPlace3.csv')
data3 = np.loadtxt('urPicknPlace4.csv')

plt.title('URpickNplace_Dataset')
plt.xlabel('Batches')
plt.ylabel('Loss')

# plt.plot(data, label= 'Neural_Network_1', alpha=0.5, color='blue')
plt.plot(data1, label='Neural_Network_2', alpha=0.5, color='red')
plt.plot(data2, label='Neural_Network_3', alpha=0.5, color='green')
plt.legend()
plt.savefig("urPicknPlace2.pdf", format="pdf")
plt.show()
