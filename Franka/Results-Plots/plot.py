from cProfile import label
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pylab import *

data = np.loadtxt('Franka1.csv')
data1 = np.loadtxt('Franka2.csv')
data2 = np.loadtxt('Franka3.csv')
data3 = np.loadtxt('Franka4.csv')

plt.title('Franka_Dataset')
plt.xlabel('Batches')
plt.ylabel('Loss')

plt.plot(data, label= 'Neural_Network_1', alpha=0.5, color='blue')
plt.plot(data1, label='Neural_Network_2', alpha=0.5, color='red')
plt.plot(data2, label='Neural_Network_3', alpha=0.5, color='green')
plt.legend()
plt.savefig("Franka1.pdf", format="pdf")
plt.show()
