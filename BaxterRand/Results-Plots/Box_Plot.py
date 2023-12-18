from distutils.log import error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('BaxterRandTest.csv')
data1 = np.loadtxt('BaxterRandTrainingTest3')

error_x = []
error_y = []
error_z = []
error_k = []
error_m = []
error_n = []
error_l = []
summation = 0  
n = len(data) 
for i in range (0,n):  
  difference = data[i] - data1[i]  
  squared_difference = difference**2
  error_x.append(squared_difference[0])
  error_y.append(squared_difference[1])
  error_z.append(squared_difference[2])
  error_k.append(squared_difference[3])
  error_m.append(squared_difference[4])
  error_n.append(squared_difference[5])
  error_l.append(squared_difference[6])

plt.title('Neural_Network3')
plt.xlabel('Outputs')
plt.ylabel('Squared_Difference')

green_circle = dict(markerfacecolor='blue', marker='o')

plt.boxplot([error_x, error_y, error_z, error_k, error_m, error_n, error_l],flierprops=green_circle, showfliers=False, patch_artist=True)
# plt.legend()
plt.savefig("BaxterRand3_BoxPlot.pdf", format = "pdf")
plt.show()
