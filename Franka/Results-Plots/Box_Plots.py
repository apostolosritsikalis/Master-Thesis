from distutils.log import error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('FrankaTest')
data1 = np.loadtxt('FrankaTrainingTest3')

error_x = []
error_y = []
error_z = []
summation = 0  
n = len(data) 
for i in range (0,n):  
  difference = data[i] - data1[i]  
  squared_difference = difference**2
  error_x.append(squared_difference[0])
  error_y.append(squared_difference[1])
  error_z.append(squared_difference[2])

plt.title('Neural_Network3')
plt.xlabel('Outputs')
plt.ylabel('Squared_Difference')

green_circle = dict(markerfacecolor='red', marker='o')

plt.boxplot([error_x, error_y, error_z],flierprops=green_circle, showfliers=False, patch_artist=True)
# plt.legend()
plt.savefig("Franka3_BoxPlot.pdf", format = "pdf")
plt.show()
