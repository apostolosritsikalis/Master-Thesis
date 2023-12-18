import numpy as np 
import RobotDART as rd
import dartpy
import copy
from utils import damped_pseudoinverse, AdT
import torch
import torch.nn as nn
import geotorch
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
from numpy import asarray, zeros
from numpy import save
from numpy import load
import math
import pandas as pd


class PSD_Matrix(nn.Module):   
    def __init__(self,L=None):
        super().__init__()
        self.L = L
        self.A = torch.nn.Linear(3,3, bias = False)
        geotorch.positive_definite(self.A, "weight")

        def forward(self,s):
            return self.A.mul(self.L-s)
        
   
model = PSD_Matrix()
print(model)


class PITask:
    def __init__(self, target, dt, Kp = 10., Ki = 0.1):
        self._target = target
        self._dt = dt
        self._Kp = Kp
        self._Ki = Ki
        self._sum_error = 0
    
    def set_target(self, target):
        self._target = target
    
    def error(self, tf):
        return rd.math.logMap(tf.inverse().multiply(self._target))
    
    def update(self, current):
        Ad_tf = AdT(current)
        error_in_body_frame = self.error(current)
        error_in_world_frame = Ad_tf @ error_in_body_frame

        self._sum_error = self._sum_error + error_in_world_frame * self._dt

        return self._Kp * error_in_world_frame + self._Ki * self._sum_error

packages = [("franka_description", "franka/franka_description")]
robot = rd.Robot("franka/franka.urdf", packages)
robot.set_color_mode("material")

robot.fix_to_world()
robot.set_position_enforced(False)
robot.set_actuator_types("servo")

positions = robot.positions()
positions[3] = np.pi / 1.5
positions[7] = positions[8] = 0.03
robot.set_positions(positions)

robot_ghost = robot.clone_ghost()

target_positions = copy.copy(robot.positions())
target_positions[0] = -2.
target_positions[3] = -np.pi / 2
target_positions[5] = np.pi / 2
robot.set_positions(target_positions)

eef_link_name = "panda_hand"
tf_desired = robot.body_pose(eef_link_name)

pos = robot.positions() + np.random.rand(robot.num_dofs())*np.pi / 1.5 - np.pi / 3.
pos[7] = pos[8] = 0.03
robot.set_positions(pos)

dt = 0.01
simu = rd.RobotDARTSimu(dt)

simu.set_collision_detector("fcl")

graphics = rd.gui.Graphics()
simu.set_graphics(graphics)
graphics.look_at([3., 1., 2.], [0., 0., 0.])

simu.add_robot(robot)
robot_ghost.set_positions(target_positions)
simu.add_robot(robot_ghost)
simu.add_checkerboard_floor()

L = tf_desired.translation()
A = np.array(([1, 0, 0],[0, 1, 0], [0, 0, 1]))

# input_data = torch.tensor(A)

def con(s):
    v = A @ (L - s)
    return v 


Kp = 2. 
Ki = 0.01 

controller = PITask(tf_desired, dt, Kp, Ki)

Sa = []
Va = []

while True:
    if simu.step_world():
        break
    tf= robot.body_pose(eef_link_name) 
    vel = controller.update(tf)
    vel[3:] = con(tf.translation())
    
    Sa.append(tf.translation())
    Va.append(vel[3:])
    
    jac = robot.jacobian(eef_link_name) 
    jac_pinv = damped_pseudoinverse(jac) 
    cmd = jac_pinv @ vel

    robot.set_commands(cmd)

Sa = np.asarray(Sa)
print(Sa)
Va = np.asarray(Va)

data = Sa[:,0],Sa[:,1],Sa[:,2],Va[:,0],Va[:,1],Va[:,2]
print(np.shape(data))

df = pd.DataFrame(list(data))
df_T = df.T
print(df_T)
df_T.to_csv('franka.csv', index=False)  


# print(Sa.shape)
# print(Sa)
# print(Va.shape)
# print(Va)

#dataset

class S_Dataset():
    def __init__(self):
        self.x = Sa
        self.y = Va
        self.Sa = Sa
        self.Va = Va
       
        
    def __len__(self):
        return len(self.Sa)

    def __getitem__(self, idx):
        return [self.Sa[idx], self.Va[idx]]  
    
             
dataset = S_Dataset()


# training_loop

# epochs = 2
# samples = len(dataset)
# batch = 8
# print(samples)
# j = samples%batch
# print(j)

    
# for epoch in range(epochs):
#     if j == 0:
#         for i in range(0, samples, batch):
#             print(f'epoch: {epoch + 1}/{epochs}') 
#             print(f'batch: {i}/{samples} , {dataset[i:i+batch]}')
#     else:
#         for i in range (0, samples, j):
#             print(f'epoch: {epoch + 1}/{epochs}') 
#             print(f'batch: {i+j}/{samples} , {dataset[i:i+batch]}')       
     
