# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 09:51:05 2021

@author: omkar_mulekar

Runs "Peeled Back" closed loop ANN controller and simulates

"""


# ============================================================================
# Import Dependencies
# ============================================================================
from keras import models
from scipy.io import loadmat
import numpy as np
from scipy import integrate
import LanderDynamics as LD
from matplotlib import pyplot as plt


# ============================================================================
# Load Data
# ============================================================================
print("Loading mat file")

trajToRun = 0 # Indexing variable


matfile = loadmat('ANN2_data.mat')
idxs = range((trajToRun-1)*100,(trajToRun-1)*100 + 100)
ctrlProfile = matfile['tfull_2'][idxs,:]
trajFromOCL = matfile['Xfull_2'][idxs,:]*np.array([-1,-1,-1,-1,-1,-1,1])
times = matfile['times'][idxs,:]

# matfile = loadmat('../TrajectoryGeneration/ToOrigin_Trajectories/d20210512_15o18_genTrajs.mat')
# # matfile = loadmat('../TrajectoryGeneration/ToOrigin_Trajectories/d20210512_15o21_genTrajs.mat')
# ctrlProfile = matfile['ctrlOut'].reshape(100,2,-1)[:,:,trajToRun]
# trajFromOCL = matfile['stateOut'].reshape(100,8,-1)[:,1:8,trajToRun]
# times = matfile['stateOut'].reshape(100,8,-1)[:,0,trajToRun]

# Load ANN
filename = 'ANN2_703_relu.h5'
# filename = '../ImitationLearning/FirstIL_ANN.h5'
ANN2 = models.load_model(filename)


# ============================================================================
# Simulation Setup/Parameters
# ============================================================================

x0 = trajFromOCL[0,:]

n_times   =  len(times)
nState    =  7
nCtrl     =  2


t  = np.zeros(times.size)
x  = np.zeros((n_times,nState))
Fi = np.zeros((n_times,nCtrl))


# ============================================================================
# Run Simulation
# ============================================================================
# x0[2] = 0;
# x0[3] = 0;
# x0[4] = 0;
# x0[5] = 0;
x[0,:] = x0


for i in range(n_times-1):
    # print(i)
    
    controller_input = 

    Fi[i,:] = ctrlProfile[i,:]

        
    # Integrate dynamics
    sol = integrate.solve_ivp(fun=lambda t, y: LD.LanderEOM(t,y,Fi[i,:]),\
                                   t_span=(times[i],times[i+1]), \
                                   y0=x[i,:]) # Default method: rk45
    
    # Pull out integration solution
    xsol = sol.y
    tsol = sol.t
                
    # 
    x[i+1,:] = xsol[:,xsol.shape[1]-1]
    
#END i LOOP
    
    
    
# ============================================================================
# Plotting
# ============================================================================


plt.figure(1)
plt.plot(trajFromOCL[:,0],trajFromOCL[:,1])
plt.plot(x[:,0],x[:,1],'--')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(['OpenOCL','ANN'],loc='best')


plt.figure(2)
plt.subplot(221)
plt.plot(times,trajFromOCL[:,0])
plt.plot(times,x[:,0],'--')
plt.ylabel('x [m]')
plt.legend(['OpenOCL','ANN'],loc='best')

plt.subplot(222)
plt.plot(times,trajFromOCL[:,1])
plt.plot(times,x[:,1],'--')
plt.ylabel('y [m]')

plt.subplot(223)
plt.plot(times,trajFromOCL[:,2])
plt.plot(times,x[:,2],'--')
plt.xlabel('Time [s]')
plt.ylabel('phi [rad]')

plt.subplot(224)
plt.plot(times,trajFromOCL[:,6])
plt.plot(times,x[:,6],'--')
plt.xlabel('Time [s]')
plt.ylabel('m [kg]')

plt.suptitle('Trajectory',y=1.05)
plt.tight_layout()




plt.figure(3)
plt.subplot(221)
plt.plot(times,trajFromOCL[:,3])
plt.plot(times,x[:,3],'--')
plt.ylabel('x-dot [m/s]')
plt.legend(['OpenOCL','ANN'],loc='best')

plt.subplot(222)
plt.plot(times,trajFromOCL[:,4])
plt.plot(times,x[:,4],'--')
plt.ylabel('y-dot [m/s]')

plt.subplot(223)
plt.plot(times,trajFromOCL[:,5])
plt.plot(times,x[:,5],'--')
plt.xlabel('Time [s]')
plt.ylabel('phi-dot [rad/s]')

plt.suptitle('Trajectory',y=1.05)
plt.tight_layout()




plt.figure(4)
plt.subplot(211)
plt.plot(times,ctrlProfile[:,0])
plt.plot(times,Fi[:,0],'--')
plt.ylabel('Fx [N]')
plt.legend(['OpenOCL','ANN'],loc='best')

plt.subplot(212)
plt.plot(times,ctrlProfile[:,1])
plt.plot(times,Fi[:,1],'--')
plt.ylabel('Fy [N]')



