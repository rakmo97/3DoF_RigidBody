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
%matplotlib inline

# ============================================================================
# Load Network
# ============================================================================
# filename = 'ANN2_703_tanh.h5'
# filename = 'ANN2_703_relu.h5'
filename = 'E:/Research_Data/3DoF_RigidBody/Code_coupled_constmass/NetworkTraining/ANN2_703_relu_n100.h5'
ANN2 = models.load_model(filename)



# ============================================================================
# Load Data
# ============================================================================
print("Loading mat file")
# matfile = loadmat('ANN2_data.mat')
# matfile = loadmat('ANN2_decoupled_data.mat')
matfile = loadmat('E:/Research_Data/3DoF_RigidBody/Code_coupled_constmass/TrajectoryGeneration/ToOrigin_Trajectories/d20211214_11o55_genTrajs.mat')
print("Mat file loaded!")

# # matfile = loadmat('ANN1_data_notaug.mat')
# Xfull = matfile['Xfull_2']
# tfull = matfile['tfull_2']
# X_train = matfile['Xtrain2'].reshape(-1,7)
# t_train = matfile['ttrain2']
# times_train = matfile['times_train']
# # t_train = matfile['ttrain2'][:,1]
# X_test = matfile['Xtest2'].reshape(-1,7)
# t_test = matfile['ttest2']
# times_test = matfile['times_test']

trajToRun = 0 # Indexing variable

# idxs = range((trajToRun-1)*100,(trajToRun-1)*100 + 100)

# ctrlProfile = t_test[idxs,:]
# trajFromOCL = -X_test[idxs,:]
# trajFromOCL[:,6] = -trajFromOCL[:,6]
# timesOCL = times_test[idxs]

# ctrlProfile = t_train[idxs,:]
# trajFromOCL = -X_train[idxs,:]
# trajFromOCL[:,6] = -trajFromOCL[:,6]
# timesOCL = times_train[idxs]


ctrlProfile = matfile['ctrlOut'].reshape(100,2,-1)[:,:,trajToRun]
trajFromOCL = matfile['stateOut'].reshape(100,7,-1)[:,1:7,trajToRun]
timesOCL = matfile['stateOut'].reshape(100,7,-1)[:,0,trajToRun]


# times = np.linspace(0,48,5000)
times = timesOCL
end = len(times) # Indexing variable

# ============================================================================
# Simulation Setup/Parameters
# ============================================================================

x0 = trajFromOCL[0,:]
target = trajFromOCL[100-1,:]



n_times   = end
nState    =   6
nCtrl     =   2



t = np.zeros(times.size)
x = np.zeros((n_times,nState))
Fi = np.zeros((n_times,nCtrl))


# ============================================================================
# Run Simulation
# ============================================================================
# x0[2] = 0;
# x0[3] = 0;
# x0[4] = 0;
# x0[5] = 0;
x[0,:] = x0


error = target - x[0,:] # Error
controller_input = np.append(error[0:5],x[0,5])

# Fi[0,:]= ANN2.predict(np.reshape(controller_input,(1,-1)))
Fi[0,:] = ctrlProfile[0,:]
# Fi[0,:] = np.array([20,0])
count = 1
print('Running sim...')
for i in range(1,end):
    # print(i)
    

    

    # t = np.linspace(times[i-1],times[i])
    t = np.linspace(timesOCL[i-1],timesOCL[i])

        
        # Integrate dynamics
    sol = integrate.solve_ivp(fun=lambda t, y: LD.LanderEOM(t,y,Fi[i-1,:]),\
                                   t_span=(times[i-1],times[i]), \
                                   y0=x[i-1,:],\
                                   t_eval=t.reshape(-1)) # Default method: rk45
    

    xsol = sol.y
    tsol = sol.t
                
    
    x[i,:] = xsol[:,xsol.shape[1]-1]
    
    error = target - x[i,:] # Error
    controller_input = error[0:6]
    
    Fi[i,:]= ANN2.predict(np.reshape(controller_input,(1,-1)))
    # Fi[i,:]= ctrlProfile[i,:]
    # Fi[i,:]= np.array([20,0])
    
#END i LOOP
    
print('Sim complete!')
    
# ============================================================================
# Plotting
# ============================================================================
print("Plotting")

plt.figure(1)
plt.plot(trajFromOCL[:,0],trajFromOCL[:,1])
plt.plot(x[:,0],x[:,1],'--')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(['OpenOCL','ANN'],loc='best')


plt.figure(2)
plt.subplot(221)
plt.plot(timesOCL,trajFromOCL[:,0])
plt.plot(times,x[:,0],'--')
plt.ylabel('x [m]')
plt.legend(['OpenOCL','ANN'],loc='best')

plt.subplot(222)
plt.plot(timesOCL,trajFromOCL[:,1])
plt.plot(times,x[:,1],'--')
plt.ylabel('y [m]')

plt.subplot(223)
plt.plot(timesOCL,trajFromOCL[:,2])
plt.plot(times,x[:,2],'--')
plt.xlabel('Time [s]')
plt.ylabel('phi [rad]')

plt.suptitle('Trajectory',y=1.05)
plt.tight_layout()




plt.figure(3)
plt.subplot(221)
plt.plot(timesOCL,trajFromOCL[:,3])
plt.plot(times,x[:,3],'--')
plt.ylabel('x-dot [m/s]')
plt.legend(['OpenOCL','ANN'],loc='best')

plt.subplot(222)
plt.plot(timesOCL,trajFromOCL[:,4])
plt.plot(times,x[:,4],'--')
plt.ylabel('y-dot [m/s]')

plt.subplot(223)
plt.plot(timesOCL,trajFromOCL[:,5])
plt.plot(times,x[:,5],'--')
plt.xlabel('Time [s]')
plt.ylabel('phi-dot [rad/s]')



plt.suptitle('Trajectory',y=1.05)
plt.tight_layout()




plt.figure(4)
plt.subplot(211)
plt.plot(timesOCL,ctrlProfile[:,0])
plt.plot(times,Fi[:,0],'--')
plt.ylabel('Fx [N]')
plt.legend(['OpenOCL','ANN'],loc='best')

plt.subplot(212)
plt.plot(timesOCL,ctrlProfile[:,1])
plt.plot(times,Fi[:,1],'--')
plt.ylabel('Fy [N]')
plt.xlabel('Time [s]')



