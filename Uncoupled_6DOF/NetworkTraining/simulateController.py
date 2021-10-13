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
# filename = 'ANN2_relu_n1000.h5'
# filename = 'ANN2_relu_n1500.h5'
# filename = 'ANN2_relu_n1000_nA1500_nB1500.h5'
# filename = 'ANN2_relu_n1000_nA200_nB200.h5'
# filename = 'ANN2_relu_n1000_nA750_nB750.h5'
filename = 'ANN2_relu_n1000_nA750_nB750_specnorm.h5'
# filename = '../ImitationLearning/FirstIL_ANN.h5'
ANN2 = models.load_model(filename)


# ============================================================================
# Load Data
# ============================================================================
print("Loading mat file")

trajToRun = 1 # Indexing variable


matfile = loadmat('ANN2_data.mat')
idxs = range((trajToRun-1)*101,(trajToRun-1)*101 + 101)
ctrlProfile = matfile['tfull_2'][idxs,:]
trajFromOCL = matfile['Xfull_2'][idxs,:]*np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1])
timesOCL = matfile['times'][idxs,:].reshape(-1)

# ============================================================================
# Simulation Setup/Parameters
# ============================================================================

extra_time = 0
delta_t = np.diff(timesOCL,axis=0)[0]
timesANN = np.hstack((timesOCL,np.arange(timesOCL[99]+delta_t,timesOCL[99]+extra_time,delta_t)))

n_times   =  len(timesANN)
nState    =  13
nCtrl     =  6

x0 = trajFromOCL[0,:]
target = np.array([0,0,0,0,0,0,0,0,0,0,0,0])

x  = np.zeros((1,nState))
Fi = np.zeros((1,nCtrl))

# ============================================================================
# Run Simulation
# ============================================================================
x[0,:] = x0


for i in range(n_times-1):

    controller_input = np.hstack((target - x[i,0:12],x[i,12])).reshape(1,-1)
    prediction = ANN2.predict(controller_input)
    Fi_prep = np.hstack((prediction[0],prediction[1]))
    Fi = np.vstack((Fi,Fi_prep))
    # Fi[i,:] = ctrlProfile[i,:]
    
    
    # if(i == 30):
    #     print("Applying disturbance at Time {} s".format(timesANN[i]))
    #     Fi[i,0] += 1000
    
        
    # Integrate dynamics
    sol = integrate.solve_ivp(fun=lambda t, y: LD.LanderEOM(t,y,Fi[i,:]),\
                                   t_span=(timesANN[i],timesANN[i+1]), \
                                   y0=x[i,:]) # Default method: rk45
    
    # Pull out integration solution
    xsol = sol.y
    tsol = sol.t
                
    # x[i+1,:] = xsol[:,xsol.shape[1]-1]
    x = np.vstack((x,xsol[:,xsol.shape[1]-1]))
    

    
    
    
# ============================================================================
# Plotting
# ============================================================================
plt.close('all')

plt.figure(1)
ax = plt.axes(projection='3d')
ax.plot3D(trajFromOCL[:,0],trajFromOCL[:,1],trajFromOCL[:,2])
ax.plot3D(x[:,0],x[:,1],x[:,2],'--')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(['OpenOCL','ANN'],loc='best')



plt.figure(2)
plt.subplot(231)
plt.plot(timesOCL,trajFromOCL[:,0])
plt.plot(timesANN,x[:,0],'--')
plt.ylabel('x [m]')
plt.legend(['OpenOCL','ANN'],loc='best')

plt.subplot(232)
plt.plot(timesOCL,trajFromOCL[:,1])
plt.plot(timesANN,x[:,1],'--')
plt.ylabel('y [m]')

plt.subplot(233)
plt.plot(timesOCL,trajFromOCL[:,2])
plt.plot(timesANN,x[:,2],'--')
plt.xlabel('Time [s]')
plt.ylabel('z [m]')

plt.subplot(234)
plt.plot(timesOCL,trajFromOCL[:,3])
plt.plot(timesANN,x[:,3],'--')
plt.xlabel('Time [s]')
plt.ylabel('vx [m/s]')

plt.subplot(235)
plt.plot(timesOCL,trajFromOCL[:,4])
plt.plot(timesANN,x[:,4],'--')
plt.xlabel('Time [s]')
plt.ylabel('vy [m/s]')

plt.subplot(236)
plt.plot(timesOCL,trajFromOCL[:,5])
plt.plot(timesANN,x[:,5],'--')
plt.xlabel('Time [s]')
plt.ylabel('vz [m/s]')
plt.suptitle('Trajectory',y=1.05)
plt.tight_layout()








plt.figure(3)
plt.subplot(231)
plt.plot(timesOCL,np.rad2deg(1)*trajFromOCL[:,6])
plt.plot(timesANN,np.rad2deg(1)*x[:,6],'--')
plt.ylabel('Roll(phi) [deg]')
plt.legend(['OpenOCL','ANN'],loc='best')

plt.subplot(232)
plt.plot(timesOCL,np.rad2deg(1)*trajFromOCL[:,7])
plt.plot(timesANN,np.rad2deg(1)*x[:,7],'--')
plt.ylabel('Pitch(theta) [deg]')

plt.subplot(233)
plt.plot(timesOCL,np.rad2deg(1)*trajFromOCL[:,8])
plt.plot(timesANN,np.rad2deg(1)*x[:,8],'--')
plt.xlabel('Time [s]')
plt.ylabel('Yaw(psi) [deg]')

plt.subplot(234)
plt.plot(timesOCL,trajFromOCL[:,9])
plt.plot(timesANN,x[:,9],'--')
plt.xlabel('Time [s]')
plt.ylabel('P [deg/s]')

plt.subplot(235)
plt.plot(timesOCL,trajFromOCL[:,10])
plt.plot(timesANN,x[:,10],'--')
plt.xlabel('Time [s]')
plt.ylabel('Q [deg/s]')

plt.subplot(236)
plt.plot(timesOCL,trajFromOCL[:,11])
plt.plot(timesANN,x[:,11],'--')
plt.xlabel('Time [s]')
plt.ylabel('R [deg/s]')
plt.suptitle('Trajectory',y=1.05)
plt.tight_layout()


plt.figure(4)
plt.plot(timesOCL,trajFromOCL[:,12])
plt.plot(timesANN,x[:,12],'--')
plt.xlabel('Time [s]')
plt.ylabel('m [kg]')


plt.figure(5)
plt.subplot(231)
plt.plot(timesOCL,ctrlProfile[:,0])
plt.plot(timesANN,Fi[:,0],'--')
plt.ylabel('Fx [N]')
plt.legend(['OpenOCL','ANN'],loc='best')

plt.subplot(232)
plt.plot(timesOCL,ctrlProfile[:,1])
plt.plot(timesANN,Fi[:,1],'--')
plt.ylabel('Fy [N]')

plt.subplot(233)
plt.plot(timesOCL,ctrlProfile[:,2])
plt.plot(timesANN,Fi[:,2],'--')
plt.ylabel('Fz [N]')

plt.subplot(234)
plt.plot(timesOCL,ctrlProfile[:,3])
plt.plot(timesANN,Fi[:,3],'--')
plt.ylabel('L [Nm]')

plt.subplot(235)
plt.plot(timesOCL,ctrlProfile[:,4])
plt.plot(timesANN,Fi[:,4],'--')
plt.ylabel('M [Nm]')

plt.subplot(236)
plt.plot(timesOCL,ctrlProfile[:,5])
plt.plot(timesANN,Fi[:,5],'--')
plt.ylabel('N [Nm]')

plt.suptitle('Trajectory',y=1.05)
plt.tight_layout()


