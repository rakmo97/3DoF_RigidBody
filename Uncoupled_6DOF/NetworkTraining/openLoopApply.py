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
%matplotlib auto

# ============================================================================
# Load Data
# ============================================================================
print("Loading mat file")

trajToRun = 0 # Indexing variable


matfile = loadmat('ANN2_data.mat')
idxs = range((trajToRun-1)*100,(trajToRun-1)*100 + 100)
ctrlProfile = matfile['tfull_2'][idxs,:]
trajFromOCL = matfile['Xfull_2'][idxs,:]*np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1])
times = matfile['times'][idxs,:]

# matfile = loadmat('../TrajectoryGeneration/ToOrigin_Trajectories/d20210512_15o18_genTrajs.mat')
# # matfile = loadmat('../TrajectoryGeneration/ToOrigin_Trajectories/d20210512_15o21_genTrajs.mat')
# ctrlProfile = matfile['ctrlOut'].reshape(100,2,-1)[:,:,trajToRun]
# trajFromOCL = matfile['stateOut'].reshape(100,8,-1)[:,1:8,trajToRun]
# times = matfile['stateOut'].reshape(100,8,-1)[:,0,trajToRun]



# ============================================================================
# Simulation Setup/Parameters
# ============================================================================

x0 = trajFromOCL[0,:]

n_times   =  len(times)
nState    =  13
nCtrl     =  6


t  = np.zeros(times.size)
x  = np.zeros((n_times,nState))
Fi = np.zeros((n_times,nCtrl))


# ============================================================================
# Run Simulation
# ============================================================================

x[0,:] = x0


for i in range(n_times-1):
   
    Fi[i,:] = ctrlProfile[i,:]

        
    # Integrate dynamics
    sol = integrate.solve_ivp(fun=lambda t, y: LD.LanderEOM(t,y,Fi[i,:]),\
                                   t_span=(times[i],times[i+1]), \
                                   y0=x[i,:]) # Default method: rk45
    
    # Pull out integration solution
    xsol = sol.y
    tsol = sol.t
                
    
    x[i+1,:] = xsol[:,xsol.shape[1]-1]
    
    
    
# ============================================================================
# Plotting
# ============================================================================

plt.figure(1)
ax = plt.axes(projection='3d')
ax.plot3D(trajFromOCL[:,0],trajFromOCL[:,1],trajFromOCL[:,2])
ax.plot3D(x[:,0],x[:,1],x[:,2],'--')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(['OpenOCL','ANN'],loc='best')



plt.figure(2)
plt.subplot(231)
plt.plot(times,trajFromOCL[:,0])
plt.plot(times,x[:,0],'--')
plt.ylabel('x [m]')
plt.legend(['OpenOCL','ANN'],loc='best')

plt.subplot(232)
plt.plot(times,trajFromOCL[:,1])
plt.plot(times,x[:,1],'--')
plt.ylabel('y [m]')

plt.subplot(233)
plt.plot(times,trajFromOCL[:,2])
plt.plot(times,x[:,2],'--')
plt.xlabel('Time [s]')
plt.ylabel('z [m]')

plt.subplot(234)
plt.plot(times,trajFromOCL[:,3])
plt.plot(times,x[:,3],'--')
plt.xlabel('Time [s]')
plt.ylabel('vx [m/s]')

plt.subplot(235)
plt.plot(times,trajFromOCL[:,4])
plt.plot(times,x[:,4],'--')
plt.xlabel('Time [s]')
plt.ylabel('vy [m/s]')

plt.subplot(236)
plt.plot(times,trajFromOCL[:,5])
plt.plot(times,x[:,5],'--')
plt.xlabel('Time [s]')
plt.ylabel('vz [m/s]')
plt.suptitle('Trajectory',y=1.05)
plt.tight_layout()








plt.figure(3)
plt.subplot(231)
plt.plot(times,np.rad2deg(1)*trajFromOCL[:,6])
plt.plot(times,np.rad2deg(1)*x[:,6],'--')
plt.ylabel('Roll(phi) [deg]')
plt.legend(['OpenOCL','ANN'],loc='best')

plt.subplot(232)
plt.plot(times,np.rad2deg(1)*trajFromOCL[:,7])
plt.plot(times,np.rad2deg(1)*x[:,7],'--')
plt.ylabel('Pitch(theta) [deg]')

plt.subplot(233)
plt.plot(times,np.rad2deg(1)*trajFromOCL[:,8])
plt.plot(times,np.rad2deg(1)*x[:,8],'--')
plt.xlabel('Time [s]')
plt.ylabel('Yaw(psi) [deg]')

plt.subplot(234)
plt.plot(times,trajFromOCL[:,9])
plt.plot(times,x[:,9],'--')
plt.xlabel('Time [s]')
plt.ylabel('P [deg/s]')

plt.subplot(235)
plt.plot(times,trajFromOCL[:,10])
plt.plot(times,x[:,10],'--')
plt.xlabel('Time [s]')
plt.ylabel('Q [deg/s]')

plt.subplot(236)
plt.plot(times,trajFromOCL[:,11])
plt.plot(times,x[:,11],'--')
plt.xlabel('Time [s]')
plt.ylabel('R [deg/s]')
plt.suptitle('Trajectory',y=1.05)
plt.tight_layout()


plt.figure(4)
plt.plot(times,trajFromOCL[:,12])
plt.plot(times,x[:,12],'--')
plt.xlabel('Time [s]')
plt.ylabel('m [kg]')


plt.figure(5)
plt.subplot(231)
plt.plot(times,ctrlProfile[:,0])
plt.plot(times,Fi[:,0],'--')
plt.ylabel('Fx [N]')
plt.legend(['OpenOCL','ANN'],loc='best')

plt.subplot(232)
plt.plot(times,ctrlProfile[:,1])
plt.plot(times,Fi[:,1],'--')
plt.ylabel('Fy [N]')

plt.subplot(233)
plt.plot(times,ctrlProfile[:,2])
plt.plot(times,Fi[:,2],'--')
plt.ylabel('Fz [N]')

plt.subplot(234)
plt.plot(times,ctrlProfile[:,3])
plt.plot(times,Fi[:,3],'--')
plt.ylabel('L [Nm]')

plt.subplot(235)
plt.plot(times,ctrlProfile[:,4])
plt.plot(times,Fi[:,4],'--')
plt.ylabel('M [Nm]')

plt.subplot(236)
plt.plot(times,ctrlProfile[:,5])
plt.plot(times,Fi[:,5],'--')
plt.ylabel('N [Nm]')

plt.suptitle('Trajectory',y=1.05)
plt.tight_layout()
