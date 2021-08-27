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

trajToRun = 1 # Indexing variable


matfile = loadmat('ANN2_data.mat')
idxs = range((trajToRun-1)*100,(trajToRun-1)*100 + 100)
ctrlProfile = matfile['tfull_2'][idxs,:]
ctrlProfileOrig = matfile['t_orig'][idxs,:]
trajFromOCL = matfile['Xfull_2'][idxs,:]*np.array([-1,-1,-1,-1,-1,-1,1])
# trajFromOCL = matfile['Xfull_2'][idxs,:]
times = matfile['times'][idxs,:]

matfile = loadmat('../TrajectoryGeneration/ToOrigin_Trajectories/d20210826_20o12_genTrajs.mat')
# ctrlProfile = matfile['ctrlOut'].reshape(100,2,-1)[:,:,trajToRun-1]
# ctrlProfileOrig = matfile['ctrlOut'].reshape(100,2,-1)[:,:,trajToRun-1]
# trajFromOCL = matfile['stateOut'].reshape(100,8,-1)[:,1:8,trajToRun-1]
# times = matfile['stateOut'].reshape(100,8,-1)[:,0,trajToRun-1]
target = matfile['stateFinal'][trajToRun-1,:]

# Load ANN
# filename = 'ANN2_703_relu_n50.h5'
# filename = 'ANN2_703_relu_n750.h5'
# filename = 'ANN2_703_relu_n100.h5'
# filename = 'ANN2_703_relu_n75.h5'
# filename = 'ANN2_split_703_relu_n200.h5'
# filename = 'ANN2_split_703_relu_n10.h5'
# filename = 'ANN2_split_703_relu_n7.h5'
# filename = 'ANN2_split_703_relu_n25.h5'
# filename = 'ANN2_split_703_relu_n25_75_2000.h5'
# filename = 'ANN2_split_703_relu_n25_75_500.h5'
# filename = 'ANN2_split_703_relu_n25_75_2000.h5'
# filename = 'ANN2_split_703_relu_n25_75_2000_WORKING.h5's
# filename = '../ImitationLearning/FirstIL_ANN.h5'
# filename = 'ANN2_703_relu_n2000.h5'
# filename = 'ANN2_703_relu_n75.h5'
filename = 'ANN2_split_703_relu_n25_75_2000.h5'
ANN2 = models.load_model(filename)


# ============================================================================
# Simulation Setup/Parameters
# ============================================================================

x0 = trajFromOCL[0,:]

n_times   =  len(times)
nState    =  7
nCtrl     =  2


t  = np.zeros(times.size)
phiSave  = np.zeros(times.size)
condNumbers  = np.zeros(times.size)
x  = np.zeros((n_times,nState))
Fi = np.zeros((n_times,nCtrl))
TxTyMsave = np.zeros((n_times,3))
r = 1

# ============================================================================
# Run Simulation
# ============================================================================
# x0[2] = 0;
# x0[3] = 0;
# x0[4] = 0;
# x0[5] = 0;
x[0,:] = x0


for i in range(n_times-1):

    error = target - x[i,:]
    controller_input = np.hstack((error[:6],x[i,6])).reshape(1,-1)

    prediction = ANN2.predict(controller_input)
    TxTyM = np.hstack((prediction[0],prediction[1])).reshape(-1)
    # TxTyM = prediction.reshape(-1)
    
    phi = x[i,2]
    phiSave[i] = phi
    D = np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)],[r,0]])
    condNumbers[i] = np.linalg.cond(D)
    # print(D)
    # print('-----------')
    # print("time {} s".format(times[i]))
    # print("Phi: {}".format(phi))
    # Fi[i,:] = np.linalg.inv((D.transpose()@D))@D.transpose()@TxTyM
    
    TxTyMsave[i,:] = TxTyM
    # TxTyMsave[i,:] = D@ctrlProfileOrig[i,:]
    
    # print('3 vector from Python: {}'.format(TxTyMsave[i,:]))
    # print('3 vect from Matlab: {}'.format(ctrlProfile[i,:]))
    
    print('')
    
    # if i == 70:
    #     print('')
        

    # Fi[i,:] = np.linalg.inv(D.transpose() @ D) @ D.transpose() @ (D@ctrlProfileOrig[i,:])
    Fi[i,:] = np.linalg.inv(D.transpose() @ D) @ D.transpose() @ ctrlProfile[i,:]
    # Fi[i,:] = ctrlProfile[i,:]

    # if i == 30:
    #     Fi[i,0] += 200
            
    # Integrate dynamics
    sol = integrate.solve_ivp(fun=lambda t, y: LD.LanderEOM_coupled(t,y,Fi[i,:]),\
                                   t_span=(times[i],times[i+1]), \
                                   y0=x[i,:]) # Default method: rk45
    
    # Pull out integration solution
    xsol = sol.y
    tsol = sol.t
                
    # 
    x[i+1,:] = xsol[:,xsol.shape[1]-1]
    

    
# ============================================================================
# Evaluate
# ============================================================================
J_ANN = LD.calculatePathCost(times, Fi)
J_OCL = LD.calculatePathCost(times, ctrlProfileOrig)

print("Cost ANN: {}".format(J_ANN))
print("Cost OCL: {}".format(J_OCL))

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
plt.plot(times,ctrlProfileOrig[:,0])
plt.plot(times,Fi[:,0],'--')
plt.ylabel('Fx [N]')
plt.legend(['OpenOCL','ANN'],loc='best')

plt.subplot(212)
plt.plot(times,ctrlProfileOrig[:,1])
plt.plot(times,Fi[:,1],'--')
plt.ylabel('Fy [N]')


plt.figure(5)
plt.subplot(311)
plt.plot(times,ctrlProfile[:,0])
plt.plot(times,TxTyMsave[:,0],'--')
plt.ylabel('Tx [N]')
plt.legend(['OpenOCL','ANN'],loc='best')

plt.subplot(312)
plt.plot(times,ctrlProfile[:,1])
plt.plot(times,TxTyMsave[:,1],'--')
plt.ylabel('Ty [N]')

plt.subplot(313)
plt.plot(times,ctrlProfile[:,2])
plt.plot(times,TxTyMsave[:,2],'--')
plt.ylabel('M [Nm]')
