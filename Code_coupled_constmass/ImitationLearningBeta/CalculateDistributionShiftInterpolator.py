# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 13:42:22 2021

@author: omkar_mulekar
"""

# ============================================================================
# Import Dependencies
# ============================================================================

from keras import models
# import ImitationLearningFunctions as ILF
from scipy.io import loadmat
import numpy as np
from keras.callbacks import EarlyStopping
import time
from matplotlib import pyplot as plt
%matplotlib inline
import DistributionShiftFunctions as DSF
from scipy import integrate
import pickle

# ============================================================================
# Load Data
# ============================================================================

traj_plotting = True


'''Load Initial ANN'''
print("Loading controller")

# filename = 'E:/Research_Data/3DoF_RigidBody/Code_coupled_constmass/NetworkTraining/ANN2_703_relu_n100.h5'
# filename =  open('RBF_save.pkl', 'rb') 
# filename =  open('linear_save.pkl', 'rb') 
# filename =  open('nearest_save.pkl', 'rb') 
filename =  open('nearest_save.pkl', 'rb') 
interp_list = pickle.load(filename)
interp_Fx = interp_list[0]
interp_Fy = interp_list[1]
ctrlsOCLfull = interp_list[2]
trajsOCLfull = interp_list[3]
times_full = interp_list[4]

print("Controller Loaded")



# ============================================================================
# Beta Iteration Loop
# ============================================================================
num_betas = 10;
num_trajs = 100;
num_states = 6;

Pdata = np.zeros([num_trajs*100,num_states])
Qbetadata = np.zeros([num_betas,num_trajs*100,num_states])
D_KL_beta = np.zeros([num_betas])

# Set up Pdata array
count = 0
for i in range(num_trajs):
    for j in range(100):
        Pdata[count,:] = trajsOCLfull[j,:,i]
        count += 1


# Values of Beta to simulate
betas = np.linspace(0,1,num=num_betas)

# Run sims and set up Qbeta Data arrays
for i in range(num_betas):
# for i in range(1):

    print("+===============================")

    beta = betas[i]
    print("Running Trajs for Beta = {}".format(beta))

    count = 0
    
    for j in range(num_trajs):
    # for j in range(1):
        
        if (j % 5 == 0):
            print("Trajectory {} of {}".format(j,num_trajs))
        
        t_OCL = times_full[:,j]
        u_OCL = ctrlsOCLfull[:,:,j]
        x_OCL = trajsOCLfull[:,:,j]
        x0 = x_OCL[0]
        
        # Run sim
        times, states, Fapplied, beta = DSF.RunPolicyWithBetaInterpolator(x0,x_OCL,u_OCL,t_OCL,interp_Fx,interp_Fy,beta)
        
        for k in range(100):
            Qbetadata[i,count,:] = states[k]
            count += 1
        
        # Plotting
        if traj_plotting:
            
            plt.figure(1)
            plt.plot(x_OCL[:,0],x_OCL[:,1])
            plt.plot(states[:,0],states[:,1],'--')
            plt.xlabel('X [m]')
            plt.ylabel('Y [m]')
            plt.title("Trajectory {} | Beta = {}".format(j, beta))
            plt.legend(['OCL','IL to query','IL to ignore'])
            plt.show()
            
        
    # Get kl divergence for current Qbeta data
    D_KL_beta[i] = DSF.CalculateKLDivergenceNormals(Pdata,Qbetadata[i])
    

plt.figure()
plt.plot(betas,D_KL_beta)
plt.xlabel("Beta")
plt.ylabel("D_KL(beta)")


M_pi = integrate.trapz(D_KL_beta, betas)
print("Shift Metric: {}".format(M_pi))


# ============================================================================
# Beta Iteration Loop
# ============================================================================







