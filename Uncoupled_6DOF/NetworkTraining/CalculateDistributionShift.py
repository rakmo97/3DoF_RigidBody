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

# ============================================================================
# Load Data
# ============================================================================

traj_plotting = False

'''Load Data'''
print("Loading trajectory file")
matfile = loadmat('E:/Research_Data/3DoF_RigidBody/Pointmass_3DOF/TrajectoryGeneration/ToOrigin_Trajectories/d20211014_16o06_genTrajs.mat')


# matfile = loadmat('ANN1_data_notaug.mat')
Xfull = matfile['stateOut']
ctrlsOCLfull = matfile['ctrlOut']
times_full = Xfull[:,0,:]
trajsOCLfull = Xfull[:,1:14,:]
# X_train = matfile['Xtrain2'].reshape(-1,7)
# t_train = matfile['ttrain2']
# times_train = matfile['times_train']
# X_test = matfile['Xtest2'].reshape(-1,7)
# t_test = matfile['ttest2']
# times_test = matfile['times_test']



print("Trajectory file loaded")


'''Load Initial ANN'''
print("Loading controller")

# filename = '../NetworkTraining/ANN2_703_relu.h5'
filename = 'E:/Research_Data/3DoF_RigidBody/Pointmass_3DOF/NetworkTraining/ANN2_relu_n1000_nA500_nB500.h5'
policy = models.load_model(filename)

print("Controller Loaded")



# ============================================================================
# Beta Iteration Loop
# ============================================================================
num_betas = 10;
num_trajs = 100;
num_states = 13;

Pdata = np.zeros([num_trajs*100,num_states])
Qbetadata = np.zeros([num_betas,num_trajs*100,num_states])
D_KL_beta = np.zeros([num_betas])
D_KL_beta_upperbound = np.zeros([num_betas])

# Set up Pdata array
count = 0
for i in range(num_trajs):
    for j in range(100):
        Pdata[count,:] = Xfull[j,1:num_states+1,i]
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
        times, states, Fapplied, beta = DSF.RunPolicyWithBeta(x0,x_OCL,u_OCL,t_OCL,policy,beta)
        
        for k in range(100):
            Qbetadata[i,count,:] = states[k]
            count += 1
        
        # Plotting
        if traj_plotting:
            
            plt.figure(1)
            ax = plt.axes(projection='3d')
            ax.plot(x_OCL[:,0],x_OCL[:,2],x_OCL[:,1])
            ax.plot(states[:,0],states[:,2],states[:,1])
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Z [m]')
            ax.set_zlabel('Y [m]')
            plt.title("Trajectory {} | Beta = {}".format(j, beta))
            plt.legend(['OCL','IL to query','IL to ignore'])
            plt.show()
            
        
    # Get kl divergence for current Qbeta data
    # D_KL_beta[i] = DSF.CalculateKLDivergenceNormals(Pdata,Qbetadata[i])
    D_KL_beta[i], D_KL_beta_upperbound[i] = DSF.CalculateKLDivergenceGMMs_Variational(Pdata,Qbetadata[i])
    

plt.figure()
plt.plot(betas,D_KL_beta)
plt.xlabel("Beta")
plt.ylabel("D_KL(beta)")


M_pi = integrate.trapz(D_KL_beta, betas)
print("Shift Metric: {}".format(M_pi))


# ============================================================================
# Beta Iteration Loop
# ============================================================================







