# -*- coding: utf-8 -*-
"""
Created on Tue May 25 10:36:05 2021

@author: Omkar
"""

# ============================================================================
# Import Dependencies
# ============================================================================
import ImitationLearningFunctions as ILF
from scipy.io import loadmat
import numpy as np
import time
from matplotlib import pyplot as plt
%matplotlib inline
import pickle

import sys
sys.path.insert(0,'../NetworkTraining')
from KNNregressor import KNNReg

# ============================================================================
# Load Data
# ============================================================================

'''Load Data'''
# print("Loading trajectory file")
# matfile = loadmat('../NetworkTraining/ANN2_data.mat')


# # matfile = loadmat('ANN1_data_notaug.mat')
# Xfull = matfile['Xfull_2']
# tfull = matfile['tfull_2']
# times_full = matfile['times']
# X_train = matfile['Xtrain2'].reshape(-1,7)
# t_train = matfile['ttrain2']
# times_train = matfile['times_train']
# X_test = matfile['Xtest2'].reshape(-1,7)
# t_test = matfile['ttest2']
# times_test = matfile['times_test']

# print("Trajectory file loaded")


'''Load Initial Policy'''
print("Loading controller")
load_folder = 'E:/Research_Data/3DoF_RigidBody/Code_coupled_constmass/NetworkTraining/SimulationHistories/'
load_filename = 'simulation_data_k9_traj1.pkl'

with open(load_folder+load_filename, 'rb')  as f:   
    interp_list = pickle.load(f)
    
knn_interp = interp_list[-4]

print("Controller Loaded")

Xfull = knn_interp.x_test_data
tfull = knn_interp.y_test_data
times_full = knn_interp.t_test_data

# ============================================================================
# Define IL Parameters
# ============================================================================

numTrajsInFile = Xfull.shape[0]/100;
starting_episode = 0
num_episodes = 100
first_to_query = 64
last_to_query = 98 # in each trajectory
num_to_query = 10 # per trajectory
num_trajs_to_query = 10
nx = 6 # size of state

# idx_to_query = list(range(0,last_to_query,int(last_to_query/num_to_query)))
# idx_rem = np.delete(list(range(0,100)),list(idx_to_query))

# ============================================================================
# IL Iteration Loop
# ============================================================================

start_time = time.time()
for epis in range(starting_episode,starting_episode+num_episodes):
    
    idxs = np.random.randint(0, high=numTrajsInFile, size=num_trajs_to_query)
    print('---------------------------')
    print("Running Episode " + str(epis+1) + " of " + str(starting_episode+num_episodes))
    
    # idx_to_query = list(range(0,last_to_query,int(last_to_query/num_to_query)))
    idx_to_query = list(np.linspace(first_to_query,last_to_query,num=num_to_query,dtype=int))
    idx_rem = np.delete(list(range(0,100)),list(idx_to_query))
    
    episICs_to_query = np.empty((0,nx))
    
    for j in range(num_trajs_to_query):
        print("Running Trajectory {} of {}".format(j+1, num_trajs_to_query))
        
        '''Setup'''
        x_OCL = Xfull[idxs[j]*100:idxs[j]*100+100,:]; print("y0 = {}".format(x_OCL[0,1]))
        if (x_OCL[0,1] < 0):
            x_OCL[:,0:nx-1] = -x_OCL[:,0:nx-1]; print("Negated y0 = {}".format(x_OCL[0,1]))
        x0 = x_OCL[0,:]; print("To use y0 = {}".format(x_OCL[0,1]))
        
        u_OCL = tfull[idxs[j]*100:idxs[j]*100+100,:]
        t_OCL = times_full[idxs[j]*100:idxs[j]*100+100]
        
        
     
            
        '''Run Policy'''
        times, ICs, Fapplied, beta = ILF.RunPolicyWithBeta(x0, x_OCL, u_OCL, t_OCL, knn_interp, epis)
        episICs_to_query = np.vstack((episICs_to_query,ICs[idx_to_query]))
        
        if (j == 1):
            plt.figure()
            plt.plot(x_OCL[:,0],x_OCL[:,1])
            plt.plot(ICs[idx_to_query,0],ICs[idx_to_query,1],'.')
            plt.plot(ICs[idx_rem,0],ICs[idx_rem,1],'.')
            plt.xlabel('X [m]')
            plt.ylabel('Y [m]')
            plt.title("Trajectory for Episode {} | Beta = {}".format(epis, beta))
            plt.legend(['OCL','IL to query','IL to ignore'])
            plt.show()
            
            
            
            plt.figure()
            plt.subplot(211)
            plt.plot(times,u_OCL[:,0])
            plt.plot(times,Fapplied[:,0],'--')
            plt.ylabel('Fx [N]')
            plt.legend(['OCL','F_applied'])
            plt.title("Thrust Profiles for Episode {} | Beta = {}".format(epis, beta))
            plt.subplot(212)
            plt.plot(times,u_OCL[:,1])
            plt.plot(times,Fapplied[:,1],'--')
            plt.xlabel('Time [s]')
            plt.ylabel('Fy [N]')
            plt.tight_layout()
            plt.show()
    
    
    plt.figure()
    plt.plot(episICs_to_query[:,0],episICs_to_query[:,1],'.')
    plt.title('before deleting')
    plt.show()
    
    episICs_to_query = np.delete(episICs_to_query,np.argwhere(episICs_to_query[:,1]<0),axis=0) # delete negative ICs
    
    # if x0[1] < 0:
    #     x_OCL = Xfull[idxs[j-1]*100:idxs[j-1]*100+100,:]
    #     x_OCL[:,0:nx-1] = -x_OCL[:,0:nx-1]
    #     x0 = x_OCL[0,:]
        
    #     u_OCL = tfull[idxs[j-1]*100:idxs[j-1]*100+100,:]
    #     t_OCL = times_full[idxs[j-1]*100:idxs[j-1]*100+100,:]
    
    plt.figure()
    plt.plot(episICs_to_query[:,0],episICs_to_query[:,1],'.')
    plt.title('after deleting')
    plt.show()
        
        # idx_to_remove = []
        # for i in range(num_to_query):
        #     if ICs[idx_to_query[i],1] <= 0:
        #         idx_to_remove.append(i)
        # idx_to_query = np.delete(idx_to_query,idx_to_remove)
    
    
    '''Query "Expert" (OpenOCL)'''
    # ILF.QueryExpert(ICs[idx_to_query])
    ILF.QueryExpert(episICs_to_query)
    
    '''Retrain Using Aggregated Dataset'''
    knn_interp = ILF.RetrainOnAggregatedDataset(knn_interp)



end_time = time.time()
run_time = end_time - start_time

# ============================================================================
# Post-training Housekeeping
# ============================================================================




# ============================================================================
# Performance Evaluation
# ============================================================================





# ============================================================================
# Plotting
# ============================================================================