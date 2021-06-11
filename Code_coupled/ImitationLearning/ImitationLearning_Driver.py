# -*- coding: utf-8 -*-
"""
Created on Tue May 25 10:36:05 2021

@author: Omkar
"""

# ============================================================================
# Import Dependencies
# ============================================================================
from keras import models
import ImitationLearningFunctions as ILF
from scipy.io import loadmat
import numpy as np
from keras.callbacks import EarlyStopping
import time
# ============================================================================
# Load Data
# ============================================================================

'''Load Data'''
print("Loading trajectory file")
matfile = loadmat('../NetworkTraining/ANN2_data.mat')


# matfile = loadmat('ANN1_data_notaug.mat')
Xfull = matfile['Xfull_2']
tfull = matfile['tfull_2']
times_full = matfile['times']
X_train = matfile['Xtrain2'].reshape(-1,7)
t_train = matfile['ttrain2']
times_train = matfile['times_train']
X_test = matfile['Xtest2'].reshape(-1,7)
t_test = matfile['ttest2']
times_test = matfile['times_test']

print("Trajectory file loaded")


'''Load Initial ANN'''
print("Loading controller")

filename = '../NetworkTraining/ANN2_703_relu.h5'
policy = models.load_model(filename)

print("Controller Loaded")


# ============================================================================
# Define IL Parameters
# ============================================================================

numTrajsInFile = Xfull.shape[0]/100;
num_episodes = 10
num_trajPerEp = 15


# ============================================================================
# IL Iteration Loop
# ============================================================================


start_time = time.time()
for epis in range(num_episodes):
    
    print("Running Episode " + str(epis+1) + " of " + str(num_episodes))
    
    
    
    '''Setup'''
    idxs = np.random.randint(0,high=numTrajsInFile,size=num_trajPerEp)
    
    for traj in range(num_trajPerEp):
        x_OCL = Xfull[idxs[traj]*100:idxs[traj]*100+100,:]
        x_OCL[:,0:6] = -x_OCL[:,0:6]
        x0 = x_OCL[0,:]
        t_OCL = times_full[idxs[traj]*100:idxs[traj]*100+100,:]
        
        '''Run Policy'''
        times, ICout, idx_dev = ILF.RunPolicyToDeviation(x0, x_OCL, t_OCL, policy)
        if traj == 0:
            ICs = ICout.reshape(-1,7)
        elif idx_dev >= 20:
            ICs = np.vstack((ICs,ICout))
        
    '''Query "Expert" (OpenOCL)'''
    ILF.QueryExpert(ICs)
    
    '''Retrain Using Aggregated Dataset'''
    policy = ILF.RetrainOnAggregatedDataset(policy)



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