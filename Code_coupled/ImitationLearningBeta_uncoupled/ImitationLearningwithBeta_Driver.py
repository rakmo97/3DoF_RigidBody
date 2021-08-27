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
from matplotlib import pyplot as plt

# ============================================================================
# Load Data
# ============================================================================

'''Load Data'''


print("Loading trajectory file")
matfile = loadmat('../NetworkTraining_uncoupled/ANN2_data.mat')


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
torig_full = matfile['t_orig']

print("Trajectory file loaded")


'''Load Initial ANN'''
print("Loading controller")

# filename = '../NetworkTraining_uncoupled/ANN2_703_relu_n75.h5'
filename = '../NetworkTraining_uncoupled/ANN2_split_703_relu_n25_75_2000.h5'
policy = models.load_model(filename)

print("Controller Loaded")


# ============================================================================
# Define IL Parameters
# ============================================================================

numTrajsInFile = Xfull.shape[0]/100;
num_episodes = 10
last_to_query = 80
num_to_query = 20

# idx_to_query = list(range(0,last_to_query,int(last_to_query/num_to_query)))
# idx_rem = np.delete(list(range(0,100)),list(idx_to_query))

# ============================================================================
# IL Iteration Loop
# ============================================================================

idxs = np.random.randint(0, high=numTrajsInFile, size=num_episodes)

start_time = time.time()
for epis in range(num_episodes):
    
    print("Running Episode " + str(epis+1) + " of " + str(num_episodes))
    
    idx_to_query = list(range(0,last_to_query,int(last_to_query/num_to_query)))
    idx_rem = np.delete(list(range(0,100)),list(idx_to_query))
    
    '''Setup'''
    x_OCL = Xfull[idxs[epis]*100:idxs[epis]*100+100,:]
    x_OCL[:,0:6] = -x_OCL[:,0:6]
    x0 = x_OCL[0,:]
    
    # u_OCL = tfull[idxs[epis]*100:idxs[epis]*100+100,:]
    u_orig = torig_full[idxs[epis]*100:idxs[epis]*100+100,:]
    t_OCL = times_full[idxs[epis]*100:idxs[epis]*100+100,:]
        
    '''Run Policy'''
    # times, ICs, Fapplied, Fpolicy, Focl = ILF.RunPolicyWithBeta(x0, x_OCL, u_OCL, t_OCL, policy, epis)
    times, ICs, Fapplied, Fpolicy, Focl = ILF.RunPolicyWithBetaOrig(x0, x_OCL, u_orig, t_OCL, policy, epis)
    
    
    plt.figure()
    plt.plot(x_OCL[:,0],x_OCL[:,1])
    plt.plot(ICs[idx_to_query,0],ICs[idx_to_query,1],'x')
    plt.plot(ICs[idx_rem,0],ICs[idx_rem,1],'x')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title("Trajectory for Episode {}".format(epis))
    plt.legend(['OCL','IL to query','IL to ignore'])
    plt.show()
    
    
    
    plt.figure()
    plt.subplot(211)
    plt.plot(times,Focl[:,0])
    plt.plot(times,Fapplied[:,0],'--')
    plt.plot(times,Fpolicy[:,0],'--')
    plt.ylabel('Fx [N]')
    plt.legend(['OCL','Applied','Policy'])
    plt.title("Thrust Profiles for Episode {}".format(epis))
    plt.subplot(212)
    plt.plot(times,Focl[:,1])
    plt.plot(times,Fapplied[:,1],'--')
    plt.plot(times,Fpolicy[:,1],'--')
    plt.xlabel('Time [s]')
    plt.ylabel('Fy [N]')
    plt.tight_layout()
    plt.show()
    
    idx_to_remove = []
    for i in range(num_to_query):
        if ICs[idx_to_query[i],1] <= 0:
            idx_to_remove.append(i)
    idx_to_query = np.delete(idx_to_query,idx_to_remove)
    
    
    '''Query "Expert" (OpenOCL)'''
    ILF.QueryExpert(ICs[idx_to_query])
    
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