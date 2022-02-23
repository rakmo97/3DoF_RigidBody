# -*- coding: utf-8 -*-
"""
Created on Tue May 25 10:49:13 2021

@author: Omkar
"""

# ============================================================================
# Import Dependencies
# ============================================================================

from scipy import integrate
import LanderDynamics as LD
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import GridSearchCV
# from keras.models import Sequential
# from keras import layers
# from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np
# from keras.metrics import RootMeanSquaredError
# from sklearn.preprocessing import MinMaxScaler
# from keras.optimizers import Adam
# from keras.optimizers import Nadam
# from keras.optimizers import sgd
from keras.callbacks import EarlyStopping
import matlab.engine
import sys
sys.path.insert(0,'../NetworkTraining')
from KNNregressor import KNNReg



# ============================================================================
# Define Functions
# ============================================================================

# ============================================================================
def RunPolicyToDeviation(x0,x_OCL,t_OCL,policy):
    """Function Description"""    
   
    onTrack = True
    states = x0.reshape(1,-1)
    times = np.array(t_OCL[0])
    i = 0
    
    
    
    while onTrack:
        
        
        times = np.vstack((times, t_OCL[i+1]))
        # t = np.linspace(t_OCL[i], t_OCL[i+1])

        controller_input = np.hstack((-states[i,0:6],states[i,6])).reshape(1,-1)
        
        Fi = policy.predict(controller_input).reshape(-1)

        # Integrate dynamics
        sol = integrate.solve_ivp(fun=lambda t, y: LD.LanderEOM(t,y,Fi),\
                                      t_span=(times[i],times[i+1]), \
                                      y0=states[i,:]) # Default method: rk45
        xsol = sol.y
        # tsol = sol.t
        
        states = np.vstack((states, xsol[:,xsol.shape[1]-1]))
               
        deviatedCondition = checkForDeviation(states, x_OCL, t_OCL)
        
        if deviatedCondition:
            onTrack = False
    
    
        i += 1
    
    print("Made it to time step {} of 100".format(i-1))
    return times[i-2], states[i-2,:], i-1


# ============================================================================
def RunPolicyWithBeta(x0,x_OCL,u_OCL,t_OCL,policy,i):
    
    states = x0.reshape(1,-1)
    times = np.array(t_OCL[0])
    Fapplied = np.zeros(u_OCL.shape)
    
    beta = calculateBeta(i);

    for j in range(t_OCL.shape[0]-1):
        
        times = np.vstack((times, t_OCL[j+1]))
        # t = np.linspace(t_OCL[i], t_OCL[i+1])

        # controller_input = np.hstack((-states[j,0:5],states[j,5])).reshape(1,-1)
        controller_input = states[j,:]
        
        Fi = policy.predict(controller_input, print_density_info=False)[0]
        
        F_input = (beta)*u_OCL[j,:] + (1-beta)*Fi
        Fapplied[j,:] = F_input

        # Integrate dynamics
        sol = integrate.solve_ivp(fun=lambda t, y: LD.LanderEOM(t,y,F_input),\
                                      t_span=(times[j],times[j+1]), \
                                      y0=states[j,:]) # Default method: rk45
        
        xsol = sol.y
        
        states = np.vstack((states, xsol[:,xsol.shape[1]-1]))


    
    return times, states, Fapplied, beta
    
    
# ============================================================================

def QueryExpert(ICs):
    """Function Description"""
    print('Querying Expert for {} trajectories'.format(ICs.shape[0]))
    eng = matlab.engine.start_matlab() # Start Matlab Engine before loop
    ICs_matlab = matlab.double(ICs.tolist())
    proxyOut = eng.QueryExpert(ICs_matlab)  
    eng.exit() # Stop matlab engine

    print("Done Querying Expert")
    return 0    
# ============================================================================
    
def RetrainOnAggregatedDataset(policy):
    """Function Description"""    
    
    # Load data
    matfile = loadmat('E:/Research_Data/3DoF_RigidBody/Code_coupled_constmass/ImitationLearningBeta/ANN2_aggregated_data.mat')
    ctrls_all = matfile['ctrlOut'].reshape(100,2,-1)
    states_all = matfile['stateOut'].reshape(100,7,-1)[:,1:7,:]
    times_all = matfile['stateOut'].reshape(100,7,-1)[:,0,:]
    
    # shuffler = np.random.permutation(ctrls_all.shape[2])
    # ctrls_all = ctrls_all[:,:,shuffler]
    # states_all = states_all[:,:,shuffler]
    # times_all = times_all[:,shuffler]

    train_split = int(policy.xData.shape[0]/100)
    split_add = 40
    train_split += split_add
    train_split = int(train_split)
    
    ctrls_train = ctrls_all[:,:,0:train_split]
    ctrls_test = ctrls_all[:,:,train_split:]
    states_train = states_all[:,:,0:train_split]
    states_test = states_all[:,:,train_split:]
    times_train = times_all[:,0:train_split]
    times_test = times_all[:,train_split:]
    
    ctrls_all_list = np.empty([0,2])
    ctrls_train_list = np.empty([0,2])
    ctrls_test_list = np.empty([0,2])
    states_all_list = np.empty([0,6])
    states_train_list = np.empty([0,6])
    states_test_list = np.empty([0,6])
    times_all_list = np.empty([0,1])
    times_train_list = np.empty([0,1])
    times_test_list = np.empty([0,1])

    ctrls_all_list = np.concatenate(ctrls_all.T,1).T
    states_all_list = np.concatenate(states_all.T,1).T
    times_all_list = np.concatenate(times_all.T,0)
    
    ctrls_train_list = np.concatenate(ctrls_train.T,1).T
    states_train_list = np.concatenate(states_train.T,1).T
    times_train_list = np.concatenate(times_train.T,0)
    
    ctrls_test_list = np.concatenate(ctrls_test.T,1).T
    states_test_list = np.concatenate(states_test.T,1).T
    times_test_list = np.concatenate(times_test.T,0)
    
    knn_interp = KNNReg(states_train_list, ctrls_train_list, k=policy.k,
                        mahalanobis=True, weight_by_distance=True,
                        t_train_data=times_train_list, 
                        x_test_data=states_test_list, y_test_data=ctrls_test_list, t_test_data=times_test_list)
    

    
    return knn_interp
# ============================================================================

def checkForDeviation(states, x_OCL, t_OCL):
    """Function Description"""   
    
    n_soFar = np.shape(states)[0]
    
    deviation = np.linalg.norm(states-x_OCL[0:n_soFar,:])
    
    if abs(deviation) > 1e2 or abs(states[n_soFar-1,2])>np.math.pi/3: # Check if deviation is too large
        return True
    else:
        return False
   
# ============================================================================

def calculateBeta(i):
    beta = 0.9925**(i) if i>0 else 0.9925
    print("Beta: {}".format(beta))
    return beta

















