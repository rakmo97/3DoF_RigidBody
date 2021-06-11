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
    
    beta = calculateBeta(i);

    for j in range(t_OCL.shape[0]):
        
        times = np.vstack((times, t_OCL[j+1]))
        # t = np.linspace(t_OCL[i], t_OCL[i+1])

        controller_input = np.hstack((-states[i,0:6],states[i,6])).reshape(1,-1)
        
        Fi = policy.predict(controller_input).reshape(-1)
        
        F_input = (beta)*u_OCL[j,:] + (1-beta)*Fi
        

        # Integrate dynamics
        sol = integrate.solve_ivp(fun=lambda t, y: LD.LanderEOM(t,y,F_input),\
                                      t_span=(times[j],times[j+1]), \
                                      y0=states[j,:]) # Default method: rk45
        
        xsol = sol.y
        states = np.vstack((states, xsol[:,xsol.shape[1]-1]))


    
    return times, states
    
    
# ============================================================================

def QueryExpert(ICs):
    """Function Description"""
    print('Querying Expert for {} trajecoties'.format(ICs.shape[0]))
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
    matfile = loadmat('ANN2_aggregated_data.mat')
    Xagg = matfile['Xagg']
    tagg = matfile['tagg']
    
    shuffler = np.random.permutation(len(Xagg))
    Xagg_shuffled = Xagg[shuffler]
    tagg_shuffled = tagg[shuffler]
    
    # Re-train
    es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=10)
    policy.fit(Xagg_shuffled, tagg_shuffled, batch_size=100, epochs=10000, validation_split=0.05,callbacks=[es])

    # Evaluating model
    results = policy.evaluate(Xagg,tagg)
    print("Test Loss: ", results[0])
    print("Test Accuracy: ", results[1])
    
    plt.figure(1)
    plt.plot(policy.history.history['loss'])
    plt.plot(policy.history.history['val_loss'])
    plt.legend(['train', 'validation'], loc='best')
    plt.title('Loss')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Cost (MSE)')
    # plt.show()
    
    return policy
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
    beta = 0.45**i
    return beta

















