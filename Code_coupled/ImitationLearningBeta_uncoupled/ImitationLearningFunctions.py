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
    Fapplied = np.zeros((u_OCL.shape[0],u_OCL.shape[1]-1))
    Fi       = np.zeros((u_OCL.shape[0],u_OCL.shape[1]-1))
    F_ocl    = np.zeros((u_OCL.shape[0],u_OCL.shape[1]-1))
    # print((u_OCL.shape[0],u_OCL.shape[1]-1))
    beta = calculateBeta(i);
    
    r = 1

    for j in range(t_OCL.shape[0]-1):
        
        times = np.vstack((times, t_OCL[j+1]))
        # t = np.linspace(t_OCL[i], t_OCL[i+1])

        controller_input = np.hstack((-states[j,0:6],states[j,6])).reshape(1,-1)
        
        prediction = policy.predict(controller_input).reshape(-1)
        TxTyM = prediction.reshape(-1)
        phi = states[j,2]
        D = np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)],[r,0]])
        
        
        Fi[j,:]  = np.linalg.inv((D.transpose()@D))@D.transpose()@TxTyM
        Fi[j,0] = np.clip(Fi[j,0],-15000,15000)
        Fi[j,1] = np.clip(Fi[j,1],0,15000)
        # print("Fi shape: {}, in shape, {}".format( Fi[i,:].shape, (np.linalg.inv((D.transpose()@D))@D.transpose()@TxTyM).shape))
        F_ocl[j,:] = np.linalg.inv((D.transpose()@D))@D.transpose()@u_OCL[j,:]
        F_input = (beta)*F_ocl[j,:] + (1-beta)*Fi[j,:]
        
        Fapplied[j,:] = F_input

        # Integrate dynamics
        sol = integrate.solve_ivp(fun=lambda t, y: LD.LanderEOM_coupled(t,y,F_input),\
                                      t_span=(times[j],times[j+1]), \
                                      y0=states[j,:]) # Default method: rk45
        
        xsol = sol.y
        
        states = np.vstack((states, xsol[:,xsol.shape[1]-1]))




    
    return times, states, Fapplied, Fi, F_ocl
    

# ============================================================================
def RunPolicyWithBetaOrig(x0,x_OCL,u_orig,t_OCL,policy,i):
    
    states = x0.reshape(1,-1)
    times = np.array(t_OCL[0])
    Fapplied = np.zeros((u_orig.shape[0],u_orig.shape[1]))
    Fi       = np.zeros((u_orig.shape[0],u_orig.shape[1]))
    F_ocl    = np.zeros((u_orig.shape[0],u_orig.shape[1]))
    print((u_orig.shape[0],u_orig.shape[1]-1))
    beta = calculateBeta(i);
    
    r = 1

    for j in range(t_OCL.shape[0]-1):
        
        times = np.vstack((times, t_OCL[j+1]))
        # t = np.linspace(t_OCL[i], t_OCL[i+1])

        controller_input = np.hstack((-states[j,0:6],states[j,6])).reshape(1,-1)
        
        # prediction = policy.predict(controller_input).reshape(-1)
        prediction = np.hstack((policy.predict(controller_input)[0],policy.predict(controller_input)[1])).reshape(-1)
        
        TxTyM = prediction.reshape(-1)
        phi = states[j,2]
        D = np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)],[r,0]])
        
        
        Fi[j,:]  = np.linalg.inv((D.transpose()@D))@D.transpose()@TxTyM
        Fi[j,0] = np.clip(Fi[j,0],-15000,15000)
        Fi[j,1] = np.clip(Fi[j,1],0,15000)
        # print("Fi shape: {}, in shape, {}".format( Fi[i,:].shape, (np.linalg.inv((D.transpose()@D))@D.transpose()@TxTyM).shape))
        # F_ocl[j,:] = np.linalg.inv((D.transpose()@D))@D.transpose()@u_OCL[j,:]
        F_ocl[j,:] = u_orig[j,:]
        
        F_input = (beta)*F_ocl[j,:] + (1-beta)*Fi[j,:]
        Fapplied[j,:] = F_input

        # Integrate dynamics
        sol = integrate.solve_ivp(fun=lambda t, y: LD.LanderEOM_coupled(t,y,F_input),\
                                      t_span=(times[j],times[j+1]), \
                                      y0=states[j,:]) # Default method: rk45
        
        xsol = sol.y
        
        states = np.vstack((states, xsol[:,xsol.shape[1]-1]))




    
    return times, states, Fapplied, Fi, F_ocl
    
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
    matfile = loadmat('ANN2_aggregated_data.mat')
    Xagg = matfile['Xagg']
    tagg = matfile['tagg']
    
    shuffler = np.random.permutation(len(Xagg))
    Xagg_shuffled = Xagg[shuffler]
    tagg_shuffled = tagg[shuffler]
    
    # Re-train
    es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=10)
    policy.fit(Xagg_shuffled, [tagg_shuffled[:,0:2],tagg_shuffled[:,2]], batch_size=100, epochs=10000, validation_split=0.05,callbacks=[es])

    # Evaluating model
    results = policy.evaluate(Xagg,[tagg[:,0:2],tagg[:,2]])
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
    plt.show()
    
    idxs = range(000,300)
    yvis = policy.predict(Xagg[idxs].reshape(-1,7));
    
    
    # plt.figure(3)
    # plt.subplot(211)
    # plt.plot(tagg[idxs,0])
    # plt.plot(yvis[0],'--')
    # plt.title('Test on retrained')
    # plt.xlabel('Index (-)')
    # plt.ylabel('Tx (N)')
    # plt.legend(['ocl','ann'])
    # plt.subplot(212)
    # plt.plot(tagg[idxs,1])
    # plt.plot(yvis[1],'--')
    # plt.xlabel('Index (-)')
    # plt.ylabel('Ty (N)')
    # plt.tight_layout()
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
    # beta = 0.998**(i) if i>0 else 0.999
    beta = 0.998**(i) if i>0 else 1
    print("Beta: {}".format(beta))
    return beta

















