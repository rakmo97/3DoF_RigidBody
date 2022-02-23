# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 09:51:05 2021

@author: omkar_mulekar

Runs "Peeled Back" closed loop ANN controller and simulates

"""


#%% ============================================================================
# Import Dependencies
# ============================================================================
# from keras import models
from scipy.io import loadmat
import numpy as np
from scipy import integrate
import math
import time
from scipy.interpolate import Rbf
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
from KNNregressor import KNNReg
import LanderDynamics as LD
import InterpController as IC
from sklearn.model_selection import train_test_split
import pickle

from matplotlib import pyplot as plt
%matplotlib inline



#%% ============================================================================
# Load Network
# ============================================================================
# filename = 'ANN2_703_tanh.h5'
# filename = 'ANN2_703_relu.h5'
# filename = 'E:/Research_Data/3DoF_RigidBody/Code_coupled_constmass/NetworkTraining/ANN2_703_relu_n100.h5'
# filename = 'E:/Research_Data/3DoF_RigidBody/Code_coupled_constmass/NetworkTraining/ANN2_703_relu_n750.h5'
# ANN2 = models.load_model(filename)



#%% ============================================================================
# Load Data
# ============================================================================
tic = time.perf_counter()
print("Loading trajectory mat file")
# matfile = loadmat('E:/Research_Data/3DoF_RigidBody/Code_coupled_constmass/TrajectoryGeneration/ToOrigin_Trajectories/d20211214_11o55_genTrajs.mat')
# matfile = loadmat('E:/Research_Data/3DoF_RigidBody/Code_coupled_constmass/TrajectoryGeneration/ToOrigin_Trajectories/d20220112_22o27_genTrajs.mat')
matfile = loadmat('E:/Research_Data/3DoF_RigidBody/Code_coupled_constmass/TrajectoryGeneration/ToOrigin_Trajectories/d20220125_11o31_genTrajs.mat')
# matfile = loadmat('E:/Research_Data/3DoF_RigidBody/Code_coupled_constmass/TrajectoryGeneration/consolidatedPreformat.mat')
print("Mat file loaded!")


num_to_pull = 2000
ctrls_all = matfile['ctrlOut'].reshape(100,2,-1)[:,:,0:num_to_pull]
states_all = matfile['stateOut'].reshape(100,7,-1)[:,1:7,0:num_to_pull]
times_all = matfile['stateOut'].reshape(100,7,-1)[:,0,0:num_to_pull]

train_split = 1500
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

print('Creating list of s-a pairs for interpolation...')

ctrls_all_list = np.concatenate(ctrls_all.T,1).T
states_all_list = np.concatenate(states_all.T,1).T
times_all_list = np.concatenate(times_all.T,0)

# for i in range(ctrls_all.shape[2]):
#     for j in range(100):
#         ctrls_all_list = np.vstack((ctrls_all_list, ctrls_all[j,:,i]))
#         states_all_list = np.vstack((states_all_list, states_all[j,:,i]))
#         times_all_list = np.vstack((times_all_list, times_all[j,i]))

ctrls_train_list = np.concatenate(ctrls_train.T,1).T
states_train_list = np.concatenate(states_train.T,1).T
times_train_list = np.concatenate(times_train.T,0)

# for i in range(ctrls_train.shape[2]):
#     for j in range(100):
#         ctrls_train_list = np.vstack((ctrls_train_list, ctrls_train[j,:,i]))
#         states_train_list = np.vstack((states_train_list, states_train[j,:,i]))
#         times_train_list = np.vstack((times_train_list, times_train[j,i]))

ctrls_test_list = np.concatenate(ctrls_test.T,1).T
states_test_list = np.concatenate(states_test.T,1).T
times_test_list = np.concatenate(times_test.T,0)

# for i in range(ctrls_test.shape[2]):
#     for j in range(100):
#         ctrls_test_list = np.vstack((ctrls_test_list, ctrls_test[j,:,i]))
#         states_test_list = np.vstack((states_test_list, states_test[j,:,i]))
#         times_test_list = np.vstack((times_test_list, times_test[j,i]))

print('List created!')
runtime_loaddata = time.perf_counter() - tic
num_trajs = 15
upper_k = 12

for traj in range(0,num_trajs):
    for k in range(1,upper_k+1):
        print("\nRunning for K = {} of {}".format(k,upper_k))
        print("Running for traj {} of {}".format(traj+1,num_trajs))
        #%% ============================================================================
        # Create Interpolators
        # ============================================================================
        tic = time.perf_counter()
        print('Creating interpolators from training set...')   
        # function_rbf = 'linear'
        # function_rbf = 'thin_plate'
        # interp_Fx = Rbf(states_train_list[:,0],states_train_list[:,1],states_train_list[:,2],
        #               states_train_list[:,3],states_train_list[:,4],states_train_list[:,5],
        #               ctrls_train_list[:,0],
        #               function=function_rbf)
        # interp_Fy = Rbf(states_train_list[:,0],states_train_list[:,1],states_train_list[:,2],
        #               states_train_list[:,3],states_train_list[:,4],states_train_list[:,5],
        #               ctrls_train_list[:,1],
        #               function=function_rbf)
        # interp_Fx = LinearNDInterpolator(list(zip(states_train_list[:,0],states_train_list[:,1],states_train_list[:,2],states_train_list[:,3],states_train_list[:,4])),
        #                                   ctrls_train_list[:,0])
        # interp_Fy = LinearNDInterpolator(list(zip(states_train_list[:,0],states_train_list[:,1],states_train_list[:,2],states_train_list[:,3],states_train_list[:,4])),
        #                                   ctrls_train_list[:,1])   
        # interp_Fx = NearestNDInterpolator(list(zip(states_train_list[:,0],states_train_list[:,1],states_train_list[:,2],states_train_list[:,3],states_train_list[:,4],states_train_list[:,5])),
        #                                   ctrls_train_list[:,0])
        # interp_Fy = NearestNDInterpolator(list(zip(states_train_list[:,0],states_train_list[:,1],states_train_list[:,2],states_train_list[:,3],states_train_list[:,4],states_train_list[:,5])),
                                           # ctrls_train_list[:,1])   
        # interp_Fx = griddata(states_train_list, ctr)
        
        # knn_interp = KNNReg(states_train_list, ctrls_train_list, k=k, mahalanobis=True, weight_by_distance=True, x_test_data=states_test, y_test_data=ctrls_test)
        knn_interp = KNNReg(states_train_list, ctrls_train_list, k=k,
                            mahalanobis=True, weight_by_distance=True,
                            t_train_data=times_train_list, 
                            x_test_data=states_test_list, y_test_data=ctrls_test_list, t_test_data=times_test_list)
        
        print('Interpolators created!')
        runtime_createinterp = time.perf_counter() - tic
        
        #%% ============================================================================
        # Simulation Setup/Parameters
        # ============================================================================
        tic = time.perf_counter()
        
        trajToRun = traj  # Indexing variable
        
        ctrlProfile = ctrls_test[:,:,trajToRun]
        trajFromOCL = states_test[:,:,trajToRun]
        timesOCL = times_test[:,trajToRun]
        
        # times = np.linspace(0,60,500)
        times = timesOCL
        end = len(times) # Indexing variable
        
        
        # x0 = trajFromOCL[0,:]
        # target = trajFromOCL[100-1,:]
        x0 = states_test[0,:,trajToRun]
        target = states_test[100-1,:,trajToRun]
        
        
        n_times   = end
        nState    =   6
        nCtrl     =   2
        
        
        
        t = np.zeros(times.size)
        x = np.zeros((n_times,nState))
        Fi = np.zeros((n_times,nCtrl))
        mean_distance = np.zeros((n_times,1))
        std_distance = np.zeros((n_times,1))
        
        
        # ============================================================================
        # Run Simulation
        # ============================================================================
        # x0[2] = 0;
        # x0[3] = 0;
        # x0[4] = 0;
        # x0[5] = 0;
        x[0,:] = x0
        
        
        error = target - x[0,:] # Error
        controller_input = np.append(error[0:5],x[0,5])
        
        # Fi[0,:]= ANN2.predict(np.reshape(controller_input,(1,-1)))
        Fi[0,:] = ctrlProfile[0,:]
        # Fi[0,:] = np.array([20,0])
        count = 1
        print('Running sim...')
        for i in range(1,end):
            # print(i)
            
            # if i == 30:
            #     Fi[i-1,:] += 30
                
        
            
        
            t = np.linspace(times[i-1],times[i])
            # t = np.linspace(timesOCL[i-1],timesOCL[i])
        
                
                # Integrate dynamics
            sol = integrate.solve_ivp(fun=lambda t, y: LD.LanderEOM(t,y,Fi[i-1,:]),\
                                           t_span=(times[i-1],times[i]), \
                                           y0=x[i-1,:],\
                                           t_eval=t.reshape(-1)) # Default method: rk45
            
        
            xsol = sol.y
            tsol = sol.t
                        
            
            x[i,:] = xsol[:,xsol.shape[1]-1]
            
            error = target - x[i,:] # Error
            controller_input = error[0:6]
            
            # Fi[i,:]= ANN2.predict(np.reshape(controller_input,(1,-1)))
            # Fi[i,:]= ctrlProfile[i,:]
            # Fi[i,:]= np.array([20,0])
            # Fi[i,:] = IC.ctrl_from_interp(x[i,:], interp_Fx, interp_Fy)
            # Fipred = IC.ctrl_from_interp(x[i,:], interp_Fx, interp_Fy)
            # Fi[i,0] = Fipred[0] if not math.isnan(Fipred[0]) else 0.0
            # Fi[i,1] = Fipred[1] if not math.isnan(Fipred[1]) else 0.0
            # Fi[i,:] = IC.KNearestInterp(x[i,:], states_train_list, ctrls_train_list, k=2, mahalanobis=True, weight_by_distance=True)
            Fi[i,:], mean_distance[i], std_distance[i] = knn_interp.predict(x[i,:], print_density_info=False)
            
        #END i LOOP
            
        print('Sim complete!')
        runtime_sim = time.perf_counter() - tic
        
        # ============================================================================
        # Post-process
        # ============================================================================
        cost_ocl = LD.CalculateCost(timesOCL, ctrlProfile)
        cost_policy = LD.CalculateCost(times, Fi)
        
        miss_distance = np.linalg.norm(np.zeros(trajFromOCL[-1,:3].shape) - x[-1,:3])
        S = np.cov(states_train_list.T)
        final_mahalanobis_offset = (np.zeros(trajFromOCL[-1,:].shape) - x[-1,:]).T @ np.linalg.inv(S) @ (np.zeros(trajFromOCL[-1,:].shape) - x[-1,:])
        
        # Save off data
        saveoff_folder = 'E:/Research_Data/3DoF_RigidBody/Code_coupled_constmass/NetworkTraining/SimulationHistories/'
        saveoff_filename = 'simulation_data_k'+str(k)+'_traj'+str(traj)+'.pkl'
        saveoff_list = [timesOCL, trajFromOCL, ctrlProfile, times, x, Fi,
                        cost_ocl, cost_policy, 
                        mean_distance, std_distance,
                        miss_distance, final_mahalanobis_offset,
                        knn_interp, k, traj,
                        ['timesOCL', 'trajFromOCL', 'ctrlProfile', 'times', 'x', 'Fi',
                        'cost_ocl', 'cost_policy', 
                        'mean_distance', 'std_distance',
                        'miss_distance', 'final_mahalanobis_offset',
                        'knn_interp','k','traj']]
        
        with open(saveoff_folder+saveoff_filename, 'wb') as f:
            pickle.dump(saveoff_list, f)
            
        # ============================================================================
        # Plotting
        # ============================================================================
        print("Plotting")
        plt.close('all')
        
        plt.figure(1)
        plt.plot(trajFromOCL[:,0],trajFromOCL[:,1])
        plt.plot(x[:,0],x[:,1],'--')
        plt.title('Trajectory {} | K = {}'.format(trajToRun, knn_interp.k))
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.legend(['OpenOCL','ANN'],loc='best')
        plt.show()
        
        plt.figure(2)
        plt.subplot(221)
        plt.plot(timesOCL,trajFromOCL[:,0])
        plt.plot(times,x[:,0],'--')
        plt.ylabel('x [m]')
        plt.legend(['OpenOCL','ANN'],loc='best')
        
        plt.subplot(222)
        plt.plot(timesOCL,trajFromOCL[:,1])
        plt.plot(times,x[:,1],'--')
        plt.ylabel('y [m]')
        
        plt.subplot(223)
        plt.plot(timesOCL,trajFromOCL[:,2]*180.0/np.pi)
        plt.plot(times,x[:,2]*180.0/np.pi,'--')
        plt.xlabel('Time [s]')
        plt.ylabel('phi [deg]')
        
        plt.suptitle('Trajectory {} | K = {}'.format(trajToRun, knn_interp.k),y=1.05)
        plt.tight_layout()
        plt.show()
        
        
        
        plt.figure(3)
        plt.subplot(221)
        plt.plot(timesOCL,trajFromOCL[:,3])
        plt.plot(times,x[:,3],'--')
        plt.ylabel('x-dot [m/s]')
        plt.legend(['OpenOCL','ANN'],loc='best')
        
        plt.subplot(222)
        plt.plot(timesOCL,trajFromOCL[:,4])
        plt.plot(times,x[:,4],'--')
        plt.ylabel('y-dot [m/s]')
        
        plt.subplot(223)
        plt.plot(timesOCL,trajFromOCL[:,5]*180.0/np.pi)
        plt.plot(times,x[:,5]*180.0/np.pi,'--')
        plt.xlabel('Time [s]')
        plt.ylabel('phi-dot [deg/s]')
        
        plt.suptitle('Trajectory {} | K = {}'.format(trajToRun, knn_interp.k),y=1.05)
        plt.tight_layout()
        plt.show()
        
        
        
        plt.figure(4)
        plt.subplot(211)
        plt.plot(timesOCL,ctrlProfile[:,0])
        plt.plot(times,Fi[:,0],'--')
        plt.ylabel('Fx [N]')
        plt.legend(['OpenOCL','ANN'],loc='best')
        
        plt.subplot(212)
        plt.plot(timesOCL,ctrlProfile[:,1])
        plt.plot(times,Fi[:,1],'--')
        plt.ylabel('Fy [N]')
        plt.xlabel('Time [s]')
        
        plt.suptitle('Ctrl Profile | Trajectory  {} | K = {}'.format(trajToRun, knn_interp.k))
        plt.show()
        
        
        plt.figure(5)
        plt.subplot(211)
        plt.plot(times,mean_distance)
        plt.ylabel('Mean Mah. Distance [-]')
        
        plt.subplot(212)
        plt.plot(times,std_distance)
        plt.ylabel('Std Mah. Distance [-]')
        plt.xlabel('Time [s]')
        
        plt.suptitle('Distance Stats | Trajectory  {} | K = {}'.format(trajToRun, knn_interp.k))
        plt.tight_layout()
        plt.show()
        
        # Run Summary
        print('================================================')
        print('               RUN SUMMARY                 ')
        print('\tTrajectory {} of 7'.format(traj))
        print('\tK = {} of 8'.format(k))
        print('\tRuntime (load/format data):\t{:.4f} s'.format(runtime_loaddata))
        print('\tRuntime (create interp):\t{:.4f} s'.format(runtime_createinterp))
        print('\tRuntime (simulation):\t\t{:.4f} s'.format(runtime_sim))
        print('\tSim Time:\t\t\t\t\t{:.4f} s'.format(times[-1]))
        print('\n\tOpenOCL Cost:\t{:.4f}'.format(cost_ocl))
        print('\tPolicy Cost:\t{:.4f}'.format(cost_policy))
        print('\n\tMiss Distance:\t\t\t\t{:.4f} m'.format(miss_distance))
        print('\tFinal Mahalnobis Offset:\t{:.4f}'.format(final_mahalanobis_offset))
        np.set_printoptions(precision=3)
        print('\nFinal State: {}'.format(x[-1]))
        print('================================================')

        