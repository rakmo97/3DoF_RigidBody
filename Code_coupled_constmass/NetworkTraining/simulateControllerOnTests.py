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
# matfile = loadmat('E:/Research_Data/3DoF_RigidBody/Code_coupled_constmass/TrajectoryGeneration/TestTrajectories/d20220211_13o04_genTrajs.mat')
matfile = loadmat('E:/Research_Data/3DoF_RigidBody/Code_coupled_constmass/TrajectoryGeneration/TestTrajectories/d20220215_09o04_genTrajs.mat')
# matfile = loadmat('E:/Research_Data/3DoF_RigidBody/Code_coupled_constmass/TrajectoryGeneration/consolidatedPreformat.mat')
print("Mat file loaded!")


num_to_pull = 20
ctrls_all = matfile['ctrlOut'].reshape(100,2,-1)[:,:,0:num_to_pull]
states_all = matfile['stateOut'].reshape(100,7,-1)[:,1:7,0:num_to_pull]
times_all = matfile['stateOut'].reshape(100,7,-1)[:,0,0:num_to_pull]
traj = 0

ctrls_all_list = np.concatenate(ctrls_all.T,1).T
states_all_list = np.concatenate(states_all.T,1).T
times_all_list = np.concatenate(times_all.T,0)

#%% ============================================================================
# Load Policy
# ============================================================================
tic = time.perf_counter()
print('Loading Policy')
filename = 'E:/Research_Data/3DoF_RigidBody/Code_coupled_constmass/ImitationLearningBeta/Dagger_k9_100epis.pkl'
# filename = 'E:/Research_Data/3DoF_RigidBody/Code_coupled_constmass/ImitationLearningBeta/Dagger_k9_199epis.pkl'
with open(filename,'rb') as f:
    pickle_in = pickle.load(f)
    
knn_interp = pickle_in[0]
# knn_interp.k = 2
       



t0 = 0
tf = 60
nt = 200

useOCLtimes = True
runningSingle = False

if runningSingle:
    singletorun = 2
    trajs_to_run = range(singletorun,singletorun+1)
    ctrls_all = ctrls_all[:,:,singletorun][..., np.newaxis]
    states_all = states_all[:,:,singletorun][..., np.newaxis]
    times_all = times_all[:,singletorun][..., np.newaxis]
    runtime_sim = np.empty(1)
    cost_ocl = np.empty(1)
    cost_policy = np.empty(1)
    miss_distance = np.empty(1)
    final_mahalanobis_offset = np.empty(1)
else:
    trajs_to_run = range(num_to_pull)
    # Pre-allocate
    runtime_sim = np.empty(num_to_pull)
    cost_ocl = np.empty(num_to_pull)
    cost_policy = np.empty(num_to_pull)
    miss_distance = np.empty(num_to_pull)
    final_mahalanobis_offset = np.empty(num_to_pull)

numrunning = len(trajs_to_run)

states_sim = np.empty([100 if useOCLtimes else nt, states_all.shape[1],numrunning])
ctrls_sim = np.empty([100 if useOCLtimes else nt, ctrls_all.shape[1],numrunning])

countouter = 0
#%% ============================================================================
# Simulation Setup/Parameters
# ============================================================================
for traj in trajs_to_run:
    tic = time.perf_counter()
    
    trajToRun = traj  # Indexing variable
        
        

    
    times = np.linspace(t0,tf,nt)
    end = len(times) # Indexing variable
    
    n_times   = end
    nState    =   6
    nCtrl     =   2
    
    t = np.zeros(times.size)
    x = np.zeros((n_times,nState))
    Fi = np.zeros((n_times,nCtrl))
    mean_distance = np.zeros((n_times,1))
    std_distance = np.zeros((n_times,1))
    
    
    if runningSingle:
        ctrlProfile = ctrls_all
        trajFromOCL = states_all
        timesOCL = times_all
        x0 = states_all[0,:,0]
        target = states_all[100-1,:]
        Fi[0,:] = ctrlProfile[0,:,0]
    else:
        ctrlProfile = ctrls_all[:,:,trajToRun]
        trajFromOCL = states_all[:,:,trajToRun]
        timesOCL = times_all[:,trajToRun]
        x0 = states_all[0,:,trajToRun]
        target = states_all[100-1,:,trajToRun]
        Fi[0,:] = ctrlProfile[0,:]
        

    
    
    times = timesOCL if useOCLtimes else np.linspace(t0,tf,nt)
    end = len(times) # Indexing variable
    
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
    
    # Fi[0,:] = np.array([20,0])
    count = 1
    print('Running sim {} of {}'.format(traj, numrunning))
    for i in range(1,end):         
        
    
        t = np.linspace(times[i-1],times[i])
    
            
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
        # Fi[i,:] += 10*np.random.randn(2)
        # if i == 30:
        #     Fi[i,0] += 30
        
    #END i LOOP
    
    
    states_sim[:,:,countouter] = x
    ctrls_sim[:,:,countouter] = Fi
    
    print('Sim complete!')
    runtime_sim[countouter] = time.perf_counter() - tic
    
    # ============================================================================
    # Post-process
    # ============================================================================
    cost_ocl[countouter] = LD.CalculateCost(timesOCL[:,countouter], ctrlProfile[:,:,countouter])
    cost_policy[countouter] = LD.CalculateCost(times[:,countouter], Fi)
    
    miss_distance[countouter] = np.linalg.norm(np.zeros(trajFromOCL[-1,:3].shape) - x[-1,:3])
    S = np.cov(states_all_list.T)
    final_mahalanobis_offset[countouter] = (np.zeros(x[-1,:].shape) - x[-1,:]).T @ np.linalg.inv(S) @ (np.zeros(x[-1,:].shape) - x[-1,:])
    # Run Summary
    print('================================================')
    print('               TRAJ SUMMARY                 ')
    print('\tTrajectory {} of {}'.format(traj,num_to_pull))
    print('\tRuntime (simulation):\t\t{:.4f} s'.format(runtime_sim[countouter]))
    print('\tSim Time:\t\t\t\t\t{:.4f} s'.format(times[-1,countouter]))
    print('\n\tOpenOCL Cost:\t{:.4f}'.format(cost_ocl[countouter]))
    print('\tPolicy Cost:\t{:.4f}'.format(cost_policy[countouter]))
    print('\n\tMiss Distance:\t\t\t\t{:.4f} m'.format(miss_distance[countouter]))
    print('\tFinal Mahalnobis Offset:\t{:.4f}'.format(final_mahalanobis_offset[countouter]))
    np.set_printoptions(precision=3)
    print('\nFinal State: {}'.format(x[-1]))
    print('================================================')
    countouter += 1
    # Save off data
    # saveoff_folder = 'E:/Research_Data/3DoF_RigidBody/Code_coupled_constmass/NetworkTraining/SimulationHistories/'
    # saveoff_filename = 'simulation_data_k'+str(k)+'_traj'+str(traj)+'.pkl'
    # saveoff_list = [timesOCL, trajFromOCL, ctrlProfile, times, x, Fi,
    #                 cost_ocl, cost_policy, 
    #                 mean_distance, std_distance,
    #                 miss_distance, final_mahalanobis_offset,
    #                 knn_interp, k, traj,
    #                 ['timesOCL', 'trajFromOCL', 'ctrlProfile', 'times', 'x', 'Fi',
    #                 'cost_ocl', 'cost_policy', 
    #                 'mean_distance', 'std_distance',
    #                 'miss_distance', 'final_mahalanobis_offset',
    #                 'knn_interp','k','traj']]
    
    # with open(saveoff_folder+saveoff_filename, 'wb') as f:
    #     pickle.dump(saveoff_list, f)
    
# ============================================================================
# MC Post Processing
# ============================================================================
runtime_sim_mean = runtime_sim.mean()
cost_ocl_mean = cost_ocl.mean()
cost_policy_mean = cost_policy.mean()
miss_distance_mean = miss_distance.mean()
final_mahalanobis_offset_mean = final_mahalanobis_offset.mean()

# ============================================================================
# MC Summary
# ============================================================================
print('================================================')
print('               MC SUMMARY                 ')
print('\tMean Runtime (simulation):\t\t{:.4f} s'.format(runtime_sim_mean))
print('\n\tMean OpenOCL Cost:\t{:.4f}'.format(cost_ocl_mean))
print('\tMean Policy Cost:\t{:.4f}'.format(cost_policy_mean))
print('\n\tMean Miss Distance:\t\t\t\t{:.4f} m'.format(miss_distance_mean))
print('\tMean Final Mahalnobis Offset:\t{:.4f}'.format(final_mahalanobis_offset_mean))
np.set_printoptions(precision=3)
print('================================================')


# ============================================================================
# Plotting
# ============================================================================
print("Plotting")
plt.close('all')

plt.figure(1)
for i in range(numrunning):
    plt.plot(states_all[:,0,i],states_all[:,1,i],color="blue")
    plt.plot(states_sim[:,0,i],states_sim[:,1,i],'--',color="orange")
plt.title('K = {}'.format(knn_interp.k))
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(['OpenOCL','Policy'],loc='best')
plt.show()

plt.figure(2)
plt.subplot(221)
for i in range(numrunning):    
    plt.plot(timesOCL,states_all[:,0,i],color="blue")
    plt.plot(times,states_sim[:,0,i],'--',color="orange")
plt.ylabel('x [m]')
plt.legend(['OpenOCL','ANN'],loc='best')
    
plt.subplot(222)
for i in range(numrunning):
    plt.plot(timesOCL,states_all[:,1,i],color="blue")
    plt.plot(times,states_sim[:,1,i],'--',color="orange")
plt.ylabel('y [m]')
    
plt.subplot(223)
for i in range(numrunning):
    plt.plot(timesOCL,states_all[:,2,i]*180.0/np.pi,color="blue")
    plt.plot(times,states_sim[:,2,i]*180.0/np.pi,'--',color="orange")
plt.xlabel('Time [s]')
plt.ylabel('phi [deg]')

plt.title('K = {}'.format(knn_interp.k))
plt.tight_layout()
plt.show()



plt.figure(3)
plt.subplot(221)
for i in range(numrunning):
    plt.plot(timesOCL,states_all[:,3,i],color="blue")
    plt.plot(times,states_sim[:,3,i],'--',color="orange")
plt.ylabel('x-dot [m/s]')
plt.legend(['OpenOCL','ANN'],loc='best')
    
plt.subplot(222)
for i in range(numrunning):
    plt.plot(timesOCL,states_all[:,4,i],color="blue")
    plt.plot(times,states_sim[:,4,i],'--',color="orange")
plt.ylabel('y-dot [m/s]')
    
plt.subplot(223)
for i in range(numrunning):
    plt.plot(timesOCL,states_all[:,5,i]*180.0/np.pi,color="blue")
    plt.plot(times,states_sim[:,5,i]*180.0/np.pi,'--',color="orange")
plt.xlabel('Time [s]')
plt.ylabel('phi-dot [deg/s]')

plt.title('K = {}'.format(knn_interp.k))
plt.tight_layout()
plt.show()



plt.figure(4)

plt.subplot(211)
for i in range(numrunning):
    plt.plot(timesOCL,ctrls_all[:,0,i],color="blue")
    plt.plot(times,ctrls_sim[:,0,i],'--',color="orange")
plt.ylabel('Fx [N]')
plt.legend(['OpenOCL','ANN'],loc='best')
    
plt.subplot(212)
for i in range(numrunning):
    plt.plot(timesOCL,ctrls_all[:,1,i],color="blue")
    plt.plot(times,ctrls_sim[:,1,i],'--',color="orange")
plt.ylabel('Fy [N]')
plt.xlabel('Time [s]')

plt.suptitle('Ctrl Profile |  K = {}'.format(knn_interp.k))
plt.show()


# plt.figure(5)
# plt.subplot(211)
# plt.plot(times,mean_distance)
# plt.ylabel('Mean Mah. Distance [-]')

# plt.subplot(212)
# plt.plot(times,std_distance)
# plt.ylabel('Std Mah. Distance [-]')
# plt.xlabel('Time [s]')

# plt.suptitle('Distance Stats | Trajectory  {} | K = {}'.format(trajToRun, knn_interp.k))
# plt.tight_layout()
# plt.show()




