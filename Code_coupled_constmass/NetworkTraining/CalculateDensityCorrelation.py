#%% ==========================================================================
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
# %matplotlib inline

#%% ==========================================================================
# Load Files
# ============================================================================
from os import listdir
from os.path import isfile, join

historyfolder = 'E:/Research_Data/3DoF_RigidBody/Code_coupled_constmass/NetworkTraining/SimulationHistories/'

onlyfiles = [f for f in listdir(historyfolder) if isfile(join(historyfolder, f))]

numfiles = len(onlyfiles)

num_k = 12
num_traj = int(numfiles/num_k)

interp_idx = -4

mean_distance_all = np.empty([num_k,num_traj,100])
std_distance_all = np.empty([num_k,num_traj,100])
miss_distance_all = np.empty([num_k,num_traj])
final_mahalanobis_offset_all = np.empty([num_k,num_traj])
k_all = np.empty([num_k,num_traj])
traj_all = np.empty([num_k,num_traj])

interp_list_all = []
for i in range(numfiles):
    with open(historyfolder+onlyfiles[i], 'rb')  as f:   
        interp_list_all.append(pickle.load(f))
        

count = 0;
for i in range(num_k):
    for j in range(num_traj):
        mean_distance_all[i,j,:] = interp_list_all[count][8].reshape(-1)
        std_distance_all[i,j,:] = interp_list_all[count][9].reshape(-1)
        miss_distance_all[i,j] = interp_list_all[count][10].reshape(-1)
        final_mahalanobis_offset_all[i,j] = interp_list_all[count][11].reshape(-1)
        k_all[i,j] = interp_list_all[count][13]
        traj_all[i,j] = interp_list_all[count][14]
        
        count += 1



avg_miss_distance_for_k = np.mean(miss_distance_all, axis=1)
avg_final_mahalanobis_offset_for_k = np.mean(final_mahalanobis_offset_all, axis=1)

plt.figure(1)
plt.subplot(121)
plt.plot(np.arange(1,num_k+1), avg_miss_distance_for_k,'x')
plt.xlabel('K [-]')
plt.ylabel('Miss Distance [m]')

plt.subplot(122)
plt.plot(np.arange(1,num_k+1), avg_final_mahalanobis_offset_for_k,'x')
plt.xlabel('K [-]')
plt.ylabel('Final Mahalanobis Offset [-]')

plt.suptitle('Final Errors')
plt.tight_layout()