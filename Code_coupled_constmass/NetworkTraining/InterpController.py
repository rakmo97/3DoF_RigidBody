
from scipy.interpolate import Rbf
from scipy.interpolate import griddata

import numpy as np


def ctrl_from_interp(state_in, interp_Fx, interp_Fy):
    
    
    Fx = interp_Fx(state_in[0],state_in[1],state_in[2],
                state_in[3],state_in[4],state_in[5])
    
    Fy = interp_Fy(state_in[0],state_in[1],state_in[2],
                state_in[3],state_in[4],state_in[5])
    
    # Fx = interp_Fx(state_in[0],state_in[1],state_in[2],state_in[3],state_in[4])
    
    # Fy = interp_Fy(state_in[0],state_in[1],state_in[2],state_in[3],state_in[4])
    
    ctrl_out = np.array([Fx,Fy])
    
    return ctrl_out
    
def KNearestInterp(state_in, state_data, ctrl_data, k=1, mahalanobis=True, weight_by_distance=True):
    
    if mahalanobis:
        S = np.cov(state_data.T)
        distances = np.empty(state_data.shape[0])
        for i in range(state_data.shape[0]):
            distances[i] = np.sqrt((state_in-state_data[i]).T@np.linalg.inv(S)@(state_in-state_data[i]))
    
    else:
        distances = np.linalg.norm(state_data - state_in,axis=1)
        
        
    if k==1:
        low_i = np.argmin(distances)   
        ctrl_out = ctrl_data[low_i]
    else:
        kSmallest_i = sorted(range(len(distances)), key = lambda sub: distances[sub])[:k]
        kSmallest = ctrl_data[kSmallest_i]
        
        if weight_by_distance:
            weights_for_avg = 1.0/distances[kSmallest_i]
            ctrl_out = np.average(kSmallest,axis=0, weights=weights_for_avg)
            
        else:
            ctrl_out = np.mean(kSmallest,axis=0)
    
    return ctrl_out


