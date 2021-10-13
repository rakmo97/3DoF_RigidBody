# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 13:42:22 2021

@author: omkar_mulekar
"""

import numpy as np
from scipy import integrate
import LanderDynamics as LD

# ============================================================================
def CalculateKLDivergenceNormals(Pdata,Qdata):
    
    muP = np.mean(Pdata,axis=0)
    covP = np.cov(Pdata.T)
    muQ = np.mean(Qdata,axis=0)
    covQ = np.cov(Qdata.T)
    
    d = Pdata.shape[1]
    
    KLDivergence = 0.5*(np.trace(np.linalg.inv(covQ)@covP) + (muQ-muP).T@np.linalg.inv(covQ)@(muQ-muP) - d + np.log(np.linalg.det(covQ)/np.linalg.det(covP)))
    
    return KLDivergence

# ============================================================================
def RunPolicyWithBeta(x0,x_OCL,u_OCL,t_OCL,policy,beta):
    
    states = x0.reshape(1,-1)
    times = np.array(t_OCL[0])
    Fapplied = np.zeros(u_OCL.shape)
    

    for j in range(t_OCL.shape[0]-1):
        
        times = np.vstack((times, t_OCL[j+1]))
        # t = np.linspace(t_OCL[i], t_OCL[i+1])
        
        n_x = x0.shape[0];

        controller_input = np.hstack((-states[j,0:n_x-1],states[j,n_x-1])).reshape(1,-1)
        
        Fi = policy.predict(controller_input).reshape(-1)
        
        F_input = (beta)*u_OCL[j,:] + (1-beta)*Fi
        Fapplied[j,:] = F_input

        # Integrate dynamics
        sol = integrate.solve_ivp(fun=lambda t, y: LD.LanderEOM(t,y,F_input),\
                                      t_span=(times[j],times[j+1]), \
                                      y0=states[j,:]) # Default method: rk45
        
        xsol = sol.y
        
        states = np.vstack((states, xsol[:,xsol.shape[1]-1]))


    
    return times, states, Fapplied, beta
    