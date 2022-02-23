# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 13:42:22 2021

@author: omkar_mulekar
"""
import sys
sys.path.append('../NetworkTraining')
import InterpController as IC
import math

import numpy as np
from scipy import integrate
import LanderDynamics as LD
import InterpController as IC
from sklearn.mixture import GaussianMixture

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
def CalculateKLDivergenceNormals_fromMeanCov(muP, covP, muQ, covQ):
    
    
    d = covP.shape[1]
    
    KLDivergence = 0.5*(np.trace(np.linalg.inv(covQ)@covP) + (muQ-muP).T@np.linalg.inv(covQ)@(muQ-muP) - d + np.log(np.linalg.det(covQ)/np.linalg.det(covP)))
    
    return KLDivergence

# ============================================================================
def CalculateKLDivergenceGMMs_MC(Pdata,Qdata, numComponents=4, MC_samples=1000):
    
    gmP = GaussianMixture(n_components=numComponents, covariance_type='full').fit(Pdata)
    gmQ = GaussianMixture(n_components=numComponents, covariance_type='full').fit(Qdata)
    
    xi = np.random.rand(MC_samples,6)*2000-1000
    
    samplesum = 0
    for i in range(MC_samples):
        if abs(CalculateGMMProbabilityDensity(gmQ,xi[i])) > 1e-6:
            samplesum += np.log(CalculateGMMProbabilityDensity(gmP,xi[i])/CalculateGMMProbabilityDensity(gmQ,xi[i]))
    
        else:
            samplesum += 0
            
            
    d = Pdata.shape[1]
    
    KLDivergence = (1/MC_samples)*samplesum
    
    return KLDivergence

# ============================================================================
def CalculateKLDivergenceGMMs_Variational(Pdata,Qdata, numComponents=4, MC_samples=1000):
    
    gmP = GaussianMixture(n_components=numComponents, covariance_type='full').fit(Pdata)
    gmQ = GaussianMixture(n_components=numComponents, covariance_type='full').fit(Qdata)
       
    KLDivergence_upperbound = 0
    for a in range(numComponents):
        for b in range(numComponents):
            KLDivergence_upperbound += gmP.weights_[a]*gmQ.weights_[b]*CalculateKLDivergenceNormals_fromMeanCov(gmP.means_[a],gmP.covariances_[a],gmQ.means_[b],gmQ.covariances_[b])
    
    KLDivergence = 0
    for a in range(numComponents):
        numerator = 0
        for ap in range(numComponents):
            numerator += gmP.weights_[ap]*np.exp(-CalculateKLDivergenceNormals_fromMeanCov(gmP.means_[a],gmP.covariances_[a],gmP.means_[ap],gmP.covariances_[ap]))
            
        denominator = 0
        for b in range(numComponents):
            denominator += gmQ.weights_[b]*np.exp(-gmQ.weights_[b]*CalculateKLDivergenceNormals_fromMeanCov(gmP.means_[a],gmP.covariances_[a],gmQ.means_[b],gmQ.covariances_[b]))
            
        KLDivergence += gmP.weights_[a]*np.log(numerator/denominator)
        
    return KLDivergence, KLDivergence_upperbound


# ============================================================================
def CalculateGMMProbabilityDensity(gm,Xin):
    
    k = gm.means_[0].shape[0]
    
    probability_density = 0
    for i in range(gm.n_components):
        probability_density += gm.weights_[i]*(np.exp(-0.5*(Xin-gm.means_[i]).T@np.linalg.inv(gm.covariances_[i])@(Xin-gm.means_[i]))/(np.sqrt(((2*np.pi)**k)*np.linalg.det(gm.covariances_[i]))))
    
    
    return probability_density

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
    

# ============================================================================
def RunPolicyWithBetaInterpolator(x0,x_OCL,u_OCL,t_OCL,interp_Fx, interp_Fy,beta):
    
    states = x0.reshape(1,-1)
    times = np.array(t_OCL[0])
    Fapplied = np.zeros(u_OCL.shape)
    

    for j in range(t_OCL.shape[0]-1):
        
        times = np.vstack((times, t_OCL[j+1]))
        # t = np.linspace(t_OCL[i], t_OCL[i+1])
        
        n_x = x0.shape[0];

        
        # Fi = IC.ctrl_from_interp(states[j,:], interp_Fx, interp_Fy)
        Fipred = IC.ctrl_from_interp(states[j,:], interp_Fx, interp_Fy); Fi = np.array([0,0])
        Fi[0] = Fipred[0] if not math.isnan(Fipred[0]) else 0.0
        Fi[1] = Fipred[1] if not math.isnan(Fipred[1]) else 0.0
    
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
def RunPolicyWithBetaInterpolatorSelf(x0,x_OCL,u_OCL,t_OCL,interp,beta):
    
    states = x0.reshape(1,-1)
    times = np.array(t_OCL[0])
    Fapplied = np.zeros(u_OCL.shape)
    

    for j in range(t_OCL.shape[0]-1):
        
        times = np.vstack((times, t_OCL[j+1]))
        # t = np.linspace(t_OCL[i], t_OCL[i+1])
        
        n_x = x0.shape[0];

        
        Fi = interp.predict(states[j,:],print_density_info=False)[0]
    
        F_input = (beta)*u_OCL[j,:] + (1-beta)*Fi
        Fapplied[j,:] = F_input

        # Integrate dynamics
        sol = integrate.solve_ivp(fun=lambda t, y: LD.LanderEOM(t,y,F_input),\
                                      t_span=(times[j],times[j+1]), \
                                      y0=states[j,:]) # Default method: rk45
        
        xsol = sol.y
        
        states = np.vstack((states, xsol[:,xsol.shape[1]-1]))


    
    return times, states, Fapplied, beta
    
    
    
    
    
    
    
    
    