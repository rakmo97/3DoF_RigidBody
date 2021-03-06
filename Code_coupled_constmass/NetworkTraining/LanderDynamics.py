# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 13:42:22 2021

@author: omkar_mulekar
"""

import numpy as np
from scipy import integrate


def LanderEOM(t,x,u):
    
    # States:
    #   x[0]: x
    #   x[1]: y
    #   x[2]: phi
    #   x[3]: dx
    #   x[4]: dy
    #   x[5]: dphi
    #   x[6]: m
    
    # Parameters
    g = 9.81 /6.0
    g0 = 9.81
    Isp = 300
    r = 0.25
    m = 500
    
    J =  m


    Fx = np.clip(u[0],-15000,15000)
    Fy = np.clip(u[1],0,15000)
    # T  = np.clip(u[2],-15000,15000)
    
    
    dx    = x[3]
    dy    = x[4]
    dphi  = x[5]
    ddx   = (1/m) * (Fx*np.cos(x[2]) - Fy*np.sin(x[2]))
    ddy   = (1/m) * (Fx*np.sin(x[2]) + Fy*np.cos(x[2])) - g
    ddphi = (1/J) * (r*Fx)
    
    # dx    = x[3]
    # dy    = x[4]
    # dphi  = x[5]
    # ddx   = (1/x[6])*Fx
    # ddy   = (1/x[6])*Fy - g
    # ddphi = (1/J) * (T)
    # dm    = - np.sqrt(Fx**2 + Fy**2) / (Isp*g0)

    xdot = np.array([dx,dy,dphi,ddx,ddy,ddphi])
    # xdot = [dx,dy,dphi,ddx,ddy,ddphi,dm]
    
    
    return xdot


def CalculateCost(t,u):
    
    L = np.empty(t.shape)
    
    for i in range(t.shape[0]):
        Lt = 0
        for j in range(u.shape[1]):
            Lt += u[i,j]**2
    
        L[i] = np.sqrt(Lt)
        
    J = integrate.trapz(L, x=t)
    
    return J
    