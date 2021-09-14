# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 13:42:22 2021

@author: omkar_mulekar
"""

import numpy as np
from scipy import integrate


def LanderEOM(t,x,u):
    
    # States Key:
    #   x[0] : x
    #   x[1] : y
    #   x[2] : z
    #   x[3] : dx
    #   x[4] : dy
    #   x[5] : dz
    #   x[6] : phi
    #   x[7] : theta
    #   x[8] : psi
    #   x[9] : p
    #   x[10]: q
    #   x[11]: r
    #   x[12]: m
    
    # Parameters
    g = 9.81 /6.0
    g0 = 9.81
    Isp = 300
    a = 1;
    b = 1;
    c = 1.5;
    

    # Control Input
    Fmax = 15000
    Fx = np.clip(u[0],-Fmax,Fmax)
    Fy = np.clip(u[1],-Fmax,Fmax)
    Fz = np.clip(u[2],    0,Fmax)
    L  = np.clip(u[3],-Fmax,Fmax)
    M  = np.clip(u[4],-Fmax,Fmax)
    N  = np.clip(u[5],-Fmax,Fmax)
    
    
    # Mass props (mass and MOI)
    m = x[12]
    Ix = (m/5)*(b**2 + c**2)
    Iy = (m/5)*(c**2 + a**2)
    Iz = (m/5)*(a**2 + b**2)
    
    # Equations of Motion: 
    dx     = x[3]
    dy     = x[4]
    dz     = x[5]
    ddx    = (1/m)*Fx
    ddy    = (1/m)*Fy
    ddz    = (1/m)*Fz - g
    dphi   = x[9] + x[10]*np.sin(x[6])*np.tan(x[7]) + x[11]*np.cos(x[6])*np.tan(x[7])
    dtheta =        x[10]*np.cos(x[6])              - x[11]*np.sin(x[6])
    dpsi   =        x[10]*np.sin(x[6])/np.cos(x[7]) + x[11]*np.cos(x[6])/np.cos(x[7])
    dp     = (1/Ix)*(L + (Iy-Iz)*x[10]*x[11])
    dq     = (1/Iy)*(M + (Iz-Ix)*x[11]*x[9] )
    dr     = (1/Iz)*(N + (Ix-Iy)*x[9] *x[10])
    dm     = - np.sqrt(Fx**2 + Fy**2 + Fz**2 + L**2 + M**2 + N**2) / (Isp*g0)
    
    
    # Format Output
    xdot = np.array([dx,dy,dz,ddx,ddy,ddz,dphi,dtheta,dpsi,dp,dq,dr,dm])
    
    
    return xdot



# def CalculateCost(t,u):
    


