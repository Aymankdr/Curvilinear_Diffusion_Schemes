#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:44:07 2024

@author: khaddari
"""

import numpy as np
import numpy.linalg as alg
import matplotlib.pyplot as plt
import assimilation1 as as1


# plate size, mm
R = 1.
# intervals in x-, y- directions, mm
Nt,Np = 50,100
dt,dp = np.pi/Nt, 2*np.pi/Np
dt2, dp2 = dt*dt, dp*dp

## Constants
D = .3
ds = 0.01
M = 300 + int(D**2 * (dt2 + dp2)/(dt2 * dp2))
k0 = D**2/((2*M)*ds) # kappa_max

# Mesh
t_secu = np.deg2rad(20)

Tcirc = np.linspace(-dt/2 + t_secu, np.pi - t_secu + dt/2, Nt + 2)
Tinsc = np.linspace(0, np.pi, Nt + 1)
Pcirc = np.linspace(-dp/2, 2*np.pi + dp/2, Np + 2)

# Normalization diagonal Matrix
NDM = []
NDM_filled = False

# Thermal diffusivity of steel, mm2.s-1
def kpp(x,y): return k0
def ktt(x,y): return k0
def kpt(x,y): return 0.0

Kpp = np.array([[kpp(i*dp,j*dt) for j in range(Nt+1)] for i in range(Np+1)])
Ktt = np.array([[ktt(i*dp,j*dt) for j in range(Nt+1)] for i in range(Np+1)])
Kpt = np.array([[kpt(i*dp,j*dt) for j in range(Nt+1)] for i in range(Np+1)])

# Amplitudes
Tcool, Thot = 0, 1

# Initial conditions - circle of radius r centred at (cx,cy) (mm)
r, cp, ct = np.pi/4, np.pi, np.pi/2 
r2 = r**2
            
def wi(x,y):
    p2 = (x - cp)**2 + (y - ct)**2
    if p2 < r2: return Thot
    else: return Tcool
    
u0 = Tcool * np.ones((Np,Nt))
    
for i in range(Np):
    for j in range(Nt):
        u0[i,j] = wi((i+0.5)*dp,(j+0.5)*dt)
        
u0.resize(Np*Nt)
         
# Boundary conditions

def north(x,CL='Neumann'):
    if CL=='Neumann':
        return 0
    if CL=='Dirichlet':
        return 0
    else: # condition mixte
        return 0
    
def south(x,CL='Neumann'):
    if CL=='Neumann':
        return 0
    if CL=='Dirichlet':
        return 0
    else: # condition mixte
        return 0

def east(y,CL='Neumann'):
    if CL=='Neumann':
        return 0
    if CL=='Dirichlet':
        return 0
    else: # condition mixte
        return 0
    
def west(y,CL='Neumann'):
    if CL=='Neumann':
        return 0
    if CL=='Dirichlet':
        return 0
    else: # condition mixte
        return 0

BC = ['Neumann','Neumann','Periodic','Periodic']

# Explicit scheme

def fill_edges(W):
    # East
    W[0,1:-1]  = W[-2,1:-1]
    # West
    W[-1,1:-1] = W[1,1:-1]
    # North
    W[:,-1] = W[:,-2]
    # South
    W[:,0] = W[:,1]
    
    return W

def explicit_step(W):
    
    # Rectangle centers
    dpW   = (-1/R)*(W[1:,1:] + W[1:,:-1] - W[:-1,1:] - W[:-1,:-1])/(2*dp)
    dtW   = (0.5/R)*((1/np.sin(Tcirc[1:]))*(W[1:,1:]-W[1:,:-1])+(1/np.sin(Tcirc[:-1]))*(W[:-1,1:]-W[:-1,:-1]))/(2*dt)
    
    Qp    = - (Kpp * dpW + Kpt * dtW)
    Qt    = - (Kpt * dpW + Ktt * dtW)
    
    divQ  = (Qp[1:,1:] + Qp[1:,:-1] - Qp[:-1,1:] - Qp[:-1,:-1])/(2*dp)
    divQ += ((Qt[1:,1:] - Qt[1:,:-1])*np.sin(Tinsc[1:]) + (Qt[:-1,1:] - Qt[:-1,:-1])*np.sin(Tinsc[:-1]))/(2*dt)
    divQ *= 1/(R*np.sin(Tcirc[1:-1]))
    
    W[1:-1,1:-1] = W[1:-1,1:-1] - 0.5 * ds * divQ
    
    W = fill_edges(W)
    W0 = W.copy()
    
    return W0

def previous_step(v0):
    W0 = v0.copy()
    W0.resize((Np,Nt))
    W  = np.zeros((Np+2,Nt+2))
    W[1:-1,1:-1] = W0
    W  = fill_edges(W)
    
    # Rectangle centers
    dpW   = (-1/R)*(W[1:,1:] + W[1:,:-1] - W[:-1,1:] - W[:-1,:-1])/(2*dp)
    dtW   = (0.5/R)*((1/np.sin(Tcirc[1:]))*(W[1:,1:]-W[1:,:-1])+(1/np.sin(Tcirc[:-1]))*(W[:-1,1:]-W[:-1,:-1]))/(2*dt)
    
    Qp    = - (Kpp * dpW + Kpt * dtW)
    Qt    = - (Kpt * dpW + Ktt * dtW)
    
    divQ  = (Qp[1:,1:] + Qp[1:,:-1] - Qp[:-1,1:] - Qp[:-1,:-1])/(2*dp)
    divQ += ((Qt[1:,1:] - Qt[1:,:-1])*np.sin(Tinsc[1:]) + (Qt[:-1,1:] - Qt[:-1,:-1])*np.sin(Tinsc[:-1]))/(2*dt)
    divQ *= 1/(R*np.sin(Tcirc[1:-1]))
    
    W[1:-1,1:-1] += ds * divQ
    
    W0 = W[1:-1,1:-1]
    v  = W0.copy() 
    v.resize(Np*Nt)
    return v

Nstep = 100
def explicit_operator(v0, nstep=M):
    w0 = v0.copy()
    w0.resize((Np,Nt))
    w  = np.zeros((Np+2,Nt+2))
    w[1:-1,1:-1] = w0
    w = fill_edges(w)
    global Nstep
    Nstep = nstep
    for m in range(nstep):
        w = explicit_step(w)
    w0 = w[1:-1,1:-1]
    v  = w0.copy() 
    v.resize(Np*Nt)
    return v


def implicit_operator(v0, nstep=M):
    w = v0.copy()
    global Nstep
    Nstep = nstep
    for m in range(nstep):
        w = as1.conjgrad(previous_step,w,w)
    return w

Lambda = []
Lambda_created = False
def create_Lambda():
    global Lambda, Lambda_created
    z = np.zeros(Np*Nt)
    print(1,end="")
    for k in range(Np*Nt):
        z[k], z[k-1] = 1.0, 0.0
        val = explicit_operator(z)[k]
        Lambda.append(1/np.sqrt(val))
        print(".",end="")
    print(Np*Nt,end="\n")
    Lambda = np.array(Lambda)
    Lambda_created = True

def correlation_operator(u):
    if not Lambda_created: create_Lambda()
    v = Lambda * u
    w = explicit_operator(v)
    x = Lambda * w
    return x

def test_L(mu=0,sig=1):
    u = np.random.normal(mu,sig,size = Np*Nt)
    v = np.random.normal(mu,sig,size = Np*Nt)
    return u.dot(explicit_operator(v)) - v.dot(explicit_operator(u))

def test_C(mu=0,sig=1):
    u = np.random.normal(mu,sig,size = Np*Nt)
    v = np.random.normal(mu,sig,size = Np*Nt)
    return u.dot(correlation_operator(v)) - v.dot(correlation_operator(u))
            

def colormap(v):
    w = v.copy()
    w.resize((Np, Nt))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('${\Phi / \Delta \Phi }$')  # Add x-axis label
    ax.set_ylabel('${\Theta / \Delta \Theta }$')  # Add y-axis label
    ax.set_xticks(np.arange(0, Np, Np//6))
    ax.set_yticks(np.arange(0, Nt, Nt//6))
    im = ax.imshow(w.T[::-1], cmap=plt.get_cmap('hot'), vmin=Tcool, vmax=Thot)
    #ax.set_axis_off()
    ax.set_title('{:.1f} s'.format((Nstep) * ds))
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    cbar_ax.set_xlabel('$T$ / K', labelpad=20)
    fig.colorbar(im, cax=cbar_ax)
    plt.show()