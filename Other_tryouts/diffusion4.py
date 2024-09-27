#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:47:46 2024

@author: khaddari
"""

import numpy as np
import numpy.linalg as alg
import matplotlib.pyplot as plt

# plate size, mm
Lx = Ly = 10.
# intervals in x-, y- directions, mm
nx, ny = 100, 100
dx, dy = Lx/nx, Ly/ny

xedge = np.linspace(-dx/2, Lx + dx/2, nx + 2)
yedge = np.linspace(-dy/2, Ly + dy/2, ny + 2)

# Thermal diffusivity of steel, mm2.s-1
def kxx(x,y): return 4.0*y
def kyy(x,y): return 4.0*x
def kxy(x,y): return 0.0

Kxx = np.array([[kxx(i*dx,j*dy) for j in range(ny+1)] for i in range(nx+1)])
Kyy = np.array([[kyy(i*dx,j*dy) for j in range(ny+1)] for i in range(nx+1)])
Kxy = np.array([[kxy(i*dx,j*dy) for j in range(ny+1)] for i in range(nx+1)])

k0 = 80.1 # maximal eigenvalue of kappa 

Tcool, Thot = 300, 700

# CFL condition
dx2, dy2 = dx*dx, dy*dy
dt = dx2 * dy2 / ((2 * k0) * (dx2 + dy2))

# Boundary conditions

def north(x,t,CL='Neumann'):
    if CL=='Neumann':
        return 0
    if CL=='Dirichlet':
        return 0
    else: # condition mixte
        return 0
    
def south(x,t,CL='Neumann'):
    if CL=='Neumann':
        return 0
    if CL=='Dirichlet':
        return 0
    else: # condition mixte
        return 0

def east(y,t,CL='Neumann'):
    if CL=='Neumann':
        return 0
    if CL=='Dirichlet':
        return 0
    else: # condition mixte
        return 0
    
def west(y,t,CL='Neumann'):
    if CL=='Neumann':
        return 0
    if CL=='Dirichlet':
        return 0
    else: # condition mixte
        return 0

BC = ['Neumann','Neumann','Neumann','Neumann']

# Initialisation

u0 = Tcool * np.ones((nx+2, ny+2))

# Initial conditions - circle of radius r centred at (cx,cy) (mm)
r, cx, cy = 2, 5, 5
r2 = r**2
            
def wi(x,y):
    p2 = (x - cx)**2 + (y - cy)**2
    if p2 < r2:
        return Thot
    else: return Tcool
    
for i in range(nx + 2):
    for j in range(ny + 2):
        u0[i,j] = wi((i+0.5)*dx,(j+0.5)*dy)
            
u = u0.copy()

# Explicit scheme

def fill_edges(W,t):
    # North
    if BC[0] == 'Dirichlet': 
        W[:,ny+1] = 2*north(xedge,t,CL=BC[0]) - W[:,ny]
    elif BC[0] == 'Neumann': 
        W[:,ny+1] = dy*north(xedge,t,CL=BC[0]) + W[:,ny]
    else: 
        print('method not available')
    # South
    if BC[1] == 'Dirichlet': 
        W[:,0] = 2*south(xedge,t,CL=BC[1]) - W[:,1]
    elif BC[1] == 'Neumann': 
        W[:,0] = dy*south(xedge,t,CL=BC[1]) + W[:,1]
    else: 
        print('method not available')
    # East
    if BC[2] == 'Dirichlet': 
        W[nx+1,:] = 2*east(yedge,t,CL=BC[2]) - W[nx,:]
    elif BC[2] == 'Neumann': 
        W[nx+1,:] = dx*east(yedge,t,CL=BC[2]) + W[nx,:]
    else: 
        print('method not available')
    # West
    if BC[3] == 'Dirichlet': 
        W[0,:] = 2*west(yedge,t,CL=BC[3]) - W[1,:]
    elif BC[3] == 'Neumann': 
        W[0,:] = dx*west(yedge,t,CL=BC[3]) + W[1,:]
    else: 
        print('method not available')
    return W
        

def explicit_step(W,W0,t):
    
    # Rectangle centers
    dxW   = (W[1:,1:] + W[1:,:-1] - W[:-1,1:] - W[:-1,:-1])/(2*dx)
    dyW   = (W[1:,1:] + W[:-1,1:] - W[1:,:-1] - W[:-1,:-1])/(2*dy)
    
    Qx    = - (Kxx * dxW + Kxy * dyW)
    Qy    = - (Kxy * dxW + Kyy * dyW)
    
    divQ  = (Qx[1:,1:] + Qx[1:,:-1] - Qx[:-1,1:] - Qx[:-1,:-1])/(2*dx)
    divQ += (Qy[1:,1:] - Qy[1:,:-1] + Qy[:-1,1:] - Qy[:-1,:-1])/(2*dy)
    
    W[1:-1,1:-1] = W0[1:-1,1:-1] - dt * divQ
    
    W = fill_edges(W,t)
    W0 = W.copy()
    
    return W,W0
            
def explicit_scheme(T,u,u0):
    M = int(T/dt)
    t = 0
    for m in range(M):
        t   += dt
        u,u0 = explicit_step(u,u0,t)
    return u,u0 

## Solution exacte fourier pour kappa == kappa0

def we(x,y,T):
    if T > 0:
        ff    = lambda y : np.exp(-(x-y)**2/(4*k0*T)) * wi(y)
        integ = 1 # gauss3.integrate(ff,borneINF,borneSUP)
        return integ/np.sqrt(4*np.pi*k0*T)
    else:
        return wi(x,y)


# Number of timesteps
T = 64e-3
nsteps = int(T/dt)#101
t      = 0.0
# Output 4 figures at these timesteps
mfig = [0, nsteps//4-1, nsteps//2-1, nsteps-1]
fignum = 0
fig = plt.figure()
for m in range(nsteps):
    t    += dt 
    u0, u = explicit_step(u0, u, t)
    if m in mfig:
        fignum += 1
        print("Iteration : ",m, "\t Figure's number : ", fignum)
        ax = fig.add_subplot(220 + fignum)
        im = ax.imshow(u0.copy().T[::-1], cmap=plt.get_cmap('hot'), vmin=Tcool,vmax=Thot)
        ax.set_axis_off()
        ax.set_title('{:.1f} ms'.format((m+1)*dt*1000))
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
cbar_ax.set_xlabel('$T$ / K', labelpad=20)
fig.colorbar(im, cax=cbar_ax)
plt.show()
