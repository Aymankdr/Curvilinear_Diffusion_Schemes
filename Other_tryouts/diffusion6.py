#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:01:44 2024

@author: khaddari
"""

import numpy as np
import numpy.linalg as alg
import matplotlib.pyplot as plt
import assimilation1 as as1


# plate size, mm
Lx = Ly = 10.
# intervals in x-, y- directions, mm
nx, ny = 100, 100
dx, dy = Lx/nx, Ly/ny
dx2, dy2 = dx*dx, dy*dy

## Constants
D = 1
dt = 1.0
M = 300 + int(D**2 * (dx2 + dy2)/(dx2 * dy2))
k0 = D**2/((2*M-4)*dt) # kappa_max

# Mesh
xedge = np.linspace(-dx/2, Lx + dx/2, nx + 2)
yedge = np.linspace(-dy/2, Ly + dy/2, ny + 2)

# Normalization diagonal Matrix
NDM = []
NDM_filled = False

# Thermal diffusivity of steel, mm2.s-1
def kxx(x,y): return k0*y/Ly
def kyy(x,y): return k0*x/Lx
def kxy(x,y): return 0.0

Kxx = np.array([[kxx(i*dx,j*dy) for j in range(ny+1)] for i in range(nx+1)])
Kyy = np.array([[kyy(i*dx,j*dy) for j in range(ny+1)] for i in range(nx+1)])
Kxy = np.array([[kxy(i*dx,j*dy) for j in range(ny+1)] for i in range(nx+1)])

# Amplitudes
Tcool, Thot = 0, 1


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

BC = ['Neumann','Neumann','Neumann','Neumann']

# Initialisation

# Initial conditions - circle of radius r centred at (cx,cy) (mm)
r, cx, cy = 2, 5, 5
r2 = r**2
            
def wi(x,y):
    p2 = (x - cx)**2 + (y - cy)**2
    if p2 < r2:
        return Thot
    else: return Tcool

def w_ci(i,j):
    u0 = np.zeros((nx+2,ny+2))
    u0[i+1,j+1] = 1/(dx*dy)
    return u0

u0 = w_ci(50,50)          
u  = u0.copy()

# Explicit scheme

def fill_edges(W):
    # North
    if BC[0] == 'Dirichlet': 
        W[:,ny+1] = 2*north(xedge,CL=BC[0]) - W[:,ny]
    elif BC[0] == 'Neumann': 
        W[:,ny+1] = dy*north(xedge,CL=BC[0]) + W[:,ny]
    else: 
        print('method not available')
    # South
    if BC[1] == 'Dirichlet': 
        W[:,0] = 2*south(xedge,CL=BC[1]) - W[:,1]
    elif BC[1] == 'Neumann': 
        W[:,0] = dy*south(xedge,CL=BC[1]) + W[:,1]
    else: 
        print('method not available')
    # East
    if BC[2] == 'Dirichlet': 
        W[nx+1,:] = 2*east(yedge,CL=BC[2]) - W[nx,:]
    elif BC[2] == 'Neumann': 
        W[nx+1,:] = dx*east(yedge,CL=BC[2]) + W[nx,:]
    else: 
        print('method not available')
    # West
    if BC[3] == 'Dirichlet': 
        W[0,:] = 2*west(yedge,CL=BC[3]) - W[1,:]
    elif BC[3] == 'Neumann': 
        W[0,:] = dx*west(yedge,CL=BC[3]) + W[1,:]
    else: 
        print('method not available')
    return W
        

def explicit_step(W,W0):
    
    # Rectangle centers
    dxW   = (W[1:,1:] + W[1:,:-1] - W[:-1,1:] - W[:-1,:-1])/(2*dx)
    dyW   = (W[1:,1:] + W[:-1,1:] - W[1:,:-1] - W[:-1,:-1])/(2*dy)
    
    Qx    = - (Kxx * dxW + Kxy * dyW)
    Qy    = - (Kxy * dxW + Kyy * dyW)
    
    divQ  = (Qx[1:,1:] + Qx[1:,:-1] - Qx[:-1,1:] - Qx[:-1,:-1])/(2*dx)
    divQ += (Qy[1:,1:] - Qy[1:,:-1] + Qy[:-1,1:] - Qy[:-1,:-1])/(2*dy)
    
    W[1:-1,1:-1] = W0[1:-1,1:-1] - dt * divQ
    
    W = fill_edges(W)
    W0 = W.copy()
    
    return W,W0

def explicit_scheme(u):
    u0 = np.copy(u)
    for m in range(M):
        u,u0 = explicit_step(u,u0)
    return u0 


def previous_step(W):
    W0 = np.copy(W)
    
    # Rectangle centers
    dxW   = (W[1:,1:] + W[1:,:-1] - W[:-1,1:] - W[:-1,:-1])/(2*dx)
    dyW   = (W[1:,1:] + W[:-1,1:] - W[1:,:-1] - W[:-1,:-1])/(2*dy)
    
    Qx    = - (Kxx * dxW + Kxy * dyW)
    Qy    = - (Kxy * dxW + Kyy * dyW)
    
    divQ  = (Qx[1:,1:] + Qx[1:,:-1] - Qx[:-1,1:] - Qx[:-1,:-1])/(2*dx)
    divQ += (Qy[1:,1:] - Qy[1:,:-1] + Qy[:-1,1:] - Qy[:-1,:-1])/(2*dy)
    
    W[1:-1,1:-1] = W0[1:-1,1:-1] + dt * divQ
    
    W = fill_edges(W)
    W0 = W.copy()
    
    return W0
            

def implicit_scheme(u):
    u0 = np.copy(u)
    for m in range(M):
        u0 = as1.conjgrad(previous_step,u0,u0)
    return u0 

def Fill_NDM():
    global NDM, NDM_filled
    NDM_filled = True
    for i in range(nx):
        for j in range(ny):
            NDM.append(explicit_scheme(w_ci(i,j))[i+1,j+1])
    

def explicit(u,normalize = False):
    if normalize:
        if not NDM_filled: Fill_NDM()
        u_core = u[1:-1,1:-1].resize(nx*ny)
        v_core = NDM * u_core
        v_tens = v_core.resize((nx,ny))
        v_core = np.zeros((nx+2,ny+2))
        v_core[1:-1,1:-1] = v_tens
        v0 = explicit_scheme(v_core)
        w0 = NDM * v0
        return w0
    else:
        u0 = explicit_scheme(u)
        w  = u0[1:-1,1:-1].resize(nx*ny)
        return w

def test_explicit():
    u = np.random.normal(size=nx*ny)
    return u.dot(u)


## Solution exacte fourier pour kappa == kappa0

def we(x,y,T):
    if T > 0:
        ff    = lambda y : np.exp(-(x-y)**2/(4*k0*T)) * wi(y)
        integ = 1 # gauss3.integrate(ff,borneINF,borneSUP)
        return integ/np.sqrt(4*np.pi*k0*T)
    else:
        return wi(x,y)


# Output 4 figures at these timesteps
mfig = [nx*ny//8-1, 3*nx*ny//8-1, 5*nx*ny//8-1, 7*nx*ny//8-1]
fignum = 0
fig = plt.figure()
for r in mfig:
    u0 = w_ci(r//nx,r - r//nx*nx)          
    u  = u0.copy()
    u0 = explicit_scheme(u)
    fignum += 1
    print("Iteration : ",r, "\t Figure's number : ", fignum)
    ax = fig.add_subplot(220 + fignum)
    im = ax.imshow(u0.copy().T[::-1], cmap=plt.get_cmap('hot'), vmin=Tcool,vmax=Thot)
    ax.set_axis_off()
    ax.set_title('{:.1f} s'.format((M)*dt))
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
cbar_ax.set_xlabel('$T$ / K', labelpad=20)
fig.colorbar(im, cax=cbar_ax)
plt.show()
