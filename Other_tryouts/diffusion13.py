#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:23:42 2024

@author: khaddari
"""

import numpy as np
import numpy.linalg as alg
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.sparse import lil_matrix, csr_matrix, eye, diags
from scipy.sparse.linalg import spsolve, eigs, norm
from diffusion7 import colormap
import diffusion7 as df7
from diffusion12 import func2array
from diffusion14 import spherical
import assimilation1 as as1
import cartopy.crs as ccrs

# plate size, mm
Lx, Ly = 10., 7.
# intervals in x-, y- directions, mm
nx, ny = 75, 75
dx, dy = Lx/nx, Ly/ny
dx2, dy2 = dx*dx, dy*dy

## Constants
D = 0.6
dt = 1.0
S = 25 # + int(D**2 * (dx2 + dy2)/(dx2 * dy2))
k0 = D**2/(2*S*dt) # kappa_max

## Boundary Condition
BC = 'periodic'

## Diffusivity
diffusivity='isotropic_and_homogeneous'
#diffusivity='anisotropic_and_inhomogeneous'

## Gradient expression
finite_diff_scheme = 'symmetric'
finite_diff_scheme = 'adaptable'

## Initial conditions
# Amplitudes
Tcool, Thot = 0, 1

# Initial conditions - circle of radius r centred at (cx,cy) (mm)

CIcase = 'modal'
CIcase = 'circle'
ang1, ang2 = 2, 2
if CIcase == 'modal':
    chosen_type = complex
else:
    chosen_type = float
            
def wi(x,y,r,cx,cy,case=CIcase):
    if case == 'circle':
        p2 = (x - cx)**2 + (y - cy)**2
        r2 = r**2
        if p2 < r2:
            return Thot
        else: return Tcool
    else:
        z = x * ang1 + y * ang2
        return np.cos(z) + 1j * np.sin(z)
    
def U0(r,cx,cy,case=CIcase):
    u0 = Tcool * np.ones((nx, ny+1), dtype=chosen_type)
    for i in range(nx):
        for j in range(ny+1):
            u0[i,j] = wi((i+0.5)*dx,(j+0.5)*dy,r,cx,cy)
    u0.resize(nx*(ny+1))
    return u0


######################################
## Curvilinear coordinates (sphere) ##
######################################
n1, n2 = nx, ny
N = n1 * (n2 + 1)

## Sphere
R = 2 # radius

theta_secu = np.deg2rad(10) 
theta_init, theta_fin = theta_secu, np.pi - theta_secu

phi_redu = np.deg2rad(0)
dphi, dtheta = (2*np.pi-phi_redu)/n1, (np.pi - 2*theta_secu)/n2

#def e1(i,j): return (R*dphi)
def e1(i,j): return (R*np.sin(j*dtheta + theta_secu)*dphi)
def e2(i,j): return (R*dtheta)

if diffusivity=='isotropic_and_homogeneous':
    def k11(i,j): return k0 #k0*i/n1
    def k22(i,j): return k0 #k0*j/n2
    def k12(i,j): return 0. #k0*i*j/(n1*n2)
if diffusivity=='anisotropic_and_inhomogeneous':
    def k11(i,j): return 4*k0*(i-n1//2)**2/n1**2
    def k22(i,j): return 4*k0*(j-n2//2)**2/n2**2
    def k12(i,j): return 8*k0*(i-n1//2)**2*(j-n2//2)**2/(n1*n2)**2

def a11(i,j): return e2(i,j)/e1(i,j)*k11(i,j)
def a22(i,j): return e1(i,j)/e2(i,j)*k22(i,j)
def a12(i,j): return k12(i,j)

A11 = np.array([[a11(i+0.5,j+0.5) for j in range(n2)] for i in range(n1)])
A22 = np.array([[a22(i+0.5,j+0.5) for j in range(n2)] for i in range(n1)])
A12 = np.array([[a12(i+0.5,j+0.5) for j in range(n2)] for i in range(n1)])

A11r = np.array([[a11(i+0.5,j) for j in range(n2+1)] for i in range(n1)])
A22r = np.array([[a22(i,j+0.5) for j in range(n2)] for i in range(n1+1)])

E1  = np.array([[e1(i,j) for j in range(n2+1)] for i in range(1,n1+1)])
E2  = np.array([[e2(i,j) for j in range(n2+1)] for i in range(1,n1+1)])
E1E2 = E1 * E2 # G[i,j] = G_i+1,j

## Construct K's coefficients
Aa11 = np.zeros((n1+2,n2+2)); Aa11[1:-1,1:-1] = A11; Aa11[0,:] = Aa11[-2,:]; Aa11[-1,:] = Aa11[1,:]
Aa22 = np.zeros((n1+2,n2+2)); Aa22[1:-1,1:-1] = A22; Aa22[0,:] = Aa22[-2,:]; Aa22[-1,:] = Aa22[1,:]
Aa12 = np.zeros((n1+2,n2+2)); Aa12[1:-1,1:-1] = A12; Aa12[0,:] = Aa12[-2,:]; Aa12[-1,:] = Aa12[1,:]

Aa11r = np.zeros((n1+2,n2+1)); Aa11r[1:-1,:] = A11r;
Aa11r[0,:] = Aa11r[-2,:]; Aa11r[-1,:] = Aa11r[1,:]
Aa22r = np.zeros((n1+1,n2+2)); Aa22r[:,1:-1] = A22r

if finite_diff_scheme == 'symmetric':
    Kcc = Aa11[1:,1:] + 2 * Aa12[1:,1:] + Aa22[1:,1:] +\
        Aa11[:-1,1:] - 2 * Aa12[:-1,1:] + Aa22[:-1,1:] +\
        Aa11[1:,:-1] - 2 * Aa12[1:,:-1] + Aa22[1:,:-1] +\
        Aa11[:-1,:-1] + 2 * Aa12[:-1,:-1] + Aa22[:-1,:-1] 
    Krr = - Aa11[1:,1:] + Aa22[1:,1:] - Aa11[1:,:-1] + Aa22[1:,:-1]
    Kll = - Aa11[:-1,1:] + Aa22[:-1,1:] - Aa11[:-1,:-1] + Aa22[:-1,:-1]
    Kuu = Aa11[1:,1:] - Aa22[1:,1:] + Aa11[:-1,1:] - Aa22[:-1,1:]
    Kdd = Aa11[1:,:-1] - Aa22[1:,:-1] + Aa11[:-1,:-1] - Aa22[:-1,:-1]
    Kru = - Aa11[1:,1:] - Aa22[1:,1:] - 2 * Aa12[1:,1:]
    Klu = - Aa11[:-1,1:] - Aa22[:-1,1:] + 2 * Aa12[:-1,1:]
    Krd = - Aa11[1:,:-1] - Aa22[1:,:-1] + 2 * Aa12[1:,:-1]
    Kld = - Aa11[:-1,:-1] - Aa22[:-1,:-1] - 2 * Aa12[:-1,:-1]
    Kcc*=2;Krr*=2;Kll*=2;Kuu*=2;Kdd*=2;Kru*=2;Klu*=2;Krd*=2;Kld*=2
else:
    Kcc = 8 * (Aa11r[:-1,:] + Aa11r[1:,:] + Aa22r[:,:-1] + Aa22r[:,1:]) +\
        2 * (Aa12[:-1,:-1] - Aa12[:-1,1:]) +\
        2 * (Aa12[1:,1:] - Aa12[1:,:-1]) 
    Krr = - 8 * Aa11r[1:,:]
    Kll = - 8 * Aa11r[:-1,:]
    Kuu = - 8 * Aa22r[:,1:]
    Kdd = - 8 * Aa22r[:,:-1]
    Kru = - 2 * Aa12[1:,1:]
    Klu = 2 * Aa12[:-1,1:]
    Krd = 2 * Aa12[1:,:-1]
    Kld = - 2 * Aa12[:-1,:-1]

## Construct M's coefficients
#Gmet = np.zeros((n1+1,n2+2)); Gmet[1:-1,1:-1] = E1E2
#Gmoy = Gmet[1:,1:]+Gmet[1:,:-1]+Gmet[:-1,1:]+Gmet[:-1,:-1]
Gmet = E1E2; Gmet[1:-1,1:-1] *= 4;
Gmet[1:-1,0] *= 2; Gmet[1:-1,-1] *= 2; Gmet[0,1:-1] *= 2; Gmet[0,1:-1] *= 2 

## Construct K
K = lil_matrix((N,N))
for i in range(n1):
    for j in range(n2+1):
        k = j + i * (n2 + 1) # true index is (i+1,j)
        # Logicals
        ru = (i == n1 - 1 or j == n2); lu = (i == 0 or j == n2)
        rd = (i == n1 - 1 or j == 0); ld = (i == 0 or j == 0)
        rr = (i == n1 - 1); ll = (i == 0)
        uu = (j == n2); dd = (j == 0)
        cc = (0 < i < n1 - 1 and 0 < j < n2)
        
        K[k,k] = Kcc[i+1,j]
        if not uu: K[k,k+1] = Kuu[i+1,j]
        if not dd: K[k,k-1] = Kdd[i+1,j]
        if not rr: K[k,k+n2+1] = Krr[i+1,j]
        else: K[k,k+n2+1 - N] = Krr[i+1,j]
        if not ll: K[k,k-n2-1] = Kll[i+1,j]
        else: K[k,k-n2-1 + N] = Kll[i+1,j]
        if not rr and not ru: K[k,k+n2+2] = Kru[i+1,j]
        elif rr and (j!=n2): K[k,k+n2+2 - N] = Kru[i+1,j]
        if not ll and not ld: K[k,k-n2-2] = Kld[i+1,j]
        elif ll and (j!=0): K[k,k-n2-2 + N] = Kld[i+1,j]
        if not rr and not rd: K[k,k+n2] = Krd[i+1,j]
        elif rr and (j!=0): K[k,k+n2 - N] = Krd[i+1,j]
        if not ll and not lu: K[k,k-n2] = Klu[i+1,j]
        elif ll and (j!=n2): K[k,k-n2 + N] = Klu[i+1,j]
        
K = K.tocsr()
        
## Construct M
Gmet *= 2
M = diags(Gmet.flatten())
Mdir = Gmet.flatten()
Minv = 1/Mdir

## Construct A=M+K
A = M + K

## Energy scheme
def energy_iteration(u):
    v = M.dot(u)
    b = spsolve(A,v)
    #b = as1.conjgrad(A, v, v, nature=0)
    return b

def energy(u,up):
    return (M @ u) @ up - 0.5 * (M @ u) @ u - 0.5 * (K @ u) @ u

def energy_scheme(u,nb_iter=S):
    E_list = []
    v = np.copy(u)
    for s in range(nb_iter):
        w = energy_iteration(v)
        e = energy(w, v)
        E_list.append(e)
        v = w
    return v,E_list

## Algorithm
def algo(nb_iter=S,x_cursor=0.5,y_cursor=0.5,u0=None):
    r, cx, cy = 0.2*min(Lx,Ly), x_cursor*Lx, y_cursor*Ly
    if type(u0)!=type(np.zeros(2)): u0 = U0(r,cx,cy)
    vv,E_list = energy_scheme(u0,nb_iter=nb_iter)
    print(vv.dot(vv))
    colormap(vv,nx=n1,ny=n2+1,nb_iter=nb_iter) 
    plt.show()
    spherical(vv, nx=n1, ny=n2+1, theta_secu=theta_secu)
    plt.show()
    s_list = list(range(nb_iter))
    plt.plot(s_list,E_list)
    #plt.show()
    return vv, E_list
    
def sparsity_pattern(X):
    x = np.array(X.todense())
    fig, axs = plt.subplots(2, 2)
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]
    ax1.spy(x, markersize=5)
    ax2.spy(x, precision=1e-10, markersize=5)
    ax3.spy(x)
    ax4.spy(x, precision=1e-10)
    plt.show()
    
def random_spectral_radius(A,Minv):
    u=np.random.normal(size=N); s=(Minv*(A.dot(u))).dot(u)/(u.dot(u));
    return s

# Dirac
def Dirac(i,j):
    v = np.zeros(N)
    v[j + (ny+1) * i] = 1.0 / (e1(i+0.5,j+0.5) * e2(i+0.5,j+0.5))
    return v

