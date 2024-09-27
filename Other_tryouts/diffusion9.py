#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:24:13 2024

@author: khaddari
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, eye
from scipy.sparse.linalg import spsolve
import diffusion7 as df7

# plate size, mm
Lx, Ly = df7.Lx, df7.Ly
# intervals in x-, y- directions, mm
nx, ny = df7.nx, df7.ny
dx, dy = Lx/nx, Ly/ny
dx2, dy2 = dx*dx, dy*dy

## Constants
D = 3
dt = 1.0
M = 300 + int(D**2 * (dx2 + dy2)/(dx2 * dy2))
k0 = D**2/((2*M-4)*dt) # kappa_max


Kxx, Kyy, Kxy = df7.Kxx, df7.Kyy, df7.Kxy

Alpha_x, Alpha_y = 0.25*(Kxx/dx + Kxy/dy)/dx, 0.25*(Kxy/dx + Kyy/dy)/dy
Beta_x, Beta_y   = 0.25*(Kxx/dx - Kxy/dy)/dx, 0.25*(Kxy/dx - Kyy/dy)/dy

reordered_matrix = lambda matrix: np.block([
    [matrix[1:, 1:], matrix[1:, :1]],
    [matrix[:1, 1:], matrix[:1, :1]]
])

#Alpha_x = reordered_matrix(Alpha_x); Alpha_y = reordered_matrix(Alpha_y)
#Beta_x = reordered_matrix(Alpha_x); Beta_y = reordered_matrix(Alpha_y)

Alpha_x.resize((nx+1)*(ny+1)); Alpha_y.resize((nx+1)*(ny+1))
Beta_x.resize((nx+1)*(ny+1)); Beta_y.resize((nx+1)*(ny+1))

BC = ['Neumann','Neumann','periodic','periodic']

def k_to_ij(k):
    i = k//nx
    j = k -i*nx
    return i,j

def kp(k):
    i,j = k_to_ij(k)
    return (j+1) + (i+1)*(ny+1)

def construct_D(BC):
    N = nx * ny
    D = lil_matrix((N, N))
    ## Central Nods
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            
            k = j + i*ny
            # Center
            D[k,k]  = Alpha_x[kp(k)] + Beta_x[kp(k)-1] + Beta_x[kp(k)-ny-1] + Alpha_x[kp(k)-ny-2]
            D[k,k] += Alpha_y[kp(k)] - Beta_y[kp(k)-1] - Beta_y[kp(k)-ny-1] + Alpha_y[kp(k)-ny-2]
            # Right Up
            D[k,k+ny+1] = - Alpha_x[kp(k)] - Alpha_y[kp(k)]
            # Right Center
            D[k,k+ny] = - Beta_x[kp(k)] - Alpha_x[kp(k)-1] - Beta_y[kp(k)] + Alpha_y[kp(k)-1]
            # Up Center
            D[k,k+1] = Beta_x[kp(k)] + Alpha_x[kp(k)-ny-1] + Beta_y[kp(k)] - Alpha_y[kp(k)-ny-1]
            # Right Down
            D[k,k+ny-1] = - Beta_x[kp(k)-1] + Beta_y[kp(k)-1]
            # Down Center 
            D[k,k-1] = Alpha_x[kp(k)-1] + Beta_x[kp(k)-ny-2] - Alpha_y[kp(k)-1] + Beta_y[kp(k)-ny-2]
            # Left Up
            D[k,k-ny+1] = - Beta_x[kp(k)-ny-1] + Beta_y[kp(k)-ny-1]
            # Left Center
            D[k,k-ny] = - Alpha_x[kp(k)-ny-1] - Beta_x[kp(k)-ny-2] + Alpha_y[kp(k)-ny-1] - Beta_y[kp(k)-ny-2]
            # Left Down
            D[k,k-ny-1] = - Alpha_x[kp(k)-ny-2] - Alpha_y[kp(k)-ny-2]

    ## Up & Down boundaries
    
    if BC[0] == 'Neumann' and BC[1] == 'Neumann':
        for i in range(1,nx-1):
            ## Up (j == ny-1)
            k = ny - 1 + i*ny 
            # Center
            D[k,k]  = Alpha_x[kp(k)] + Beta_x[kp(k)-1] + Beta_x[kp(k)-ny-1] + Alpha_x[kp(k)-ny-2]
            D[k,k] += Alpha_y[kp(k)] - Beta_y[kp(k)-1] - Beta_y[kp(k)-ny-1] + Alpha_y[kp(k)-ny-2]
            D[k,k] += Beta_x[kp(k)] + Alpha_x[kp(k)-ny-1] + Beta_y[kp(k)] - Alpha_y[kp(k)-ny-1]
            # Right Center
            D[k,k+ny] = - Beta_x[kp(k)] - Alpha_x[kp(k)-1] - Beta_y[kp(k)] + Alpha_y[kp(k)-1]
            D[k,k+ny] += - Alpha_x[kp(k)] - Alpha_y[kp(k)]
            # Right Down
            D[k,k+ny-1] = - Beta_x[kp(k)-1] + Beta_y[kp(k)-1]
            # Down Center 
            D[k,k-1] = Alpha_x[kp(k)-1] + Beta_x[kp(k)-ny-2] - Alpha_y[kp(k)-1] + Beta_y[kp(k)-ny-2]
            # Left Center
            D[k,k-ny] = - Alpha_x[kp(k)-ny-1] - Beta_x[kp(k)-ny-2] + Alpha_y[kp(k)-ny-1] - Beta_y[kp(k)-ny-2]
            D[k,k-ny] += - Beta_x[kp(k)-ny-1] + Beta_y[kp(k)-ny-1]
            # Left Down
            D[k,k-ny-1] = - Alpha_x[kp(k)-ny-2] - Alpha_y[kp(k)-ny-2]
            
            ## Down (j == 0)
            k = i*ny 
            # Center
            D[k,k]  = Alpha_x[kp(k)] + Beta_x[kp(k)-1] + Beta_x[kp(k)-ny-1] + Alpha_x[kp(k)-ny-2]
            D[k,k] += Alpha_y[kp(k)] - Beta_y[kp(k)-1] - Beta_y[kp(k)-ny-1] + Alpha_y[kp(k)-ny-2]
            D[k,k] += Alpha_x[kp(k)-1] + Beta_x[kp(k)-ny-2] - Alpha_y[kp(k)-1] + Beta_y[kp(k)-ny-2]
            # Right Up
            D[k,k+ny+1] = - Alpha_x[kp(k)] - Alpha_y[kp(k)]
            # Right Center
            D[k,k+ny] = - Beta_x[kp(k)] - Alpha_x[kp(k)-1] - Beta_y[kp(k)] + Alpha_y[kp(k)-1]
            D[k,k+ny] += - Beta_x[kp(k)-1] + Beta_y[kp(k)-1]
            # Up Center
            D[k,k+1] = Beta_x[kp(k)] + Alpha_x[kp(k)-ny-1] + Beta_y[kp(k)] - Alpha_y[kp(k)-ny-1]
            # Left Up
            D[k,k-ny+1] = - Beta_x[kp(k)-ny-1] + Beta_y[kp(k)-ny-1]
            # Left Center
            D[k,k-ny] = - Alpha_x[kp(k)-ny-1] - Beta_x[kp(k)-ny-2] + Alpha_y[kp(k)-ny-1] - Beta_y[kp(k)-ny-2]
            D[k,k-ny] += - Alpha_x[kp(k)-ny-2] - Alpha_y[kp(k)-ny-2]
            
        
            
    ## Right & Left boundaries
    if BC[2] == 'periodic' or BC[3] == 'periodic':
        for j in range(1,ny-1):
            ## Left (i == 0)
            k = j 
            # Center
            D[k,k]  = Alpha_x[kp(k)] + Beta_x[kp(k)-1] + Beta_x[kp(k)-ny-1] + Alpha_x[kp(k)-ny-2]
            D[k,k] += Alpha_y[kp(k)] - Beta_y[kp(k)-1] - Beta_y[kp(k)-ny-1] + Alpha_y[kp(k)-ny-2]
            # Right Up
            D[k,k+ny+1] = - Alpha_x[kp(k)] - Alpha_y[kp(k)]
            # Right Center
            D[k,k+ny] = - Beta_x[kp(k)] - Alpha_x[kp(k)-1] - Beta_y[kp(k)] + Alpha_y[kp(k)-1]
            # Up Center
            D[k,k+1] = Beta_x[kp(k)] + Alpha_x[kp(k)-ny-1] + Beta_y[kp(k)] - Alpha_y[kp(k)-ny-1]
            # Right Down
            D[k,k+ny-1] = - Beta_x[kp(k)-1] + Beta_y[kp(k)-1]
            # Down Center 
            D[k,k-1] = Alpha_x[kp(k)-1] + Beta_x[kp(k)-ny-2] - Alpha_y[kp(k)-1] + Beta_y[kp(k)-ny-2]
            # Left Up
            D[k,k-ny+1+ny*(nx)] = - Beta_x[kp(k)-ny-1] + Beta_y[kp(k)-ny-1]
            # Left Center
            D[k,k-ny+ny*(nx)] = - Alpha_x[kp(k)-ny-1] - Beta_x[kp(k)-ny-2] + Alpha_y[kp(k)-ny-1] - Beta_y[kp(k)-ny-2]
            # Left Down
            D[k,k-ny-1+ny*(nx)] = - Alpha_x[kp(k)-ny-2] - Alpha_y[kp(k)-ny-2]
            
            
            ## Right (i == nx-1)
            k = j + (nx-1)*ny 
            # Center
            D[k,k]  = Alpha_x[kp(k)] + Beta_x[kp(k)-1] + Beta_x[kp(k)-ny-1] + Alpha_x[kp(k)-ny-2]
            D[k,k] += Alpha_y[kp(k)] - Beta_y[kp(k)-1] - Beta_y[kp(k)-ny-1] + Alpha_y[kp(k)-ny-2]
            # Right Up
            D[k,k+ny+1-ny*(nx)] = - Alpha_x[kp(k)] - Alpha_y[kp(k)]
            # Right Center
            D[k,k+ny-ny*(nx)] = - Beta_x[kp(k)] - Alpha_x[kp(k)-1] - Beta_y[kp(k)] + Alpha_y[kp(k)-1]
            # Up Center
            D[k,k+1] = Beta_x[kp(k)] + Alpha_x[kp(k)-ny-1] + Beta_y[kp(k)] - Alpha_y[kp(k)-ny-1]
            # Right Down
            D[k,k+ny-1-ny*(nx)] = - Beta_x[kp(k)-1] + Beta_y[kp(k)-1]
            # Down Center 
            D[k,k-1] = Alpha_x[kp(k)-1] + Beta_x[kp(k)-ny-2] - Alpha_y[kp(k)-1] + Beta_y[kp(k)-ny-2]
            # Left Up
            D[k,k-ny+1] = - Beta_x[kp(k)-ny-1] + Beta_y[kp(k)-ny-1]
            # Left Center
            D[k,k-ny] = - Alpha_x[kp(k)-ny-1] - Beta_x[kp(k)-ny-2] + Alpha_y[kp(k)-ny-1] - Beta_y[kp(k)-ny-2]
            # Left Down
            D[k,k-ny-1] = - Alpha_x[kp(k)-ny-2] - Alpha_y[kp(k)-ny-2]
            
    ## Right Up (i == nx-1 ; j == ny-1)
    k = ny - 1 + (nx-1)*ny 
    # Center
    D[k,k]  = Alpha_x[kp(k)] + Beta_x[kp(k)-1] + Beta_x[kp(k)-ny-1] + Alpha_x[kp(k)-ny-2]
    D[k,k] += Alpha_y[kp(k)] - Beta_y[kp(k)-1] - Beta_y[kp(k)-ny-1] + Alpha_y[kp(k)-ny-2]
    D[k,k] += Beta_x[kp(k)] + Alpha_x[kp(k)-ny-1] + Beta_y[kp(k)] - Alpha_y[kp(k)-ny-1]
    # Right Center
    D[k,k+ny-ny*(nx)] = - Beta_x[kp(k)] - Alpha_x[kp(k)-1] - Beta_y[kp(k)] + Alpha_y[kp(k)-1]
    D[k,k+ny-ny*(nx)] += - Alpha_x[kp(k)] - Alpha_y[kp(k)]
    # Right Down
    D[k,k+ny-1-ny*(nx)] = - Beta_x[kp(k)-1] + Beta_y[kp(k)-1]
    # Down Center 
    D[k,k-1] = Alpha_x[kp(k)-1] + Beta_x[kp(k)-ny-2] - Alpha_y[kp(k)-1] + Beta_y[kp(k)-ny-2]
    # Left Center
    D[k,k-ny] = - Alpha_x[kp(k)-ny-1] - Beta_x[kp(k)-ny-2] + Alpha_y[kp(k)-ny-1] - Beta_y[kp(k)-ny-2]
    D[k,k-ny] += - Beta_x[kp(k)-ny-1] + Beta_y[kp(k)-ny-1]
    # Left Down
    D[k,k-ny-1] = - Alpha_x[kp(k)-ny-2] - Alpha_y[kp(k)-ny-2]
    
    ## Left Up (i == 0 ; j == ny-1)
    k = ny - 1 
    # Center
    D[k,k]  = Alpha_x[kp(k)] + Beta_x[kp(k)-1] + Beta_x[kp(k)-ny-1] + Alpha_x[kp(k)-ny-2]
    D[k,k] += Alpha_y[kp(k)] - Beta_y[kp(k)-1] - Beta_y[kp(k)-ny-1] + Alpha_y[kp(k)-ny-2]
    D[k,k] += Beta_x[kp(k)] + Alpha_x[kp(k)-ny-1] + Beta_y[kp(k)] - Alpha_y[kp(k)-ny-1]
    # Right Center
    D[k,k+ny] = - Beta_x[kp(k)] - Alpha_x[kp(k)-1] - Beta_y[kp(k)] + Alpha_y[kp(k)-1]
    D[k,k+ny] += - Alpha_x[kp(k)] - Alpha_y[kp(k)]
    # Right Down
    D[k,k+ny-1] = - Beta_x[kp(k)-1] + Beta_y[kp(k)-1]
    # Down Center 
    D[k,k-1] = Alpha_x[kp(k)-1] + Beta_x[kp(k)-ny-2] - Alpha_y[kp(k)-1] + Beta_y[kp(k)-ny-2]
    # Left Center
    D[k,k-ny+ny*(nx)] = - Alpha_x[kp(k)-ny-1] - Beta_x[kp(k)-ny-2] + Alpha_y[kp(k)-ny-1] - Beta_y[kp(k)-ny-2]
    D[k,k-ny+ny*(nx)] += - Beta_x[kp(k)-ny-1] + Beta_y[kp(k)-ny-1]
    # Left Down
    D[k,k-ny-1+ny*(nx)] = - Alpha_x[kp(k)-ny-2] - Alpha_y[kp(k)-ny-2]
    
    ## Right Down (i == nx-1 ; j == 0)
    k = (nx-1)*ny 
    # Center
    D[k,k]  = Alpha_x[kp(k)] + Beta_x[kp(k)-1] + Beta_x[kp(k)-ny-1] + Alpha_x[kp(k)-ny-2]
    D[k,k] += Alpha_y[kp(k)] - Beta_y[kp(k)-1] - Beta_y[kp(k)-ny-1] + Alpha_y[kp(k)-ny-2]
    D[k,k] += Alpha_x[kp(k)-1] + Beta_x[kp(k)-ny-2] - Alpha_y[kp(k)-1] + Beta_y[kp(k)-ny-2]
    # Right Up
    D[k,k+ny+1-ny*(nx)] = - Alpha_x[kp(k)] - Alpha_y[kp(k)]
    # Right Center
    D[k,k+ny-ny*(nx)] = - Beta_x[kp(k)] - Alpha_x[kp(k)-1] - Beta_y[kp(k)] + Alpha_y[kp(k)-1]
    D[k,k+ny-ny*(nx)] += - Beta_x[kp(k)-1] + Beta_y[kp(k)-1]
    # Up Center
    D[k,k+1] = Beta_x[kp(k)] + Alpha_x[kp(k)-ny-1] + Beta_y[kp(k)] - Alpha_y[kp(k)-ny-1]
    # Left Up
    D[k,k-ny+1] = - Beta_x[kp(k)-ny-1] + Beta_y[kp(k)-ny-1]
    # Left Center
    D[k,k-ny] = - Alpha_x[kp(k)-ny-1] - Beta_x[kp(k)-ny-2] + Alpha_y[kp(k)-ny-1] - Beta_y[kp(k)-ny-2]
    D[k,k-ny] += - Alpha_x[kp(k)-ny-2] - Alpha_y[kp(k)-ny-2]
    
    ## Left Down (i == 0 ; j == 0)
    k = 0 
    # Center
    D[k,k]  = Alpha_x[kp(k)] + Beta_x[kp(k)-1] + Beta_x[kp(k)-ny-1] + Alpha_x[kp(k)-ny-2]
    D[k,k] += Alpha_y[kp(k)] - Beta_y[kp(k)-1] - Beta_y[kp(k)-ny-1] + Alpha_y[kp(k)-ny-2]
    D[k,k] += Alpha_x[kp(k)-1] + Beta_x[kp(k)-ny-2] - Alpha_y[kp(k)-1] + Beta_y[kp(k)-ny-2]
    # Right Up
    D[k,k+ny+1] = - Alpha_x[kp(k)] - Alpha_y[kp(k)]
    # Right Center
    D[k,k+ny] = - Beta_x[kp(k)] - Alpha_x[kp(k)-1] - Beta_y[kp(k)] + Alpha_y[kp(k)-1]
    D[k,k+ny] += - Beta_x[kp(k)-1] + Beta_y[kp(k)-1]
    # Up Center
    D[k,k+1] = Beta_x[kp(k)] + Alpha_x[kp(k)-ny-1] + Beta_y[kp(k)] - Alpha_y[kp(k)-ny-1]
    # Left Up
    D[k,k-ny+1+ny*(nx)] = - Beta_x[kp(k)-ny-1] + Beta_y[kp(k)-ny-1]
    # Left Center
    D[k,k-ny+ny*(nx)] = - Alpha_x[kp(k)-ny-1] - Beta_x[kp(k)-ny-2] + Alpha_y[kp(k)-ny-1] - Beta_y[kp(k)-ny-2]
    D[k,k-ny+ny*(nx)] += - Alpha_x[kp(k)-ny-2] - Alpha_y[kp(k)-ny-2]
    
    k = 0; l = 7; print(k_to_ij(k),'-->',k_to_ij(l),'or',k,'->',l,": ",1e4*D[k,l])
    k = 7; l = 0; print(k_to_ij(k),'-->',k_to_ij(l),'or',k,'->',l,": ",1e4*D[k,l])
    
    
    
    return D.tocsr()

autoadj = df7.as1.autoadj

D = construct_D(BC)
h = 10000 * (D-D.T).todense()
htf = h==np.zeros((nx*ny,nx*ny))
print("Percentage: ",htf.sum()/(nx*ny)**2)
print(h.round(1))