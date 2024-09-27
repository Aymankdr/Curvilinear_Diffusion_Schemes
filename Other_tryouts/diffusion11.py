#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:35:32 2024

@author: khaddari
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, eye, diags, dia_matrix, bmat
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

#Alpha_x.resize((nx+1)*(ny+1)); Alpha_y.resize((nx+1)*(ny+1))
#Beta_x.resize((nx+1)*(ny+1)); Beta_y.resize((nx+1)*(ny+1))

BC = ['Neumann','Neumann','periodic','periodic']

def k_to_ij(k):
    i = k//nx
    j = k -i*nx
    return i,j

def kp(k):
    i,j = k_to_ij(k)
    return (j+1) + (i+1)*(ny+1)

"""
                                 [A  B           C]
                                 [C  A  B         ]
                                 [   C  A  B      ]
                                 [      .  .  .   ]
                                 [         .  .  .]
                                 [B           C  A]
"""

def Eta_A(i:int):
    return Alpha_x[i+1,1:] + Beta_x[i+1,:-1] + Beta_x[i,1:] + Alpha_x[i,:-1] +\
           Alpha_y[i+1,1:] + Alpha_y[i,:-1]  - Beta_y[i,:-1]- Beta_y[i,1:]
           
def Upsilon_A(i:int):
    return (Alpha_x[i,1:] + Beta_x[i+1,1:] - Alpha_y[i,1:] + Beta_y[i+1,1:])[:-1]
           
def Omega_A(i:int):
    return (Alpha_x[i+1,:-1] + Beta_x[i,:-1] - Alpha_y[i+1,:-1] + Beta_y[i,:-1])[1:]

def block_A(i:int):
    Ctr = Eta_A(i)
    # Neumann y==0:
    Ctr[0]  += Alpha_x[i+1,0] + Beta_x[i,0] - Alpha_y[i+1,0] + Beta_y[i,0]
    # Neumann y==Ly:
    Ctr[-1] += Alpha_x[i,-1] + Beta_x[i+1,-1] - Alpha_y[i,-1] - Beta_y[i+1,-1]
    # Tridiagonal block
    A = diags([Omega_A(i),Ctr,Upsilon_A(i)],[-1,0,1])
    return A



def Eta_B(i:int):
    return - Alpha_x[i+1,:-1] - Beta_x[i+1,1:] + Alpha_y[i+1,1:] - Beta_y[i+1,1:]
           
def Upsilon_B(i:int):
    return (Alpha_x[i+1,1:] - Alpha_y[i+1,1:])[:-1]
           
def Omega_B(i:int):
    return (- Beta_x[i+1,:-1] + Beta_y[i+1,:-1])[1:]

def block_B(i:int):
    Ctr = Eta_B(i)
    # Neumann y==0:
    Ctr[0]  += - Beta_x[i+1,0] + Beta_y[i+1,0]
    # Neumann y==Ly:
    Ctr[-1] += Alpha_x[i+1,-1] - Alpha_y[i+1,-1]
    # Tridiagonal block
    B = diags([Omega_B(i),Ctr,Upsilon_B(i)],[-1,0,1])
    return B



def Eta_C(i:int):
    return - Alpha_x[i,1:] - Beta_x[i,:-1] + Alpha_y[i,1:] - Beta_y[i,:-1]
           
def Upsilon_C(i:int):
    return (- Beta_x[i,1:] + Beta_y[i,1:])[:-1]
           
def Omega_C(i:int):
    return (Alpha_x[i,:-1] - Alpha_y[i,:-1])[1:]


def block_C(i:int):
    Ctr = Eta_C(i)
    # Neumann y==0:
    Ctr[0]  += Alpha_x[i,0] - Alpha_y[i,0]
    # Neumann y==Ly:
    Ctr[-1] += - Beta_x[i,-1] + Beta_y[i,-1]
    # Tridiagonal block
    C = diags([Omega_C(i),Ctr,Upsilon_C(i)],[-1,0,1])
    return C



def create_block_matrix_D():
    blocks = [[None for _ in range(nx)] for _ in range(nx)]
    
    for i in range(nx):
        blocks[i][i] = block_A(i)
        if i<nx-1: blocks[i][i+1] = block_B(i)
        else: blocks[i][0] = block_B(i)
        blocks[i][i-1] = block_C(i)

    return bmat(blocks, format='csr')




'''  Section Assembly  '''

def construct_A(i):
    Eta = np.zeros(ny)
    
    
    k = i*ny
    Eta[0] = Alpha_x[kp(k)] + Beta_x[kp(k)-1] + Beta_x[kp(k)-ny-1] + Alpha_x[kp(k)-ny-2]
    Eta[0] += Alpha_y[kp(k)] - Beta_y[kp(k)-1] - Beta_y[kp(k)-ny-1] + Alpha_y[kp(k)-ny-2]
    Eta[0] += Alpha_x[kp(k)-1] + Beta_x[kp(k)-ny-2] - Alpha_y[kp(k)-1] + Beta_y[kp(k)-ny-2]
    
    for j in range(1,ny-1):
        k = j + i*ny
        Eta[j]  = Alpha_x[kp(k)] + Beta_x[kp(k)-1] + Beta_x[kp(k)-ny-1] + Alpha_x[kp(k)-ny-2]
        Eta[j] += Alpha_y[kp(k)] - Beta_y[kp(k)-1] - Beta_y[kp(k)-ny-1] + Alpha_y[kp(k)-ny-2]
        
    k = ny - 1 + i*ny 
    Eta[-1]  = Alpha_x[kp(k)] + Beta_x[kp(k)-1] + Beta_x[kp(k)-ny-1] + Alpha_x[kp(k)-ny-2]
    Eta[-1] += Alpha_y[kp(k)] - Beta_y[kp(k)-1] - Beta_y[kp(k)-ny-1] + Alpha_y[kp(k)-ny-2]
    Eta[-1] += Beta_x[kp(k)] + Alpha_x[kp(k)-ny-1] + Beta_y[kp(k)] - Alpha_y[kp(k)-ny-1]


def construct_D():
    N = nx * ny
    D = lil_matrix((N, N))
    
    return D