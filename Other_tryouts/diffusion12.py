#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:24:57 2024

@author: khaddari
"""

import numpy as np
import numpy.linalg as alg
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from scipy.sparse import lil_matrix, csr_matrix, eye, diags
from scipy.sparse.linalg import spsolve, eigs
import assimilation1 as as1
import diffusion7 as df7

# plate size, mm
Lx, Ly = df7.Lx, df7.Ly
# intervals in x-, y- directions, mm
nx, ny = 75,76
dx, dy = Lx/nx, Ly/ny
dx2, dy2 = dx*dx, dy*dy

## Constants
D = df7.D
dt = df7.dt
S = df7.M
k0 = df7.k0

# Mesh
xedge = np.linspace(-dx/2, Lx + dx/2, nx + 2)
yedge = np.linspace(-dy/2, Ly + dy/2, ny + 2)

# Basis Hilbert function
def phi(x,y,corner):
    i,j = corner[0] - 0.5, corner[1] - 0.5
    if i-0.5 <= x < i+0.5:
        if j-0.5 <= y < j+0.5: # Kij
            return (x - i + 0.5)*(y - j + 0.5)
        elif j+0.5 <= y < j+1.5: # Kij+1
            return - (x - i + 0.5)*(y - j - 1.5)
        else:
            return 0.0
    elif i+0.5 <= x < i+1.5:
        if j-0.5 <= y < j+0.5: # Ki+1j
            return - (x - i - 1.5)*(y - j + 0.5)
        elif j+0.5 <= y < j+1.5: # Ki+1j+1
            return (x - i - 1.5)*(y - j - 1.5)
        else:
            return 0.0
    else:
        return 0.0

def dx_phi(x,y,corner):
    i,j = corner[0] - 0.5, corner[1] - 0.5
    if i-0.5 <= x < i+0.5:
        if j-0.5 <= y < j+0.5: # Kij
            return (y - j + 0.5)
        elif j+0.5 <= y < j+1.5: # Kij+1
            return - (y - j - 1.5)
        else:
            return 0.0
    elif i+0.5 <= x < i+1.5:
        if j-0.5 <= y < j+0.5: # Ki+1j
            return - (y - j + 0.5)
        elif j+0.5 <= y < j+1.5: # Ki+1j+1
            return (y - j - 1.5)
        else:
            return 0.0
    else:
        return 0.0
    
def dy_phi(x,y,corner):
    i,j = corner[0] - 0.5, corner[1] - 0.5
    if i-0.5 <= x < i+0.5:
        if j-0.5 <= y < j+0.5: # Kij
            return (x - i + 0.5)
        elif j+0.5 <= y < j+1.5: # Kij+1
            return - (x - i + 0.5)
        else:
            return 0.0
    elif i+0.5 <= x < i+1.5:
        if j-0.5 <= y < j+0.5: # Ki+1j
            return - (x - i - 1.5)
        elif j+0.5 <= y < j+1.5: # Ki+1j+1
            return (x - i - 1.5)
        else:
            return 0.0
    else:
        return 0.0
    
# Dirac
def Dirac(i,j):
    v = np.zeros(nx*ny)
    v[j + ny * i] = 1.0 / (df7.e1(i+0.5,j+0.5) * df7.e2(i+0.5,j+0.5))
    return v
              
# The forms
def d_form(corner1, corner2):
    a,b = min(corner1[0]-1.,corner2[0]-1.),max(corner1[0]+1.,corner2[0]+1.)
    c,d = min(corner1[1]-1.,corner2[1]-1.),max(corner1[1]+1.,corner2[1]+1.)
    func = lambda x,y:dx_phi(x,y,corner1)*dy_phi(x,y,corner2)+dy_phi(x,y,corner1)*dx_phi(x,y,corner2)#+\
       # dy_phi(x,y,corner1)*dy_phi(x,y,corner2)
    return dblquad(func, c, d, a, b)

def c_form(corner1, corner2):
    a,b = min(corner1[0]-1.,corner2[0]-1.),max(corner1[0]+1.,corner2[0]+1.)
    c,d = min(corner1[1]-1.,corner2[1]-1.),max(corner1[1]+1.,corner2[1]+1.)
    func = lambda x,y:phi(x,y,corner1)*phi(x,y,corner2)
    return dblquad(func, c, d, a, b)

def b_form(corner1, corner2, method='fast'):
    if method=='fast':
        xc,yc = (corner1[0]+corner2[0])/2,(corner1[1]+corner2[1])/2
        e1c, e2c = df7.e1(xc,yc), df7.e2(xc,yc)
        diag_index = (corner1[0]-corner2[0])*(corner1[1]-corner2[1])
        x_diff,y_diff = abs(corner1[0]-corner2[0]),abs(corner1[1]-corner2[1])
        if diag_index == 0.0:
            if x_diff == 0.0 and y_diff == 0.0:
                return e1c * e2c * 4/9
            elif (x_diff == 1.0 and y_diff == 0.0) or (x_diff == 0.0 and y_diff == 1.0):
                return e1c * e2c * 1/9
        elif abs(diag_index) == 1.0:
            return e1c * e2c * 1/36
        else:
            return 0.0
    else:
        a,b = min(corner1[0]-1.,corner2[0]-1.),max(corner1[0]+1.,corner2[0]+1.)
        c,d = min(corner1[1]-1.,corner2[1]-1.),max(corner1[1]+1.,corner2[1]+1.)
        func = lambda x,y:phi(x,y,corner1)*phi(x,y,corner2)*df7.e1(x,y)*df7.e2(x,y)
        return dblquad(func, c, d, a, b)[0]

def a_form(corner1, corner2, method='fast'):
    if method=='fast':
        xc,yc = (corner1[0]+corner2[0])/2,(corner1[1]+corner2[1])/2
        a11c, a12c, a22c = df7.a11(xc,yc), df7.a12(xc,yc), df7.a22(xc,yc)
        diag_index = (corner1[0]-corner2[0])*(corner1[1]-corner2[1])
        x_diff,y_diff = abs(corner1[0]-corner2[0]),abs(corner1[1]-corner2[1])
        if diag_index == 0.0:
            if x_diff == 0.0:
                if y_diff == 0.0:
                    return a11c * 4/3 + a22c * 4/3
                elif y_diff == 1.0:
                    return a11c * 1/3 - a22c * 2/3
                else:
                    return 0.0
            elif x_diff == 1.0:
                return - a11c * 2/3 + a22c * 1/3
            else:
                return 0.0
        elif diag_index == 1.0:
            return - a11c * 1/6 - a12c * 1/2 - a22c * 1/6
        elif diag_index == -1.0:
            return - a11c * 1/6 + a12c * 1/2 - a22c * 1/6
        else:
            return 0.0
    else:
        a,b = min(corner1[0]-1.,corner2[0]-1.),max(corner1[0]+1.,corner2[0]+1.)
        c,d = min(corner1[1]-1.,corner2[1]-1.),max(corner1[1]+1.,corner2[1]+1.)
        func1 = lambda x,y:phi(x,y,corner1)*phi(x,y,corner2)*df7.e1(x,y)*df7.e2(x,y)
        term1 = dblquad(func1, c, d, a, b)
        func2 = lambda x,y:df7.a11(x,y)*dx_phi(x,y,corner1)*dx_phi(x,y,corner2)+\
            df7.a12(x,y)*(dx_phi(x,y,corner1)*dy_phi(x,y,corner2)+dy_phi(x,y,corner1)*dx_phi(x,y,corner2))+\
                df7.a22(x,y)*dy_phi(x,y,corner1)*dy_phi(x,y,corner2)
        term2 = dblquad(func2, c, d, a, b)
        return (term1[0] + term2[0]) #,term1[1] + term2[1])

# Function to array
def func2array(func,bounds,nx,ny,*args):
    a,b = bounds[0][0],bounds[0][1]
    c,d = bounds[1][0],bounds[1][1]
    x,y = np.linspace(a, b, nx), np.linspace(c, d, ny)
    l   = [[func(xi,yj,*args) for yj in y] for xi in x]
    return np.array(l)

def create_stack(mask):
    mc = df7.Mp(mask).T
    #mc = mask().T[1:,1:]
    global index_stack, stack_size
    stack_size = sum(sum(mc))
    index_stack=[]
    for i in range(nx):
        for j in range(ny):
            if mc[j,i]:
                k = j + i*ny
                index_stack.append(k)
    return index_stack

# Matrices
def construct_A(nx,ny,mask=None):
    N = nx * ny
    A = lil_matrix((N, N))
    if mask==None:
        for i in range(nx):
            for j in range(ny):
                k = j + i*ny
                A[k,k] = a_form([i,j], [i,j])
                if j>0: A[k,k-1] = a_form([i,j], [i,j-1])
                if j<ny-1: A[k,k+1] = a_form([i,j], [i,j+1])
                if i>0: A[k,k-ny] = a_form([i,j], [i-1,j])
                if i<nx-1: A[k,k+ny] = a_form([i,j], [i+1,j])
                if j>0 and i>0: A[k,k-ny-1] = a_form([i,j], [i-1,j-1])
                if j<ny-1 and i>0: A[k,k-ny+1] = a_form([i,j], [i-1,j+1])
                if j>0 and i<nx-1: A[k,k+ny-1] = a_form([i,j], [i+1,j-1])
                if j<ny-1 and i<nx-1: A[k,k+ny+1] = a_form([i,j], [i+1,j+1])
    else:
        mc = df7.Mp(mask).T
        #mc = mask().T[1:,1:]
        for i in range(nx):
            for j in range(ny):
                if mc[j,i]:
                    k = j + i*ny
                    A[k,k] = a_form([i,j], [i,j])
                    if j>0 and mc[j-1,i]: A[k,k-1] = a_form([i,j], [i,j-1])
                    if j<ny-1 and mc[j+1,i]: A[k,k+1] = a_form([i,j], [i,j+1])
                    if i>0 and  mc[j,i-1]: A[k,k-ny] = a_form([i,j], [i-1,j])
                    if i<nx-1 and mc[j,i+1]: A[k,k+ny] = a_form([i,j], [i+1,j])
                    if j>0 and i>0 and mc[j-1,i-1]: A[k,k-ny-1] = a_form([i,j], [i-1,j-1])
                    if j<ny-1 and i>0 and mc[j+1,i-1]: A[k,k-ny+1] = a_form([i,j], [i-1,j+1])
                    if j>0 and i<nx-1 and mc[j-1,i+1]: A[k,k+ny-1] = a_form([i,j], [i+1,j-1])
                    if j<ny-1 and i<nx-1 and mc[j+1,i+1]: A[k,k+ny+1] = a_form([i,j], [i+1,j+1])
    return A.tocsr()

def construct_B(nx,ny,mask=None):
    N = nx * ny
    B = lil_matrix((N, N))
    if mask==None:
        for i in range(nx):
            for j in range(ny):
                k = j + i*ny
                B[k,k] = b_form([i,j], [i,j])
                if j>0: B[k,k-1] = b_form([i,j], [i,j-1])
                if j<ny-1: B[k,k+1] = b_form([i,j], [i,j+1])
                if i>0: B[k,k-ny] = b_form([i,j], [i-1,j])
                if i<nx-1: B[k,k+ny] = b_form([i,j], [i+1,j])
                if j>0 and i>0: B[k,k-ny-1] = b_form([i,j], [i-1,j-1])
                if j<ny-1 and i>0: B[k,k-ny+1] = b_form([i,j], [i-1,j+1])
                if j>0 and i<nx-1: B[k,k+ny-1] = b_form([i,j], [i+1,j-1])
                if j<ny-1 and i<nx-1: B[k,k+ny+1] = b_form([i,j], [i+1,j+1])
    else:
        mc = df7.Mp(mask).T
        #mc = mask().T[1:,1:]
        for i in range(nx):
            for j in range(ny):
                if mc[j,i]:
                    k = j + i*ny
                    B[k,k] = b_form([i,j], [i,j])
                    if j>0 and mc[j-1,i]: B[k,k-1] = b_form([i,j], [i,j-1])
                    if j<ny-1 and mc[j+1,i]: B[k,k+1] = b_form([i,j], [i,j+1])
                    if i>0 and  mc[j,i-1]: B[k,k-ny] = b_form([i,j], [i-1,j])
                    if i<nx-1 and mc[j,i+1]: B[k,k+ny] = b_form([i,j], [i+1,j])
                    if j>0 and i>0 and mc[j-1,i-1]: B[k,k-ny-1] = b_form([i,j], [i-1,j-1])
                    if j<ny-1 and i>0 and mc[j+1,i-1]: B[k,k-ny+1] = b_form([i,j], [i-1,j+1])
                    if j>0 and i<nx-1 and mc[j-1,i+1]: B[k,k+ny-1] = b_form([i,j], [i+1,j-1])
                    if j<ny-1 and i<nx-1 and mc[j+1,i+1]: B[k,k+ny+1] = b_form([i,j], [i+1,j+1])
    return B.tocsr()


def FEscheme(u0,mask=None,diag_M=False,solver='spsolve',nb_iter=S):
    global K,M
    K = construct_A(nx,ny,mask=mask)
    M = construct_B(nx,ny,mask=mask)
    C = M+K
    u = np.copy(u0)
    if diag_M:
        # Compute the row sums
        row_sums = M.sum(axis=1).A1  # .A1 converts it to a 1D array
        # Create a diagonal matrix with these sums
        M = diags(row_sums)
        M = M.tocsr()
    if mask == None:
        for m in range(S):
            v = M.dot(u)
            if solver=='conjgrad': u = as1.conjgrad(C, v, v, nature=0)
            else: u = spsolve(C,v)
    else:
        create_stack(mask)
        # Create the A_small matrix
        K_small = K[index_stack, :][:, index_stack]
        M_small = M[index_stack, :][:, index_stack]
        # Convert to csr_matrix for efficiency if needed
        K_small = csr_matrix(K_small)
        M_small = csr_matrix(M_small)
        C_small = K_small + M_small
        # Extract the subvector
        u_small = u[index_stack]
        for m in range(S):
            v_small = M_small.dot(u_small)
            if solver=='conjgrad': u_small = as1.conjgrad(C_small, v_small, v_small, nature=0)
            else: u_small = spsolve(C_small,v_small)
        u[:] = 0
        u[index_stack] = u_small
    return u
    
def FEiteration(u):
    v = M.dot(u)
    b = spsolve(M+K,v)
    return b

## Autoadjoint test
def W_operator(v0):
    # W0 = (df7.E1E2) * W0
    W0 = 9/4 * (M @ v0)
    return W0

def test_op(oper,mu=0,sig=1,mask=None,nb_iter=S):
    u = np.random.normal(mu,sig,size = nx*ny)
    v = np.random.normal(mu,sig,size = nx*ny)
    return u.dot(W_operator(oper(v,mask=mask,nb_iter=nb_iter))) -\
        v.dot(W_operator(oper(v,mask=mask,nb_iter=nb_iter)))
        
def test_mat(A):
    u = np.random.normal(0,10,size = nx*ny)
    v = np.random.normal(0,10,size = nx*ny)
    K = construct_A(nx,ny)
    M = construct_B(nx,ny)
    W = 9/4 * M
    p1, p2 = u, v
    for s in range(S):
        s1 = spsolve(M + K, M @ p1)
        s2 = spsolve(M + K, M @ p2)
        p1, p2 = s1, s2
    s1, s2 = W @ p1, W @ p2
    return s1.dot(v) - s2.dot(u)
    return A.dot(u).dot(v) - A.dot(v).dot(u) 

create_stack(df7.mask1)