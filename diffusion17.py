#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:45:49 2024

@author: khaddari
"""

# Usual packages
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as alg
from diffusion14 import spherical, colormap
import matplotlib.pyplot as plt

# Data
L = Lx = Ly = 1.0
Nl = 100
T = 1e-1
S = 10
dt = T/S
dx = dy = 2*L/Nl

kxx, kyy, kxy = 3.6, 3.6, 1.0
kdet          = kxx * kyy - kxy * kxy
kinv          = np.array([[kyy,-kxy],[-kxy,kxx]]) / kdet

lbdx, lbdy = 0.5 * np.pi / Lx, 0.5 * np.pi / Ly
lbdt = lbdx ** 2 * kxx + lbdy ** 2 * kyy
lbd  = 0 #lbdx ** 2 * kxx + lbdy ** 2 * kyy

domain1 = [[-Lx,Lx],[-Ly,Ly]]

# Fields
def k11(x,y): return kxx
def k22(x,y): return kyy
def k12(x,y): return kxy

def e1(x,y): return 1.0
def e2(x,y): return 1.0
def gamma(x,y): return 1.0 # e1 * e2

def a11(x,y): return k11(x,y) * e2(x,y) / e1(x,y)
def a22(x,y): return k22(x,y) * e1(x,y) / e2(x,y)
def a12(x,y): return k12(x,y)

def mu(t,x,y): return -2 * lbdx * lbdy * kxy \
    * np.sin(lbdx * x) * np.sin(lbdy * y) * np.exp(-lbdt * t) \
    + lbd * np.cos(lbdx * x) * np.cos(lbdy * y)
    
def eta_ex(t,x,y): return np.cos(lbdx * x) * np.cos(lbdy * y) * np.exp(-lbdt * t)

def eta_gauss(t,x,y,x0=0.,y0=0.): 
    z = np.array([x-x0,y-y0])
    g = (kinv @ z) @ z / (4 * t)
    return (1/(4*np.pi*t*np.sqrt(kdet))) * np.exp(-g)

# Functions
def meshing_domain(Nx,Ny,domain=[[0,1],[0,1]]):
    x_inf, x_sup, y_inf, y_sup = domain[0][0], domain[0][1], domain[1][0], domain[1][1]
    x_length, y_length = x_sup - x_inf, y_sup - y_inf
    dx, dy = x_length/Nx, y_length/Ny
    # i,j
    x_nodes, y_nodes =\
        np.linspace(x_inf+0.5*dx,x_sup-0.5*dx,Nx), np.linspace(y_inf+0.5*dy,y_sup-0.5*dy,Ny)
    # i+1/2,j+1/2
    x_mesh, y_mesh =\
    np.linspace(x_inf,x_sup,Nx+1), np.linspace(y_inf,y_sup,Ny+1)
    return x_mesh,y_mesh, x_nodes, y_nodes

def build_A(mesh):
    x_mesh, y_mesh, x_nodes, y_nodes = mesh
    A11 = np.array([[a11(x,y) for y in y_nodes] for x in x_mesh])
    A22 = np.array([[a22(x,y) for y in y_mesh] for x in x_nodes])
    A12 = np.array([[a12(x,y) for y in y_mesh] for x in x_mesh])
    return A11, A22, A12

def build_G(mesh):
    x_mesh, y_mesh, x_nodes, y_nodes = mesh
    G = np.array([[gamma(x,y) for y in y_nodes] for x in x_nodes])
    return G

def diff_mesh(mesh):
    x_mesh, y_mesh, x_nodes, y_nodes = mesh
    dx, dy = x_mesh[1]-x_mesh[0], y_mesh[1]-y_mesh[0]
    return dx, dy

def build_Kdiags(A,mesh):
    A11,A22,A12 = build_A(mesh)
    dx,dy       = diff_mesh(mesh)
    o           = np.zeros(1)
    Kcore = (dy/dx) * (A11[1:,:] + A11[:-1,:]) +\
            (dx/dy) * (A22[:,1:] + A22[:,:-1]) +\
            0.5 * (A12[:-1,:-1] + A12[1:,1:] - A12[1:,:-1] - A12[:-1,1:])
    Kcore = Kcore.flatten()
    Kiipm = (-dy/dx) * A11[1:-1,:]
    Kiipm = Kiipm.flatten()
    Kjjpm = (-dx/dy) * A22
    Kjjpm[:,0]  *= 0.0
    Kjjpm = Kjjpm[:,:-1]
    Kjjpm = Kjjpm.flatten()
    Kjjpm = np.concatenate((Kjjpm, o))
    Kijpm = 0.5 * A12[1:-1,:]
    Kijpm[:,0]  *= 0.0
    Kijpm = Kijpm[:,:-1]
    Kijpm = Kijpm.flatten()
    Kijpm = np.concatenate((Kijpm, o))
    return Kcore, Kiipm, Kjjpm, Kijpm

def build_K(Kdiags, Nx, Ny):
    N = Nx * Ny
    Kcore, Kiipm, Kjjpm, Kijpm = Kdiags
    #print('Length : ',len(Kijpm))
    #print('Length : ',N - Ny - 1)
    K = sp.diags([-Kijpm[1:-1], Kiipm, Kijpm, Kjjpm[1:-1],\
    Kcore, Kjjpm[1:-1], Kijpm, Kiipm, -Kijpm[1:-1]],
              [-Ny-1,-Ny,-Ny+1,-1, 0, 1,Ny-1,Ny,Ny+1], format = 'coo')
    # Boundary conditions (Dirichlet)
    '''data = np.concatenate((K_diags.data, BC_data))
    row = np.concatenate((K_diags.row, BC_row))
    col = np.concatenate((K_diags.col, BC_col))
    K = sp.csc_matrix((data, (row, col)), shape = (N, N))'''
    return K

def build_M(G, mesh):
    dx, dy = diff_mesh(mesh)
    M = sp.diags(dx * dy * G.flatten(), format = 'coo')
    return M

def build_y(t, mesh):
    x_mesh, y_mesh, x_nodes, y_nodes = mesh
    y = np.array([[mu(t,x,y) for y in y_nodes] for x in x_nodes]).flatten()
    return y

def build_u0(mesh):
    x_mesh, y_mesh, x_nodes, y_nodes = mesh
    u0 = np.array([[eta_ex(0.0,x,y) for y in y_nodes] for x in x_nodes]).flatten()
    return u0

def build_u(t, mesh):
    x_mesh, y_mesh, x_nodes, y_nodes = mesh
    u0 = np.array([[eta_ex(t,x,y) for y in y_nodes] for x in x_nodes]).flatten()
    return u0

def find_u(T, S, Nx, Ny, domain):
    mesh   = meshing_domain(Nx, Ny, domain = domain)
    A      = build_A(mesh)
    G      = build_G(mesh)
    Kdiags = build_Kdiags(A, mesh)
    K      = build_K(Kdiags, Nx, Ny)
    M      = build_M(G, mesh)
    dt     = T/S if S > 0 else 0
    t      = 0.0
    C      = M + dt * K
    u0     = build_u0(mesh)
    up     = np.copy(u0)
    uex    = build_u(T, mesh)
    for s in range(S):
        t  += dt
        y   = build_y(t, mesh)
        ssh = M @ (up + dt * y)
        u,e = alg.cg(C,ssh,up,tol=1e-50,atol=1e-12)
        if e != 0: print("Convergence issue at iteration: ", s)
        up  = np.copy(u)
    return up, uex

## Dirac diffusion
def build_ug(t, mesh, x0=0., y0=0.):
    x_mesh, y_mesh, x_nodes, y_nodes = mesh
    ug = np.array([[eta_gauss(t,x,y,x0,y0) for y in y_nodes] for x in x_nodes]).flatten()
    return ug

def Dirac(x,y,mesh):
    x_mesh, y_mesh, x_nodes, y_nodes = mesh
    dx,dy = diff_mesh(mesh)
    xmin, xmax, ymin, ymax = x_mesh[0], x_mesh[-1], y_mesh[0], y_mesh[-1]
    xhat, yhat = (x-xmin)/(xmax-xmin), (y-ymin)/(ymax-ymin)
    Nx, Ny = len(x_nodes), len(y_nodes)
    i, j = int(Nx * xhat - 0.5), int(Ny * yhat - 0.5)
    u0   = np.zeros(Nx * Ny)
    u0[j + Ny * i] = 1/(dx*dy)
    return u0

def diffuse_Dirac(x, y, T, S, Nx, Ny, domain):
    mesh   = meshing_domain(Nx, Ny, domain = domain)
    A      = build_A(mesh)
    G      = build_G(mesh)
    Kdiags = build_Kdiags(A, mesh)
    K      = build_K(Kdiags, Nx, Ny)
    M      = build_M(G, mesh)
    dt     = T/S if S > 0 else 0
    t      = 0.0
    C      = M + dt * K
    ug     = build_ug(T, mesh, x, y)
    u0     = Dirac(x, y, mesh)
    up     = np.copy(u0)
    for s in range(S):
        t  += dt
        ssh = M @ (up)
        u,e = alg.cg(C,ssh,up,tol=1e-50,atol=1e-12)
        if e != 0: print("Convergence issue at iteration: ", s)
        up  = np.copy(u)
    return up, ug

def precision_graph(T,domain,S,Nl):
    if type(S) is list:
        assert type(Nl) is int
        H,L = [],[]
        for s in S:
            u,uex=find_u(T,s,Nl,Nl,domain)
            err  = (u - uex) @ (u - uex)
            dt   = T/s
            H.append(dt)
            L.append(err)
        return H,L
    if type(Nl) is list:
        assert type(S) is int 
        H,L,L1,L2 = [],[], [], []
        for nl in Nl:
            #u,uex=find_u(T,S,nl,nl,domain)
            u,uex=diffuse_Dirac(0,0,T,S,nl,nl,domain)
            h    = 1/nl
            err  = (u - uex) @ (u - uex)
            H.append(h)
            L.append(err)
            L1.append(h*err)
            L2.append(h*h*err)
        return H,L,L1,L2

