#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 10:06:12 2024

@author: khaddari
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, eye
from scipy.sparse.linalg import spsolve, eigs, norm
from scipy.integrate import dblquad
import diffusion7 as df7
import diffusion9 as df9
import diffusion11 as df11

# plate size, mm
Lx, Ly = df7.Lx, df7.Ly
# intervals in x-, y- directions, mm
nx, ny = df7.nx, df7.ny
dx, dy = Lx/nx, Ly/ny
dx2, dy2 = dx*dx, dy*dy

# Virtual time step
ds = df7.dt

## Constants
M = df7.M


BC = df9.BC

# Diffusion term
D0 = df9.construct_D(BC)

# Forced matrix symmetry
D  = (D0 + D0.T)/2
D  = D0
#D  = df11.create_block_matrix()

def implicit_scheme(u):
    b = u.copy()
    L = ds*D + eye(nx*ny).tocsr()
    for m in range(M):
        b = spsolve(L,b)
    return b

def explicit_scheme(u):
    b = u.copy()
    L = eye(nx*ny).tocsr() - ds*D
    for m in range(M):
        b = L.dot(b)
    return b

# Output
vv=implicit_scheme(df7.u0)
df7.colormap(vv)

# Spectre of D
spD  = eigs(D,k=nx*ny-2)[0]
rhoD = max(np.abs(spD))
minspD = min(np.real(spD))
errAutoAdj = norm(D0 - D0.T)
errAutoAdjRelative = errAutoAdj/norm(D0)

print("rho (D) = ", rhoD,'\n',"Minimal real part of spectre: ", minspD)
print("| D - D^T |/|D| = ", errAutoAdjRelative)