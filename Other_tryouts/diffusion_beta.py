#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:36:28 2024

@author: khaddari
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# Define parameters
L = 1.0        # Length of the domain
N = 100        # Number of spatial points
dx = L / (N - 1)  # Spatial step size
dt = 0.01      # Time step size
T = 1.0        # Total time
alpha = 1.0    # Diffusivity (constant in this example; can be a function of x)

# Grid points
x = np.linspace(0, L, N)

# Variable diffusivity example (if needed)
def diffusivity(x):
    return 1 + 0. * np.sin(2 * np.pi * x / L)

# Construct the tridiagonal matrix
def construct_matrix(N, dx, dt, diffusivity_func):
    main_diag = np.zeros(N)
    off_diag = np.zeros(N-1)

    for i in range(1, N-1):
        diff_ip = diffusivity_func(x[i] + dx / 2)
        diff_im = diffusivity_func(x[i] - dx / 2)
        main_diag[i] = 1 + dt * (diff_ip + diff_im) / dx**2
        off_diag[i-1] = -dt * diff_im / dx**2
        off_diag[i] = -dt * diff_ip / dx**2

    # Boundary conditions
    main_diag[0] = main_diag[-1] = 1  # Dirichlet boundary conditions
    off_diag[0] = off_diag[-1] = 0   # No off-diagonal terms at boundaries

    return diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr')

# Construct the matrix
A = construct_matrix(N, dx, dt, diffusivity)

# Initial condition (example)
u = np.sin(np.pi * x)

# Boundary conditions
u[0] = u[-1] = 0  # Assuming Dirichlet boundary conditions at both ends

# Time-stepping loop
t = 0
while t < T:
    # Right-hand side
    b = u.copy()

    # Solve the linear system
    u_new = spsolve(A, b)

    # Update the solution
    u = u_new

    # Update time
    t += dt

# Final solution
print("Final solution:", u)

plt.plot(x,u)