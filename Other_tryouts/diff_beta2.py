#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:49:54 2024

@author: khaddari
"""

import numpy as np
from scipy.sparse import diags, kron, identity
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# Define parameters
Lx, Ly = 1.0, 1.0  # Length of the domain in x and y directions
Nx, Ny = 50, 50    # Number of spatial points in x and y directions
dx = Lx / (Nx - 1)  # Spatial step size in x direction
dy = Ly / (Ny - 1)  # Spatial step size in y direction
dt = 0.01          # Time step size
T = 1.0            # Total time

# Grid points
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Variable diffusivity example (if needed)
def diffusivity(x, y):
    return 1 + 0.5 * np.sin(2 * np.pi * x / Lx) * np.sin(2 * np.pi * y / Ly)

# Construct the tridiagonal matrix
def construct_matrix(Nx, Ny, dx, dy, dt, diffusivity_func):
    main_diag_x = np.zeros(Nx)
    off_diag_x = np.zeros(Nx-1)
    
    main_diag_y = np.zeros(Ny)
    off_diag_y = np.zeros(Ny-1)
    
    for i in range(1, Nx-1):
        diff_ip = diffusivity_func(x[i] + dx / 2, y[i])
        diff_im = diffusivity_func(x[i] - dx / 2, y[i])
        main_diag_x[i] = 1 + dt * (diff_ip + diff_im) / dx**2
        off_diag_x[i-1] = -dt * diff_im / dx**2
        off_diag_x[i] = -dt * diff_ip / dx**2

    for j in range(1, Ny-1):
        diff_jp = diffusivity_func(x[j], y[j] + dy / 2)
        diff_jm = diffusivity_func(x[j], y[j] - dy / 2)
        main_diag_y[j] = 1 + dt * (diff_jp + diff_jm) / dy**2
        off_diag_y[j-1] = -dt * diff_jm / dy**2
        off_diag_y[j] = -dt * diff_jp / dy**2

    # Boundary conditions
    main_diag_x[0] = main_diag_x[-1] = 1  # Dirichlet boundary conditions
    main_diag_y[0] = main_diag_y[-1] = 1  # Dirichlet boundary conditions
    off_diag_x[0] = off_diag_x[-1] = 0    # No off-diagonal terms at boundaries
    off_diag_y[0] = off_diag_y[-1] = 0    # No off-diagonal terms at boundaries
    
    # Construct the 1D tridiagonal matrices
    Tx = diags([off_diag_x, main_diag_x, off_diag_x], [-1, 0, 1], format='csr')
    Ty = diags([off_diag_y, main_diag_y, off_diag_y], [-1, 0, 1], format='csr')
    
    # Construct the 2D matrix using Kronecker products
    Ix = identity(Nx)
    Iy = identity(Ny)
    A = kron(Iy, Tx) + kron(Ty, Ix)
    
    return A

# Construct the matrix
A = construct_matrix(Nx, Ny, dx, dy, dt, diffusivity)

# Initial condition (example)
u = np.sin(np.pi * X) * np.sin(np.pi * Y)

# Boundary conditions
u[:, 0] = u[:, -1] = 0  # Dirichlet boundary conditions in x direction
u[0, :] = u[-1, :] = 0  # Dirichlet boundary conditions in y direction

# Flatten the 2D array to a 1D array for the solver
u_flat = u.flatten()

# Time-stepping loop
t = 0
while t < T:
    # Right-hand side
    b = u_flat.copy()

    # Solve the linear system
    u_new_flat = spsolve(A, b)

    # Update the solution
    u_flat = u_new_flat

    # Update time
    t += dt

# Reshape the solution back to 2D
u_final = u_flat.reshape((Ny, Nx))

# Final solution
plt.imshow(u_final, extent=[0, Lx, 0, Ly], origin='lower', cmap='hot')
plt.colorbar()
plt.title('Final solution')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
