#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:18:35 2024

@author: khaddari
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, eye
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

# Construct the sparse matrix
def construct_matrix(Nx, Ny, dx, dy, dt, diffusivity_func):
    N = Nx * Ny
    A = lil_matrix((N, N))
    
    def idx(i, j):
        return i * Nx + j

    for i in range(Ny):
        for j in range(Nx):
            index = idx(i, j)
            if i == 0 or i == Ny-1 or j == 0 or j == Nx-1:
                A[index, index] = 1  # Boundary conditions
            else:
                diff_ip = diffusivity_func(x[j] + dx / 2, y[i])
                diff_im = diffusivity_func(x[j] - dx / 2, y[i])
                diff_jp = diffusivity_func(x[j], y[i] + dy / 2)
                diff_jm = diffusivity_func(x[j], y[i] - dy / 2)
                
                A[index, index] = 1 + dt * (diff_ip + diff_im) / dx**2 + dt * (diff_jp + diff_jm) / dy**2
                A[index, idx(i, j-1)] = -dt * diff_im / dx**2
                A[index, idx(i, j+1)] = -dt * diff_ip / dx**2
                A[index, idx(i-1, j)] = -dt * diff_jm / dy**2
                A[index, idx(i+1, j)] = -dt * diff_jp / dy**2

    return A.tocsr()

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
