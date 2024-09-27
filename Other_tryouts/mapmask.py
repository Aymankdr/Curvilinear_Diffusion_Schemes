#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:21:11 2024

@author: khaddari
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def colormap(v, mask, nx, ny, Tcool, Thot, M, dt):
    # Reshape the input array
    w = v.copy()
    w.resize((nx, ny))
    
    # Create the figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('X / dx')  # Add x-axis label
    ax.set_ylabel('Y / dy')  # Add y-axis label
    
    # Set ticks if necessary
    if nx > 6:
        ax.set_xticks(np.arange(0, nx, nx // 6))
    if ny > 6:
        ax.set_yticks(np.arange(0, ny, ny // 6))
    
    # Create a masked array where the mask is True
    masked_w = np.ma.masked_where(mask, w)
    
    # Create a custom colormap that includes blue for the masked areas
    cmap = plt.get_cmap('hot')
    cmap.set_bad(color='blue')
    
    # Plot the data with the colormap and masked areas
    im = ax.imshow(masked_w.T[::-1], cmap=cmap, vmin=Tcool, vmax=Thot)
    ax.set_title('{:.1f} iter'.format(M * dt))
    
    # Create a colorbar
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    cbar_ax.set_xlabel('$T$ / K', labelpad=20)
    fig.colorbar(im, cax=cbar_ax)
    
    # Show the plot
    plt.show()
    
# Example parameters
nx, ny = 10, 10
Tcool, Thot = 0, 100
M, dt = 1, 0.1

# Example data
v = np.random.rand(nx * ny) * (Thot - Tcool) + Tcool
mask = np.random.choice([True, False], size=(nx, ny), p=[0.1, 0.9])  # Random mask for demonstration

# Call the colormap function
colormap(v, mask, nx, ny, Tcool, Thot, M, dt)

