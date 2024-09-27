#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:03:52 2024

@author: khaddari
"""

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.special import sph_harm

def spherical(vv,nx,ny,mask=None,theta_secu=0,scale='automatic'):
    phi = np.linspace(0, 2 * np.pi, nx)
    theta = np.linspace(theta_secu, np.pi - theta_secu, ny)
    theta, phi = np.meshgrid(theta, phi)
    
    # The Cartesian coordinates of the unit sphere
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    w = np.copy(vv)
    w.resize((nx,ny))
    
    # Apply the mask if provided
    if mask is not None:
        masked_w = np.ma.masked_where(mask[1:-1,1:-1], w)
        # Fill the spherical grid
        fcolors = masked_w[::-1]
    else:
        # Fill the spherical grid
        fcolors = w[::-1]
    
    if scale=='automatic':
        fmax, fmin = fcolors.max(), fcolors.min()
        if not fmax == fmin: fcolors = (fcolors - fmin) / (fmax - fmin) 
        else: fcolors = fcolors - fmin
    else:
        assert type(scale) is tuple 
        fmin, fmax = scale[0], scale[1]
        if not fmax == fmin: fcolors = (fcolors - fmin) / (fmax - fmin) 
        else: fcolors = fcolors - fmin
    
    # Set the aspect ratio to 1 so our sphere looks spherical
    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface with the colormap 'viridis'
    cmap = cm.viridis
    cmap.set_bad(color='red')  # Set color for masked values
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cmap(fcolors))

    # Add a colorbar which maps values to colors
    norm = colors.Normalize(vmin=fmin, vmax=fmax)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(fcolors)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)

    # Turn off the axis planes
    ax.set_axis_off()

    '''
    # Calculate the spherical harmonic Y(l,m) and normalize to [0,1]
    fcolors = w.T[::-1]
    #sph_harm(m, l, theta, phi).real
    fmax, fmin = fcolors.max(), fcolors.min()
    fcolors = (fcolors - fmin) / (fmax - fmin)
    
    # Set the aspect ratio to 1 so our sphere looks spherical
    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface with the colormap 'viridis'
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cm.viridis(fcolors))
    
    # Add a colorbar which maps values to colors
    norm = colors.Normalize(vmin=fmin, vmax=fmax)
    mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    mappable.set_array(fcolors)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
    '''

    ''' SNIPET FROM FUNCTION THAT ADDS A MASK
    masked_w = np.ma.masked_where(mask()[1:-1,1:-1], w)
    im = ax.imshow(w.T[::-1], cmap=plt.get_cmap(), vmin=Tcool, vmax=Thot)
    cmap.set_bad(color='red')
    # Plot the data with the colormap and masked areas
    im = ax.imshow(masked_w.T[::-1], cmap=cmap, vmin=Tcool, vmax=Thot)
    ax.set_title('{:.1f} iter'.format(nb_iter * dt))
    '''

def colormap(v,nx,ny,nb_iter,mask=None,scale='automatic'):
    w = v.copy()
    w.resize((nx, ny))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('X / dx')  # Add x-axis label
    ax.set_ylabel('Y / dy')  # Add y-axis label
    if nx>6:ax.set_xticks(np.arange(0, nx, nx//6))
    if ny>6:ax.set_yticks(np.arange(0, ny, ny//6))
    if scale == 'automatic':
        im = ax.imshow(w.T[::-1], cmap=plt.get_cmap('hot'), vmin=0, vmax=v.max())
    else:
        fmin, fmax = scale[0], scale[1]
        im = ax.imshow(w.T[::-1], cmap=plt.get_cmap('hot'), vmin=fmin, vmax=fmax)
    is_null = (v.min() == v.max())

    # Create a masked array where the mask is True
    if not mask is None:
        masked_w = np.ma.masked_where(mask[1:-1,1:-1], w)
        cmap = plt.get_cmap('hot')
        cmap.set_bad(color='cyan')
        # Plot the data with the colormap and masked areas
        if is_null:
            im = ax.imshow(masked_w.T[::-1], cmap=cmap, vmin=0, vmax=1)
        else:
            im = ax.imshow(masked_w.T[::-1], cmap=cmap, vmin=0, vmax=1)
        ax.set_title('{:.1f} iter'.format(nb_iter * 1.0))
        # Create a colorbar
        cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.show()
    else:
        ax.set_title('{:.1f} iter'.format(nb_iter * 1.0))
        cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.show()